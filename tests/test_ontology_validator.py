"""
OntologyValidator 单元测试
=========================
覆盖：实体抽取、类型推断、写时验证、读时验证、边界情况。

注意：中文实体抽取的"回退"策略因中文无词边界而存在固有噪声。
可靠路径：英文命名实体、中文组织名（公司/大学等后缀）、
          中文人名（"出生于/生于"等动词触发）。
"""
from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pytest

from core.ontology_validator import (
    OntologyValidator,
    OntologyConfig,
    ValidationResult,
    ReadValidationResult,
    ONTOLOGY_TYPES,
)


# ─── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def validator() -> OntologyValidator:
    """无Kuzu连接的纯逻辑验证器（写时验证降级为pass）。"""
    return OntologyValidator(config=OntologyConfig(enabled=True))


@pytest.fixture
def validator_disabled() -> OntologyValidator:
    """禁用的验证器。"""
    return OntologyValidator(config=OntologyConfig(enabled=False))


@pytest.fixture
def kuzu_validator(kuzu_store) -> OntologyValidator:
    """带Kuzu的验证器（可做真正的Cypher矛盾检测）。"""
    return OntologyValidator(
        kuzu_store=kuzu_store,
        config=OntologyConfig(enabled=True, reject_on_contradiction=True),
    )


# ─── Helpers ──────────────────────────────────────────────────


def _make_result_dict(
    content: str,
    score: float = 0.8,
    tau: float = 0.6,
    trust: float = 0.7,
    ep_id: str = "",
) -> dict[str, Any]:
    return {
        "id": ep_id or str(uuid.uuid4()),
        "score": score,
        "tau_value": tau,
        "trust_score": trust,
        "content": content,
    }


def _create_episode_with_ontology(kuzu_store, content: str, ontology_type: str,
                                   trust: float = 0.8) -> str:
    """用 query_cypher 创建含额外属性的节点。"""
    ep_id = str(uuid.uuid4())
    kuzu_store.query_cypher(
        "CREATE (e:EpisodeNode {id: $id, content: $content, created_at: $t, "
        "tau_initial: 1.0, tau_value: 0.6, source: 'test', "
        "trust_score: $trust, ontology_type: $otype})",
        {"id": ep_id, "content": content, "t": 1.0, "trust": trust, "otype": ontology_type},
    )
    return ep_id


# ─── 实体抽取测试 ────────────────────────────────────────────


class TestExtractEntities:
    """_extract_entities 的边界覆盖。"""

    @pytest.mark.parametrize("text,expected_any", [
        # 英文命名实体（可靠路径：re.ASCII \b）
        ("Elon Musk is the CEO of Tesla", "Elon Musk"),
        ("John lives in Beijing", "John"),
        ("Neural Networks are the future", "Neural Networks"),
        ("I met Sam Altman yesterday", "Sam Altman"),
        ("Sam Altman访问上海总部", "Sam Altman"),
        # 中文组织名（可靠路径：公司/大学后缀）
        ("华为公司发布新产品", "华为公司"),
        ("清华大学是顶级学府", "清华大学"),
        ("北京大学位于海淀", "北京大学"),
        ("中国银行提供贷款", "中国银行"),
        # 中文人名 — 动词触发（可靠路径）
        ("毛泽东出生于1893年", "毛泽东"),
        ("周恩来生于1898年", "周恩来"),
        ("牛顿出生于1643年", "牛顿"),
        # 中英混合
        ("特斯拉CEO Elon Musk", "特斯拉"),
        ("阿里巴巴集团CEO 张勇", "阿里巴巴集团"),
    ])
    def test_extract_entities_finds_expected(self, validator: OntologyValidator,
                                              text: str, expected_any: str):
        """各层级正则应正确捕获目标实体。"""
        entities = validator._extract_entities(text)
        assert expected_any in entities, (
            f"Expected '{expected_any}' in entities for text='{text}', got {entities}"
        )

    @pytest.mark.parametrize("text,expected_count", [
        ("", 0),
        ("hello world", 0),
        ("this is a test", 0),
        ("nothing special", 0),
        ("今天天气不错", 0),
        ("我吃了一个苹果", 0),
    ])
    def test_no_entities(self, validator: OntologyValidator, text: str, expected_count: int):
        """无实体时应返回空列表。"""
        entities = validator._extract_entities(text)
        assert len(entities) == expected_count

    def test_entities_deduplicated(self, validator: OntologyValidator):
        """重复的实体应去重。"""
        text = "华为公司成立多年，华为公司一直领先"
        entities = validator._extract_entities(text)
        assert entities.count("华为公司") == 1

    def test_stop_words_filtered(self, validator: OntologyValidator):
        """停用词不应出现在实体中。"""
        text = "我们可以进行一个测试"
        entities = validator._extract_entities(text)
        for stop_word in {"我们", "他们", "这个", "那个", "一个"}:
            assert stop_word not in entities, f"Stop word '{stop_word}' leaked into entities"

    def test_english_re_ascii_boundary(self, validator: OntologyValidator):
        """中英混排下英文实体边界应正确（\b 不应被Unicode中文干扰）。"""
        text = "Sam Altman访问阿里巴巴集团在上海的总部"
        entities = validator._extract_entities(text)
        assert "Sam Altman" in entities, f"Expected full name 'Sam Altman', got {entities}"


# ─── 本体类型推断测试 ──────────────────────────────────────


class TestClassifyOntologyType:

    def test_trigger_word_maps_type(self, validator: OntologyValidator):
        """文本含触发词时应正确映射本体类型。"""
        cases = [
            ("张三出生于1990年", "person_birth"),
            ("爱因斯坦死于1955年", "person_death"),
            ("苹果公司成立", "organization_founded"),
            ("全球变暖导致海平面上升", "scientific_claim"),
            ("会议将于2024年举行", "event_date"),
            ("巴黎位于法国", "location_fact"),
            ("两人之间的关系很好", "relationship"),
        ]
        for text, expected in cases:
            entities = validator._extract_entities(text)
            otype = validator._classify_ontology_type(text, entities)
            assert otype == expected, f"Expected {expected}, got {otype} for '{text}'"

    def test_no_trigger_returns_generic(self, validator: OntologyValidator):
        """无触发词时应返回 generic_fact。"""
        otype = validator._classify_ontology_type("今天天气不错", [])
        assert otype == "generic_fact"

    def test_case_insensitive_match(self, validator: OntologyValidator):
        """触发词匹配应大小写不敏感。"""
        otype = validator._classify_ontology_type("BIRTH", ["BIRTH"])
        assert otype == "person_birth"


# ─── 值提取测试 ──────────────────────────────────────────────


class TestExtractValues:

    @pytest.mark.parametrize("text,expected_year", [
        ("张三出生于1990年", "1990"),
        ("爱因斯坦死于1955年4月18日", "1955"),
        ("苹果公司成立于1976年", "1976"),
    ])
    def test_extract_year(self, validator: OntologyValidator, text: str, expected_year: str):
        """应能正确提取年份。"""
        values = validator._extract_values(text, "person_birth")
        assert values.get("year") == expected_year

    @pytest.mark.parametrize("text,expected_date", [
        ("会议于2024-03-15举行", "2024-03-15"),
        ("生于2024/03/15", "2024/03/15"),
    ])
    def test_extract_date(self, validator: OntologyValidator, text: str, expected_date: str):
        """应能正确提取日期。"""
        values = validator._extract_values(text, "event_date")
        assert values.get("date") == expected_date

    def test_no_year(self, validator: OntologyValidator):
        """无年份时返回空字典。"""
        values = validator._extract_values("张三喜欢音乐", "generic_fact")
        assert values == {}


# ─── 写时验证测试 ─────────────────────────────────────────────


class TestWriteValidate:

    def test_passes_when_no_kuzu(self, validator: OntologyValidator):
        """无Kuzu连接时应直接返回passed=True。"""
        result = validator.write_validate("张三出生于1990年")
        assert result.passed is True

    def test_disabled_returns_pass(self, validator_disabled: OntologyValidator):
        """禁用时write_validate应返回passed=True。"""
        result = validator_disabled.write_validate("张三出生于1990年")
        assert result.passed is True

    def test_empty_content(self, validator: OntologyValidator):
        """空文本应正常通过。"""
        result = validator.write_validate("")
        assert result.passed is True

    def test_no_entity_text_passes(self, kuzu_validator: OntologyValidator, kuzu_store):
        """无实体的文本不应触发矛盾检测。"""
        result = kuzu_validator.write_validate("今天天气不错")
        assert result.passed is True
        assert result.conflict_count == 0

    @pytest.mark.skip(reason="需要真实 GraphLite 引擎（mock 无法测 Cypher 矛盾检测）")
    def test_same_entity_diff_value_detected(self, kuzu_validator: OntologyValidator, kuzu_store):
        """同一实体写入矛盾年份时应检测到冲突（置信度降低但默认不拒绝）。"""
        _create_episode_with_ontology(
            kuzu_store, "张三出生于1990年", ontology_type="person_birth")

        result = kuzu_validator.write_validate(
            "张三出生于2000年", episode_id=str(uuid.uuid4()))
        # 有矛盾但默认阈值(0.3)下不拒绝(graceful degradation)
        assert result.conflict_count >= 1
        assert result.ontology_type == "person_birth"
        assert result.entity_name == "张三"
        assert result.confidence < 1.0  # 置信度已降低
        assert result.confidence > 0.3  # 但未低于拒绝阈值
        assert result.passed is True   # 优雅降级：置信度降低但写操作通过

    def test_different_entity_no_conflict(self, kuzu_validator: OntologyValidator, kuzu_store):
        """不同实体不应触发冲突。"""
        _create_episode_with_ontology(
            kuzu_store, "张三出生于1990年", ontology_type="person_birth")

        result = kuzu_validator.write_validate(
            "李四出生于1990年", episode_id=str(uuid.uuid4()))
        assert result.passed is True
        assert result.conflict_count == 0
        assert result.entity_name != "张三"

    @pytest.mark.skip(reason="需要真实 GraphLite 引擎（mock 无法测 Cypher 矛盾检测）")
    def test_confidence_penalty_multiple_conflicts(self, kuzu_validator: OntologyValidator, kuzu_store):
        """多条矛盾事实应有置信度累积衰减。"""
        _create_episode_with_ontology(
            kuzu_store, "张三出生于1990年", ontology_type="person_birth", trust=0.8)
        _create_episode_with_ontology(
            kuzu_store, "张三出生于1985年", ontology_type="person_birth", trust=0.8)

        result = kuzu_validator.write_validate(
            "张三出生于2000年", episode_id=str(uuid.uuid4()))
        assert result.conflict_count >= 2
        # 2条 × conflict_penalty_factor(0.5) = 1.0 → confidence = max(0.1, 0) = 0.1
        assert result.confidence <= 0.5

    def test_graceful_on_kuzu_error(self, validator: OntologyValidator):
        """Kuzu查询出错时不应崩溃，应优雅降级。"""
        result = validator.write_validate("张三出生于1990年", episode_id="test")
        assert result.passed is True


# ─── 读时验证测试 ─────────────────────────────────────────────


class TestReadValidate:

    def test_empty_results(self, validator: OntologyValidator):
        """空结果列表应返回空。"""
        validated = validator.read_validate([])
        assert validated == []

    def test_disabled_returns_identity(self, validator_disabled: OntologyValidator):
        """禁用时分数不变。"""
        results = [_make_result_dict(content="张三出生于1990年", score=0.8)]
        validated = validator_disabled.read_validate(results)
        assert len(validated) == 1
        assert validated[0].adjusted_score == 0.8

    def test_no_kuzu_passthrough(self, validator: OntologyValidator):
        """无Kuzu应直接返回原始分数。"""
        results = [_make_result_dict(content="张三出生于1990年", score=0.8)]
        validated = validator.read_validate(results)
        assert validated[0].original_score == 0.8

    def test_sorts_by_adjusted_score(self, validator: OntologyValidator):
        """应按照调整后的分数降序排列。"""
        results = [
            _make_result_dict(content="张三出生于1990年", score=0.5),
            _make_result_dict(content="毛泽东出生于1893年", score=0.9),
            _make_result_dict(content="Elon Musk founded Tesla", score=0.7),
        ]
        validated = validator.read_validate(results)
        scores = [v.adjusted_score for v in validated]
        assert scores == sorted(scores, reverse=True), "Results not sorted descending"

    @pytest.mark.skip(reason="需要真实 GraphLite 引擎（mock 无法测 Cypher 矛盾检测）")
    def test_ontology_confidence_penalized_on_conflict(self, kuzu_validator: OntologyValidator, kuzu_store):
        """包含冲突实体的事实应被降低 ontology_confidence。"""
        ep1 = _create_episode_with_ontology(
            kuzu_store, "张三出生于1990年", ontology_type="person_birth")
        _create_episode_with_ontology(
            kuzu_store, "张三出生于2000年", ontology_type="person_birth")

        results = [
            _make_result_dict(content="张三出生于1990年", score=0.9, ep_id=ep1),
            _make_result_dict(content="Elon Musk founded Tesla", score=0.8),
        ]
        validated = kuzu_validator.read_validate(results)

        # 第一条涉及冲突实体，置信度下调
        v_zhang = [v for v in validated if v.episode_id == ep1][0]
        assert v_zhang.ontology_confidence < 1.0
        assert v_zhang.adjusted_score < v_zhang.original_score

        # 第二条无冲突
        v_other = [v for v in validated if v.episode_id != ep1][0]
        assert v_other.ontology_confidence == 0.5


# ─── 集成测试：写时验证 → 读时验证 全流程 ─────────────────


class TestWriteThenRead:

    @pytest.mark.skip(reason="需要真实 GraphLite 引擎（mock 无法测 Cypher 矛盾检测）")
    def test_write_read_integration(self, kuzu_validator: OntologyValidator, kuzu_store):
        """写时验证 → 写入 → 读时验证 完整闭环。"""
        # 1. 写入两条矛盾事实
        ep1 = _create_episode_with_ontology(
            kuzu_store, "张三出生于1990年", ontology_type="person_birth")
        ep2 = _create_episode_with_ontology(
            kuzu_store, "张三出生于2000年", ontology_type="person_birth")

        # 2. 写时验证新矛盾
        result = kuzu_validator.write_validate(
            "张三出生于2010年", episode_id=str(uuid.uuid4()))
        assert result.conflict_count >= 2

        # 3. 读时验证应惩罚已有矛盾
        results = [
            _make_result_dict(content="张三出生于1990年", score=0.9, ep_id=ep1),
            _make_result_dict(content="张三出生于2000年", score=0.9, ep_id=ep2),
            _make_result_dict(content="Elon Musk founded Tesla", score=0.7),
        ]
        validated = kuzu_validator.read_validate(results)

        # 张三相关事实应被惩罚
        zhang_results = [v for v in validated if v.episode_id in (ep1, ep2)]
        for v in zhang_results:
            assert v.ontology_confidence < 1.0
            assert v.adjusted_score < v.original_score

        # Elon Musk 不受影响
        musk_results = [v for v in validated if v.episode_id not in (ep1, ep2)]
        assert all(v.ontology_confidence == 0.5 for v in musk_results)
