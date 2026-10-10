"""本体暂存台账 —— 候选聚合验收测试（v6.27.0+ 聚合修复）

覆盖（对应任务书 AC1–AC6 的可判定断言）：
- AC1 模板聚合：语义模板相同、仅实例槽位（人名/序号）不同 → 1 个候选
- AC2 负样本保护：语义不同、仅字面相近的文本 → 必须产生不同键
- AC3 可回溯：聚合候选保留全部原始样本，聚合不得丢弃证据
- AC5 入口统一：validator 与 evolution 两个分类入口共用同一实现
- 数据迁移：v1 旧键台账加载后标 legacy、绝不静默丢弃
"""

from __future__ import annotations

import json

import pytest

from core.ontology_staging import (
    OntologyStaging,
    candidate_key,
    get_staging,
    reset_staging,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """每个用例独立台账文件 + 清洁单例 + 开关默认开。"""
    monkeypatch.delenv("SHM_ONTOLOGY_STAGING", raising=False)
    reset_staging(str(tmp_path / "staging.json"))
    yield
    reset_staging(str(tmp_path / "staging.json"))


_TEMPLATE = "{name} 认为量子浮点校准仪需要每季度做一次非线性漂移补偿，这是第{n}条独特观测"
_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Kate", "Leo",
]


# ─────────────────────── AC1 模板聚合 ───────────────────────

def test_ac1_template_instances_aggregate_to_single_candidate():
    """12 个同模板实例（人名 + 序号不同）→ 台账只占 1 个候选，count 累加。"""
    s = get_staging()
    for i, name in enumerate(_NAMES):
        s.record(_TEMPLATE.format(name=name, n=i))
    items = s.list_candidates()
    assert len(items) == 1
    assert items[0]["count"] == 12
    assert items[0]["sample_text"] == _TEMPLATE.format(name="Alice", n=0)


def test_ac1_aggregation_generalizes_beyond_the_12_samples():
    """聚合是真实语义分组，不是硬编码 12 个样本：换人名/换序号仍同键。"""
    base = candidate_key(_TEMPLATE.format(name="Alice", n=0))
    other = candidate_key(_TEMPLATE.format(name="Zoe", n=99))
    assert base == other


# ─────────────────────── AC2 负样本保护 ───────────────────────

def test_ac2_negative_same_subject_different_predicate():
    """(a) 主语相同但谓词不同 → 不同键。"""
    a = candidate_key("Alice 认为量子浮点校准仪需要每季度校准")
    b = candidate_key("Alice 否认量子浮点校准仪需要每季度校准")
    assert a != b


def test_ac2_negative_same_syntax_different_object():
    """(b) 同一句式但讨论不同对象 → 不同键。"""
    a = candidate_key("Alice 认为量子浮点校准仪需要每季度补偿")
    b = candidate_key("Alice 认为量子振荡器需要每季度补偿")
    assert a != b


def test_ac2_negative_name_is_topic_not_source():
    """(c) 人名不同但主题确实不同（人名即主题，非报告来源）→ 不同键。"""
    a = candidate_key("Alice 是项目负责人")
    b = candidate_key("Bob 是项目负责人")
    assert a != b


def test_ac2_negative_bare_number_is_topic_not_ordinal():
    """(c) 纯数字不同但主题确实不同（型号非第<N>序数）→ 不同键。"""
    a = candidate_key("量子浮点校准仪型号 747 存在漂移")
    b = candidate_key("量子浮点校准仪型号 380 存在漂移")
    assert a != b


def test_ac2_negative_capitalized_topic_words_not_masked():
    """(c) 句首大写英文词可能是主题（Python/Java、Exa/API），不可当人名抹掉。"""
    assert candidate_key("量化平台使用 Python 作为主语言") != \
        candidate_key("量化平台使用 Java 作为主语言")
    assert candidate_key("Exa works. Running a batch of searches.") != \
        candidate_key("API works. Running a batch of searches.")


# ─────────────────────── AC3 可回溯 ───────────────────────

def test_ac3_aggregation_preserves_all_original_samples():
    """聚合不得丢弃证据：samples 长度 = 实际出现次数，每条原文可找回。"""
    s = get_staging()
    texts = [_TEMPLATE.format(name=name, n=i) for i, name in enumerate(_NAMES)]
    for t in texts:
        s.record(t)
    c = s.list_candidates()[0]
    assert len(c["samples"]) == len(texts) == 12
    assert c["samples"] == texts  # 顺序、内容逐一对应，全部原文可回溯


# ─────────────────────── AC5 入口统一 ───────────────────────

def test_ac5_both_entries_produce_identical_classification():
    """两个分类入口对既有输入分类结果完全等价（不改变分类结果）。"""
    from core.ontology_evolution import classify_with_extended
    from core.ontology_validator import OntologyValidator

    v = OntologyValidator()
    for text in ("他出生于1985年", "会议于2024年举行"):
        assert v._classify_ontology_type(text, []) == classify_with_extended(text, [])
    for text in ("绝对不命中的独特文本_ac5_7777",):
        assert v._classify_ontology_type(text, []) == "generic_fact"
        assert classify_with_extended(text, []) == "generic_fact"


def test_ac5_both_entries_delegate_to_shared_impl(monkeypatch):
    """两个入口都必须委托同一实现（抽公共函数），而非各自复制逻辑。"""
    from core import ontology_evolution as oe
    from core import ontology_validator as ov

    seen: list[str] = []

    def fake(text, entities, types):
        seen.append(text)
        return "generic_fact"

    monkeypatch.setattr(ov, "classify_ontology_type", fake)
    monkeypatch.setattr(oe, "classify_ontology_type", fake)

    ov.OntologyValidator()._classify_ontology_type("入口1", [])
    oe.classify_with_extended("入口2", [])
    assert seen == ["入口1", "入口2"]


# ─────────────────────── 数据迁移（不得静默丢弃） ───────────────────────

def test_legacy_v1_ledger_is_marked_not_dropped(tmp_path):
    """v1 旧键台账加载后保留旧键、标 legacy、stats() 明确区分，绝不丢弃。"""
    p = tmp_path / "legacy.json"
    sample = _TEMPLATE.format(name="Alice", n=0)
    p.write_text(json.dumps({
        "version": 1,
        "updated_at": 0,
        "counters": {"matched": 1, "unmatched": 2},
        "candidates": {
            "deadbeefdeadbeef": {
                "key": "deadbeefdeadbeef",
                "sample_text": sample,
                "samples": [sample],
                "count": 1,
                "first_seen": 0.0,
                "last_seen": 0.0,
                "status": "pending",
            },
        },
    }, ensure_ascii=False), encoding="utf-8")

    s = OntologyStaging(str(p))
    items = s.list_candidates()
    assert len(items) == 1                       # 不丢弃
    assert items[0]["key"] == "deadbeefdeadbeef"  # 保留旧键
    assert items[0]["legacy"] is True             # 标记 legacy
    st = s.stats()
    assert st["total_candidates"] == 1
    assert st["legacy_candidates"] == 1
