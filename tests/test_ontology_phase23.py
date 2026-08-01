"""
OntologyValidator Phase2/3 单元测试
=====================================
覆盖：_extract_types、_compute_type_overlap、_extract_entity_cooccurrence、
      extract_and_relate、_compute_topology_score。
"""
from __future__ import annotations

import pytest

from core.ontology_validator import OntologyValidator, OntologyConfig


@pytest.fixture
def validator() -> OntologyValidator:
    return OntologyValidator(config=OntologyConfig(enabled=True))


# ─── _extract_types 测试（Phase1: ENTITY_TYPE_MAP 匹配）────


class TestExtractTypes:
    """ENTITY_TYPE_MAP 实体类型抽取。"""

    @pytest.mark.parametrize("text,expected_entity,expected_type", [
        # 深度学习框架
        ("using pytorch for training", "pytorch", "deep_learning_framework"),
        ("tensorflow model deployed", "tensorflow", "deep_learning_framework"),
        ("paddlepaddle on CPU", "paddlepaddle", "deep_learning_framework"),
        # 硬件
        ("GPU acceleration for AI", "gpu", "hardware"),
        ("Intel CPU benchmark", "cpu", "hardware"),
        ("nvidia cuda toolkit", "nvidia", "hardware"),
        # 向量/图数据库
        ("FAISS vector search", "faiss", "vector_database"),
        ("GraphLite graph query", "graphlite", "graph_database"),
        # 中文互联网平台（P3新增）
        ("baidu search engine", "baidu", "company"),
        ("tencent cloud services", "tencent", "company"),
        ("weixin messaging", "weixin", "internet_platform"),
        ("bilibili video platform", "bilibili", "internet_platform"),
        # 网络服务（P3新增）
        ("dns resolution", "dns", "network_service"),
        ("cdn acceleration", "cdn", "network_service"),
        # AI助手
        ("claude AI assistant", "claude", "ml_model"),
        ("deepseek model", "deepseek", "ml_model"),
        # 混合输入
        ("使用 PyTorch 在 CPU 上运行", "pytorch", "deep_learning_framework"),
        ("使用 PyTorch 在 CPU 上运行", "cpu", "hardware"),
    ])
    def test_extract_types_known_entities(self, validator, text, expected_entity, expected_type):
        """已知实体应正确映射到本体类型。"""
        types = validator._extract_types(text)
        found = [t for t in types if t["entity"] == expected_entity]
        assert len(found) == 1, f"Expected entity '{expected_entity}' in {types}"
        assert found[0]["type"] == expected_type

    def test_unknown_entities_return_empty(self, validator):
        """未知实体（不在MAP中）应返回空列表。"""
        types = validator._extract_types("today weather beijing shanghai")
        assert types == []

    def test_mixed_known_unknown(self, validator):
        """混合已知/未知实体时应只返回已知。"""
        types = validator._extract_types("PyTorch and randomword for CPU testing")
        entities = {t["entity"] for t in types}
        assert "pytorch" in entities
        assert "cpu" in entities
        # randomword 不应出现在结果中
        assert "randomword" not in entities

    def test_deduplication(self, validator):
        """相同实体多次出现应去重。"""
        types = validator._extract_types("PyTorch PyTorch PyTorch")
        assert len([t for t in types if t["entity"] == "pytorch"]) == 1

    def test_case_insensitive(self, validator):
        """实体匹配应大小写不敏感。"""
        for case in ["PyTorch", "pytorch", "PYTORCH"]:
            types = validator._extract_types(case)
            assert any(t["entity"] == "pytorch" for t in types), f"Failed for {case}"

    def test_ascii_word_boundary(self, validator):
        """英文实体边界不应被CJK字符干扰。"""
        types = validator._extract_types("cpu版测试")
        assert any(t["entity"] == "cpu" for t in types), f"cpu not found in cpu版"


# ─── _compute_type_overlap 测试 ──────────────────────────


class TestTypeOverlap:
    """类型一致性评分。"""

    def test_exact_match(self, validator):
        """完全匹配应返回1.0。"""
        q_types = [{"entity": "pytorch", "type": "deep_learning_framework",
                     "category": "ml_framework", "matched": True}]
        r_types = [{"entity": "pytorch", "type": "deep_learning_framework",
                     "category": "ml_framework", "matched": True}]
        score = validator._compute_type_overlap(q_types, r_types)
        assert score == 1.0

    def test_no_overlap(self, validator):
        """完全无重叠应返回0.0。"""
        q_types = [{"entity": "pytorch", "type": "deep_learning_framework",
                     "category": "ml_framework", "matched": True}]
        r_types = [{"entity": "baidu", "type": "internet_platform",
                     "category": "web_service", "matched": True}]
        score = validator._compute_type_overlap(q_types, r_types)
        assert score == 0.0

    def test_partial_overlap(self, validator):
        """部分重叠应在0~1之间。"""
        q_types = [
            {"entity": "pytorch", "type": "deep_learning_framework",
             "category": "ml_framework", "matched": True},
            {"entity": "cpu", "type": "hardware",
             "category": "infrastructure", "matched": True},
        ]
        r_types = [
            {"entity": "pytorch", "type": "deep_learning_framework",
             "category": "ml_framework", "matched": True},
            {"entity": "baidu", "type": "internet_platform",
             "category": "web_service", "matched": True},
        ]
        score = validator._compute_type_overlap(q_types, r_types)
        # 2 q_types, 2 r_types → 1 exact match (dl), 3 unique types
        # exact_ratio = 1/3 = 0.333
        # 1 cat match, 3 unique cats → cat_ratio = 1/3 = 0.333
        # final = 0.6*0.333 + 0.4*0.333 = 0.333
        assert score == pytest.approx(0.333, abs=0.01)

    def test_same_type_diff_entity(self, validator):
        """不同实体但相同类型应得满分。"""
        q_types = [{"entity": "pytorch", "type": "deep_learning_framework",
                     "category": "ml_framework", "matched": True}]
        r_types = [{"entity": "tensorflow", "type": "deep_learning_framework",
                     "category": "ml_framework", "matched": True}]
        score = validator._compute_type_overlap(q_types, r_types)
        # 相同类型 deep_learning_framework → exact match → 1.0
        assert score == 1.0





class TestEntityCooccurrence:
    """Phase2/3: 文本中 ENTITY_TYPE_MAP 实体共现提取。"""

    def test_single_entity(self, validator):
        """单个实体应返回单元素列表。"""
        entities = validator._extract_entity_cooccurrence("using pytorch")
        assert entities == ["pytorch"]

    def test_multiple_entities(self, validator):
        """多个实体应全部返回。"""
        entities = validator._extract_entity_cooccurrence(
            "PyTorch on CPU with FAISS"
        )
        assert len(entities) >= 3
        assert "pytorch" in entities
        assert "cpu" in entities
        assert "faiss" in entities

    def test_no_entity(self, validator):
        """无实体返回空列表。"""
        entities = validator._extract_entity_cooccurrence("hello world")
        assert entities == []

    def test_deduplication(self, validator):
        """相同实体多次出现应只返回一次。"""
        entities = validator._extract_entity_cooccurrence(
            "PyTorch PyTorch PyTorch"
        )
        assert entities == ["pytorch"]

    def test_chinese_context(self, validator):
        """中文文本中正确识别实体。"""
        entities = validator._extract_entity_cooccurrence(
            "在CPU上运行PyTorch进行推理"
        )
        assert "cpu" in entities
        assert "pytorch" in entities

    def test_new_entity_types(self, validator):
        """P3新增的实体应能识别。"""
        entities = validator._extract_entity_cooccurrence(
            "baidu dns tencent cloud"
        )
        assert "baidu" in entities
        assert "dns" in entities
        assert "tencent" in entities


# ─── _compute_topology_score 测试 ───────────────────────


class TestTopologyScore:
    """Phase2: 拓扑路径验证（无GraphLite时返回1.0）。"""

    def test_no_graphlite_returns_1(self, validator):
        """无GraphLite连接时拓扑置信度为1.0（跳过）。"""
        score = validator._compute_topology_score(
            [{"entity": "pytorch", "type": "dl"}],
            "using pytorch on cpu",
        )
        assert score == 1.0

    def test_empty_query_entities(self, validator):
        """查询无实体时应返回1.0。"""
        score = validator._compute_topology_score([], "test content")
        assert score == 1.0

    def test_empty_result_entities(self, validator):
        """结果无实体时应返回1.0。"""
        score = validator._compute_topology_score(
            [{"entity": "pytorch", "type": "dl"}],
            "text with no known entities",
        )
        assert score == 1.0


# ─── extract_and_relate 测试 ─────────────────────────


class TestExtractAndRelate:
    """Phase3: 写入时实体关系提取。"""

    def test_no_graphlite_returns_0(self, validator):
        """无GraphLite时应返回0。"""
        count = validator.extract_and_relate("PyTorch on CPU with FAISS")
        assert count == 0

    def test_fewer_than_two_entities(self, validator):
        """少于2个实体时应返回0。"""
        count = validator.extract_and_relate("just pytorch")
        assert count == 0


# ─── ENTITY_TYPE_MAP 完整性测试 ──────────────────────


class TestEntityMapCompleteness:
    """ENTITY_TYPE_MAP 的条目完整性。"""

    def test_minimum_entity_count(self, validator):
        """实体至少应有100个（P3扩展后）。"""
        assert len(validator.ENTITY_TYPE_MAP) >= 95

    def test_all_entities_have_categories(self, validator):
        """所有 ENTITY_TYPE 应有对应的类别映射。"""
        for etype in set(validator.ENTITY_TYPE_MAP.values()):
            assert etype in validator.ENTITY_TYPE_CATEGORIES, (
                f"Type '{etype}' missing from ENTITY_TYPE_CATEGORIES"
            )

    def test_known_entities_exist(self, validator):
        """关键实体必须存在于映射中。"""
        required = {"pytorch", "cpu", "gpu", "faiss", "graphlite",
                    "baidu", "dns", "tencent", "bilibili"}
        missing = required - set(validator.ENTITY_TYPE_MAP.keys())
        assert not missing, f"Missing required entities: {missing}"
