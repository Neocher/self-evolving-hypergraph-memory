"""
Ontology Matcher 单元测试
==========================
覆盖 design_ontology_gaps.md v2 模块2：
  · 自匹配 → all(method=="exact" and score==1.0)（修复 #4：exact 二元）
  · 词法匹配（改名副本，阈值 0.75）
  · 结构匹配（邻居 Jaccard，阈值 0.6）
  · max_types=100 车挡器（修复 #7）
  · match_report 汇总
"""
from __future__ import annotations

import pytest

from core.ontology_matcher import OntologyMatcher, MatchResult
from core.ontology_v2 import EdgeTypeDef, EntityTypeDef, OntologyService


def _baseline() -> OntologyService:
    """标准本体：Person/Organization/Location + FOLLOWS/LOCATED_IN 边。"""
    svc = OntologyService()
    svc.register_entity_type(EntityTypeDef(name="Person", parent="_BaseNode"))
    svc.register_entity_type(EntityTypeDef(name="Organization", parent="_BaseNode"))
    svc.register_entity_type(EntityTypeDef(name="Location", parent="_BaseNode"))
    svc.register_edge_type(EdgeTypeDef(
        name="FOLLOWS", source_types=["Person"], target_types=["Person", "Organization"]))
    svc.register_edge_type(EdgeTypeDef(
        name="LOCATED_IN", source_types=["Person", "Organization"], target_types=["Location"]))
    return svc


@pytest.fixture
def src() -> OntologyService:
    return _baseline()


# ─── 自匹配（验收标准：exact=1.0）────────────────────────


class TestSelfMatch:
    def test_self_match_all_exact_score_1(self, src: OntologyService):
        """自匹配：所有结果 method==exact 且 score==1.0（设计文档验收标准）。"""
        results = OntologyMatcher().match(src, src)
        assert len(results) > 0
        assert all(m.method == "exact" and m.score == 1.0 for m in results), results

    def test_self_match_covers_all_types(self, src: OntologyService):
        """自匹配应覆盖本体中每个类型。"""
        results = OntologyMatcher().match(src, src)
        matched_sources = {m.source for m in results}
        matched_targets = {m.target for m in results}
        names = {t.name for t in src.list_entity_types()}
        assert matched_sources == names
        assert matched_targets == names


# ─── 词法匹配 ───────────────────────────────────────────


class TestLexical:
    def test_renamed_copy_matches_lexically(self):
        """改名副本：Personne vs Person 应词法匹配且 score >= 0.75。"""
        # 自定义本体（避免删除 baseline 类型 Person）：Personne + Group vs Person + Group
        a = OntologyService()
        a.register_entity_type(EntityTypeDef(name="Personne", parent="_BaseNode"))
        a.register_entity_type(EntityTypeDef(name="Group", parent="_BaseNode"))
        a.register_edge_type(EdgeTypeDef(
            name="MEMBER_OF", source_types=["Personne"], target_types=["Group"]))
        b = OntologyService()
        b.register_entity_type(EntityTypeDef(name="Person", parent="_BaseNode"))
        b.register_entity_type(EntityTypeDef(name="Group", parent="_BaseNode"))
        b.register_edge_type(EdgeTypeDef(
            name="MEMBER_OF", source_types=["Person"], target_types=["Group"]))

        results = OntologyMatcher().match(a, b)
        lexical = [m for m in results if m.method == "lexical"]
        assert lexical, "expected lexical matches"
        assert any(m.source == "Personne" and m.target == "Person" for m in lexical)
        assert all(m.score >= 0.75 for m in lexical)

    def test_identical_names_are_exact_not_lexical(self, src: OntologyService):
        """同名类型对应是 exact 而非 lexical。"""
        results = OntologyMatcher().match(src, src)
        assert all(m.method == "exact" for m in results)
        assert not any(m.method == "lexical" for m in results)


# ─── 结构匹配 ───────────────────────────────────────────


def _with_edge_type(name: str, srcs: list, tgts: list) -> OntologyService:
    svc = OntologyService()
    svc.register_entity_type(EntityTypeDef(name=name, parent="_BaseNode"))
    svc.register_entity_type(EntityTypeDef(name="Group", parent="_BaseNode"))
    svc.register_edge_type(EdgeTypeDef(name="MEMBER_OF",
                                       source_types=[name], target_types=["Group"]))
    return svc


class TestStructural:
    def test_same_structure_different_names_matches_structural(self):
        """同结构异名类型（Researcher/Scientist 都 MEMBER_OF Group）应结构匹配。"""
        a = _with_edge_type("Researcher", ["Researcher"], ["Group"])
        b = _with_edge_type("Scientist", ["Scientist"], ["Group"])
        results = OntologyMatcher().match(a, b)
        structural = [m for m in results if m.method == "structural"]
        assert structural, f"expected structural match, got {results}"
        assert any(m.source == "Scientist" or m.target == "Scientist" for m in structural)
        # 邻居 {group, member_of} 完全一致 → Jaccard = 1.0
        assert all(m.score >= 0.6 for m in structural)
        assert max(m.score for m in structural) == pytest.approx(1.0)

    def test_disjoint_structure_no_structural_match(self):
        """结构完全不同（无边连接）不应产生结构匹配。"""
        svc = OntologyService()
        svc.register_entity_type(EntityTypeDef(name="Isolated", parent="_BaseNode"))
        other = _baseline()
        results = OntologyMatcher().match(svc, other)
        assert not any(m.method == "structural" for m in results)


# ─── max_types 车挡器（修复 #7）──────────────────────────


class TestMaxTypesGuard:
    def test_structural_skipped_when_types_exceed_limit(self):
        """类型数超过 max_types 时跳过结构匹配并 warning。"""
        big = OntologyService()
        for i in range(12):
            big.register_entity_type(EntityTypeDef(
                name=f"Type{i:02d}", parent="_BaseNode"))
        matcher = OntologyMatcher(max_types=10)
        results = matcher.match(big, big)
        assert not any(m.method == "structural" for m in results)

    def test_report_flags_structural_computed_false(self):
        big = OntologyService()
        for i in range(12):
            big.register_entity_type(EntityTypeDef(
                name=f"Type{i:02d}", parent="_BaseNode"))
        report = OntologyMatcher(max_types=10).match_report(big, big)
        assert report["structural_computed"] is False
        assert report["max_types"] == 10


# ─── match_report ───────────────────────────────────────


class TestMatchReport:
    def test_report_summary_fields(self, src: OntologyService):
        """match_report 返回汇总：总数、方法分布、平均分、匹配明细。"""
        report = OntologyMatcher().match_report(src, src)
        assert report["total_matches"] == len(report["matches"])
        assert report["by_method"]["exact"] == report["total_matches"]
        assert report["avg_score"] == pytest.approx(1.0)
        assert report["thresholds"] == {"lexical": 0.75, "structural": 0.6}
        assert report["structural_computed"] is True
        assert all(m["score"] == 1.0 and m["method"] == "exact"
                   for m in report["matches"])

    def test_match_result_is_dataclass(self):
        m = MatchResult(source="A", target="B", score=0.9, method="lexical")
        assert m.score == 0.9
        assert m.method == "lexical"
