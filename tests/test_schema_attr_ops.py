"""
P2 Schema 演化深化（v5.50.0 Schema-AttrOps）测试
================================================
覆盖（任务书第 5 项实施清单）：
  1. get_distinct_attr_names：mock GraphLite 返回 attr_name 列表；失败 → []
  2. _apply_attr_ops：canonical ∈ distinct_attrs 才写入；canonical 不在 → skip；
     泛词 → skip；max 1/轮
  3. _expand_attr_aliases：term 命中 alias → 扩展 canonical；空表 → 原样；去重保序
  4. 通道内消费：构造 QueryRouter 注入 attr_aliases，属性查询命中 canonical
     （走公共入口 retrieve()，防假绿）

运行: python -m pytest tests/test_schema_attr_ops.py -v
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.dream_pipeline import DreamPipeline
from core.ontology_evolution import (
    OntologyEvolution,
    _apply_attr_ops,
    _apply_merge,
    _apply_new_type,
    _build_prompt,
    evolve_once,
    load_extended,
)
from graph.graphlite_store import GraphLiteStore
from retrieval.query_router import QueryRouter, QueryRouterConfig
from retrieval.self_evolving import SelfEvolvingRetrieval


def run(coro):
    return asyncio.run(coro)


def _llm(response: str) -> MagicMock:
    client = MagicMock()
    client.api_key = "test-key"
    client.chat = AsyncMock(return_value=response)
    return client


def _summaries() -> list[dict]:
    return [
        {
            "topics": ["quantum entanglement", "nonlocality"],
            "report": "Community about quantum entanglement experiments and nonlocality.",
        },
    ]


def _year_ts(year: int) -> float:
    return datetime(year, 1, 1).timestamp()


def _seed_result(nid: str, content: str, score: float = 0.9) -> dict:
    return {
        "node_id": nid, "content": content, "score": score,
        "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss",
    }


def _make_router(store, hypergraph_results: list[dict], attr_aliases=None):
    """构造 QueryRouter：mock _hypergraph_retrieve 控制种子；属性通道走真实 store。"""
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router._zh_en_tech_map = {}
    router._time_keywords = set()
    router.graphlite_store = store
    router._attr_aliases = attr_aliases or {}
    router._hypergraph_retrieve = MagicMock(return_value=hypergraph_results)
    return router


# ─── 1. get_distinct_attr_names ─────────────────────────────


class TestGetDistinctAttrNames:

    def test_returns_distinct_attr_names(self):
        store = MagicMock()
        store.query_cypher.return_value = [
            {"attr_name": "revenue"},
            {"attr_name": "market_cap"},
            {"attr_name": "revenue"},  # 重复 → 去重
        ]
        names = GraphLiteStore.get_distinct_attr_names(store)
        assert names == ["revenue", "market_cap"]

    def test_failure_returns_empty(self):
        store = MagicMock()
        store.query_cypher.return_value = []  # GraphLite 失败 → query_cypher 返回 []
        assert GraphLiteStore.get_distinct_attr_names(store) == []

    def test_non_dict_rows_skipped(self):
        store = MagicMock()
        store.query_cypher.return_value = [
            {"attr_name": "revenue"},
            "garbage-row",
            {"attr_name": None},
            {"attr_name": ""},
        ]
        assert GraphLiteStore.get_distinct_attr_names(store) == ["revenue"]


# ─── 2. _apply_attr_ops ─────────────────────────────────────


class TestApplyAttrOps:

    def test_writes_when_canonical_in_distinct(self):
        parsed = {
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["income", "营收"]},
            ],
        }
        current = {}
        new_current, result = _apply_attr_ops(parsed, current, ["revenue", "market_cap"])
        assert result["action"] == "attr_op"
        assert result["canonical"] == "revenue"
        assert new_current["attr_aliases"]["revenue"] == ["income", "营收"]

    def test_skip_canonical_not_in_distinct(self):
        parsed = {
            "attr_ops": [
                {"op": "merge_alias", "canonical": "unknown_attr", "aliases": ["income"]},
            ],
        }
        current = {}
        new_current, result = _apply_attr_ops(parsed, current, ["revenue"])
        assert result["action"] == "skip"
        assert new_current is None
        assert "attr_aliases" not in current

    def test_skip_generic_aliases(self):
        parsed = {
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["data", "信息"]},
            ],
        }
        current = {}
        new_current, result = _apply_attr_ops(parsed, current, ["revenue"])
        assert result["action"] == "skip"
        assert new_current is None

    def test_skip_generic_canonical(self):
        parsed = {
            "attr_ops": [
                {"op": "merge_alias", "canonical": "data", "aliases": ["income"]},
            ],
        }
        current = {}
        new_current, result = _apply_attr_ops(parsed, current, ["data", "revenue"])
        assert result["action"] == "skip"
        assert new_current is None

    def test_max_one_per_round(self):
        parsed = {
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["income"]},
                {"op": "merge_alias", "canonical": "market_cap", "aliases": ["市值"]},
            ],
        }
        current = {}
        new_current, result = _apply_attr_ops(parsed, current, ["revenue", "market_cap"])
        assert result["canonical"] == "revenue"
        assert list(new_current["attr_aliases"].keys()) == ["revenue"]

    def test_merges_existing_aliases_dedup(self):
        current = {"attr_aliases": {"revenue": ["income"]}}
        parsed = {
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["income", "营收"]},
            ],
        }
        new_current, result = _apply_attr_ops(parsed, current, ["revenue"])
        assert result["action"] == "attr_op"
        assert new_current["attr_aliases"]["revenue"] == ["income", "营收"]

    def test_canonical_self_not_in_alias_table(self):
        """【P3-1】canonical 自身出现在 aliases → 过滤，不进别名表。"""
        parsed = {
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["revenue", "income"]},
            ],
        }
        current = {}
        new_current, result = _apply_attr_ops(parsed, current, ["revenue"])
        assert result["action"] == "attr_op"
        assert new_current["attr_aliases"]["revenue"] == ["income"]

    def test_canonical_only_aliases_skip(self):
        """【P3-1】aliases 仅含 canonical 自身 → 过滤后为空 → skip。"""
        parsed = {
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["revenue"]},
            ],
        }
        current = {}
        new_current, result = _apply_attr_ops(parsed, current, ["revenue"])
        assert result["action"] == "skip"
        assert new_current is None

    def test_no_ops_skip(self):
        new_current, result = _apply_attr_ops({}, {}, ["revenue"])
        assert result["action"] == "skip"
        assert new_current is None

    def test_distinct_none_skip(self):
        parsed = {
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["income"]},
            ],
        }
        new_current, result = _apply_attr_ops(parsed, {}, None)
        assert result["action"] == "skip"
        assert new_current is None


# ─── evolve_once 集成（attr_ops 与类型决策正交）──────────────


class TestEvolveAttrOps:

    def test_attr_ops_with_type_skip(self, tmp_path):
        """类型 skip 但 attr_op 通过守卫 → 落盘，返回 attr_op。"""
        p = tmp_path / "ontology_extended.json"
        client = _llm(json.dumps({
            "action": "skip",
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["income"]},
            ],
        }))
        result = run(evolve_once(_summaries(), client, str(p),
                                 distinct_attrs=["revenue", "market_cap"]))
        assert result["action"] == "attr_op"
        assert result["canonical"] == "revenue"
        assert load_extended(str(p))["attr_aliases"]["revenue"] == ["income"]

    def test_no_attr_ops_when_distinct_none(self, tmp_path):
        """distinct_attrs=None → 跳过 attr_ops，行为与类型决策一致。"""
        p = tmp_path / "ontology_extended.json"
        client = _llm(json.dumps({
            "action": "skip",
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["income"]},
            ],
        }))
        result = run(evolve_once(_summaries(), client, str(p)))
        assert result["action"] == "skip"
        assert result["reason"] == "llm_decided"
        assert not p.exists()

    def test_attr_ops_orthogonal_with_new_type(self, tmp_path):
        """new_type 与 attr_op 同轮发生（正交），两者均落盘。"""
        p = tmp_path / "ontology_extended.json"
        client = _llm(json.dumps({
            "action": "new_type",
            "type": "quantum_result",
            "description": "量子实验",
            "conflict_keys": ["quantum", "entanglement"],
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["income"]},
            ],
        }))
        result = run(evolve_once(_summaries(), client, str(p),
                                 distinct_attrs=["revenue", "market_cap"]))
        assert result["action"] == "new_type"
        extended = load_extended(str(p))
        assert "quantum_result" in extended
        assert extended["attr_aliases"]["revenue"] == ["income"]


# ─── P0-1：OntologyEvolution.evolve 生产路径取 distinct 清单 ─────


class TestOntologyEvolutionProduction:

    def test_evolve_gets_distinct_and_triggers_attr_ops(self, tmp_path):
        """【P0-1】生产路径：evolve() 从 store 取 distinct 清单 → 触发 attr_ops 落盘。"""
        p = tmp_path / "ontology_extended.json"
        store = MagicMock()
        store.get_distinct_attr_names.return_value = ["revenue", "market_cap"]
        client = _llm(json.dumps({
            "action": "skip",
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["income"]},
            ],
        }))
        evo = OntologyEvolution(extended_path=str(p), llm_client=client,
                                graphlite_store=store)
        result = run(evo.evolve(_summaries()))
        assert result["action"] == "attr_op"
        assert result["canonical"] == "revenue"
        assert load_extended(str(p))["attr_aliases"]["revenue"] == ["income"]

    def test_evolve_no_store_skips_attr_ops(self, tmp_path):
        """【P0-1】无 store → distinct_attrs=None → 跳过 attr_ops（仅类型决策）。"""
        p = tmp_path / "ontology_extended.json"
        client = _llm(json.dumps({
            "action": "skip",
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["income"]},
            ],
        }))
        evo = OntologyEvolution(extended_path=str(p), llm_client=client)
        result = run(evo.evolve(_summaries()))
        assert result["action"] == "skip"
        assert result["reason"] == "llm_decided"
        assert not p.exists()

    def test_evolve_empty_distinct_skips_orphan_canonical(self, tmp_path):
        """【P0-1】distinct 清单不含 canonical → attr_op 被守卫拒绝（孤儿 alias）。"""
        p = tmp_path / "ontology_extended.json"
        store = MagicMock()
        store.get_distinct_attr_names.return_value = []
        client = _llm(json.dumps({
            "action": "skip",
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["income"]},
            ],
        }))
        evo = OntologyEvolution(extended_path=str(p), llm_client=client,
                                graphlite_store=store)
        result = run(evo.evolve(_summaries()))
        assert result["action"] == "skip"
        assert not p.exists()


# ─── P1-2：attr_aliases 不进类型决策命名空间 ─────────────────────


class TestReservedKeys:

    def test_build_prompt_filters_attr_aliases(self):
        """【P1-2】_build_prompt 不把 attr_aliases 当类型渲染。"""
        current = {
            "attr_aliases": {"revenue": ["income"]},
            "event_date": {"description": "日期", "conflict_keys": ["date"]},
        }
        prompt = _build_prompt(_summaries(), current)
        assert "attr_aliases" not in prompt
        assert "event_date" in prompt

    def test_merge_rejects_reserved_target(self):
        """【P1-2】merge_existing 目标为 attr_aliases → 拒绝（不污染 alias 表）。"""
        current = {"attr_aliases": {"revenue": ["income"]}}
        parsed = {"action": "merge_existing", "type": "attr_aliases",
                  "conflict_keys": ["income"]}
        new_current, result = _apply_merge(parsed, current)
        assert result["action"] == "skip"
        assert result["reason"] == "merge_target_reserved"
        assert new_current is None

    def test_new_type_rejects_reserved_name(self):
        """【P1-2】new_type 名为 attr_aliases → 拒绝。"""
        parsed = {"action": "new_type", "type": "attr_aliases",
                  "description": "x", "conflict_keys": ["income", "revenue"]}
        new_current, result = _apply_new_type(parsed, {})
        assert result["action"] == "skip"
        assert result["reason"] == "type_reserved"
        assert new_current is None


# ─── 3. _expand_attr_aliases ────────────────────────────────


class TestExpandAttrAliases:

    def test_term_hits_alias_expands_canonical(self):
        assert QueryRouter._expand_attr_aliases(
            ["income"], {"revenue": ["income"]},
        ) == ["income", "revenue"]

    def test_empty_table_unchanged(self):
        assert QueryRouter._expand_attr_aliases(
            ["income", "revenue"], {},
        ) == ["income", "revenue"]

    def test_dedup_preserve_order(self):
        assert QueryRouter._expand_attr_aliases(
            ["income", "revenue", "income"], {"revenue": ["income"]},
        ) == ["income", "revenue"]

    def test_canonical_term_unchanged(self):
        assert QueryRouter._expand_attr_aliases(
            ["revenue"], {"revenue": ["income"]},
        ) == ["revenue"]

    def test_underscore_case_normalized(self):
        assert QueryRouter._expand_attr_aliases(
            ["market cap"], {"market_cap": ["market cap", "市值"]},
        ) == ["market cap", "market_cap"]


# ─── 3a. _extract_property_terms alias 子串匹配边界（R3 P1-6）──


class TestExtractPropertyTermsAliasBoundary:

    def test_ascii_single_token_alias_no_substring(self):
        """纯 ASCII 单 token alias "income" 不子串命中 "incoming"（词边界保持）。"""
        terms = QueryRouter._extract_property_terms(
            "Apple incoming", {"revenue"}, {"revenue": ["income"]},
        )
        assert "income" not in terms

    def test_chinese_alias_still_hits(self):
        """中文 alias "营业额" 仍子串命中（含 CJK → 走子串通道）。"""
        terms = QueryRouter._extract_property_terms(
            "Apple 营业额", {"revenue"}, {"revenue": ["营业额"]},
        )
        assert "营业额" in terms

    def test_multiword_alias_still_hits(self):
        """多词英文 alias "annual income" 仍子串命中（含空格 → 走子串通道）。"""
        terms = QueryRouter._extract_property_terms(
            "Apple annual income", {"revenue"}, {"revenue": ["annual income"]},
        )
        assert "annual income" in terms


# ─── 3b. set_attr_aliases 非 dict 降级（R3 P3-2）─────────────


class TestSetAttrAliasesValidation:

    def test_non_dict_aliases_degrade_to_empty(self):
        """set_attr_aliases 非 dict（list/string）→ 降级空 dict，不抛异常。"""
        router = QueryRouter.__new__(QueryRouter)
        router._attr_aliases = {"revenue": ["income"]}
        router.set_attr_aliases(["income"])  # list
        assert router._attr_aliases == {}
        router.set_attr_aliases("revenue")  # string
        assert router._attr_aliases == {}

    def test_dict_and_none_still_work(self):
        router = QueryRouter.__new__(QueryRouter)
        router.set_attr_aliases(None)
        assert router._attr_aliases == {}
        router.set_attr_aliases({"revenue": ["income"]})
        assert router._attr_aliases == {"revenue": ["income"]}


# ─── 3c. 构造注入非 dict 降级（R4 P1-7）──────────────────────


class TestAttrAliasesConstructorValidation:

    def test_non_dict_attr_aliases_degrade_to_empty(self):
        """【R4 P1-7】构造注入非 dict attr_aliases（list/string）→ 降级空 dict，不抛异常。"""
        router = QueryRouter(None, None, None, attr_aliases=["x"])  # list
        assert router._attr_aliases == {}
        router = QueryRouter(None, None, None, attr_aliases="revenue")  # string
        assert router._attr_aliases == {}

    def test_dict_and_none_still_work(self):
        router = QueryRouter(None, None, None, attr_aliases=None)
        assert router._attr_aliases == {}
        router = QueryRouter(None, None, None, attr_aliases={"revenue": ["income"]})
        assert router._attr_aliases == {"revenue": ["income"]}


# ─── 4. 通道内消费（公共入口 retrieve()）──────────────────────


class TestAttrAliasChannelConsumption:

    def _seed_two_attrs(self, store):
        """Apple 双属性版本：revenue(10B) + income(9B)（income 为 revenue 的别名）。"""
        store.create_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2020))
        store.create_property_version("Apple", "income", "9B", valid_from=_year_ts(2020))

    def test_alias_expands_to_canonical_via_retrieve(self, overgraph_store):
        """注入 attr_aliases 后 "Apple income" 命中 revenue + income 双属性（公共入口）。"""
        self._seed_two_attrs(overgraph_store)
        router = _make_router(
            overgraph_store, [_seed_result("ep1", "Apple 的业务情况")],
            attr_aliases={"revenue": ["income"]},
        )
        out = router.retrieve("Apple income")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert {p["attr_name"] for p in props} == {"revenue", "income"}

    def test_no_alias_no_expansion(self, overgraph_store):
        """无别名（空表）时 "Apple income" 只命中 income（零回归）。"""
        self._seed_two_attrs(overgraph_store)
        router = _make_router(
            overgraph_store, [_seed_result("ep1", "Apple 的业务情况")],
            attr_aliases={},
        )
        out = router.retrieve("Apple income")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert {p["attr_name"] for p in props} == {"income"}

    def test_alias_not_existing_attr_name_via_retrieve(self, overgraph_store):
        """【P1-1】alias 不是现存 attr_name：只存 revenue，查 "Apple income" 命中 revenue。

        income 不作为真实 attr_name 写入（修复前该缺口使 _extract_property_terms
        收不到 income → 别名学习失效）；别名表提取阶段识别 income → 扩展出
        canonical revenue。
        """
        overgraph_store.create_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2020))
        router = _make_router(
            overgraph_store, [_seed_result("ep1", "Apple 的业务情况")],
            attr_aliases={"revenue": ["income"]},
        )
        out = router.retrieve("Apple income")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert {p["attr_name"] for p in props} == {"revenue"}

    def test_chinese_alias_matches_canonical_via_retrieve(self, overgraph_store):
        """【P1-4】中文 alias "营业额" → 命中 canonical revenue（公共入口）。

        只存 revenue + market_cap，查 "Apple 营业额" 应只返回 revenue（market_cap
        被属性词过滤）。修复前中文 alias 收不进 terms → 不过滤 → 双属性都返回。
        """
        overgraph_store.create_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2020))
        overgraph_store.create_property_version("Apple", "market_cap", "2T", valid_from=_year_ts(2020))
        router = _make_router(
            overgraph_store, [_seed_result("ep1", "Apple 的业务情况")],
            attr_aliases={"revenue": ["营业额"]},
        )
        out = router.retrieve("Apple 营业额")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert {p["attr_name"] for p in props} == {"revenue"}


# ─── 5. 梦境 attr_op 写盘后 alias map 刷新（P1-5）──────────────


class TestAttrAliasDreamRefresh:

    def test_dream_attr_op_refreshes_router_alias_map(self, tmp_path):
        """【P1-5】梦境 attr_op 写盘后 → retrieval_guard 刷新内层 _qr alias map。"""
        p = tmp_path / "ontology_extended.json"
        store = MagicMock()
        store.get_distinct_attr_names.return_value = ["revenue"]
        client = _llm(json.dumps({
            "action": "skip",
            "attr_ops": [
                {"op": "merge_alias", "canonical": "revenue", "aliases": ["营业额"]},
            ],
        }))
        evo = OntologyEvolution(extended_path=str(p), llm_client=client,
                                graphlite_store=store)
        inner = QueryRouter.__new__(QueryRouter)
        inner.config = QueryRouterConfig()
        inner._attr_aliases = {}
        guard = SelfEvolvingRetrieval(inner)
        pipe = DreamPipeline(llm_client=client, ontology_evolution=evo,
                             retrieval_guard=guard)
        run(pipe._ontology_evolution_step(_summaries()))
        assert load_extended(str(p))["attr_aliases"]["revenue"] == ["营业额"]
        assert inner._attr_aliases == {"revenue": ["营业额"]}

    def test_refresh_no_guard_silent(self):
        """无 retrieval_guard / 无 evolve → _refresh_attr_aliases 静默降级（不抛异常）。"""
        pipe = DreamPipeline(llm_client=None, ontology_evolution=None,
                             retrieval_guard=None)
        pipe._refresh_attr_aliases()
