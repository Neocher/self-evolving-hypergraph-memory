"""
v5.53.0 P3c 实体扩召回（Entity-Expansion）测试
============================================
覆盖（任务书 5 类用例，全部走公共入口 retrieve(level=FUSION)）：
  1. 跨会话聚合召回：mock GraphLite 返回含实体多会话消息 → append + level="entity_expansion"
  2. 时间锚过滤：session_ts 传入 → GQL 含 created_at 上界过滤（spy）；不传 → 无过滤
  3. 纯中文查询 → 提取不到 ASCII 专名 → 直接返回（query_cypher 不被调用）；
     【R1 P0】中英混合查询（"Apple 最近做了什么"）→ 先提取专名再走 CONTAINS
  4. enabled=false 零回归：关闭时与现状完全一致（无 entity_expansion 节点，
     其余补充通道调用与基线一致）
  5. boost 钳制：expanded score = min(全部种子分) × 0.5，严格低于种子分
     【R1 P1】6 种子 [0.9×5, 0.1] → 扩展分 0.05 不排在 0.1 前

运行: .venv/bin/python -m pytest tests/test_entity_expansion.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel
from config.settings import EntityExpansionConfig


def _seed(nid: str, content: str, score: float = 0.9) -> dict:
    return {
        "node_id": nid, "content": content, "score": score,
        "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss",
    }


def _row(nid: str, content: str, tau: float = 0.8) -> dict:
    return {"node_id": nid, "content": content, "tau_value": tau, "fact_track": "active"}


def _make_router(store, seeds, entity_expansion: EntityExpansionConfig | None = None):
    """构造 QueryRouter：mock _fusion_retrieve 控制种子；其余补充通道恒等。

    走生产公共入口 retrieve(level=FUSION)：_finish 闭包内真实执行
    _entity_expansion → _filter_archived → _deduplicate_and_sort（rerank 关闭）。
    """
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig(
        rerank_enabled=False,
        entity_expansion=entity_expansion or EntityExpansionConfig(),
    )
    router._zh_en_tech_map = {}
    router._time_keywords = set()
    router.graphlite_store = store
    router._fusion_retrieve = MagicMock(return_value=seeds)
    router._community_expansion = MagicMock(side_effect=lambda r, q, qe, rq: r)
    router._mesa_synthesis = MagicMock(side_effect=lambda r, q, rq: r)
    router._visual_recall = MagicMock(side_effect=lambda r, q, rq: r)
    router._property_temporal_retrieve = MagicMock(
        side_effect=lambda r, q, rq, now_ts=None, at_ts=None: r
    )
    return router


class TestEntityExpansionConfig:
    def test_config_loaded_from_yaml(self):
        """defaults.yaml 的 entity_expansion 段正确接线为 EntityExpansionConfig（默认开）。"""
        from config.settings import load_settings
        ecfg = load_settings().retrieval.entity_expansion
        assert ecfg.enabled is True
        assert ecfg.boost == 0.5
        assert ecfg.max_results == 10
        assert ecfg.max_entities == 3
        assert ecfg.time_filter is True

    def test_build_router_config_passthrough(self):
        """api/app.py 透传 entity_expansion 嵌套配置（enabled=False 不被吞成默认）。"""
        from api.app import _build_router_config

        class FakeRcfg:
            tau_weight = 0.4
            vector_weight = 0.6
            top_k_l1 = 5
            top_k_vector = 20
            top_k_keyword = 20
            entity_expansion = EntityExpansionConfig(enabled=False, boost=0.3)

        qcfg = _build_router_config(FakeRcfg())
        assert qcfg.entity_expansion.enabled is False
        assert qcfg.entity_expansion.boost == 0.3

    def test_missing_entity_expansion_defaults(self):
        """旧配置对象无 entity_expansion 字段 → QueryRouterConfig 默认 factory 生效。"""
        from api.app import _build_router_config

        class NoEE:
            tau_weight = 0.4
            vector_weight = 0.6
            top_k_l1 = 5
            top_k_vector = 20
            top_k_keyword = 20

        qcfg = _build_router_config(NoEE())
        assert qcfg.entity_expansion.enabled is True
        assert qcfg.entity_expansion.boost == 0.5
        assert qcfg.entity_expansion.time_filter is True

    def test_config_validation_invalid_values(self):
        """【R1 P2-1】boost>=1 / boost<0 / max_results=0 / max_entities=0 → ValueError。"""
        with pytest.raises(ValueError):
            EntityExpansionConfig(boost=1.2)
        with pytest.raises(ValueError):
            EntityExpansionConfig(boost=-0.1)
        with pytest.raises(ValueError):
            EntityExpansionConfig(max_results=0)
        with pytest.raises(ValueError):
            EntityExpansionConfig(max_entities=0)

    def test_config_validation_boundaries_ok(self):
        """【R1 P2-1】边界合法值不抛：boost=0 / 0.99，max_results/max_entities=1。"""
        ecfg = EntityExpansionConfig(boost=0.0, max_results=1, max_entities=1)
        assert ecfg.boost == 0.0 and ecfg.max_results == 1 and ecfg.max_entities == 1
        assert EntityExpansionConfig(boost=0.99).boost == 0.99


class TestExtractProperNouns:
    def test_extracts_capital_sequences_normalized(self):
        """大写专名序列提取 + normalize_entity_name 规范化 + 停用词过滤。"""
        ents = QueryRouter._extract_proper_nouns(
            "What did Apple Inc announce about Tesla Motors?"
        )
        assert "apple" in ents  # "Apple Inc" → "apple"（去尾词后缀）
        assert "tesla motors" in ents
        assert all(e != "what" for e in ents)  # 句首 What 被停用词过滤
        assert len(ents) <= 3

    def test_sentence_start_stopwords_filtered(self):
        """句首 How/Where 等非实体不进候选。"""
        ents = QueryRouter._extract_proper_nouns(
            "How did Tesla solve the battery problem?"
        )
        assert "tesla" in ents
        assert all(e not in ("how", "the", "battery") for e in ents)

    def test_sentence_start_the_stripped_from_sequence(self):
        """【R1 P2-2】句首 The 混入大写序列 → token 级首词剥离（不再提出 the apple）。"""
        ents = QueryRouter._extract_proper_nouns(
            "The Apple Store opened a new flagship."
        )
        assert "apple store" in ents  # 中间大写词 Store 保留
        assert all(e != "the" and not e.startswith("the ") for e in ents)


class TestEntityExpansion:
    def test_cross_session_aggregate_append(self):
        """跨会话聚合召回：含实体的多会话消息 append，level=_source=entity_expansion，
        score = min(种子分 0.9) × 0.5 = 0.45。"""
        store = MagicMock()
        store.query_cypher.return_value = [
            _row("ep2", "In a previous session, Apple announced the M4 chip at WWDC."),
            _row("ep3", "Another session noted Apple's revenue grew 12% last quarter."),
        ]
        router = _make_router(store, [_seed("ep1", "What did Apple do recently?", score=0.9)])
        out = router.retrieve("What did Apple do recently?", level=RetrievalLevel.FUSION)
        exp = [r for r in out if r.get("level") == "entity_expansion"]
        assert len(exp) == 2
        assert {r["node_id"] for r in exp} == {"ep2", "ep3"}
        assert all(r["_source"] == "entity_expansion" for r in exp)
        assert all(r["score"] == pytest.approx(round(0.9 * 0.5, 6)) for r in exp)
        # 实体词经 normalize_entity_name 小写化作 CONTAINS 参数（对齐 _entity_match）
        cypher, params = store.query_cypher.call_args[0]
        assert "e.content CONTAINS $t0" in cypher
        assert params["t0"] == "apple"
        # 【R1 P2-3】单条合并查询 LIMIT 下推：每实体 max_results × 实体数（1 实体 → 10）
        assert params["limit"] == 10

    def test_time_anchor_adds_created_at_filter(self):
        """now_ts（session_ts）传入 → GQL 含 created_at 上界过滤，参数 int 秒。"""
        store = MagicMock()
        store.query_cypher.return_value = []
        router = _make_router(store, [_seed("ep1", "Apple revenue", score=0.9)])
        router.retrieve("Apple revenue", level=RetrievalLevel.FUSION,
                        session_ts=1700000000.5)
        cypher, params = store.query_cypher.call_args[0]
        assert "AND e.created_at <= $at_ts" in cypher
        assert params["at_ts"] == 1700000000

    def test_no_time_filter_without_ts(self):
        """session_ts 为 None → GQL 无 created_at 过滤（时间锚上界仅按需注入）。"""
        store = MagicMock()
        store.query_cypher.return_value = []
        router = _make_router(store, [_seed("ep1", "Apple revenue", score=0.9)])
        router.retrieve("Apple revenue", level=RetrievalLevel.FUSION)
        cypher, params = store.query_cypher.call_args[0]
        assert "created_at <=" not in cypher
        assert "at_ts" not in params

    def test_cjk_query_skipped(self):
        """纯中文查询 → 提取不到大写 ASCII 专名 → 不查 GraphLite，零回归。"""
        store = MagicMock()
        store.query_cypher.return_value = [_row("ep2", "Apple 发布新芯片")]
        router = _make_router(store, [_seed("ep1", "苹果 公司 最近 发布 了 什么", score=0.9)])
        out = router.retrieve("苹果 公司 最近 发布 了 什么", level=RetrievalLevel.FUSION)
        assert all(r.get("level") != "entity_expansion" for r in out)
        store.query_cypher.assert_not_called()

    def test_mixed_cjk_english_query_runs_expansion(self):
        """【R1 P0】中英混合查询：先提取 ASCII 专名再走 CONTAINS（不再被全查询
        CJK 判定短路）——"Apple 最近做了什么" 提取 apple 并调用 query_cypher。"""
        store = MagicMock()
        store.query_cypher.return_value = [_row("ep2", "Apple 发布新芯片 M4")]
        router = _make_router(store, [_seed("ep1", "Apple 最近做了什么", score=0.9)])
        out = router.retrieve("Apple 最近做了什么", level=RetrievalLevel.FUSION)
        exp = [r for r in out if r.get("level") == "entity_expansion"]
        assert len(exp) == 1
        assert exp[0]["node_id"] == "ep2"
        assert exp[0]["_source"] == "entity_expansion"
        cypher, params = store.query_cypher.call_args[0]
        assert "e.content CONTAINS $t0" in cypher
        assert params["t0"] == "apple"

    def test_disabled_zero_regression(self):
        """enabled=false → 不查 GraphLite、无 entity_expansion 节点，种子原样返回；
        其余补充通道调用与基线一致（各恰好 1 次 identity 透传）——关闭扩展不影响
        生产链路（不 stub 全部补充通道，验证完整链路等价）。"""
        store = MagicMock()
        store.query_cypher.return_value = [_row("ep2", "Apple 发布新芯片")]
        router = _make_router(
            store,
            [_seed("ep1", "What did Apple do recently?", score=0.9)],
            entity_expansion=EntityExpansionConfig(enabled=False),
        )
        out = router.retrieve("What did Apple do recently?", level=RetrievalLevel.FUSION)
        assert len(out) == 1
        assert out[0]["node_id"] == "ep1"
        assert out[0]["score"] == 0.9
        assert all(r.get("level") != "entity_expansion" for r in out)
        store.query_cypher.assert_not_called()
        assert router._community_expansion.call_count == 1
        assert router._mesa_synthesis.call_count == 1
        assert router._visual_recall.call_count == 1
        assert router._property_temporal_retrieve.call_count == 1

    def test_boost_clamped_below_seed(self):
        """boost 钳制：expanded score = min(种子分) × 0.5，严格低于全部种子分。"""
        store = MagicMock()
        store.query_cypher.return_value = [
            _row("ep2", "Apple announced the M4 chip at WWDC."),
        ]
        router = _make_router(store, [
            _seed("ep1", "Apple revenue grew", score=0.9),
            _seed("ep3", "Apple supply chain", score=0.4),
        ])
        out = router.retrieve("Apple revenue", level=RetrievalLevel.FUSION)
        exp = [r for r in out if r.get("level") == "entity_expansion"]
        seed_scores = [r["score"] for r in out if r.get("level") != "entity_expansion"]
        assert len(exp) == 1
        assert exp[0]["score"] == pytest.approx(round(0.4 * 0.5, 6))  # min 种子 0.4
        assert all(e["score"] < min(seed_scores) for e in exp)

    def test_p1_seed_score_min_over_all_seeds(self):
        """【R1 P1】6 种子 [0.9×5, 0.1]：扩展分 = min(全部种子 0.1)×0.5 = 0.05，
        严格低于 0.1 且排在 0.1 种子之后（旧逻辑取前 5 → 0.45 反超 0.1）。"""
        store = MagicMock()
        store.query_cypher.return_value = [
            _row("ep9", "Apple announced the M4 chip at WWDC."),
        ]
        seeds = [_seed(f"ep{i}", f"Apple session {i}", score=0.9) for i in range(1, 6)]
        seeds.append(_seed("ep6", "Apple low score node", score=0.1))
        router = _make_router(store, seeds)
        out = router.retrieve("Apple revenue", level=RetrievalLevel.FUSION)
        exp = [r for r in out if r.get("level") == "entity_expansion"]
        assert len(exp) == 1
        assert exp[0]["score"] == pytest.approx(round(0.1 * 0.5, 6))  # min(全部种子)=0.1
        seed6_pos = next(i for i, r in enumerate(out) if r.get("node_id") == "ep6")
        exp_pos = next(i for i, r in enumerate(out) if r.get("level") == "entity_expansion")
        assert exp_pos > seed6_pos  # 扩展分 0.05 严格低于种子 0.1 → 排序在 0.1 之后

    def test_total_append_capped_at_20(self):
        """总 append 硬上限 20：25 条命中 → 只追加 20 条。"""
        store = MagicMock()
        store.query_cypher.return_value = [
            _row(f"ep{i}", f"Apple session message number {i}.") for i in range(25)
        ]
        router = _make_router(store, [_seed("ep1", "What did Apple do?", score=0.9)])
        out = router.retrieve("What did Apple do?", level=RetrievalLevel.FUSION)
        exp = [r for r in out if r.get("level") == "entity_expansion"]
        assert len(exp) == 20

    def test_graphlite_exception_silent_degrade(self):
        """GraphLite 异常 → 静默降级返回原 results（主检索零回归）。"""
        store = MagicMock()
        store.query_cypher.side_effect = RuntimeError("graphlite down")
        router = _make_router(store, [_seed("ep1", "Apple revenue", score=0.9)])
        out = router.retrieve("Apple revenue", level=RetrievalLevel.FUSION)
        assert len(out) == 1
        assert out[0]["node_id"] == "ep1"
        assert all(r.get("level") != "entity_expansion" for r in out)


class TestEntityExpansionAgenticPath:
    """P3c：agentic 路径下实体扩召回生效（_agentic_round 补 _entity_expansion）。"""

    def test_agentic_round_appends_entity_expansion(self):
        store = MagicMock()
        store.query_cypher.return_value = [_row("ep2", "Apple announced the M4 chip.")]
        router = QueryRouter.__new__(QueryRouter)
        router.config = QueryRouterConfig(
            agentic_enabled=True, agentic_max_steps=1, rerank_enabled=False,
        )
        router._zh_en_tech_map = {}
        router._time_keywords = set()
        router.graphlite_store = store
        router._episode_cache = {}
        router._services = None
        router.faiss_index = None
        router.faiss_id_map = {}
        router.encoder = None
        router._fusion_retrieve = MagicMock(
            return_value=[_seed("ep1", "What did Apple do recently?", score=0.9)]
        )
        router._community_expansion = MagicMock(side_effect=lambda r, q, qe, rq: r)
        router._mesa_synthesis = MagicMock(side_effect=lambda r, q, rq: r)
        router._visual_recall = MagicMock(side_effect=lambda r, q, rq: r)
        router._property_temporal_retrieve = MagicMock(
            side_effect=lambda r, q, rq, now_ts=None, at_ts=None: r
        )
        router._hypergraph_supplement = MagicMock(side_effect=lambda r: r)

        out = router.retrieve("What did Apple do recently?", level=RetrievalLevel.FUSION)
        exp = [r for r in out if r.get("level") == "entity_expansion"]
        assert exp, "agentic 路径应追加实体扩召回节点（P3c）"
        assert exp[0]["node_id"] == "ep2"
        assert exp[0]["_source"] == "entity_expansion"
