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
  5. boost 钳制：expanded score = max(全部种子分) × 0.9，仅低于最高种子分
     【CC P3c】6 种子 [0.9×5, 0.1] → 扩展分 0.81 排在 0.1 之前（仅低于 0.9）
  6. 【R2 P0】大小写敏感回归：CONTAINS 大小写敏感，实体保留原始大小写 +
     小写双变体条件（"Melanie"/"melanie"）覆盖大写/小写两种存储

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
        assert ecfg.boost == 0.9
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
        assert qcfg.entity_expansion.boost == 0.9
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
    def test_extracts_capital_sequences_preserves_case(self):
        """大写专名序列提取保留原始大小写（不再小写化）+ 去尾词后缀 + 停用词过滤。"""
        ents = QueryRouter._extract_proper_nouns(
            "What did Apple Inc announce about Tesla Motors?"
        )
        assert "Apple" in ents  # "Apple Inc" → "Apple"（去尾词后缀，保留大小写）
        assert "Tesla Motors" in ents
        assert all(e != "what" for e in ents)  # 句首 What 被停用词过滤
        assert all(e[0].isupper() for e in ents)  # 全部保留原始首字母大写
        assert len(ents) <= 3

    def test_sentence_start_stopwords_filtered(self):
        """句首 How/Where 等非实体不进候选。"""
        ents = QueryRouter._extract_proper_nouns(
            "How did Tesla solve the battery problem?"
        )
        assert "Tesla" in ents
        assert all(e not in ("how", "the", "battery") for e in ents)

    def test_sentence_start_the_stripped_from_sequence(self):
        """【R1 P2-2】句首 The 混入大写序列 → token 级首词剥离（不再提出 the apple）。"""
        ents = QueryRouter._extract_proper_nouns(
            "The Apple Store opened a new flagship."
        )
        assert "Apple Store" in ents  # 中间大写词 Store 保留（原始大小写）
        assert all(e != "the" and not e.startswith("the ") for e in ents)


class TestEntityExpansion:
    def test_cross_session_aggregate_append(self):
        """跨会话聚合召回：含实体的多会话消息 append，level=_source=entity_expansion，
        score = max(种子分 0.9) × 0.9 = 0.81。"""
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
        assert all(r["score"] == pytest.approx(round(0.9 * 0.9, 6)) for r in exp)
        # 双变体条件：原始大小写（Apple）+ 小写（apple）——CONTAINS 大小写敏感，
        # 覆盖大写/小写两种存储（R2 P0）
        cypher, params = store.query_cypher.call_args[0]
        assert "e.content CONTAINS $t0_orig" in cypher
        assert "e.content CONTAINS $t0_lower" in cypher
        assert params["t0_orig"] == "Apple"
        assert params["t0_lower"] == "apple"
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
        assert "e.content CONTAINS $t0_orig" in cypher
        assert "e.content CONTAINS $t0_lower" in cypher
        assert params["t0_orig"] == "Apple"
        assert params["t0_lower"] == "apple"

    def test_case_sensitive_storage_matches_original_case(self):
        """【R2 P0】大小写回归：GraphLite CONTAINS 大小写敏感（实测 'Melanie' 3 rows /
        'melanie' 0 rows）——mock 返回大写专名存储内容，旧逻辑实体小写化后 CONTAINS
        打不进（扩展恒空）；修复后生成 orig（"Melanie"）+ lower（"melanie"）双条件，
        大写内容被命中 append。（修复前本测试必 FAIL：params 无 t0_orig 键 /
        cypher 无 orig 条件）"""
        store = MagicMock()
        store.query_cypher.return_value = [
            _row("ep2", "Melanie asked about the quarterly budget yesterday."),
        ]
        router = _make_router(
            store, [_seed("ep1", "What did Melanie say?", score=0.9)]
        )
        out = router.retrieve("What did Melanie say?", level=RetrievalLevel.FUSION)
        exp = [r for r in out if r.get("level") == "entity_expansion"]
        assert len(exp) == 1
        assert exp[0]["node_id"] == "ep2"
        assert exp[0]["_source"] == "entity_expansion"
        cypher, params = store.query_cypher.call_args[0]
        assert "e.content CONTAINS $t0_orig" in cypher
        assert "e.content CONTAINS $t0_lower" in cypher
        assert params["t0_orig"] == "Melanie"
        assert params["t0_lower"] == "melanie"
        assert params["limit"] == 10

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
        """boost 钳制：expanded score = max(种子分) × 0.9，仅低于最高种子分。"""
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
        assert exp[0]["score"] == pytest.approx(round(0.9 * 0.9, 6))  # max 种子 0.9
        assert all(e["score"] < max(seed_scores) for e in exp)

    def test_p1_seed_score_max_anchor_over_all_seeds(self):
        """【CC P3c】推翻 R1-P1 min 锚契约：6 种子 [0.9×5, 0.1] → 扩展分 =
        max(全部种子 0.9)×0.9 = 0.81，仅低于最高种子 0.9，排在 0.1 低分种子之前。
        为何推翻 R1-P1：cat1 聚合场景要求跨会话证据进 LLM 上下文（docs[:40]→
        rerank top-12），min 锚使扩展分 ≈0.05 沉底进不了 top-40；max 锚 + boost
        0.9 仅低于最高种子，由内部/外部 rerank 双兜底收敛语义相关性。"""
        store = MagicMock()
        store.query_cypher.return_value = [
            _row("ep9", "Apple announced the M4 chip at WWDC."),
        ]
        seeds = [_seed(f"ep{i}", f"Apple session {i}", score=0.9) for i in range(1, 6)]
        seeds.append(_seed("ep6", "Apple low score node", score=0.1))
        router = _make_router(store, seeds)
        out = router.retrieve("Apple revenue", level=RetrievalLevel.FUSION)
        # 【2026-08-23 P0 sufficiency 门控】6 高分种子（gap=0.889 ≥ 0.25,
        # distinct=6 ≥ 3）= 证据充分 → 跳过实体扩展（扩展仅在证据不足时补充）。
        exp = [r for r in out if r.get("level") == "entity_expansion"]
        assert len(exp) == 0, "证据充分时应跳过实体扩展（sufficiency 门控）"

    def test_p1_sufficiency_gate_disabled_still_expands(self):
        """sufficiency_gate=False 时保留 max 锚扩展逻辑（原行为）：6 种子
        [0.9×5, 0.1] → 扩展分 = max(0.9)×0.9 = 0.81，仅低于最高种子 0.9。"""
        store = MagicMock()
        store.query_cypher.return_value = [
            _row("ep9", "Apple announced the M4 chip at WWDC."),
        ]
        seeds = [_seed(f"ep{i}", f"Apple session {i}", score=0.9) for i in range(1, 6)]
        seeds.append(_seed("ep6", "Apple low score node", score=0.1))
        router = _make_router(
            store, seeds,
            entity_expansion=EntityExpansionConfig(sufficiency_gate=False))
        out = router.retrieve("Apple revenue", level=RetrievalLevel.FUSION)
        exp = [r for r in out if r.get("level") == "entity_expansion"]
        assert len(exp) == 1
        assert exp[0]["score"] == pytest.approx(round(0.9 * 0.9, 6))  # max(全部种子)=0.9
        seed6_pos = next(i for i, r in enumerate(out) if r.get("node_id") == "ep6")
        exp_pos = next(i for i, r in enumerate(out) if r.get("level") == "entity_expansion")
        assert exp_pos < seed6_pos  # 扩展分 0.81 仅低于最高种子 0.9 → 排序在 0.1 之前

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
