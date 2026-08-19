"""
P0-2 Agentic 检索测试
=====================
规则编排 + session_ts 时间锚注入（方案 B）。覆盖任务书验收：
  1. 默认关（agentic_enabled=False）→ 单轮 FUSION 全路径，字节级等价
  2. session_ts 注入 → 相对时间词按 session 时间锚解析（非墙钟）
  3. agentic_max_steps 硬上限（死循环防护）
  4. agentic_min_new 锚点枯竭提前停

走公共入口 QueryRouter.retrieve()（mock 基础通道 _fusion_retrieve 控制种子，
同 test_property_temporal 的 _hypergraph_retrieve mock 模式），编排逻辑走真实实现。

运行: python -m pytest tests/test_agentic_retrieve.py -v
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from retrieval.query_router import (
    QueryRouter, QueryRouterConfig, RetrievalLevel, _TIME_ANCHOR_SENTINEL,
)


def _r(nid: str, content: str, score: float = 0.9) -> dict:
    """构造一条种子结果（fusion 通道输出形态）。"""
    return {
        "node_id": nid, "content": content, "score": score,
        "fact_track": "active", "tau_value": 1.0, "level": "fusion_vector",
    }


def _make_fusion_router(
    fusion_results: list[dict] | None = None,
    agentic_enabled: bool = False,
    **cfg,
) -> QueryRouter:
    """构造 QueryRouter：mock 基础通道 + 补充通道（隔离 settings/store/视觉依赖）。

    补充通道 mock 为恒等（返回输入），编排逻辑（分类/路由/充分性/锚点/多轮循环）
    走真实实现。_fusion_retrieve 每次调用返回独立 dict 副本（防跨调用 score 突变）。
    """
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig(agentic_enabled=agentic_enabled, **cfg)
    router._zh_en_tech_map = {}
    router._time_keywords = set()
    router.graphlite_store = None
    router._cjk_warned = False
    router._episode_cache = {}
    router._services = None
    router.faiss_index = None
    router.faiss_id_map = {}
    router.encoder = None
    router._fusion_retrieve = MagicMock(
        side_effect=lambda q, qe=None, rq=None, now_ts=None: [dict(r) for r in (fusion_results or [])]
    )
    router._community_expansion = MagicMock(side_effect=lambda r, q, qe, rq: r)
    router._visual_recall = MagicMock(side_effect=lambda r, q, rq: r)
    router._property_temporal_retrieve = MagicMock(side_effect=lambda r, q, rq, now_ts=None, at_ts=None: r)
    router._hypergraph_supplement = MagicMock(side_effect=lambda r: r)
    return router


class TestAgenticDisabled:
    """验收 1：默认关 → 单轮 FUSION 全路径，_agentic_retrieve 不调用。"""

    def test_agentic_retrieve_disabled_equals_baseline(self):
        router = _make_fusion_router(
            fusion_results=[_r("a", "content A", 0.9), _r("b", "content B", 0.5)],
        )
        baseline = router.retrieve("test query", level=RetrievalLevel.FUSION)
        with patch.object(router, "_agentic_retrieve") as agentic:
            out = router.retrieve("test query", level=RetrievalLevel.FUSION)
            agentic.assert_not_called()
        assert out == baseline, "agentic_enabled=False 应字节级等价于单轮 FUSION 路径"


class TestAgenticTimeInjection:
    """验收 2：session_ts 注入 → 相对时间词按 session 时间锚解析（非墙钟）。"""

    def test_relative_time_anchored_to_session_ts(self):
        router = _make_fusion_router()
        session_ts = 1_600_000_000.0  # 固定历史时刻，远离当前墙钟
        # 相对时间词相对 session_ts 解析（cat=2 根治：历史 session 不再按墙钟错算）
        assert QueryRouter._relative_time_at_ts("3 days ago", session_ts) == pytest.approx(
            session_ts - 3 * 86400, abs=1
        )
        assert QueryRouter._relative_time_at_ts("昨天", session_ts) == pytest.approx(
            session_ts - 86400, abs=1
        )
        # 无 session_ts 回落墙钟（向后兼容）
        assert QueryRouter._relative_time_at_ts("3 days ago", None) == pytest.approx(
            time.time() - 3 * 86400, abs=5
        )

    def test_property_time_mode_threads_session_ts(self):
        router = _make_fusion_router()
        session_ts = 1_600_000_000.0
        mode, at_ts = router._property_time_mode("Apple 3 days ago revenue", session_ts)
        assert mode == "at_time"
        assert at_ts == pytest.approx(session_ts - 3 * 86400, abs=1)
        # agentic 路径首步（_classify_intent）同样以 session_ts 锚定时间
        plan = router._classify_intent("Apple 3 days ago revenue", session_ts)
        assert plan.time_mode == "at_time"
        assert plan.at_ts == pytest.approx(session_ts - 3 * 86400, abs=1)


class TestAgenticMaxSteps:
    """验收 3：agentic_max_steps 硬上限（死循环防护）。"""

    def test_max_steps_hard_limit(self):
        router = _make_fusion_router(
            agentic_enabled=True, agentic_max_steps=3, agentic_min_new=1,
        )
        # 每轮返回不同 node_id + 不同实体（持续有新锚点）+ 拥挤分数（gap=0 证据不足）
        # → 只能靠 max_steps 硬上限停止（min_new 不触发）
        entities = ["Apple", "Google", "Microsoft"]
        rounds = [
            [_r(f"r{r}-{i}", f"{entities[r]} revenue", 0.9) for i in range(3)]
            for r in range(3)
        ]
        router._fusion_retrieve = MagicMock(side_effect=rounds)
        router.retrieve("cross message vague query", level=RetrievalLevel.FUSION)
        assert router._fusion_retrieve.call_count == 3, (
            f"max_steps=3 应恰好执行 3 轮，实际 {router._fusion_retrieve.call_count} 轮"
        )


class TestAgenticMinNewStop:
    """验收 4：agentic_min_new 锚点枯竭提前停。"""

    def test_min_new_stop(self):
        router = _make_fusion_router(
            agentic_enabled=True, agentic_max_steps=5, agentic_min_new=3,
        )
        # 首轮结果内容无锚点（小写词/动词不当实体）+ 拥挤分数（证据不足）
        # → 锚点枯竭（new=0 < 3）提前停，第二轮不执行
        rounds = [
            [_r("a1", "please tell me why", 0.9), _r("a2", "it was the same", 0.9)],
            [_r("b1", "should not run", 0.9)],
        ]
        router._fusion_retrieve = MagicMock(side_effect=rounds)
        router.retrieve("vague query", level=RetrievalLevel.FUSION)
        assert router._fusion_retrieve.call_count == 1, (
            f"锚点枯竭应首轮后即停，实际 {router._fusion_retrieve.call_count} 轮"
        )


class TestIncrementalAnchorStop:
    """P1-1 增量枯竭：相同锚点不再重复满足 min_new（seen_anchors 差集）。"""

    def test_same_anchors_do_not_resatisfy_min_new(self):
        router = _make_fusion_router(
            agentic_enabled=True, agentic_max_steps=5, agentic_min_new=1,
        )
        # 每轮内容相同（锚点无增量）+ 新 node_id（seen 不拦）+ 拥挤分数
        # → 第二轮后锚点差集为空，min_new 提前停（第三轮不执行）
        rounds = [
            [_r("a1", "Apple revenue", 0.9)],
            [_r("b1", "Apple revenue", 0.9)],
            [_r("c1", "Apple revenue", 0.9)],
        ]
        router._fusion_retrieve = MagicMock(side_effect=rounds)
        out = router.retrieve("vague query", level=RetrievalLevel.FUSION)
        assert isinstance(out, list), "agentic 路径应返回去重排序后的列表"
        assert router._fusion_retrieve.call_count == 2, (
            f"锚点无增量应第二轮后停，实际 {router._fusion_retrieve.call_count} 轮"
        )


class TestSeenAnchorsCaseNormalize:
    """N5-P3: seen_anchors 大小写归一——Apple vs apple 计同一锚点。"""

    def test_case_variation_not_new_anchor(self):
        router = _make_fusion_router(
            agentic_enabled=True, agentic_max_steps=5, agentic_min_new=1,
        )
        # 首轮 "Apple"（大写）→ 次轮 "apple"（小写）应计同一锚点 → 枯竭停
        rounds = [
            [_r("a1", "Apple revenue", 0.9)],
            [_r("b1", "apple revenue", 0.9)],
            [_r("c1", "please tell me why", 0.9)],
        ]
        router._fusion_retrieve = MagicMock(side_effect=rounds)
        router.retrieve("vague query", level=RetrievalLevel.FUSION)
        assert router._fusion_retrieve.call_count == 2, (
            f"Apple/apple 大小写差异应计同一锚点，第 2 轮枯竭停，"
            f"实际 {router._fusion_retrieve.call_count} 轮"
        )


class TestAgenticLevelGate:
    """P1-4：agentic_enabled 仅劫持 FUSION，不劫持 HYPERGRAPH/VECTOR/KEYWORD。"""

    def test_agentic_does_not_hijack_other_levels(self):
        router = _make_fusion_router(
            agentic_enabled=True,
            fusion_results=[_r("a", "content A", 0.9)],
        )
        with patch.object(router, "_agentic_retrieve") as agentic:
            router.retrieve("test query", level=RetrievalLevel.HYPERGRAPH)
            agentic.assert_not_called()


class TestP13IntentClassification:
    """P1-3：英文属性词识别 + 伪实体过滤。"""

    def test_english_property_terms_recognized(self):
        router = _make_fusion_router()
        terms = router._classify_property_terms("Apple market cap revenue")
        assert "revenue" in terms
        assert "market_cap" in terms

    def test_pseudo_entities_filtered(self):
        router = _make_fusion_router()
        assert router._extract_query_entities("What happened in 2023") == []
        plan = router._classify_intent("What happened in 2023", None)
        assert plan.intent == "time"

    def test_lowercase_english_entity_restored(self):
        """N1-P1: "apple 收入" 能提取小写实体 apple（对齐写侧 Apple Inc）。"""
        router = _make_fusion_router()
        ents = router._extract_query_entities("apple 收入")
        assert "apple" in ents, f"小写 apple 应被提取: {ents}"

    def test_lowercase_time_words_still_filtered(self):
        """N1-P1: 恢复小写提取但时间词/动词仍过滤——"happened year" 不当实体。"""
        router = _make_fusion_router()
        assert router._extract_query_entities("what happened last year") == []

    def test_word_boundary_not_substring(self):
        """N2-P2: "age" 不命中 "agent"/"manager"，"sales" 不命中 "salesforce"。"""
        router = _make_fusion_router()
        assert router._classify_property_terms("the agent and manager") == []
        assert router._classify_property_terms("salesforce crm") == []


class TestP12AnchorSessionTs:
    """P1-2：证据时间锚按 session_ts 解析 + 时间锚计入 new。"""

    def test_extract_anchors_threads_session_ts(self):
        router = _make_fusion_router()
        plan = router._classify_intent("some entity query", None)
        session_ts = 1_600_000_000.0
        anchors = router._extract_anchors(
            [{"content": "Apple 3 days ago revenue", "node_id": "x"}], plan, session_ts,
        )
        assert anchors.time_anchor == pytest.approx(session_ts - 3 * 86400, abs=1)
        # 【R2 N3-P2】时间锚 key 带 timestamp（哨兵前缀 + 解析后 ts），值唯一
        assert any(a.startswith(_TIME_ANCHOR_SENTINEL) for a in anchors.all)

    def test_distinct_time_anchors_distinct_keys(self):
        """N3-P2: 不同时间锚（"3 days ago" vs "2023"）→ 不同 key，可各自计数。"""
        router = _make_fusion_router()
        plan = router._classify_intent("some entity query", None)
        session_ts = 1_600_000_000.0
        a1 = router._extract_anchors(
            [{"content": "Apple 3 days ago revenue", "node_id": "x"}], plan, session_ts,
        )
        a2 = router._extract_anchors(
            [{"content": "Apple 2023 revenue", "node_id": "x"}], plan, session_ts,
        )
        keys1 = [a for a in a1.all if a.startswith(_TIME_ANCHOR_SENTINEL)]
        keys2 = [a for a in a2.all if a.startswith(_TIME_ANCHOR_SENTINEL)]
        assert keys1 and keys2
        assert keys1 != keys2, "不同时间锚应产生不同 key"


class TestP21HypergraphSupplementSort:
    """P2-1：_hypergraph_supplement 排序后取头尾。"""

    def test_sorts_before_head_tail(self):
        router = _make_fusion_router()
        router.graphlite_store = MagicMock()
        captured = {}

        def fake_expansion(seeds, existing_ids, tail_score):
            captured["seeds"] = seeds
            captured["tail_score"] = tail_score
            return []

        router._graph_expansion = MagicMock(side_effect=fake_expansion)
        results = [
            _r("low", "low content", 0.1),
            _r("high", "high content", 0.9),
            _r("mid", "mid content", 0.5),
        ]
        QueryRouter._hypergraph_supplement(router, results)
        assert captured["seeds"] == ["high", "mid", "low"]
        assert captured["tail_score"] == pytest.approx(0.1, abs=1e-6)


class TestSessionTsPropagation:
    """P2-2 集成：retrieve() 公共入口 session_ts 透传至 _property_temporal_retrieve。"""

    def test_retrieve_session_ts_propagates(self):
        router = _make_fusion_router(
            fusion_results=[_r("a", "Apple 收入", 0.9)],
        )
        captured = {}

        def spy_pt(r, q, rq, now_ts=None, at_ts=None):
            captured["now_ts"] = now_ts
            captured["at_ts"] = at_ts
            return r

        router._property_temporal_retrieve = MagicMock(side_effect=spy_pt)
        session_ts = 1_700_000_000.0
        router.retrieve(
            "Apple 收入是多少", level=RetrievalLevel.FUSION, session_ts=session_ts,
        )
        assert captured["now_ts"] == session_ts, (
            f"单轮 FUSION 应透传 session_ts={session_ts}，实际 {captured['now_ts']}"
        )

    def test_agentic_at_ts_propagates(self):
        router = _make_fusion_router(
            agentic_enabled=True, agentic_max_steps=1, agentic_min_new=1,
            fusion_results=[_r("a", "Apple revenue", 0.9)],
        )
        captured = {}

        def spy_pt(r, q, rq, now_ts=None, at_ts=None):
            captured["now_ts"] = now_ts
            captured["at_ts"] = at_ts
            return r

        router._property_temporal_retrieve = MagicMock(side_effect=spy_pt)
        session_ts = 1_700_000_000.0
        router.retrieve(
            "Apple 3 days ago revenue", level=RetrievalLevel.FUSION,
            session_ts=session_ts,
        )
        assert captured["now_ts"] == session_ts
        assert captured["at_ts"] == pytest.approx(session_ts - 3 * 86400, abs=1)


class TestR3PseudoEntityControl:
    """R3 P2-1: 小写英文词伪实体控制——停用词扩充 + 撇号缩写还原。"""

    def test_prepositions_and_auxiliaries_filtered(self):
        """of/to/has/had/have/by/an/as/am 等常见词不当实体。"""
        router = _make_fusion_router()
        assert router._extract_query_entities("of to has had have by an as am") == []
        # 停用词被过滤但真实小写实体 apple 仍保留
        assert "apple" in router._extract_query_entities("the price of apple has changed")

    def test_contraction_not_split_into_pseudo_entity(self):
        """don't → do not（do/not 均在停用表），不再切出 "don"。"""
        router = _make_fusion_router()
        ents = router._extract_query_entities("I don't know the price")
        assert "don" not in ents, f"don't 不应切出 don: {ents}"
        assert "do" not in ents
        assert router._extract_query_entities("it's fine") == ["fine"]

    def test_possessive_apostrophe_still_extracts_entity(self):
        """所有格 Apple's 不在缩写表，专名 apple 仍被提取。"""
        router = _make_fusion_router()
        assert "apple" in router._extract_query_entities("apple's revenue")

    def test_lets_contraction_not_split_into_let(self):
        """R4 P2-1: let's → let us（let 为停用词），"Let's find Apple" 不产出 "let" 伪实体。"""
        router = _make_fusion_router()
        ents = router._extract_query_entities("Let's find Apple")
        lower = [e.lower() for e in ents]
        assert "let" not in lower, f"Let's 不应切出 let: {ents}"
        assert "apple" in lower, f"真实实体 apple 应保留: {ents}"


class TestR3PropertyFilterWordBoundary:
    """R3 P2-2: 生产属性过滤链词边界匹配——age 不命中 manager/sales 不命中 salesforce。"""

    def test_attr_name_matches_word_boundary(self):
        router = _make_fusion_router()
        assert router._attr_name_matches("age", "age") is True
        assert router._attr_name_matches("age", "manager") is False
        assert router._attr_name_matches("age", "agent") is False
        assert router._attr_name_matches("sales", "salesforce") is False
        # 下划线归一：market_cap ↔ market cap
        assert router._attr_name_matches("market_cap", "market_cap") is True
        assert router._attr_name_matches("market cap", "market_cap") is True

    def test_extract_property_terms_no_substring(self):
        router = _make_fusion_router()
        # "age" 是 "manager" 子串，但词边界下不应提取为属性词
        assert "age" not in router._extract_property_terms("Apple age", {"manager", "revenue"})
        # 精确 attr_name 命中
        assert "age" in router._extract_property_terms("Apple age", {"age", "manager"})


class TestR3RelativeTimeAnchorStableKey:
    """R3 P3-1: 相对时间锚稳定语义键，绝对时间保留 timestamp。"""

    def test_time_anchor_key_semantic(self):
        router = _make_fusion_router()
        assert QueryRouter._time_anchor_key("today") == "today"
        assert QueryRouter._time_anchor_key("昨天") == "yesterday"
        assert QueryRouter._time_anchor_key("3 days ago") == "3_day_ago"
        assert QueryRouter._time_anchor_key("last year") == "last_year"
        assert QueryRouter._time_anchor_key("2023") is None
        assert QueryRouter._time_anchor_key("Apple revenue") is None

    def test_relative_anchor_stable_across_rounds(self):
        """无 session_ts 时相对时间锚键按语义稳定（不随 time.time() 变化）。"""
        router = _make_fusion_router()
        plan = router._classify_intent("some entity query", None)
        a1 = router._extract_anchors(
            [{"content": "Apple today revenue", "node_id": "x"}], plan, None,
        )
        a2 = router._extract_anchors(
            [{"content": "Apple today revenue", "node_id": "x"}], plan, None,
        )
        k1 = [a for a in a1.all if a.startswith(_TIME_ANCHOR_SENTINEL)]
        k2 = [a for a in a2.all if a.startswith(_TIME_ANCHOR_SENTINEL)]
        assert k1 == k2 == [f"{_TIME_ANCHOR_SENTINEL}:today"], f"相对锚键应稳定: {k1} vs {k2}"

    def test_absolute_anchor_keeps_timestamp(self):
        router = _make_fusion_router()
        plan = router._classify_intent("some entity query", None)
        anchors = router._extract_anchors(
            [{"content": "Apple 2023 revenue", "node_id": "x"}], plan, None,
        )
        keys = [a for a in anchors.all if a.startswith(_TIME_ANCHOR_SENTINEL)]
        assert keys, "绝对时间锚应有哨兵键"
        assert any(ch.isdigit() for ch in keys[0]), f"绝对时间锚键应含 timestamp: {keys}"


class TestR3N1PublicEntryIntegration:
    """R3 P3-3: "apple 收入" 走 retrieve() 公共入口，覆盖实体提取 → 属性时间检索链路。"""

    def test_retrieve_apple_income_triggers_property_temporal(self, overgraph_store):
        from core.entity_resolver import EntityResolver
        resolver = EntityResolver(graphlite_store=overgraph_store)
        resolver._update_property_version(
            "Apple Inc", "revenue", "10B", valid_from=1_600_000_000.0,
        )

        router = _make_fusion_router(fusion_results=[_r("ep1", "Apple 的业务情况", 0.9)])
        router.graphlite_store = overgraph_store
        router._property_temporal_retrieve = (
            QueryRouter._property_temporal_retrieve.__get__(router, QueryRouter)
        )

        out = router.retrieve("apple 收入", level=RetrievalLevel.FUSION)
        props = [r for r in out if r.get("level") == "property_temporal"]
        assert props, "retrieve() 公共入口应触发属性时间检索（小写 apple 提取实体）"
        assert props[0]["entity_id"] == "Apple Inc"
        assert "10B" in props[0]["content"]
