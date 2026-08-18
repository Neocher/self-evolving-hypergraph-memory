"""
P3a 测试 — bge-reranker 生产管道集成
=====================================
覆盖设计任务书（p3a_impl_task.md）验收点：
  - 单测（mock _get_reranker）：仅 top-k 调用链 / 重排生效 / 异常降级 / failed 标记
  - 集成：走 retrieve(level=FUSION) 公共入口断言 rerank 生效
  - 回归：retrieve(rerank=False) 与关闭前逐字节等价（golden diff）

运行: python -m pytest tests/test_query_router_rerank.py -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel
from retrieval.self_evolving import SelfEvolvingRetrieval
from api.routes.search import _level_from_strategy


def _make_router() -> QueryRouter:
    """构造零依赖 QueryRouter（None store/index，不触发真实模型/索引加载）。"""
    return QueryRouter(None, None, None)


def _tmp_persist_path() -> str:
    """临时持久化路径，避免 SelfEvolvingRetrieval 演化写盘污染仓库 data/。"""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    return path


def _passthrough(results, *args, **kwargs):
    return results


def _fusion_stack(router: QueryRouter, docs: list[dict]) -> ExitStack:
    """patch _fusion_retrieve 返回固定 docs + _finish 内部增强通道透传（只测 rerank）。"""
    stack = ExitStack()
    stack.enter_context(
        patch.object(router, "_fusion_retrieve", return_value=[dict(d) for d in docs])
    )
    for name in (
        "_community_expansion",
        "_mesa_synthesis",
        "_visual_recall",
        "_property_temporal_retrieve",
    ):
        stack.enter_context(patch.object(router, name, side_effect=_passthrough))
    return stack


class FakeReranker:
    """可编排的假 CrossEncoder：predict 返回给定 scores 或抛异常，并记录调用。"""

    def __init__(self, scores=None, error=None):
        self._scores = scores
        self._error = error
        self.pairs = None
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        self.pairs = list(pairs)
        if self._error is not None:
            raise self._error
        if callable(self._scores):
            return self._scores(pairs)
        return list(self._scores)


def _sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


class TestRerankResultsShortCircuit:
    """enabled=False / _rerank_failed / len<2 → 原列表直接返回，不触发 _get_reranker。"""

    def test_disabled_returns_original(self):
        router = _make_router()
        results = [{"node_id": "a", "content": "x", "score": 0.5}]
        with patch.object(router, "_get_reranker") as gr:
            out = router._rerank_results(results, "q", enabled=False)
        gr.assert_not_called()
        assert out is results

    def test_failed_flag_skips(self):
        router = _make_router()
        router._rerank_failed = True
        results = [
            {"node_id": "a", "content": "x", "score": 0.5},
            {"node_id": "b", "content": "y", "score": 0.4},
        ]
        with patch.object(router, "_get_reranker") as gr:
            out = router._rerank_results(results, "q", enabled=True)
        gr.assert_not_called()
        assert out is results

    def test_single_result_returns_original(self):
        router = _make_router()
        results = [{"node_id": "a", "content": "x", "score": 0.5}]
        with patch.object(router, "_get_reranker") as gr:
            out = router._rerank_results(results, "q", enabled=True)
        gr.assert_not_called()
        assert out is results


class TestRerankResultsReordering:
    """CrossEncoder 打分 → sigmoid 归一化 → 重排覆盖头部 score。"""

    def test_reorders_head_and_overwrites_scores(self):
        router = _make_router()
        router.config.rerank_input_k = 40
        # scores 升序 [0.1, 0.5, 0.9] → 重排后 c 最高、a 最低
        fake = FakeReranker(scores=[0.1, 0.5, 0.9])
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "b", "content": "beta", "score": 0.8},
            {"node_id": "c", "content": "gamma", "score": 0.7},
        ]
        with patch.object(router, "_get_reranker", return_value=fake):
            out = router._rerank_results(results, "query", enabled=True)

        assert [r["node_id"] for r in out] == ["c", "b", "a"]
        assert fake.pairs == [("query", "alpha"), ("query", "beta"), ("query", "gamma")]
        expected = _sigmoid([0.1, 0.5, 0.9])
        for r, e in zip(out, [expected[2], expected[1], expected[0]]):
            assert np.isclose(r["score"], e, atol=1e-9), f"head score 应覆盖为 sigmoid: {r['score']} != {e}"

    def test_only_top_k_candidates_reranked(self):
        router = _make_router()
        router.config.rerank_input_k = 3
        # scores [0.1, 0.3, 0.2] → 头部重排为 [n1, n2, n0]，尾部 n3..n9 原序 append
        fake = FakeReranker(scores=[0.1, 0.3, 0.2])
        results = [
            {"node_id": f"n{i}", "content": f"content{i}", "score": 1.0 - i * 0.01}
            for i in range(10)
        ]
        with patch.object(router, "_get_reranker", return_value=fake):
            out = router._rerank_results(results, "q", enabled=True)

        assert len(fake.pairs) == 3, "仅 min(rerank_input_k, len) 候选送入 reranker"
        assert [r["node_id"] for r in out[:3]] == ["n1", "n2", "n0"]
        assert [r["node_id"] for r in out[3:]] == [f"n{i}" for i in range(3, 10)]

    def test_skips_empty_content(self):
        router = _make_router()
        router.config.rerank_input_k = 10
        fake = FakeReranker(scores=[0.9, 0.1])
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "v", "content": "", "score": 0.85},  # 视觉节点无文本
            {"node_id": "b", "content": "beta", "score": 0.8},
        ]
        with patch.object(router, "_get_reranker", return_value=fake):
            out = router._rerank_results(results, "q", enabled=True)

        assert fake.pairs == [("q", "alpha"), ("q", "beta")], "空 content 应跳过不打分"
        # scores [0.9, 0.1] → 重排 [a, b]；空 content 节点 v 保持原相对位置（下标 1）
        assert [r["node_id"] for r in out] == ["a", "v", "b"]

    def test_unscorable_keeps_relative_position(self):
        router = _make_router()
        router.config.rerank_input_k = 10
        fake = FakeReranker(scores=[0.9, 0.1, 0.5])  # a 最高, b 最低, c 中间
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "v1", "content": "", "score": 0.85},
            {"node_id": "b", "content": "beta", "score": 0.8},
            {"node_id": "v2", "content": "  ", "score": 0.75},  # 纯空白也视为空
            {"node_id": "c", "content": "gamma", "score": 0.7},
        ]
        with patch.object(router, "_get_reranker", return_value=fake):
            out = router._rerank_results(results, "q", enabled=True)

        # scorable [a, b, c] 按 [0.9, 0.1, 0.5] 重排为 [a, c, b]；
        # v1/v2 保持其在 head 中的原始相对位置（下标 1/3），不被 append 到末尾
        assert [r["node_id"] for r in out] == ["a", "v1", "c", "v2", "b"]


class TestRerankResultsDegradation:
    """异常降级 / 失败永久标记 → 静默返回原列表，主检索零回归。"""

    def test_predict_exception_returns_original(self):
        router = _make_router()
        fake = FakeReranker(error=RuntimeError("boom"))
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "b", "content": "beta", "score": 0.8},
        ]
        with patch.object(router, "_get_reranker", return_value=fake):
            out = router._rerank_results(results, "q", enabled=True)
        assert out is results, "异常应返回原列表对象（静默降级）"
        assert results[0]["score"] == 0.9, "异常路径不得改写原结果 score"


class TestGetRerankerLazyLoad:
    """_get_reranker 双重检查锁 + 函数内 import + 失败永久标记。"""

    def test_lazy_load_cached(self):
        router = _make_router()
        fake_ce = MagicMock()
        st = types.SimpleNamespace(CrossEncoder=MagicMock(return_value=fake_ce))
        with patch.dict(sys.modules, {"sentence_transformers": st}):
            r1 = router._get_reranker()
            r2 = router._get_reranker()
        assert r1 is r2 is fake_ce, "懒加载结果应缓存复用"
        st.CrossEncoder.assert_called_once_with(
            "BAAI/bge-reranker-base", device="cpu"
        )

    def test_import_failure_marks_failed_permanently(self):
        router = _make_router()
        st = types.SimpleNamespace()  # 无 CrossEncoder 属性 → import 抛 ImportError
        with patch.dict(sys.modules, {"sentence_transformers": st}):
            r1 = router._get_reranker()
        assert r1 is None
        assert router._rerank_failed is True
        # 第二次调用被 _rerank_failed 短路，不再触发 import
        with patch.dict(sys.modules, {"sentence_transformers": st}):
            r2 = router._get_reranker()
        assert r2 is None


class TestRerankIntegration:
    """走 retrieve(level=FUSION) 公共入口，断言 rerank 在去重+boost+钳制之后生效。"""

    def test_fusion_rerank_reorders_via_public_entry(self):
        router = _make_router()
        router.config.rerank_enabled = True
        router.config.rerank_input_k = 40
        docs = [
            {"node_id": "a", "content": "alpha memory", "score": 0.9},
            {"node_id": "b", "content": "beta memory", "score": 0.8},
            {"node_id": "c", "content": "gamma memory", "score": 0.7},
        ]
        fake = FakeReranker(scores=[0.1, 0.5, 0.9])  # c 最高 → 重排后 [c, b, a]
        with _fusion_stack(router, docs):
            with patch.object(router, "_get_reranker", return_value=fake):
                out = router.retrieve("memory", level=RetrievalLevel.FUSION)

        assert fake.calls == 1
        assert [r["node_id"] for r in out] == ["c", "b", "a"]


class TestRerankRegression:
    """retrieve(rerank=False) / config 关 → 与关闭前逐字节等价（golden = _deduplicate_and_sort）。"""

    def test_explicit_rerank_false_equals_golden(self):
        router = _make_router()
        router.config.rerank_enabled = True  # 默认开，但显式 rerank=False 应关闭
        docs = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "b", "content": "beta", "score": 0.5},
        ]
        golden = QueryRouter._deduplicate_and_sort([dict(d) for d in docs])
        with _fusion_stack(router, docs):
            with patch.object(router, "_get_reranker") as gr:
                out = router.retrieve("alpha", level=RetrievalLevel.FUSION, rerank=False)
        gr.assert_not_called()
        assert out == golden, "rerank=False 应与关闭前（仅 _deduplicate_and_sort）逐字节等价"

    def test_config_disabled_with_rerank_none_equals_golden(self):
        router = _make_router()
        router.config.rerank_enabled = False  # rerank=None 读 config → 关
        docs = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "b", "content": "beta", "score": 0.5},
        ]
        golden = QueryRouter._deduplicate_and_sort([dict(d) for d in docs])
        with _fusion_stack(router, docs):
            with patch.object(router, "_get_reranker") as gr:
                out = router.retrieve("alpha", level=RetrievalLevel.FUSION)
        gr.assert_not_called()
        assert out == golden, "config.rerank_enabled=False 应与关闭前逐字节等价"

    def test_non_fusion_level_never_reranks(self):
        router = _make_router()
        router.config.rerank_enabled = True
        docs = [{"node_id": "a", "content": "alpha", "score": 0.9}]
        # VECTOR 降级链路径：_finish 不应触发 rerank（仅 FUSION 生效）
        with patch.object(router, "_vector_retrieve", return_value=[dict(d) for d in docs]):
            with patch.object(router, "_community_expansion", side_effect=_passthrough), \
                 patch.object(router, "_mesa_synthesis", side_effect=_passthrough), \
                 patch.object(router, "_visual_recall", side_effect=_passthrough), \
                 patch.object(router, "_property_temporal_retrieve", side_effect=_passthrough), \
                 patch.object(router, "_get_reranker") as gr:
                out = router.retrieve("alpha", level=RetrievalLevel.VECTOR)
        gr.assert_not_called()
        assert [r["node_id"] for r in out] == ["a"]


class TestLevelFromStrategy:
    """F1：策略字符串 → RetrievalLevel 映射（CC 方案 B）。"""

    def test_hybrid_maps_to_fusion(self):
        assert _level_from_strategy("hybrid") == RetrievalLevel.FUSION

    def test_hybrid_normalized_case_and_whitespace(self):
        assert _level_from_strategy("Hybrid") == RetrievalLevel.FUSION
        assert _level_from_strategy("  hybrid  ") == RetrievalLevel.FUSION

    def test_auto_maps_to_hypergraph(self):
        assert _level_from_strategy("auto") == RetrievalLevel.HYPERGRAPH

    def test_tau_first_maps_to_hypergraph(self):
        assert _level_from_strategy("tau_first") == RetrievalLevel.HYPERGRAPH

    def test_vector_first_maps_to_hypergraph(self):
        assert _level_from_strategy("vector_first") == RetrievalLevel.HYPERGRAPH

    def test_none_maps_to_hypergraph(self):
        assert _level_from_strategy(None) == RetrievalLevel.HYPERGRAPH

    def test_unknown_maps_to_hypergraph(self):
        assert _level_from_strategy("garbage") == RetrievalLevel.HYPERGRAPH


class TestSelfEvolvingPassthrough:
    """F2：SelfEvolvingRetrieval.retrieve 透传 level/rerank 到内层 QueryRouter。"""

    def test_forwards_level_and_rerank(self):
        router = _make_router()
        se = SelfEvolvingRetrieval(router, persist_path=_tmp_persist_path())
        with patch.object(router, "retrieve", return_value=[
            {"content": "x", "score": 0.7, "node_id": "r1"},
        ]) as m:
            se.retrieve("q", level=RetrievalLevel.FUSION, rerank=True)
        m.assert_called_once_with("q", include_archived=False, session_ts=None,
                                  level=RetrievalLevel.FUSION, rerank=True)

    def test_defaults_level_and_rerank(self):
        router = _make_router()
        se = SelfEvolvingRetrieval(router, persist_path=_tmp_persist_path())
        with patch.object(router, "retrieve", return_value=[
            {"content": "x", "score": 0.7, "node_id": "r1"},
        ]) as m:
            se.retrieve("q")
        m.assert_called_once_with("q", include_archived=False, session_ts=None,
                                  level=RetrievalLevel.HYPERGRAPH, rerank=None)


class TestPrewarmReranker:
    """F3：prewarm_reranker 幂等 + 失败永久标记。"""

    def test_idempotent_when_already_loaded(self):
        router = _make_router()
        router._reranker = object()
        with patch.object(router, "_get_reranker") as gr:
            asyncio.run(router.prewarm_reranker())
        gr.assert_not_called()

    def test_idempotent_when_already_failed(self):
        router = _make_router()
        router._rerank_failed = True
        with patch.object(router, "_get_reranker") as gr:
            asyncio.run(router.prewarm_reranker())
        gr.assert_not_called()

    def test_loads_reranker_once(self):
        router = _make_router()
        fake = object()
        with patch.object(router, "_get_reranker", return_value=fake) as gr:
            asyncio.run(router.prewarm_reranker())
        gr.assert_called_once()

    def test_failure_marks_failed_permanently(self):
        router = _make_router()

        def fail():
            router._rerank_failed = True
            return None

        with patch.object(router, "_get_reranker", side_effect=fail):
            asyncio.run(router.prewarm_reranker())
        assert router._rerank_failed is True
        with patch.object(router, "_get_reranker") as gr:
            asyncio.run(router.prewarm_reranker())
        gr.assert_not_called()


class TestRerankLengthMismatch:
    """F5：predict 返回长度与 scorable 不符 → 静默降级原列表。"""

    def test_predict_length_mismatch_returns_original(self):
        router = _make_router()
        router.config.rerank_input_k = 10
        fake = FakeReranker(scores=[0.9])  # 1 分 vs 3 个 scorable
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "b", "content": "beta", "score": 0.8},
            {"node_id": "c", "content": "gamma", "score": 0.7},
        ]
        with patch.object(router, "_get_reranker", return_value=fake):
            out = router._rerank_results(results, "q", enabled=True)
        assert out is results, "长度失配应返回原列表对象"
        assert [r["score"] for r in results] == [0.9, 0.8, 0.7], "不得改写原 score"


class TestRerankNaNGuard:
    """F6：SDK 返回 NaN/Inf → nan_to_num 消毒，产出有效 (0,1] score。"""

    def test_nan_inf_logits_sanitized(self):
        router = _make_router()
        router.config.rerank_input_k = 10
        fake = FakeReranker(scores=[float("nan"), float("inf"), float("-inf")])
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "b", "content": "beta", "score": 0.8},
            {"node_id": "c", "content": "gamma", "score": 0.7},
        ]
        with patch.object(router, "_get_reranker", return_value=fake):
            out = router._rerank_results(results, "q", enabled=True)
        assert len(out) == 3
        for r in out:
            assert np.isfinite(r["score"]), f"score 不得为 NaN/Inf: {r['score']}"
            assert 0.0 <= r["score"] <= 1.0


class TestHybridStrategyIntegration:
    """F7：走公共入口（SelfEvolvingRetrieval.retrieve + _level_from_strategy("hybrid")），
    断言 hybrid→FUSION 且 rerank 生效——修复 Codex 指出的「显式传 FUSION 入口假绿」。"""

    def test_hybrid_maps_to_fusion_and_reranks_via_wrapper(self):
        router = _make_router()
        router.config.rerank_enabled = True
        router.config.rerank_input_k = 40
        docs = [
            {"node_id": "a", "content": "alpha memory", "score": 0.9},
            {"node_id": "b", "content": "beta memory", "score": 0.8},
            {"node_id": "c", "content": "gamma memory", "score": 0.7},
        ]
        fake = FakeReranker(scores=[0.1, 0.5, 0.9])  # c 最高 → 重排 [c, b, a]
        se = SelfEvolvingRetrieval(router, persist_path=_tmp_persist_path())
        with _fusion_stack(router, docs):
            with patch.object(router, "_get_reranker", return_value=fake):
                out = se.retrieve("memory",
                                  level=_level_from_strategy("hybrid"),
                                  rerank=True)

        assert fake.calls == 1, "hybrid→FUSION 应触发 rerank"
        assert [r["node_id"] for r in out] == ["c", "b", "a"]


def _gateway_resp(**kw):
    base = dict(
        query="q", strategy_used="auto", results=[], total_found=0,
        latency_ms=0.0, degraded=False,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


class TestSerialStrategyCache:
    """R2 P1-2: REST cache_key 含 strategy → 同 query 先 auto 后 hybrid 不走缓存。"""

    def test_serial_strategy_bypasses_cache(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes import router, get_services, Services
        from api.routes._deps import _result_cache, _result_cache_lock

        svc = Services()
        svc.query_router = MagicMock()
        svc.query_router.retrieve.return_value = []
        svc.graphlite_store = MagicMock()
        svc.graphlite_store.query_cypher.return_value = []

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_services] = lambda: svc
        client = TestClient(app)

        try:
            with _result_cache_lock:
                _result_cache.clear()

            q = "serial-strategy-probe-query"
            r1 = client.post("/memories/retrieve", json={"query": q, "top_k": 5})
            assert r1.status_code == 200, r1.text
            r2 = client.post("/memories/retrieve", json={"query": q, "top_k": 5, "strategy": "hybrid"})
            assert r2.status_code == 200, r2.text
            assert svc.query_router.retrieve.call_count == 2, "同 query 不同 strategy 不得命中缓存"
        finally:
            with _result_cache_lock:
                _result_cache.clear()


class TestGatewayStrategyPassthrough:
    """R2 P1-1: A2A/ACP/CLI 三条网关路径透传 strategy。"""

    def test_a2a_retrieve_passes_strategy(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from gateway.a2a_server import register_routes

        api = MagicMock()
        api.retrieve = AsyncMock(return_value=_gateway_resp())
        app = FastAPI()
        register_routes(app, api)
        client = TestClient(app)
        resp = client.post("/memory/retrieve", json={"query": "q", "strategy": "hybrid"})
        assert resp.status_code == 200, resp.text
        assert api.retrieve.call_args.kwargs["strategy"] == "hybrid"

    def test_a2a_retrieve_strategy_defaults_auto(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from gateway.a2a_server import register_routes

        api = MagicMock()
        api.retrieve = AsyncMock(return_value=_gateway_resp())
        app = FastAPI()
        register_routes(app, api)
        client = TestClient(app)
        resp = client.post("/memory/retrieve", json={"query": "q"})
        assert resp.status_code == 200, resp.text
        assert api.retrieve.call_args.kwargs["strategy"] == "auto"

    def test_acp_retrieve_passes_strategy(self):
        from gateway.acp_adapter import SHMACPAdapter

        gateway = MagicMock()
        gateway.retrieve = AsyncMock(return_value=_gateway_resp())
        adapter = SHMACPAdapter(MagicMock(), gateway)
        result = asyncio.run(adapter._handle_retrieve({"query": "q", "strategy": "hybrid"}))
        assert result["status"] == "ok"
        assert gateway.retrieve.call_args.kwargs["strategy"] == "hybrid"

    def test_cli_retrieve_writes_strategy_to_body(self):
        from gateway.cli import SHMClient

        client = SHMClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch("gateway.cli.requests.post", return_value=mock_resp) as post:
            client.retrieve("q", strategy="hybrid")
        assert post.call_args.kwargs["json"]["strategy"] == "hybrid"
