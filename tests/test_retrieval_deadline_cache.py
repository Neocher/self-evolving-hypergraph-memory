"""
C1+C2 验收测试（v6.25 回移版）：协作式 deadline + degraded 不进缓存 + 缓存代际失效
================================================================================
覆盖：
  AC1  degraded=True 响应不进缓存：连续两次同 query，第二次必须真的重新检索
       （fake router 计数）
  AC2  deadline 生效：注入 sleep 超预算的假通道，retrieve(deadline=...) 预算内返回
       且结果非空、不再启动 rerank（mock 断言 _rerank_results 未被调用）
  AC3  deadline=None 零回归（默认路径）
  AC4  写入 bump generation 后旧 key 不再命中（fake router 计数）
  AC5  预算不够但还没过期 → 不启动 rerank（1.2×EWMA 预估拦截）
  AC6  连续 3 次超预算 → _is_rerank_budget_disabled() 为真且此后跳过

运行: python3.12 -m pytest tests/test_retrieval_deadline_cache.py -q
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router, Services, get_services
from api.routes._deps import (
    _result_cache, _result_cache_lock,
    bump_result_cache_generation,
    current_result_cache_generation,
)
from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel


@pytest.fixture
def mock_services():
    """最小服务容器：仅覆盖 retrieve 端点需要的字段。"""
    svc = Services()
    svc.encoder = MagicMock()
    svc.encoder.embed = MagicMock(return_value=np.zeros(384, dtype=np.float32))
    svc.faiss_index = MagicMock()
    svc.faiss_index.ntotal = 100
    svc.faiss_index.search = MagicMock(
        return_value=(np.array([[0.5]]), np.array([[0]]))
    )
    svc.faiss_id_map = {}
    svc.graph_store = MagicMock()
    svc.graph_store.query_cypher = MagicMock(return_value=[])
    svc.query_router = None
    svc.ontology_validator = None
    svc.ontology_v2 = None
    svc.dream_scheduler = None
    svc.tau_engine = None
    svc.ssm_gate = None
    svc.evidence_tracker = None
    return svc


@pytest.fixture
def client(mock_services):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_services] = lambda: mock_services
    return TestClient(app)


def _clear_cache():
    with _result_cache_lock:
        _result_cache.clear()


class TestDegradedNotCached:
    def test_degraded_response_retrieved_twice(self, client, mock_services):
        """AC1：degraded=True 的响应不得写缓存（连续两次同 query 均真检索）。"""
        _clear_cache()
        mock_qr = MagicMock()
        mock_qr.config = QueryRouterConfig(rerank_enabled=False)
        mock_qr.retrieve.return_value = [{
            "node_id": "n_degraded", "content": "degraded content",
            "score": 0.6, "level": "l1_empty", "tau_value": 0.5,
            "_degradation_level": "l1_empty",
        }]
        mock_services.query_router = mock_qr
        mock_services.graph_store.query_cypher.return_value = []

        q = "degraded-cache-probe-query"
        r1 = client.post("/memories/retrieve", json={"query": q, "top_k": 5})
        r2 = client.post("/memories/retrieve", json={"query": q, "top_k": 5})

        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text
        assert r1.json()["degraded"] is True
        assert mock_qr.retrieve.call_count == 2, (
            "degraded 响应被错误缓存：第二次未真正重新检索"
        )
        with _result_cache_lock:
            assert len(_result_cache) == 0, "degraded 响应不应残留缓存"


class TestDeadlineCooperative:
    def test_deadline_skips_slow_channel_and_rerank(self):
        """AC2：deadline 过期 → 扩展通道跳过 + rerank 不启动，返回部分结果。"""
        router = QueryRouter.__new__(QueryRouter)
        router.config = QueryRouterConfig(
            rerank_enabled=True,
            fusion_dedup_enabled=False,
        )
        router._zh_en_tech_map = {}
        router._time_keywords = set()
        router.encoder = None
        seeds = [{
            "node_id": "n0", "content": "alpha beta", "score": 0.9,
            "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss",
        }]
        router._fusion_retrieve = MagicMock(return_value=[dict(s) for s in seeds])

        # deadline 已过期 → 所有扩展通道绝不能被启动
        for name in (
            "_community_expansion", "_mesa_synthesis", "_visual_recall",
            "_property_temporal_retrieve", "_entity_expansion",
            "_attribute_expansion", "_scope_retrieve", "_schema_recall",
            "_fact_retrieve",
        ):
            setattr(router, name, MagicMock(
                side_effect=AssertionError(f"{name} should not be called")
            ))
        router._rerank_results = MagicMock()

        deadline = time.monotonic() - 0.001  # 已经过期
        t0 = time.monotonic()
        out = router.retrieve("alpha beta", level=RetrievalLevel.FUSION, deadline=deadline)
        elapsed = time.monotonic() - t0

        assert isinstance(out, list) and out, "deadline 过期应返回 deadline 前的部分结果（非空）"
        assert elapsed < 1.0, f"deadline 过期应快速返回，实际 {elapsed:.3f}s"
        assert router._community_expansion.call_count == 0, (
            "deadline 过期后扩展通道不应被启动"
        )
        assert router._rerank_results.call_count == 0, (
            "deadline 过期后 rerank 不应被启动"
        )

    def test_deadline_none_behavior_unchanged(self):
        """AC3：deadline=None 走默认路径（不跳过扩展、rerank 正常被调用）。"""
        router = QueryRouter.__new__(QueryRouter)
        router.config = QueryRouterConfig(
            rerank_enabled=True,
            fusion_dedup_enabled=False,
        )
        router._zh_en_tech_map = {}
        router._time_keywords = set()
        router.encoder = None
        seeds = [{
            "node_id": "n0", "content": "alpha beta", "score": 0.9,
            "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss",
        }]
        router._fusion_retrieve = MagicMock(return_value=[dict(s) for s in seeds])

        calls = []
        for name in (
            "_community_expansion", "_mesa_synthesis", "_visual_recall",
            "_property_temporal_retrieve", "_entity_expansion",
            "_attribute_expansion", "_scope_retrieve", "_schema_recall",
            "_fact_retrieve",
        ):
            def _passthrough(results, *args, _name=name, **kwargs):
                calls.append(_name)
                return results
            setattr(router, name, MagicMock(side_effect=_passthrough))
        router._rerank_results = MagicMock(side_effect=lambda results, *a, **k: results)

        out = router.retrieve("alpha beta", level=RetrievalLevel.FUSION)

        assert isinstance(out, list) and out
        assert all(name in calls for name in (
            "_community_expansion", "_mesa_synthesis", "_visual_recall",
        )), f"deadline=None 不应跳过扩展通道: {calls}"
        assert router._rerank_results.call_count == 1, (
            "deadline=None 时 rerank 应正常被调用"
        )

    def test_budget_too_small_for_rerank_skips_before_start(self):
        """AC5：预算不够但还没过期：remaining=0.05s < 1.2×estimate(0.5s) → 不启动。"""
        router = QueryRouter.__new__(QueryRouter)
        router.config = QueryRouterConfig(
            rerank_enabled=True,
            fusion_dedup_enabled=False,
        )
        router._zh_en_tech_map = {}
        router._time_keywords = set()
        router.encoder = None
        seeds = [{
            "node_id": "n0", "content": "alpha beta", "score": 0.9,
            "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss",
        }]
        router._fusion_retrieve = MagicMock(return_value=[dict(s) for s in seeds])
        for name in (
            "_community_expansion", "_mesa_synthesis", "_visual_recall",
            "_property_temporal_retrieve", "_entity_expansion",
            "_attribute_expansion", "_scope_retrieve", "_schema_recall",
            "_fact_retrieve",
        ):
            setattr(router, name, MagicMock(side_effect=lambda results, *a, **k: results))
        router._rerank_results = MagicMock()
        # 已学到 rerank 实测约 0.5s；deadline 只剩 0.05s → 1.2*0.5=0.6 > 0.05 → 跳过
        router._rerank_duration_estimate = 0.5

        deadline = time.monotonic() + 0.05
        t0 = time.monotonic()
        out = router.retrieve("alpha beta", level=RetrievalLevel.FUSION, deadline=deadline)
        elapsed = time.monotonic() - t0

        assert isinstance(out, list) and out
        assert elapsed < 0.4, f"预算不足应快速返回而非启动慢 rerank: {elapsed:.3f}s"
        assert router._rerank_results.call_count == 0, (
            "预算不够时不应启动 _rerank_results"
        )
        assert router._is_rerank_budget_disabled() is False, (
            "仅 1 次预算不足不应达到禁用阈值"
        )

    def test_three_consecutive_budget_exceeds_disables_rerank(self):
        """AC6：连续 3 次超预算 → _is_rerank_budget_disabled() 为真；此后预算充足也跳过。"""
        router = QueryRouter.__new__(QueryRouter)
        router.config = QueryRouterConfig(
            rerank_enabled=True,
            fusion_dedup_enabled=False,
        )
        router._zh_en_tech_map = {}
        router._time_keywords = set()
        router.encoder = None
        seeds = [{
            "node_id": "n0", "content": "alpha beta", "score": 0.9,
            "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss",
        }]
        router._fusion_retrieve = MagicMock(return_value=[dict(s) for s in seeds])
        for name in (
            "_community_expansion", "_mesa_synthesis", "_visual_recall",
            "_property_temporal_retrieve", "_entity_expansion",
            "_attribute_expansion", "_scope_retrieve", "_schema_recall",
            "_fact_retrieve",
        ):
            setattr(router, name, MagicMock(side_effect=lambda results, *a, **k: results))
        router._rerank_results = MagicMock(side_effect=lambda results, *a, **k: results)

        # 三次已过期（remaining<=0）→ 每次 note budget exceeded → 第三次禁用
        for _ in range(3):
            router.retrieve("alpha beta", level=RetrievalLevel.FUSION,
                            deadline=time.monotonic() - 0.001)
        assert router._is_rerank_budget_disabled() is True, (
            "连续 3 次超预算后应进程内禁用重排"
        )

        # 已禁用：即便预算充足（deadline 很远），rerank 也不再启动
        router._rerank_results.reset_mock()
        router.retrieve("alpha beta", level=RetrievalLevel.FUSION,
                        deadline=time.monotonic() + 10.0)
        assert router._rerank_results.call_count == 0, (
            "禁用后预算充足也不应再启动 rerank"
        )


class TestRetrieveWithDeadlineHelper:
    def test_forwards_deadline_when_callable_accepts_it(self):
        """入口 helper：签名支持 deadline 时透传（生产 QueryRouter/SelfEvolvingRetrieval）。"""
        from api.routes.search import _retrieve_with_deadline

        class R:
            def retrieve(self, query, include_archived=False, session_ts=None,
                         level=None, rerank=None, deadline=None):
                self.seen = dict(deadline=deadline, level=level)
                return []

        r = R()
        _retrieve_with_deadline(r, "q", deadline=123.0, level="fusion")
        assert r.seen["deadline"] == 123.0


class TestGenerationInvalidation:
    def test_bump_generation_misses_old_cache_key(self, client, mock_services):
        """AC4：写入 bump generation 后，旧代际缓存键不再命中。"""
        _clear_cache()
        mock_qr = MagicMock()
        mock_qr.config = QueryRouterConfig(rerank_enabled=False)
        mock_qr.retrieve.return_value = [{
            "node_id": "n_ok", "content": "cacheable content",
            "score": 0.9, "level": "hypergraph", "tau_value": 1.0,
        }]
        mock_services.query_router = mock_qr
        mock_services.graph_store.query_cypher.return_value = []

        q = "generation-probe-query"
        r1 = client.post("/memories/retrieve", json={"query": q, "top_k": 5})
        assert r1.status_code == 200, r1.text
        assert mock_qr.retrieve.call_count == 1
        with _result_cache_lock:
            assert len(_result_cache) == 1

        bump_result_cache_generation()

        r2 = client.post("/memories/retrieve", json={"query": q, "top_k": 5})
        assert r2.status_code == 200, r2.text
        assert mock_qr.retrieve.call_count == 2, (
            "bump generation 后旧缓存键不应命中，必须重新检索"
        )

    def test_generation_monotonic(self):
        g0 = current_result_cache_generation()
        g1 = bump_result_cache_generation()
        g2 = current_result_cache_generation()
        assert g1 == g0 + 1
        assert g2 == g1
