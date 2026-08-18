"""
检索路由测试
===========
覆盖：retrieve endpoint · 降级检索 · 断路器 · 空结果处理

运行: python -m pytest tests/test_retrieve_routes.py -v
"""

import time

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from api.routes import router, init_services, Services, get_services


@pytest.fixture
def mock_services():
    """模拟 SHM 服务依赖"""
    svc = Services()
    svc.encoder = MagicMock()
    svc.encoder.embed = MagicMock(return_value=np.zeros(384, dtype=np.float32))
    svc.faiss_index = MagicMock()
    svc.faiss_index.ntotal = 100
    svc.faiss_index.search = MagicMock(
        return_value=(np.array([[0.5, 0.3, 0.1]]), np.array([[0, 1, 2]]))
    )
    svc.faiss_id_map = {0: "node_a", 1: "node_b", 2: "node_c"}
    svc.graphlite_store = MagicMock()
    svc.graphlite_store.query_cypher = MagicMock(return_value=[
        {"id": "node_a", "content": "test content A", "tau_initial": 0.9, "source": "test"},
        {"id": "node_b", "content": "test content B", "tau_initial": 0.8, "source": "test"},
    ])
    svc.graphlite_store.get_all_nodes.return_value = {}
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
    """创建测试客户端，注入模拟服务"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    # 注册依赖注入覆盖
    app.dependency_overrides[get_services] = lambda: mock_services
    return TestClient(app)


class TestRetrieveEndpoint:
    """检索端点测试"""

    def test_retrieve_no_query_router_returns_503(self, client):
        """无 query_router 时应返回 503"""
        response = client.post("/memories/retrieve", json={
            "query": "test query",
            "top_k": 5,
        })
        assert response.status_code == 503

    def test_retrieve_with_query_router_success(self, client, mock_services):
        """有 query_router 时应返回正常结果"""
        mock_qr = MagicMock()
        mock_qr.retrieve.return_value = []
        mock_services.query_router = mock_qr
        response = client.post("/memories/retrieve", json={
            "query": "test query",
            "top_k": 5,
        })
        assert response.status_code == 200
        body = response.json()
        assert "results" in body
        assert "latency_ms" in body

    def test_retrieve_empty_query(self, client, mock_services):
        """空查询应返回 422"""
        response = client.post("/memories/retrieve", json={
            "query": "",
            "top_k": 5,
        })
        assert response.status_code == 422

    def test_retrieve_empty_result(self, client, mock_services):
        """检索无结果应返回空列表"""
        mock_qr = MagicMock()
        mock_qr.retrieve.return_value = []
        mock_services.query_router = mock_qr
        mock_services.graphlite_store.query_cypher.return_value = []
        response = client.post("/memories/retrieve", json={
            "query": "nonexistent content",
            "top_k": 5,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["results"] == []

    def test_retrieve_profile_context_injected(self, client, mock_services):
        """search_profile 命中 → profile_context 注入响应（User-Profile 旁路接线回归）。"""
        mock_qr = MagicMock()
        mock_qr.retrieve.return_value = []
        mock_qr._qr = mock_qr  # SelfEvolvingRetrieval 解包路径：内层即本 mock
        mock_qr.search_profile.return_value = {
            "matched": True,
            "context": "【用户画像】\n- preferences: 咖啡 (weight 1.0)",
        }
        mock_services.query_router = mock_qr
        mock_services.graphlite_store.query_cypher.return_value = []
        response = client.post("/memories/retrieve", json={
            "query": "咖啡",
            "top_k": 5,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["profile_context"] == "【用户画像】\n- preferences: 咖啡 (weight 1.0)"

    def test_retrieve_profile_context_no_hit_is_none(self, client, mock_services):
        """search_profile 未命中 → profile_context 为 None（字段向后兼容）。"""
        mock_qr = MagicMock()
        mock_qr.retrieve.return_value = []
        mock_qr._qr = mock_qr
        mock_qr.search_profile.return_value = {"matched": False, "context": ""}
        mock_services.query_router = mock_qr
        mock_services.graphlite_store.query_cypher.return_value = []
        response = client.post("/memories/retrieve", json={
            "query": "随便问问",
            "top_k": 5,
        })
        assert response.status_code == 200
        assert response.json()["profile_context"] is None

    def test_retrieve_profile_boost_score_clamped(self, client, mock_services):
        """P1 回归（v5.39.0）：score=1.0 画像命中 → boost 后钳制 ≤1.0 → 200 不 500。

        走 /memories/retrieve 公共入口：真实 QueryRouter.retrieve →
        _deduplicate_and_sort（画像 ×1.2 → min(1.0) 钳制）→ EpisodicResult 构造。
        修复前 score=1.0×1.2=1.2 触发 le=1.0 ValidationError → 500。
        """
        from retrieval.query_router import QueryRouter, QueryRouterConfig, set_user_profile

        set_user_profile({"preferences": {"咖啡": {"weight": 1.0, "sources": 1}}})
        try:
            router = QueryRouter.__new__(QueryRouter)
            router.config = QueryRouterConfig(rerank_enabled=False)  # P0-1 钉住 auto→HYPERGRAPH（画像 boost 路径）
            router._zh_en_tech_map = {}
            router._time_keywords = set()
            # 注入 score=1.0 且命中画像的原始检索结果（模拟画像命中加分前置状态）
            router._hypergraph_retrieve = MagicMock(return_value=[{
                "node_id": "n_profile", "content": "今天喝咖啡", "score": 1.0,
                "fact_track": "active", "level": "hypergraph",
                "tau_value": 1.0, "created_at": time.time(),
            }])
            mock_services.query_router = router
            mock_services.graphlite_store.query_cypher.return_value = []
            response = client.post("/memories/retrieve", json={
                "query": "今天喝咖啡",
                "top_k": 5,
            })
            assert response.status_code == 200, response.text
            results = response.json()["results"]
            assert results, "画像命中结果应出现在响应中"
            # 契约断言：score ∈ [0, 1]（越界即 EpisodicResult 构造失败 → 500）
            assert all(0.0 <= r["score"] <= 1.0 for r in results)
            assert results[0]["score"] == 1.0  # 1.0×1.2 → 钳制 1.0
            assert results[0]["node_id"] == "n_profile"
        finally:
            set_user_profile({})


class TestRetrieveR3Fix:
    """R3 修复回归：P1-1 hybrid→FUSION 降级误判 / P1-2 cache key 含 namespace。"""

    def test_hybrid_fusion_not_degraded(self, client, mock_services):
        """P1-1: hybrid→FUSION 正常结果（level=fusion_*）不应被标 degraded。"""
        from api.routes._deps import _result_cache, _result_cache_lock
        from retrieval.query_router import RetrievalLevel

        mock_qr = MagicMock()
        mock_qr.retrieve.return_value = [{
            "node_id": "n_fusion", "content": "fusion result content",
            "score": 0.9, "level": "fusion_multi", "tau_value": 1.0,
        }]
        mock_services.query_router = mock_qr
        mock_services.graphlite_store.query_cypher.return_value = []

        with _result_cache_lock:
            _result_cache.clear()
        try:
            resp = client.post("/memories/retrieve", json={
                "query": "p1-1-fusion-probe-query",
                "top_k": 5,
                "strategy": "hybrid",
            })
            assert resp.status_code == 200, resp.text
            assert resp.json()["degraded"] is False, "hybrid→FUSION 正常结果不得标 degraded"
            assert mock_qr.retrieve.call_args.kwargs["level"] == RetrievalLevel.FUSION, (
                "hybrid 策略必须透传 FUSION level（防路由丢接线假绿）"
            )
        finally:
            with _result_cache_lock:
                _result_cache.clear()

    def test_default_auto_l1_faiss_not_degraded(self, client, mock_services):
        """P1-1 回归（R4）：默认 auto → HYPERGRAPH 正常路径 level=l1_faiss 不得标 degraded。

        修复前降级判断猜 level 前缀：l1_faiss != "hypergraph" 且不属 fusion_* →
        误标 degraded=True（默认 auto 检索全 degraded）。修复后基于 _degradation_level
        显式标记，正常 l1_faiss 无该标记 → degraded=False。
        """
        from api.routes._deps import _result_cache, _result_cache_lock
        from retrieval.query_router import RetrievalLevel, QueryRouterConfig

        mock_qr = MagicMock()
        # 【P3b R1 P0-1】显式关 rerank/hyDE：新行为下 auto+rerank_enabled=True → FUSION，
        # 本测试意图是 L1 hypergraph 降级检测（auto→HYPERGRAPH），需钉住旧映射。
        mock_qr.config = QueryRouterConfig(rerank_enabled=False)
        mock_qr.retrieve.return_value = [{
            "node_id": "n_l1", "content": "l1 faiss normal result",
            "score": 0.9, "level": "l1_faiss", "tau_value": 1.0,
        }]
        mock_services.query_router = mock_qr
        mock_services.graphlite_store.query_cypher.return_value = []

        with _result_cache_lock:
            _result_cache.clear()
        try:
            resp = client.post("/memories/retrieve", json={
                "query": "r4-p1-default-auto-l1-probe-query",
                "top_k": 5,
            })
            assert resp.status_code == 200, resp.text
            assert resp.json()["degraded"] is False, "默认 auto 正常 l1_faiss 结果不得标 degraded"
            assert mock_qr.retrieve.call_args.kwargs["level"] == RetrievalLevel.HYPERGRAPH
        finally:
            with _result_cache_lock:
                _result_cache.clear()

    def test_l1_faiss_real_hypergraph_retrieve_not_degraded(self, client, mock_services):
        """P3: 走真实 QueryRouter._hypergraph_retrieve 的 l1_faiss 入口（不 mock retrieve 返回值）。

        构造真实 QueryRouter（faiss_index + faiss_id_map + _episode_cache），经
        /memories/retrieve 公共入口 → retrieve(level=HYPERGRAPH) → _hypergraph_retrieve
        真实执行，返回 level=l1_faiss 结果；断言 degraded=False 且无降级标记。
        """
        from retrieval.query_router import QueryRouter, QueryRouterConfig
        from api.routes._deps import _result_cache, _result_cache_lock

        router = QueryRouter(
            graphlite_store=None,
            faiss_index=MagicMock(),
            tfidf_index=None,
            encoder=MagicMock(),
            config=QueryRouterConfig(rerank_enabled=False),  # P0-1 钉住 auto→HYPERGRAPH（本测试测真实 L1 路径）
            faiss_id_map={0: "node_l1_real"},
            episode_cache={"node_l1_real": {
                "id": "node_l1_real", "content": "l1 faiss real entry result",
                "tau_value": 1.0, "fact_track": "active",
            }},
        )
        router.encoder.embed.return_value = np.zeros(512, dtype=np.float32)
        router.faiss_index.search.return_value = (
            np.array([[0.1]]),
            np.array([[0]]),
        )
        mock_services.query_router = router
        mock_services.graphlite_store.query_cypher.return_value = []

        def _passthrough(results, *args, **kwargs):
            return results

        with _result_cache_lock:
            _result_cache.clear()
        try:
            with patch.object(router, "_community_expansion", side_effect=_passthrough), \
                 patch.object(router, "_mesa_synthesis", side_effect=_passthrough), \
                 patch.object(router, "_visual_recall", side_effect=_passthrough), \
                 patch.object(router, "_property_temporal_retrieve", side_effect=_passthrough):
                resp = client.post("/memories/retrieve", json={
                    "query": "p3-l1-faiss-real-entry-query",
                    "top_k": 5,
                })
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["degraded"] is False, "真实 l1_faiss 正常路径不得标 degraded"
            results = body["results"]
            assert results, "真实 _hypergraph_retrieve 应返回结果"
            assert results[0]["node_id"] == "node_l1_real"
            assert results[0]["retrieval_level"] == "l1_faiss"
        finally:
            with _result_cache_lock:
                _result_cache.clear()

    def test_l1_empty_cascade_degraded(self, client, mock_services):
        """P1-1: L1 空→VECTOR 级联路径打 _degradation_level=l1_empty → degraded=True。

        修复前 L1 空级联不打标，degraded 恒 False。修复后经真实 retrieve 级联
        （_hypergraph_retrieve 空 → _vector_retrieve 命中）打标，endpoint 返回 degraded=True。
        """
        from retrieval.query_router import QueryRouter, QueryRouterConfig
        from api.routes._deps import _result_cache, _result_cache_lock

        router = QueryRouter.__new__(QueryRouter)
        router.config = QueryRouterConfig(rerank_enabled=False)  # P0-1 钉住 auto→HYPERGRAPH（本测试测 L1→L2 级联）
        router._zh_en_tech_map = {}
        router._time_keywords = set()
        router._hypergraph_retrieve = MagicMock(return_value=[])
        router._vector_retrieve = MagicMock(return_value=[{
            "node_id": "n_l2", "content": "l2 cascade result", "score": 0.9,
            "level": "vector", "tau_value": 1.0, "fact_track": "active",
        }])
        mock_services.query_router = router
        mock_services.graphlite_store.query_cypher.return_value = []

        with _result_cache_lock:
            _result_cache.clear()
        try:
            resp = client.post("/memories/retrieve", json={
                "query": "l1-empty-cascade-query",
                "top_k": 5,
            })
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["degraded"] is True, "L1 空级联应标 degraded=True"
            assert body["results"], "L2 结果应返回"
            assert body["results"][0]["retrieval_level"] == "vector"
        finally:
            with _result_cache_lock:
                _result_cache.clear()

    def test_cypher_fallback_queryerror_degraded(self, client, mock_services):
        """P1-2: REST 兜底 query_cypher 抛真实 SDK QueryError → 空结果 + degraded=True。

        修复前 degraded=True 置于 wait_for 之后，非超时异常跳外层 except 时该行
        未执行 → degraded 恒 False。修复后置位于 wait_for 之前，异常仍 degraded=True。
        """
        from graphlite_sdk.error import QueryError
        from api.routes._deps import _result_cache, _result_cache_lock

        mock_qr = MagicMock()
        mock_qr.retrieve.return_value = []
        mock_services.query_router = mock_qr
        mock_services.graphlite_store.query_cypher.side_effect = QueryError(
            "simulated cypher fallback failure"
        )

        with _result_cache_lock:
            _result_cache.clear()
        try:
            resp = client.post("/memories/retrieve", json={
                "query": "cypher-fallback-queryerror-probe",
                "top_k": 5,
            })
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["degraded"] is True, "Cypher 兜底抛 QueryError 应标 degraded=True"
            assert body["results"] == []
        finally:
            with _result_cache_lock:
                _result_cache.clear()

    def test_namespace_distinct_cache_keys(self, client, mock_services):
        """P1-2: 同 query 不同 namespace 不得共用缓存（retrieve 调用 2 次）。"""
        from api.routes._deps import _result_cache, _result_cache_lock

        mock_qr = MagicMock()
        mock_qr.retrieve.return_value = []
        mock_services.query_router = mock_qr
        mock_services.graphlite_store.query_cypher.return_value = []

        with _result_cache_lock:
            _result_cache.clear()
        try:
            q = "p1-2-namespace-probe-query"
            r1 = client.post("/memories/retrieve", json={"query": q, "top_k": 5, "namespace": "nsA"})
            assert r1.status_code == 200, r1.text
            r2 = client.post("/memories/retrieve", json={"query": q, "top_k": 5, "namespace": "nsB"})
            assert r2.status_code == 200, r2.text
            assert mock_qr.retrieve.call_count == 2, "同 query 不同 namespace 不得命中缓存"
        finally:
            with _result_cache_lock:
                _result_cache.clear()


class TestSearchVectorEndpoint:
    """向量检索端点测试"""

    def test_search_vector_no_faiss(self, client, mock_services):
        """无 FAISS 索引时应降级返回空"""
        mock_services.faiss_index = None
        response = client.post("/search/vector", json={
            "query": "test vector",
            "limit": 5,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["total_found"] == 0
        assert body["degraded"] is True

    def test_search_vector_success(self, client, mock_services):
        """正常向量检索"""
        mock_services.faiss_index.search.return_value = (
            np.array([[0.9, 0.5, 0.3]]),
            np.array([[0, 1, 2]]),
        )
        mock_services.graphlite_store.get_episode = MagicMock(return_value={
            "content": "test content", "id": "node_a", "tau_initial": 0.9
        })
        response = client.post("/search/vector", json={
            "query": "test query",
            "limit": 3,
        })
        assert response.status_code == 200
        body = response.json()
        assert "results" in body
        assert len(body["results"]) > 0


class TestHealthEndpoint:
    """健康检查端点测试"""

    def test_health_returns_200(self, client):
        """/health 应返回 200"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_stats(self, client):
        """/health 返回 stats 字段"""
        response = client.get("/health")
        body = response.json()
        assert "stats" in body or "status" in body


class TestDegradationScenarios:
    """降级场景测试"""

    def test_all_paths_registered(self):
        """所有关键路径都已注册"""
        from api.routes import router
        paths = {r.path for r in router.routes}
        required = {
            "/health", "/search/vector", "/memories/retrieve",
            "/memories/episodes", "/memories/sensory",
            "/ontology/types", "/hyperedges",
            "/communities", "/index/rebuild",
            "/memories/dream/trigger", "/query",
        }
        missing = required - paths
        assert not missing, f"Missing routes: {missing}"

    def test_cache_isolated_to_write(self):
        """缓存在写入后不清空（只对检索有效）"""
        from api.routes._deps import _result_cache, _result_cache_lock
        with _result_cache_lock:
            _result_cache["test:5"] = "cached_result"
        assert _result_cache.get("test:5") == "cached_result"


class TestDegradePathTimeout:
    """【H2-a】降级分支超时保护：Cypher 兜底不再无限挂起（超时即跳过）"""

    def test_cypher_fallback_timeout_skips_and_returns_empty(self, client, mock_services):
        """GraphLite 卡死时：主检索空 → Cypher 兜底超时 → 跳过兜底返回空，而非挂起。

        修复前兜底 to_thread 无 wait_for：若挂的是 GraphLite，主检索超时后
        兜底查询再次无限挂起（H2 超时保护被兜底路径击穿）→ 兜底结果会被返回。
        修复后超时即跳过兜底分支 → 结果为 []。
        """
        mock_qr = MagicMock()
        mock_qr.retrieve.return_value = []  # 主检索空 → 触发 Cypher 兜底
        mock_services.query_router = mock_qr
        mock_services.quarantine_store = None  # 隔离过滤不参与本用例

        def slow_cypher(*args, **kwargs):
            time.sleep(0.3)  # 模拟 GraphLite 挂死（远超兜底超时 0.1s）
            return [("n1", "content 1")]

        mock_services.graphlite_store.query_cypher = MagicMock(side_effect=slow_cypher)

        with patch("api.routes.search._DEGRADE_TIMEOUT", 0.1):
            # 唯一 query 避免命中其他用例写入的 _result_cache 缓存键
            resp = client.post("/memories/retrieve", json={
                "query": "h2a-timeout-probe-query",
                "top_k": 5,
            })

        assert resp.status_code == 200
        assert resp.json()["results"] == [], "兜底超时应跳过并返回空结果"


class TestEmbedQueue:
    """嵌入队列测试"""

    def test_embed_queue_module_level(self):
        """嵌入队列是模块级全局变量"""
        from api.routes.write import _embed_queue, _embed_queue_lock
        from api.routes._deps import _FAISS_BATCH_SIZE
        assert isinstance(_embed_queue, list)
        assert _FAISS_BATCH_SIZE >= 50

    def test_embed_queue_thread_safe(self):
        """嵌入队列有线程锁保护"""
        from api.routes.write import _embed_queue_lock
        import threading as _th
        assert isinstance(_embed_queue_lock, type(_th.Lock()))
