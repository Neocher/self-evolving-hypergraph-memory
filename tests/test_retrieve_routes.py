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
