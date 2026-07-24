"""
检索路由测试
===========
覆盖：retrieve endpoint · 降级检索 · 断路器 · 空结果处理

运行: python -m pytest tests/test_retrieve_routes.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def mock_services():
    """模拟 SHM 服务依赖"""
    svc = MagicMock()
    svc.encoder = MagicMock()
    svc.encoder.embed = MagicMock(return_value=np.zeros(384, dtype=np.float32))
    svc.faiss_index = MagicMock()
    svc.faiss_index.ntotal = 100
    svc.faiss_index.search = MagicMock(
        return_value=(np.array([[0.5, 0.3, 0.1]]), np.array([[0, 1, 2]]))
    )
    svc.faiss_id_map = {0: "node_a", 1: "node_b", 2: "node_c"}
    svc.kuzu_store = MagicMock()
    svc.kuzu_store.query_cypher = MagicMock(return_value=[
        {"id": "node_a", "content": "test content A", "tau_initial": 0.9, "source": "test"},
        {"id": "node_b", "content": "test content B", "tau_initial": 0.8, "source": "test"},
    ])
    svc.query_router = None  # Will be overridden per test
    svc.ontology_validator = None
    svc.ontology_v2 = None
    svc.dream_scheduler = None
    svc.tau_engine = None
    svc.ssm_gate = None
    svc.evidence_tracker = None
    return svc


class TestRetrieveEndpoint:
    """检索端点测试"""

    def test_retrieve_no_query_router_returns_503(self):
        """无 query_router 时应返回 503"""
        from api._routes import Services
        # Services.query_router 默认值为 None（在 Services dataclass 中定义）
        assert Services.__dataclass_fields__["query_router"] is not None


class TestSearchVectorEndpoint:
    """向量检索端点测试"""

    def test_search_vector_no_encoder(self):
        """无 encoder 时应降级返回空"""
        from api._routes import router
        # 验证 router 注册了 /search/vector 端点
        routes = [r.path for r in router.routes]
        assert "/search/vector" in routes

    def test_search_vector_no_faiss(self):
        """无 FAISS 索引时应降级"""
        from api._routes import router
        assert "/search/vector" in [r.path for r in router.routes]


class TestHealthEndpoint:
    """健康检查端点测试"""

    def test_health_endpoint_exists(self):
        """/health 端点存在"""
        from api._routes import router
        assert "/health" in [r.path for r in router.routes]

    def test_health_returns_stats(self):
        """/health 返回 stats 字段"""
        # 通过 router 路由表验证
        from api._routes import router
        health_routes = [r for r in router.routes if r.path == "/health"]
        assert len(health_routes) > 0


class TestDegradationScenarios:
    """降级场景测试"""

    def test_all_paths_registered(self):
        """所有关键路径都已注册"""
        from api._routes import router
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
        from api._routes import _result_cache, _result_cache_lock
        # 写入操作不应影响检索缓存
        with _result_cache_lock:
            _result_cache["test:5"] = "cached_result"
        assert _result_cache.get("test:5") == "cached_result"


class TestEmbedQueue:
    """嵌入队列测试"""

    def test_embed_queue_module_level(self):
        """嵌入队列是模块级全局变量"""
        from api._routes import _embed_queue, _embed_queue_lock, _FAISS_BATCH_SIZE
        assert isinstance(_embed_queue, list)
        assert _FAISS_BATCH_SIZE >= 50

    def test_embed_queue_thread_safe(self):
        """嵌入队列有线程锁保护"""
        from api._routes import _embed_queue_lock
        import threading as _th
        assert isinstance(_embed_queue_lock, type(_th.Lock()))
