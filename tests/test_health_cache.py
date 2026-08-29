"""Health check TTL cache 测试。

验证 /health 端点的 node_count/hyperedge_count 5s TTL 缓存行为：
- 首次请求 → 执行 COUNT(*) 查询 → 缓存结果
- 5s 内第二次请求 → 跳过 COUNT(*) → 使用缓存值
- 超过 5s 后请求 → 重新执行 COUNT(*) → 刷新缓存
- 写入后 ≤5s 内数值可能旧（容忍，TTL 设计）
"""
import time as _real_time
from unittest.mock import MagicMock, call

import pytest
from fastapi.testclient import TestClient

import api.routes.system as _sys_module


@pytest.fixture(autouse=True)
def _reset_health_cache():
    """每个测试前重置模块级缓存，确保测试隔离。"""
    _sys_module._HEALTH_STATS_CACHE = {"node_count": 0, "hyperedge_count": 0}
    _sys_module._HEALTH_STATS_CACHE_TIME = 0.0
    yield
    _sys_module._HEALTH_STATS_CACHE = {"node_count": 0, "hyperedge_count": 0}
    _sys_module._HEALTH_STATS_CACHE_TIME = 0.0


def _make_graph_store():
    """创建带计数的 mock graph_store，按查询内容返回不同结果。"""
    store = MagicMock()
    call_log = []

    def query_side_effect(query, params=None):
        call_log.append(query)
        if "RETURN 1 AS test" in query:
            return [[1]]
        if "count(*) AS cnt" in query and "HyperedgeNode" in query:
            return [[8]]  # hyperedge count
        if "count(*) AS cnt" in query:
            return [[42]]  # node count
        return []

    store.query_cypher.side_effect = query_side_effect
    store.circuit_breaker = None
    store._call_log = call_log
    return store


def _make_app(store):
    """构造带依赖注入的 FastAPI TestClient。"""
    from api.app import create_app
    from api.routes._deps import Services, get_services

    svc = Services()
    svc.graph_store = store
    svc.faiss_index = None
    svc.audit_chain = None
    svc.dream_scheduler = None

    app = create_app()
    app.dependency_overrides[get_services] = lambda: svc
    return TestClient(app)


class TestHealthCache:

    def test_first_call_runs_count_queries(self, monkeypatch):
        store = _make_graph_store()
        client = _make_app(store)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()

        assert data["graph_connected"] is True
        assert data["stats"]["node_count"] == 42
        assert data["stats"]["hyperedge_count"] == 8

        count_queries = [q for q in store._call_log if "count(*) AS cnt" in q]
        assert len(count_queries) == 2, (
            f"First call should run both COUNT queries, got {count_queries}"
        )

    def test_second_call_within_ttl_skips_count_queries(self, monkeypatch):
        store = _make_graph_store()
        client = _make_app(store)

        now = [_real_time.time()]

        def fake_time():
            return now[0]

        monkeypatch.setattr(_sys_module, "_now", fake_time)
        # Also patch _time.time() since our module uses it for _HEALTH_STATS_TTL
        # (but the cache check uses _now() from _deps, which maps to time.time())

        resp1 = client.get("/health")
        assert resp1.status_code == 200
        count_after_first = len([q for q in store._call_log if "count(*) AS cnt" in q])
        assert count_after_first == 2

        resp2 = client.get("/health")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["stats"]["node_count"] == 42
        assert data2["stats"]["hyperedge_count"] == 8

        count_after_second = len([q for q in store._call_log if "count(*) AS cnt" in q])
        assert count_after_second == 2, (
            f"Second call within TTL should NOT run COUNT queries, "
            f"but got {count_after_second} (first had {count_after_first})"
        )

    def test_after_ttl_expiry_refreshes_cache(self, monkeypatch):
        store = _make_graph_store()
        client = _make_app(store)

        fake_now = _real_time.time()
        monkeypatch.setattr(_sys_module, "_now", lambda: fake_now)

        resp1 = client.get("/health")
        count_after_first = len([q for q in store._call_log if "count(*) AS cnt" in q])
        assert count_after_first == 2

        fake_now += 6.0  # past 5s TTL
        monkeypatch.setattr(_sys_module, "_now", lambda: fake_now)

        resp2 = client.get("/health")
        data2 = resp2.json()
        assert data2["stats"]["node_count"] == 42

        count_after_second = len([q for q in store._call_log if "count(*) AS cnt" in q])
        assert count_after_second == 4, (
            f"After TTL expiry should re-run COUNT queries, "
            f"got {count_after_second} total (first had {count_after_first})"
        )

    def test_cache_hit_graph_connected_still_verified(self, monkeypatch):
        store = _make_graph_store()
        client = _make_app(store)

        # First call to populate cache
        resp1 = client.get("/health")
        assert resp1.status_code == 200

        # Second call within TTL — graph_connected still verified via RETURN 1
        resp2 = client.get("/health")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["graph_connected"] is True

        return1_calls = [q for q in store._call_log if "RETURN 1 AS test" in q]
        assert len(return1_calls) >= 1, "graph_connected should be verified each call"

    def test_cache_hit_circuit_breaker_not_unknown(self):
        """P1: 缓存命中时 circuit_breaker 应报告真实状态而非 unknown。

        此前 checker.graph_store=None 导致 _check_circuit_breaker()
        返回 {"state":"unknown"}。修复后从 deps.graph_store 重建。
        """
        store = _make_graph_store()
        # 设定真实断路器状态
        mock_cb = MagicMock()
        mock_cb.state = MagicMock()
        mock_cb.state.value = "closed"
        mock_cb._window = [True, True, True]
        store.circuit_breaker = mock_cb

        client = _make_app(store)

        # First call → cache miss, runs full check
        resp1 = client.get("/health")
        assert resp1.status_code == 200

        # Second call → cache hit, should still report real CB state
        resp2 = client.get("/health")
        assert resp2.status_code == 200
        data2 = resp2.json()
        cb = data2["stats"]["circuit_breaker"]
        assert cb["state"] == "closed", (
            f"cache hit should report real CB state, got {cb}"
        )
        assert cb["window_size"] == 3
        assert cb["success_rate"] == 100.0

    def test_cache_hit_graph_connected_false_when_empty(self):
        """P2: RETURN 1 返回空列表 → graph_connected=False。

        query_cypher 永不抛异常（契约），因此必须检查返回值。
        """
        store = _make_graph_store()

        # 修改 RETURN 1 返回空列表（模拟图不可达）
        call_log = []
        def query_side_effect(query, params=None):
            call_log.append(query)
            if "RETURN 1 AS test" in query:
                return []  # 图宕机，无行返回
            if "count(*) AS cnt" in query and "HyperedgeNode" in query:
                return [[8]]
            if "count(*) AS cnt" in query:
                return [[42]]
            return []
        store.query_cypher.side_effect = query_side_effect
        store._call_log = call_log

        client = _make_app(store)

        # First call → cache miss, checker has graph_store
        resp1 = client.get("/health")
        assert resp1.status_code == 200
        # _check_graph returns False because RETURN 1 returns []
        assert resp1.json()["graph_connected"] is False
        assert resp1.json()["status"] == "error"

        # Second call → cache hit, RETURN 1 check also returns []
        resp2 = client.get("/health")
        assert resp2.status_code == 200
        assert resp2.json()["graph_connected"] is False
