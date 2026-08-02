"""熔断器落地冒烟测试（H1）+ with_retry 双模式验证（H2）。

覆盖:
- CircuitBreaker 状态机: closed → open → half_open → closed/open
- 滑动窗口失败率与阈值判定
- P1-1 并发访问保护: 多线程并发 record_*/allow_request 不破坏窗口不变量（RLock）
- P1-2 half_open 探针配额按时间重新武装（探针被消耗但无 record_* 时不再永久卡 half_open）
- P2-3 record_failure 只对基础设施错误（ConnectionError/TimeoutError）计数
- P2-A execute_cypher 写路径对熔断窗口完全中立（成功/失败均不计数）
- P2-B 重试成功（F→T）不污染窗口；重试耗尽才按查询结果计 1 次失败
- P2-C 断路器 open → API 全局异常处理器返回 503 而非 500
- P2-D get_episodes_batch 加 @with_retry 重试（与 query_cypher 一致）
- GraphLiteStore.query_cypher 永不抛异常契约（open 返回 [] 而非 raise）——P0-2
- GraphLiteStore.execute_cypher open 时 raise CircuitBreakerOpen（写路径显式失败）——P0-2/P2-2
- P0-1 get_episodes_batch 熔断门控 + query_router L1 超图检索级联（L613 可触发）
- HealthChecker 报告 open 状态
- with_retry 同步/异步包装（同步包装器不改变同步调用语义）
"""
import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graph.graphlite_store import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitBreakerState,
)


def _cfg(**overrides):
    base = dict(
        failure_threshold=0.5,
        recovery_timeout=30.0,
        half_open_max_requests=1,
        window_size=10,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestCircuitBreakerStateMachine:

    def test_initial_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb._window == []
        assert cb.allow_request() is True
        assert cb.is_open() is False

    def test_partial_window_does_not_trip(self):
        """窗口不满时不跳闸：单次失败不切断图存储。"""
        cb = CircuitBreaker()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.allow_request() is True

    def test_trips_when_full_window_rate_exceeds_threshold(self):
        """窗口满且失败率 ≥ 阈值 → open，并 raise CircuitBreakerOpen。"""
        cb = CircuitBreaker(_cfg(window_size=2))
        cb.record_failure()  # [F] 窗口不满
        assert cb.state == CircuitBreakerState.CLOSED
        with pytest.raises(CircuitBreakerOpen):
            cb.record_failure()  # [F,F] 满窗口 100% ≥ 50%
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.allow_request() is False
        assert cb.is_open() is True

    def test_below_threshold_stays_closed(self):
        """阈值 1.0：满窗口 [F,F,T] 失败率 2/3 < 1.0，不跳闸。"""
        cb = CircuitBreaker(_cfg(failure_threshold=1.0, window_size=3))
        for r in (False, False, True):
            cb.record_failure() if not r else cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.allow_request() is True

    def test_window_slides_at_window_size(self):
        """window_size=3：窗口满后滑动，只保留最近 3 条。"""
        cb = CircuitBreaker(_cfg(failure_threshold=1.0, window_size=3))
        for r in (False, False, True, True):
            cb.record_failure() if not r else cb.record_success()
        assert len(cb._window) == 3
        assert cb._window == [False, True, True]

    def test_open_to_half_open_after_recovery_timeout(self):
        """open 后 recovery_timeout 已过 → 自动迁移 half_open，放行 1 个探测。"""
        cb = CircuitBreaker(_cfg(window_size=1))
        with pytest.raises(CircuitBreakerOpen):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        cb._opened_at = time.time() - 31.0  # 模拟 recovery_timeout 已过
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.allow_request() is True  # 放行第 1 个探测
        assert cb.allow_request() is False  # 探测配额耗尽（时间未到不重新武装）

    def test_half_open_probe_quota_rearms_after_timeout(self):
        """P1-2: 探针被消耗但无 record_* 时，间隔 recovery_timeout/half_open_max_requests
        后配额重新武装，不再永久卡 half_open。"""
        cb = CircuitBreaker(_cfg(window_size=1, recovery_timeout=30.0))
        with pytest.raises(CircuitBreakerOpen):
            cb.record_failure()
        cb._opened_at = time.time() - 31.0
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.allow_request() is True   # 第 1 个探测被消耗，无 record_*
        assert cb.allow_request() is False  # 配额耗尽
        assert cb.state == CircuitBreakerState.HALF_OPEN  # 未永久卡死
        # 距上次探测超过 recovery_timeout/half_open_max_requests → 重新武装
        cb._half_open_last_probe_at = time.time() - 31.0
        assert cb.allow_request() is True

    def test_half_open_probe_success_closes(self):
        cb = CircuitBreaker(_cfg(window_size=1))
        with pytest.raises(CircuitBreakerOpen):
            cb.record_failure()
        cb._opened_at = time.time() - 31.0
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb._window == []

    def test_half_open_probe_failure_reopens(self):
        cb = CircuitBreaker(_cfg(window_size=1))
        with pytest.raises(CircuitBreakerOpen):
            cb.record_failure()
        cb._opened_at = time.time() - 31.0
        assert cb.state == CircuitBreakerState.HALF_OPEN
        with pytest.raises(CircuitBreakerOpen):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_record_failure_counts_only_infra_errors(self):
        """P2-3: 应用错误（GQL 语法等 RuntimeError）不计数，不污染熔断窗口。"""
        cb = CircuitBreaker(_cfg(window_size=2))
        cb.record_failure(RuntimeError("gql syntax error"))
        cb.record_failure(RuntimeError("gql syntax error"))
        assert cb._window == []  # 应用错误全部不计数
        assert cb.state == CircuitBreakerState.CLOSED
        cb.record_failure(ConnectionError("graphlite down"))
        with pytest.raises(CircuitBreakerOpen):
            cb.record_failure(TimeoutError("graphlite down"))
        assert cb._window == [False, False]  # 基础设施错误计数
        assert cb.state == CircuitBreakerState.OPEN

    def test_record_failure_counts_real_sdk_query_error(self):
        """P0 回归: 真实 SDK QueryError 必须计数（此前被 isinstance 过滤 → 熔断器死代码）。"""
        from graphlite_sdk.error import QueryError

        cb = CircuitBreaker(_cfg(window_size=2))
        cb.record_failure(QueryError("Query failed: graphlite down"))
        assert cb._window == [False]  # 真实 SDK 异常计数（窗口 +1）
        assert cb.state == CircuitBreakerState.CLOSED
        with pytest.raises(CircuitBreakerOpen):
            cb.record_failure(QueryError("Query failed: still down"))
        assert cb._window == [False, False]
        assert cb.state == CircuitBreakerState.OPEN  # 窗口积累 → 跳闸

    def test_concurrent_state_mutations_are_thread_safe(self):
        """P1-1: 多线程并发 allow_request/record_success 不破坏窗口不变量。"""
        cb = CircuitBreaker(_cfg(window_size=100, failure_threshold=1.0))
        errors: list[BaseException] = []

        def worker():
            try:
                for _ in range(200):
                    cb.allow_request()
                    cb.record_success()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(cb._window) <= cb.window_size


class TestGraphLiteStoreCircuitBreaker:

    def test_query_success_records_success(self, graphlite_store):
        store = graphlite_store
        store.query_cypher("RETURN 1 AS test")
        assert store.circuit_breaker._window == [True]
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_query_failure_trips_breaker_returns_empty(self, graphlite_store):
        """连续基础设施失败触发 open：query_cypher 不抛，返回 []（永不抛异常契约）。"""
        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=1))
        store._session = MagicMock()
        store._session.query.side_effect = ConnectionError("graphlite down")
        assert store.query_cypher("MATCH (n) RETURN n") == []
        assert store.circuit_breaker.state == CircuitBreakerState.OPEN

    def test_query_failure_trips_breaker_with_real_sdk_query_error(self, graphlite_store):
        """P0 回归: SDK QueryError（生产真实形态）触发熔断，query_cypher 静默返回 []。

        此前 except (ConnectionError, TimeoutError) 匹配不到 SDK QueryError → 永不计数。
        """
        from graphlite_sdk.error import QueryError

        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=1))
        store._session = MagicMock()
        store._session.query.side_effect = QueryError("Query failed: connection refused")
        assert store.query_cypher("MATCH (n) RETURN n") == []
        assert store.circuit_breaker.state == CircuitBreakerState.OPEN

    def test_query_cypher_returns_empty_when_open(self, graphlite_store):
        """冒烟: query_cypher open 时返回 [] 不抛，且不再触达 session。"""
        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=1))
        store.circuit_breaker._state = CircuitBreakerState.OPEN
        store.circuit_breaker._opened_at = time.time()  # 保持 OPEN（避免属性自动迁移 half_open）
        store._session = MagicMock()
        assert store.query_cypher("MATCH (n) RETURN n") == []
        store._session.query.assert_not_called()

    def test_execute_cypher_raises_when_open(self, graphlite_store):
        """冒烟: execute_cypher open 时 raise CircuitBreakerOpen，且不再触达 session。"""
        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=1))
        store.circuit_breaker._state = CircuitBreakerState.OPEN
        store.circuit_breaker._opened_at = time.time()  # 保持 OPEN（避免属性自动迁移 half_open）
        store._session = MagicMock()
        with pytest.raises(CircuitBreakerOpen):
            store.execute_cypher("MATCH (n) RETURN n")
        store._session.query.assert_not_called()

    def test_execute_cypher_app_error_not_counted(self, graphlite_store):
        """execute_cypher 不吞异常：应用错误（GQL 语法）直接上抛，且不计数（P2-3）。"""
        store = graphlite_store
        store._session = MagicMock()
        store._session.query.side_effect = RuntimeError("gql syntax error")
        with pytest.raises(RuntimeError):
            store.execute_cypher("INVALID GQL")
        assert store.circuit_breaker._window == []  # 应用错误不计数
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_query_retry_exhausted_returns_empty(self, graphlite_store):
        """P2-1: 基础设施错误重试耗尽（max_attempts=2）→ query_cypher 返回 [] 不抛。"""
        store = graphlite_store
        store._session = MagicMock()
        store._session.query.side_effect = ConnectionError("still down")
        assert store.query_cypher("MATCH (n) RETURN n") == []
        assert store._session.query.call_count == 2  # 恰好重试 2 次
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED  # 窗口未满不跳闸

    def test_query_retry_real_sdk_query_error_calls_twice(self, graphlite_store):
        """P1-1: 真实 SDK QueryError 也必须触发 with_retry 重试（调用 2 次）。

        此前 @with_retry 用默认 retryable_exceptions（内置 ConnectionError/TimeoutError），
        匹配不到 SDK QueryError → 生产重试是死代码，只调用 1 次。
        """
        from graphlite_sdk.error import QueryError

        store = graphlite_store
        store._session = MagicMock()
        store._session.query.side_effect = QueryError("Query failed: connection refused")
        assert store.query_cypher("MATCH (n) RETURN n") == []
        assert store._session.query.call_count == 2  # P1-1: 真实 SDK 异常恰好重试 2 次
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED  # 窗口未满不跳闸

    def test_get_episodes_batch_raises_when_open(self, graphlite_store):
        """P0-1: 熔断门控——open 状态 get_episodes_batch raise CircuitBreakerOpen。"""
        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=1))
        store.circuit_breaker._state = CircuitBreakerState.OPEN
        store.circuit_breaker._opened_at = time.time()  # 保持 OPEN（避免属性自动迁移 half_open）
        store._session = MagicMock()
        with pytest.raises(CircuitBreakerOpen):
            store.get_episodes_batch(["uuid-1"])
        store._session.query.assert_not_called()

    def test_l1_cascades_to_l2_when_breaker_open(self, graphlite_store):
        """P0-1: L1 超图检索熔断 → retrieve() 级联 L2 并打标 l1_circuit_breaker。"""
        import numpy as np
        from retrieval.query_router import (
            QueryRouter,
            QueryRouterConfig,
            RetrievalLevel,
        )

        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=1))
        store.circuit_breaker._state = CircuitBreakerState.OPEN
        store.circuit_breaker._opened_at = time.time()  # 保持 OPEN（避免属性自动迁移 half_open）
        store._session = MagicMock()

        router = QueryRouter.__new__(QueryRouter)
        router.graphlite_store = store
        router.faiss_index = MagicMock()
        router.faiss_id_map = {0: "uuid-1"}
        router.config = QueryRouterConfig()
        router._episode_cache = {}
        router.encoder = None
        router._time_keywords = []
        router._zh_en_tech_map = {}
        router.faiss_index.search.return_value = (
            np.array([[0.5]]),
            np.array([[0]]),
        )

        results = router.retrieve(
            "hello", query_embedding=np.array([[0.1]]), level=RetrievalLevel.HYPERGRAPH
        )
        assert results, "L2 应返回降级结果"
        assert results[0]["_degradation_level"] == "l1_circuit_breaker"
        assert results[0]["node_id"] == "uuid-1"

    def test_health_check_reports_open_state(self, graphlite_store):
        """health 检查显示 open 状态（此前恒 not_configured）。"""
        from observability.health import HealthChecker

        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=1))
        store._session = MagicMock()
        store._session.query.side_effect = ConnectionError("graphlite down")
        store.query_cypher("MATCH (n) RETURN n")  # 跳闸，query_cypher 静默返回 []

        result = HealthChecker(graph_store=store).check()
        cb_report = result.details["circuit_breaker"]
        assert cb_report["state"] == "open"
        assert cb_report["recent_failures"] == 1
        assert cb_report["window_size"] == 1
        assert cb_report["success_rate"] == 0.0

    # ─── P2-A: 写路径对熔断窗口完全中立 ─────────────────

    def test_execute_cypher_success_neutral_to_window(self, graphlite_store):
        """P2-A: execute_cypher 成功不再 record_success——写路径不稀释读失败率。"""
        store = graphlite_store
        store._session = MagicMock()
        store._session.query.return_value = SimpleNamespace(rows=[{"n": 1}])
        assert store.execute_cypher("MATCH (n) RETURN n") == [{"n": 1}]
        assert store.circuit_breaker._window == []  # 写路径成功不写样本
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_execute_cypher_failure_neutral_to_window(self, graphlite_store):
        """P2-A: execute_cypher 失败 N 次不计数（窗口不变），读失败才计数。"""
        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=3))
        store._session = MagicMock()
        store._session.query.side_effect = ConnectionError("graphlite down")
        for _ in range(3):
            with pytest.raises(ConnectionError):
                store.execute_cypher("MATCH (n) RETURN n")
        assert store.circuit_breaker._window == []  # 写失败不污染窗口
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED
        # 读路径失败才计数（重试耗尽 → 计 1 次失败）
        assert store.query_cypher("MATCH (n) RETURN n") == []
        assert store.circuit_breaker._window == [False]
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED

    # ─── P2-B: 窗口按「查询结果」计数而非「attempt」 ─────

    def test_query_retry_success_no_failure_pollution(self, graphlite_store):
        """P2-B: 查询需重试但最终成功（F→T）→ 窗口不含失败记录。"""
        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=2))
        store._session = MagicMock()
        calls = {"n": 0}

        def flaky_query(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return SimpleNamespace(rows=[{"n": 1}])

        store._session.query.side_effect = flaky_query
        assert store.query_cypher("MATCH (n) RETURN n") == [{"n": 1}]
        assert calls["n"] == 2  # 恰好重试成功
        assert store.circuit_breaker._window == [True]  # 只 record_success，无失败污染
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_query_retry_exhausted_counts_failure_once(self, graphlite_store):
        """P2-B: 重试耗尽才计失败——按查询结果计 1 次而非每次 attempt。"""
        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=10))
        store._session = MagicMock()
        store._session.query.side_effect = ConnectionError("still down")
        assert store.query_cypher("MATCH (n) RETURN n") == []
        assert store._session.query.call_count == 2  # 恰好重试 2 次
        assert store.circuit_breaker._window == [False]  # 只 1 次失败
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED

    # ─── P2-D: get_episodes_batch 重试（与 query_cypher 一致） ──

    def test_get_episodes_batch_retry_success_no_failure_pollution(self, graphlite_store):
        """P2-D: get_episodes_batch 需重试但成功 → 窗口只含成功样本。"""
        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=2))
        store._session = MagicMock()
        calls = {"n": 0}

        def flaky_query(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return SimpleNamespace(
                rows=[{"e": {"Node": {"properties": {"id": "uuid-1"}}}}]
            )

        store._session.query.side_effect = flaky_query
        result = store.get_episodes_batch(["uuid-1"])
        assert calls["n"] == 2
        assert result == [{"id": "uuid-1"}]
        assert store.circuit_breaker._window == [True]
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_get_episodes_batch_retry_exhausted_counts_failure_once(self, graphlite_store):
        """P2-D: get_episodes_batch 重试耗尽 → 按查询结果计 1 次失败 + 返回 []。"""
        store = graphlite_store
        store.circuit_breaker = CircuitBreaker(_cfg(window_size=10))
        store._session = MagicMock()
        store._session.query.side_effect = ConnectionError("graphlite down")
        assert store.get_episodes_batch(["uuid-1"]) == []
        assert store._session.query.call_count == 2  # 与 query_cypher 一致：重试 2 次
        assert store.circuit_breaker._window == [False]  # 只 1 次失败
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED


class TestCircuitBreakerOpenHTTP:
    """P2-C: 断路器 open 时 API 返回 503 而非 500（全局异常处理器）。"""

    def test_circuit_open_returns_503_not_500(self, monkeypatch):
        """未防护读路径（如 /conflicts 用 execute_cypher）跳闸 → 503 circuit_open。"""
        monkeypatch.setenv("DEV_MODE", "true")  # 跳过认证中间件
        from fastapi.testclient import TestClient

        from api.app import create_app
        from api.routes import Services, get_services

        svc = Services()
        store = MagicMock()
        store.execute_cypher.side_effect = CircuitBreakerOpen(
            "circuit breaker open, query rejected"
        )
        svc.graphlite_store = store

        app = create_app()
        app.dependency_overrides[get_services] = lambda: svc
        client = TestClient(app)  # 不进 context → lifespan 不执行
        resp = client.get("/conflicts")
        assert resp.status_code == 503
        assert resp.json() == {"error": "circuit_open"}


class TestWithRetryDualMode:

    def test_sync_wrapper_retries_and_returns(self):
        from core.retry import with_retry

        calls = {"n": 0}

        @with_retry(max_attempts=3, base_delay=0.01, backoff=2.0)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        assert flaky() == "ok"  # 同步调用，无需 await
        assert calls["n"] == 3

    def test_sync_wrapper_not_a_coroutine(self):
        """同步函数装饰后仍是同步函数（调用方无需 await）。"""
        from core.retry import with_retry

        @with_retry(max_attempts=2, base_delay=0.01)
        def f():
            return 42

        assert f() == 42
        assert not asyncio.iscoroutinefunction(f)

    def test_async_wrapper_retries(self):
        from core.retry import with_retry

        calls = {"n": 0}

        @with_retry(max_attempts=3, base_delay=0.01, backoff=2.0)
        async def flaky_async():
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("transient")
            return "ok"

        assert asyncio.run(flaky_async()) == "ok"
        assert calls["n"] == 3

    def test_retries_exhausted_raises_last(self):
        from core.retry import with_retry

        @with_retry(max_attempts=3, base_delay=0.01, backoff=2.0)
        def always_fails():
            raise ConnectionError("still down")

        with pytest.raises(ConnectionError):
            always_fails()
