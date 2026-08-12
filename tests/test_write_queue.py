"""
v5.23 写串行化队列测试
======================
覆盖（.trio-task-v523.md §新增测试）:
  · FIFO 顺序 — 多任务按入队顺序执行
  · 异常传播 — fn 抛错 → submit 抛同异常
  · 超时 — wait_for 超时抛 asyncio.TimeoutError + 迟到完成（任务仍落库, 写线程不崩溃）
  · 队列满拒绝 — max_pending 满 → WriteQueueFullError（背压）
  · 关闭拒绝 — shutdown 后 submit → WriteQueueClosedError
  · 写线程重入直连 — 写线程内 submit 直接同步执行（不死锁）
  · 读不受写影响 — 慢写期间事件循环心跳无大间隔
  · 并发吞吐 — 8 并发 × 10 条 = 80 条, 平均壁钟 < 500ms/条（验收标准）
  · 路由集成 — Services.write_queue 注入后写路由经队列执行
  · qsubmit 回退 — 无队列 → 同步直调; 队列满/超时 → HTTPException 503

运行: python -m pytest tests/test_write_queue.py -v
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.routes import router, Services, get_services
from api.routes._deps import qsubmit
from core.write_queue import WriteQueue, WriteQueueFullError, WriteQueueClosedError


def _slow_write(delay: float, completed: list) -> None:
    """模拟单次 GraphLite 写耗时 delay 秒（单写者物理上限语义）。"""
    time.sleep(delay)
    completed.append(1)


def _record(i: int, order: list) -> None:
    time.sleep(0.005)
    order.append(i)


class TestWriteQueueFIFO:
    def test_fifo_order_preserved(self):
        """多任务按入队顺序串行执行（create_episode → ensure_session →
        link_to_session 的 FIFO 有序性基础）。"""
        q = WriteQueue(wait_timeout=10.0)
        try:
            order = []
            asyncio.run(self._run(q, order))
            assert order == list(range(20))
        finally:
            q.shutdown()

    @staticmethod
    async def _run(q, order):
        for i in range(20):
            await q.submit(_record, i, order)

    def test_concurrent_submits_stay_serialized(self):
        """并发 submit 仍被写线程串行执行（同一时刻只有一个 fn 在跑）。"""
        q = WriteQueue(wait_timeout=30.0)
        try:
            active = []
            peak = []
            seen = []

            def track(i):
                active.append(1)
                peak.append(len(active))
                time.sleep(0.01)
                active.pop()
                seen.append(i)

            async def run():
                await asyncio.gather(*[q.submit(track, i) for i in range(12)])

            asyncio.run(run())
            assert max(peak) == 1, f"writes overlapped: peak concurrency {max(peak)}"
            assert sorted(seen) == list(range(12))
        finally:
            q.shutdown()


class TestWriteQueueExceptions:
    def test_exception_propagates(self):
        """写线程 fn 抛错 → submit 抛同异常（异常经 future 原样传播）。"""
        q = WriteQueue(wait_timeout=10.0)
        try:
            def boom():
                raise RuntimeError("boom")

            with pytest.raises(RuntimeError, match="boom"):
                asyncio.run(q.submit(boom))
        finally:
            q.shutdown()

    def test_closed_rejects_new_submits(self):
        q = WriteQueue(wait_timeout=10.0)
        q.shutdown()
        with pytest.raises(WriteQueueClosedError):
            asyncio.run(q.submit(_record, 0, []))


class TestWriteQueueBackpressure:
    def test_queue_full_rejects(self):
        """max_pending 满 → WriteQueueFullError（背压拒绝, 不无限堆积）。"""
        q = WriteQueue(max_pending=1, wait_timeout=5.0)
        try:
            started = threading.Event()
            release = threading.Event()

            def blocker():
                started.set()
                release.wait(5)

            async def run():
                t0 = asyncio.create_task(q.submit(blocker))
                await asyncio.to_thread(started.wait, 5)  # 等 worker 卡在 blocker
                t1 = asyncio.create_task(q.submit(_record, 1, []))  # 入队 1（worker 忙, 排队）
                await asyncio.sleep(0.02)                 # 确保 t1 已完成 put_nowait
                with pytest.raises(WriteQueueFullError):
                    await q.submit(_record, 2, [])        # 第 3 个 → 背压拒绝（同步, 不需 worker）
                release.set()                             # 放行 worker
                await asyncio.gather(t0, t1)

            asyncio.run(run())
        finally:
            q.shutdown()

    def test_queue_recovers_after_drain(self):
        """排空后队列恢复可用（拒绝不是永久性熔断）。"""
        q = WriteQueue(max_pending=1, wait_timeout=10.0)
        try:
            async def run():
                await q.submit(_slow_write, 0.05, [])
                assert await q.submit(lambda: 7) == 7

            asyncio.run(run())
        finally:
            q.shutdown()


class TestWriteQueueTimeout:
    def test_timeout_raises_and_late_completion(self):
        """超过 wait_timeout 抛 asyncio.TimeoutError；任务仍会迟到完成（不丢写,
        写线程不崩溃）。"""
        q = WriteQueue(wait_timeout=0.1)
        try:
            done = []
            with pytest.raises(asyncio.TimeoutError):
                asyncio.run(q.submit(_slow_write, 0.3, done))
            # 迟到完成：等待写线程真正完成
            for _ in range(50):
                if done:
                    break
                time.sleep(0.02)
            assert done == [1], "写任务应在超时后仍真实落库（迟到完成语义）"
            # 队列仍可用（写线程未因 InvalidStateError 崩溃）
            assert asyncio.run(q.submit(lambda: 7)) == 7
        finally:
            q.shutdown()


class TestWriteQueueReentrancy:
    def test_reentrant_submit_runs_synchronously(self):
        """写线程内再 submit → 直接同步执行（防死锁）。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            seen = []

            def inner():
                seen.append("inner")
                return 42

            def outer():
                # 写线程内重入：asyncio.run 启动新 loop 调 submit，
                # 命中 worker_ident 守卫 → 同步执行不排队
                return asyncio.run(q.submit(inner))

            assert asyncio.run(q.submit(outer)) == 42
            assert seen == ["inner"]
        finally:
            q.shutdown()


class TestWriteQueueShutdown:
    def test_shutdown_drain_bounded_on_stuck_write(self):
        """F4: drain 限时 — 写任务卡死时 shutdown 有界返回（不无限挂起）。

        修复前: _q.join() 无超时 → 写线程死锁（GraphLite 卡死）时
        _worker.join(timeout=5.0) 兜底永远执行不到, uvicorn 关闭挂起。
        修复后: drain 以 drain_timeout 为界轮询超时, shutdown 总耗时
        上界 ≈ drain_timeout + worker.join(5s) 兜底。
        """
        q = WriteQueue(wait_timeout=30.0)
        entered = threading.Event()
        release = threading.Event()

        def blocker():
            entered.set()
            release.wait(30)  # 模拟 GraphLite 死锁: 永不返回

        async def fire():
            asyncio.ensure_future(q.submit(blocker))
            await asyncio.sleep(0.1)  # 让 worker 进入 blocker

        asyncio.run(fire())
        assert entered.is_set(), "worker did not enter blocked write"

        t0 = time.monotonic()
        q.shutdown(drain=True, drain_timeout=0.2)
        elapsed = time.monotonic() - t0
        # 有界: drain 0.2s + worker.join 5.0s 兜底 → 远小于无限挂起
        assert elapsed < 7.0, f"shutdown hung {elapsed:.1f}s on stuck write"
        release.set()  # 放行 worker 线程退出


class TestWriteQueueLoopResponsiveness:
    def test_reads_not_affected_by_writes(self):
        """慢写占据写线程期间，事件循环保持响应（读路径不受写阻塞影响）。

        修复前：同步 GraphLite 写直调卡死整个 loop，心跳间隔会被拉大到写耗时；
        修复后：写在线程池/写线程，loop 心跳间隔 ≈ 定时器间隔。
        """
        q = WriteQueue(wait_timeout=10.0)
        try:
            ticks = []

            async def heartbeat():
                for _ in range(120):
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.01)

            async def run():
                hb = asyncio.create_task(heartbeat())
                await q.submit(_slow_write, 0.4, [])
                await q.submit(_slow_write, 0.4, [])
                await hb

            asyncio.run(run())
            gaps = [b - a for a, b in zip(ticks, ticks[1:])]
            max_gap = max(gaps)
            # 若写阻塞 loop，gap ≈ 0.4s；正常 jitter ≈ 0.01-0.03s
            assert max_gap < 0.12, f"event loop stalled during writes: max gap {max_gap:.3f}s"
        finally:
            q.shutdown()


class TestWriteQueueThroughput:
    def test_8_concurrent_writers_80_items_avg_under_500ms(self):
        """队列机制基准：8 并发写 80 条在串行化下无排队风暴（机制验证）。

        ⚠️ 验收口径已重定义（Codex F3）: 绝对 avg<500ms 与物理现实矛盾——
        单写 237ms × 80 条串行 → 平均等待 ≈8s 物理不可能。本测试用 8ms
        合成写验证**队列机制**（串行化不放大排队延迟）；真实端到端口径
        =「队列开销≈0，总耗时 ≈ N × 单写耗时」见
        tests/test_write_queue_real_graphlite.py::TestRealThroughput。
        串行写 8ms/条 → 理论平均排队+写 ≈ 40.5×8ms ≈ 324ms < 500ms。
        """
        q = WriteQueue(wait_timeout=30.0)
        try:
            n_writers, n_per = 8, 10
            completed = []
            latencies = []

            async def writer(_w):
                for _ in range(n_per):
                    t0 = time.monotonic()
                    await q.submit(_slow_write, 0.008, completed)
                    latencies.append(time.monotonic() - t0)

            async def run():
                await asyncio.gather(*[writer(w) for w in range(n_writers)])

            asyncio.run(run())
            assert len(completed) == n_writers * n_per
            avg_ms = (sum(latencies) / len(latencies)) * 1000
            assert avg_ms < 500, f"avg {avg_ms:.0f}ms/条 >= 500ms (峰值 {max(latencies)*1000:.0f}ms)"
        finally:
            q.shutdown()


class TestQsubmit:
    """qsubmit 帮助函数：写队列集成点 + 降级/503 语义。"""

    def test_fallback_to_direct_call_without_queue(self):
        """无 write_queue（测试/降级）→ 同步直调, 行为与改造前一致。"""
        svc = Services()
        gstore = MagicMock()
        gstore.create_episode = MagicMock(return_value="e1")
        svc.graphlite_store = gstore

        result = asyncio.run(qsubmit(svc, gstore.create_episode, {"id": "e1"}))
        assert result == "e1"
        gstore.create_episode.assert_called_once_with({"id": "e1"})

    def test_queue_full_maps_to_503(self):
        class FullQueue:
            async def submit(self, *a, **k):
                raise WriteQueueFullError("queue full")

        svc = Services()
        svc.write_queue = FullQueue()
        with pytest.raises(HTTPException) as ei:
            asyncio.run(qsubmit(svc, lambda: None))
        assert ei.value.status_code == 503

    def test_timeout_maps_to_503(self):
        class TimeoutQueue:
            async def submit(self, *a, **k):
                raise asyncio.TimeoutError("late")

        svc = Services()
        svc.write_queue = TimeoutQueue()
        with pytest.raises(HTTPException) as ei:
            asyncio.run(qsubmit(svc, lambda: None))
        assert ei.value.status_code == 503

    def test_closed_maps_to_503(self):
        """F5: 队列关闭后 submit → WriteQueueClosedError → 503（关闭竞态不落 500）。"""
        class ClosedQueue:
            async def submit(self, *a, **k):
                raise WriteQueueClosedError("queue closed")

        svc = Services()
        svc.write_queue = ClosedQueue()
        with pytest.raises(HTTPException) as ei:
            asyncio.run(qsubmit(svc, lambda: None))
        assert ei.value.status_code == 503


# ─── 路由集成（真实 WriteQueue + mock store）──────────────────


def _make_svc(**overrides) -> Services:
    svc = Services()
    gstore = MagicMock()
    gstore.create_episode = MagicMock(return_value=None)
    gstore.execute_cypher = MagicMock(return_value=False)
    gstore.ensure_session = MagicMock()
    gstore.link_to_session = MagicMock()
    gstore.get_episode = MagicMock(return_value=None)
    gstore.get_or_create_session = MagicMock(return_value="")
    svc.graphlite_store = gstore
    for k, v in overrides.items():
        setattr(svc, k, v)
    return svc


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    def _build(svc):
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app)

    return _build


class TestWriteRouteWithQueue:
    def test_create_episode_executes_via_queue(self, client):
        """注入 write_queue 后, 写路由的 GraphLite 写调用经队列执行并返回 200。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_svc(write_queue=q)
            resp = client(svc).post("/memories/episodes", json={
                "content": "queue integration", "source": "tester", "namespace": "ns1",
            })
            assert resp.status_code == 200, resp.text
            svc.graphlite_store.create_episode.assert_called_once()
            payload = svc.graphlite_store.create_episode.call_args[0][0]
            assert payload["content"] == "queue integration"
            # 命名空间链接经队列且顺序保持（ensure 先于 link）
            svc.graphlite_store.ensure_session.assert_called_once_with("ns1")
            svc.graphlite_store.link_to_session.assert_called_once()
            assert svc.graphlite_store.link_to_session.call_args[0][0] == "ns1"
            # 请求结束后队列排空（无残留任务）
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_sensory_route_fallback_write_via_queue(self, client):
        """write_sensory 无环形缓冲区兜底路径也走队列。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_svc(write_queue=q)
            svc.graphlite_store._sensory_buffer = None  # 触发无环形缓冲区的兜底写路径
            resp = client(svc).post("/memories/sensory", json={
                "content": "sensory via queue", "source": "tester",
            })
            assert resp.status_code == 200, resp.text
            svc.graphlite_store.create_episode.assert_called_once()
            payload = svc.graphlite_store.create_episode.call_args[0][0]
            assert payload["content"] == "sensory via queue"
        finally:
            q.shutdown()
