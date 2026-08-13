"""
v5.28 写队列永久卡死修复测试
============================
覆盖（2026-08-13 修复，Codex 复核后修正）:
  · F1 — submit 不再占用 executor 线程：写线程卡死时，后续请求各自超时，
    不排队等死（旧实现 run_in_executor 单 worker 被占死 → 全 503 永不恢复）
  · F1 — 迟到完成语义保持：超时后写线程仍落库，且其他请求不受影响
  · F2 — 无 _executor/_await_future 残留（纯 wrap_future 桥接）
  · F3 — 看门狗仅线程死亡时重启（alive+慢写不重启，避免双写并发）；
        空闲队列不误判卡死（unfinished_tasks==0 守卫）
  · F3 — shutdown 满队列不永久阻塞（put_nowait + 跳过 sentinel）

运行: python -m pytest tests/test_write_queue_v528.py -v
"""

import asyncio
import threading
import time

import pytest

from core.write_queue import WriteQueue


class TestF1NoExecutorStarvation:
    """核心修复：写线程卡死不再饿死后续请求的等待路径。"""

    def test_stuck_write_does_not_starve_other_requests(self):
        """写线程卡在任务1上，请求2/3 仍能在各自 wait_timeout 内失败，
        且等待路径不排队等死（旧实现单 worker executor 被占死）。"""
        q = WriteQueue(max_pending=10, wait_timeout=0.3)
        try:
            entered = threading.Event()
            release = threading.Event()

            def blocker():
                entered.set()
                release.wait(5)  # 模拟 GraphLite 卡死：永不返回

            def fast():
                return 42

            async def run():
                t1 = asyncio.create_task(q.submit(blocker))
                await asyncio.to_thread(entered.wait, 2)
                t2 = asyncio.create_task(q.submit(fast))
                t3 = asyncio.create_task(q.submit(fast))
                results = await asyncio.wait_for(
                    asyncio.gather(t1, t2, t3, return_exceptions=True),
                    timeout=5.0,
                )
                return results

            results = asyncio.run(run())
            assert isinstance(results[0], asyncio.TimeoutError), f"t1={results[0]!r}"
            for i, r in enumerate(results[1:], start=2):
                assert isinstance(r, asyncio.TimeoutError), f"t{i}={r!r}"
            # MED-3 判别性：卡死窗口内必须只有写线程 worker（无 executor 线程）。
            # 精确匹配 name=="shm-writer-worker"（LOW-3：避免子串误伤 daemon 线程）
            writer_threads = [
                t.name for t in threading.enumerate() if "shm-writer" in t.name
            ]
            assert writer_threads == ["shm-writer-worker"], (
                f"应只有写线程(worker)，实为: {writer_threads}"
            )
            # 队列未满（可继续入队）
            assert q.pending_count() <= q.max_pending
        finally:
            release.set()
            q.shutdown(drain=True, drain_timeout=0.5)

    def test_no_executor_attribute(self):
        """F2：不再持有专用 executor（旧实现线程被占死的根源）。"""
        q = WriteQueue(wait_timeout=1.0)
        try:
            assert not hasattr(q, "_executor"), "_executor 应已删除"
            assert not hasattr(q, "_await_future"), "_await_future 应已删除"
        finally:
            q.shutdown()

    def test_late_completion_kept(self):
        """超时后写线程仍完成（迟到完成语义保持）。"""
        q = WriteQueue(wait_timeout=0.05)
        try:
            done = []

            def slow():
                time.sleep(0.2)
                done.append(1)

            with pytest.raises(asyncio.TimeoutError):
                asyncio.run(q.submit(slow))
            for _ in range(50):
                if done:
                    break
                time.sleep(0.02)
            assert done == [1], "写任务应在超时后仍真实落库"
            # 队列仍可用
            assert asyncio.run(q.submit(lambda: 7)) == 7
        finally:
            q.shutdown()


class TestF3Watchdog:
    """看门狗：仅线程死亡时重启；空闲不误判。"""

    def test_worker_death_auto_restart(self):
        """写线程死亡 → 下次 submit 检测到并重启，队列继续消费。"""
        q = WriteQueue(wait_timeout=2.0)
        try:
            old = q._worker
            q._worker = threading.Thread(
                target=lambda: None, daemon=True, name="shm-writer-worker-dead"
            )
            q._worker.start()
            q._worker.join(timeout=1)  # 让线程跑完退出 → is_alive=False
            assert not q._worker.is_alive(), "模拟死亡线程应已退出"
            result = asyncio.run(q.submit(lambda: "alive"))
            assert result == "alive"
            assert q._worker.is_alive()
            assert old is not q._worker
        finally:
            q.shutdown()

    def test_is_stuck_idle_not_stuck(self):
        """HIGH-1：空闲队列（无在途任务）即使心跳陈旧也不算卡死。"""
        q = WriteQueue(wait_timeout=0.1)
        try:
            q._last_activity = time.monotonic() - q._stuck_timeout - 1
            assert not q._is_stuck(), "空闲队列不应判定卡死（误判会 spurious 重启）"
            assert q._worker.is_alive(), "未重启（守护生效）"
        finally:
            q.shutdown()

    def test_is_stuck_detects_inflight_heartbeat_timeout(self):
        """有在途任务 + 心跳超时 → 判定卡死（诊断用，不触发重启）。"""
        q = WriteQueue(wait_timeout=0.2)
        try:
            entered = threading.Event()
            release = threading.Event()

            def blocker():
                entered.set()
                release.wait(5)

            async def fire():
                asyncio.ensure_future(q.submit(blocker))
                await asyncio.sleep(0.05)

            asyncio.run(fire())
            assert entered.is_set(), "blocker 应已进入写线程"
            # 有在途任务 + 心跳陈旧 → stuck=True（仅诊断，_restart_worker 不重启 alive）
            q._last_activity = time.monotonic() - q._stuck_timeout - 1
            assert q._is_stuck()
            worker_before = q._worker
            q._restart_worker()
            assert q._worker is worker_before, "alive+慢写不得重启（HIGH-2：避免双写并发）"
        finally:
            release.set()
            q.shutdown(drain=False)

    def test_restart_dead_worker_keeps_processing(self):
        """线程死亡 → 重启后队列任务继续被消费（FIFO 顺序保持）。"""
        q = WriteQueue(wait_timeout=2.0)
        try:
            order = []

            def record(i):
                order.append(i)

            # 制造线程死亡 → _restart_worker 真正重建
            q._worker = threading.Thread(
                target=lambda: None, daemon=True, name="shm-writer-worker-dead"
            )
            q._worker.start()
            q._worker.join(timeout=1)
            assert not q._worker.is_alive()
            old_ident = q._worker_ident
            q._restart_worker()
            assert q._worker_ident != old_ident, "死亡线程应已重建"
            asyncio.run(q.submit(record, 1))
            asyncio.run(q.submit(record, 2))
            for _ in range(50):
                if len(order) >= 2:
                    break
                time.sleep(0.02)
            assert order == [1, 2], f"重启后队列消费异常: {order}"
        finally:
            q.shutdown()


class TestShutdown:
    """MED-4：shutdown 满队列不永久阻塞。"""

    def test_shutdown_full_queue_does_not_hang(self):
        """MED-4 判别测试：worker 卡死 + **队列真满**（qsize==maxsize）
        → shutdown 走 put_nowait skip-sentinel 分支，有界返回不挂起。

        旧实现 _q.put(_SENTINEL) 阻塞 put 永久等空位 → uvicorn 关闭挂起。
        """
        q = WriteQueue(max_pending=2, wait_timeout=5.0)
        entered = threading.Event()
        release = threading.Event()

        def blocker():
            entered.set()
            release.wait(30)

        async def fill():
            # 任务1 在途（占写线程，永不返回）→ unfinished_tasks=1
            asyncio.ensure_future(q.submit(blocker))
            await asyncio.sleep(0.1)
            assert entered.is_set(), "blocker 应已进入写线程"
            # 任务2、任务3 入队（pending）→ qsize=2 == maxsize → 队列真满
            asyncio.ensure_future(q.submit(blocker))
            asyncio.ensure_future(q.submit(blocker))
            await asyncio.sleep(0.1)
            assert q.pending_count() == 2, "应有 2 条 pending 占满队列"

        asyncio.run(fill())
        t0 = time.monotonic()
        q.shutdown(drain=True, drain_timeout=0.2)  # 不应永久阻塞
        elapsed = time.monotonic() - t0
        assert elapsed < 7.0, f"shutdown hung {elapsed:.1f}s"
        release.set()

    def test_shutdown_full_queue_fails_pending_immediately(self):
        """M3：worker 卡死 + 队列真满 → shutdown 清空积压，pending future 立即收到
        WriteQueueClosedError（而非各自挂到 wait_timeout 超时）；shutdown 按时返回。

        修复前积压任务的等待方在 shutdown 后仍挂 wait_for(wait_timeout) 直到超时；
        修复后 get_nowait 清空 + fut.set_exception + task_done() + 重放 sentinel。
        """
        from core.write_queue import WriteQueueClosedError

        q = WriteQueue(max_pending=2, wait_timeout=10.0)
        entered = threading.Event()
        release = threading.Event()

        def blocker():
            entered.set()
            release.wait(30)

        async def scenario():
            t1 = asyncio.create_task(q.submit(blocker))  # 在途（占写线程）
            await asyncio.sleep(0.1)
            assert entered.is_set(), "blocker 应已进入写线程"
            t2 = asyncio.create_task(q.submit(blocker))  # pending
            t3 = asyncio.create_task(q.submit(blocker))  # pending → 队列满
            await asyncio.sleep(0.1)
            assert q.pending_count() == 2, "应有 2 条 pending 占满队列"

            t0 = time.monotonic()
            q.shutdown(drain=True, drain_timeout=0.2)
            elapsed = time.monotonic() - t0
            assert elapsed < 7.0, f"shutdown hung {elapsed:.1f}s"

            # 积压任务立即失败（WriteQueueClosedError）：wait_timeout=10s 远大于
            # shutdown 耗时（~5.2s），证明是 M3 set_exception 而非各自自然超时。
            results = await asyncio.wait_for(
                asyncio.gather(t2, t3, return_exceptions=True),
                timeout=1.0,
            )
            for r in results:
                assert isinstance(r, WriteQueueClosedError), f"pending 应收到关闭异常: {r!r}"
            # 在途任务仍由写线程持有：放行后迟到完成（迟到完成语义不破坏）
            release.set()
            r1 = await asyncio.wait_for(
                asyncio.gather(t1, return_exceptions=True), timeout=2.0
            )
            assert r1[0] is None, f"在途任务应迟到完成（blocker 返回 None）: {r1[0]!r}"
            return elapsed

        elapsed = asyncio.run(scenario())
        release.set()
        assert elapsed < 7.0, f"shutdown hung {elapsed:.1f}s"
