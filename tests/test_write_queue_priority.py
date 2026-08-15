"""
v5.40 写队列优先级 Write-Priority 测试
=====================================
覆盖（impl_v540.md 改动 4）:
  · 高优先插队 — low 先入队 / high 后入队 → high 先执行（含多任务混排）
  · 低优先不饿死 — 持续 high 流下 low 最终执行（非抢占，只重排不丢）
  · 背压 — high 满 → WriteQueueFullError；low 达准入闸 → 拒（high 容量预留）
  · 重入 — 写线程内 submit high/low 均同步执行（防死锁语义保持）
  · 切块端到端 — 写线程执行 low 长块时 high 到达 → 等待 ≤ 1 块（~块时长），
    证明梦境 PERSIST 切块后外部写不再被 30-60s 单体任务饿到 30s 超时 503

运行: python -m pytest tests/test_write_queue_priority.py -v
"""

import asyncio
import threading
import time

import pytest

from core.write_queue import WriteQueue, WriteQueueFullError


class TestHighPrioritySkipping:
    """高优先级任务插队：后入队的高优先先于先入队的低优先执行。"""

    def test_high_jumps_ahead_of_queued_low(self):
        """low 先入队，high 后入队（worker 忙）→ 放行后 high 先执行。"""
        q = WriteQueue(wait_timeout=5.0)
        entered = threading.Event()
        release = threading.Event()
        try:
            order = []

            def blocker():
                entered.set()
                release.wait(5)

            async def run():
                t0 = asyncio.create_task(q.submit(blocker))
                await asyncio.to_thread(entered.wait, 2)
                t_low = asyncio.create_task(
                    q.submit(lambda: order.append("low"), priority="low"))
                await asyncio.sleep(0.02)  # 确保 low 已入队（worker 忙）
                t_high = asyncio.create_task(
                    q.submit(lambda: order.append("high"), priority="high"))
                await asyncio.sleep(0.02)  # 确保 high 已入队
                release.set()              # 放行 blocker，worker 开始消费
                await asyncio.gather(t0, t_low, t_high)

            asyncio.run(run())
            assert order == ["high", "low"], f"high 应插队到 low 前: {order}"
        finally:
            release.set()
            q.shutdown()

    def test_high_first_among_mixed_queue(self):
        """low1(先) + high(后) + low2(最后) → high 最优先，low 间仍 FIFO。"""
        q = WriteQueue(wait_timeout=5.0)
        entered = threading.Event()
        release = threading.Event()
        try:
            order = []

            def blocker():
                entered.set()
                release.wait(5)

            async def run():
                t0 = asyncio.create_task(q.submit(blocker))
                await asyncio.to_thread(entered.wait, 2)
                t_low1 = asyncio.create_task(
                    q.submit(lambda: order.append("low1"), priority="low"))
                await asyncio.sleep(0.02)
                t_high = asyncio.create_task(
                    q.submit(lambda: order.append("high"), priority="high"))
                await asyncio.sleep(0.02)
                t_low2 = asyncio.create_task(
                    q.submit(lambda: order.append("low2"), priority="low"))
                await asyncio.sleep(0.02)
                release.set()
                await asyncio.gather(t0, t_low1, t_high, t_low2)

            asyncio.run(run())
            assert order == ["high", "low1", "low2"], (
                f"high 应排最前，同优先级保持入队序: {order}"
            )
        finally:
            release.set()
            q.shutdown()


class TestLowNotStarved:
    def test_low_eventually_executes_under_high_stream(self):
        """持续 high 流下 low 不饿死：非抢占，high 全插前面后 low 仍执行。"""
        q = WriteQueue(wait_timeout=5.0)
        entered = threading.Event()
        release = threading.Event()
        try:
            order = []

            def blocker():
                entered.set()
                release.wait(5)

            async def run():
                t0 = asyncio.create_task(q.submit(blocker))
                await asyncio.to_thread(entered.wait, 2)
                # low 先入队（worker 忙，排队在 high 流后）
                t_low = asyncio.create_task(
                    q.submit(lambda: order.append("low"), priority="low"))
                await asyncio.sleep(0.02)
                # high 流一次性全部入队（worker 忙 → 21 任务全排队，无间隙竞态）
                highs = [asyncio.create_task(
                    q.submit(lambda i=i: order.append(f"high{i}"), priority="high"))
                    for i in range(20)]
                await asyncio.sleep(0.05)
                release.set()  # 放行 blocker，worker 按优先级消费
                await asyncio.gather(t0, t_low, *highs)

            asyncio.run(run())
            assert order.count("low") == 1
            assert order[-1] == "low", f"low 应最终执行（不饿死）: {order}"
            assert order[:20] == [f"high{i}" for i in range(20)], (
                "high 流应全部排在 low 前（重排不丢写）"
            )
        finally:
            release.set()
            q.shutdown()


class TestBackpressure:
    def test_high_full_rejects(self):
        """队列真满（qsize==maxsize，含 high）→ WriteQueueFullError（与现状一致）。"""
        q = WriteQueue(max_pending=1, wait_timeout=5.0)
        entered = threading.Event()
        release = threading.Event()
        try:
            def blocker():
                entered.set()
                release.wait(5)

            async def run():
                t0 = asyncio.create_task(q.submit(blocker))
                await asyncio.to_thread(entered.wait, 2)
                t1 = asyncio.create_task(
                    q.submit(lambda: 1, priority="high"))
                await asyncio.sleep(0.02)
                with pytest.raises(WriteQueueFullError):
                    await q.submit(lambda: 2, priority="high")
                release.set()
                await asyncio.gather(t0, t1)

            asyncio.run(run())
        finally:
            release.set()
            q.shutdown()

    def test_low_gate_rejects_but_high_reserved(self):
        """低准入闸：low 达 low_max 拒；high 仍可入队（为 high 预留容量）。

        判别性：max_pending=5, low_max=3 → 3 个 low 占满 low 额度 → 第 4 个
        low 拒，但 high 仍能入队（qsize 3→4 < maxsize 5）。
        """
        q = WriteQueue(max_pending=5, wait_timeout=5.0, low_max=3)
        entered = threading.Event()
        release = threading.Event()
        try:
            def blocker():
                entered.set()
                release.wait(5)

            async def run():
                t0 = asyncio.create_task(q.submit(blocker))
                await asyncio.to_thread(entered.wait, 2)
                lows = [asyncio.create_task(
                    q.submit(lambda: None, priority="low")) for _ in range(3)]
                await asyncio.sleep(0.05)  # 确保 3 个 low 全部入队（worker 忙 → qsize=3）
                with pytest.raises(WriteQueueFullError):
                    await q.submit(lambda: None, priority="low")
                t_high = asyncio.create_task(
                    q.submit(lambda: None, priority="high"))
                await asyncio.sleep(0.05)  # high 已入队（worker 忙 → qsize=4）
                assert q.pending_count() == 4, (
                    f"high 应突破 low 准入闸（预留容量）: pending={q.pending_count()}"
                )
                release.set()
                await asyncio.gather(t0, *lows, t_high)

            asyncio.run(run())
        finally:
            release.set()
            q.shutdown()


class TestReentrancyPriority:
    def test_reentrant_submit_high_and_low_sync(self):
        """写线程内 submit(priority=high/low) 均同步执行（重入守卫优先于优先级）。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            seen = []

            def inner_high():
                seen.append("inner_high")
                return "h"

            def inner_low():
                seen.append("inner_low")
                return "l"

            def outer():
                r1 = asyncio.run(q.submit(inner_high, priority="high"))
                r2 = asyncio.run(q.submit(inner_low, priority="low"))
                return (r1, r2)

            assert asyncio.run(q.submit(outer)) == ("h", "l")
            assert seen == ["inner_high", "inner_low"], f"重入同步执行顺序: {seen}"
        finally:
            q.shutdown()


class TestChunkingEndToEnd:
    def test_high_waits_at_most_one_chunk(self):
        """切块端到端：写线程执行 low 长块时 high 到达 → 等待 ≤ 1 块（~块时长）。

        模拟梦境 PERSIST 切块（3 块 × 0.2s，原单体 0.6s）：high 在第 1 块执行期间
        提交 → 最大等待 ≈ 当前块剩余（< 0.2s）→ 远小于 30s wait_timeout → 不 503。
        同时 high 插到块 2/块 3 之前（块间排空 high 的机制验证）。
        """
        q = WriteQueue(wait_timeout=5.0)
        try:
            order = []

            def chunk(i):
                time.sleep(0.2)
                order.append(("low", i))

            async def run():
                t_low0 = asyncio.create_task(
                    q.submit(chunk, 0, priority="low"))
                await asyncio.sleep(0.05)  # 确保写线程已进入块 0（在途 low）
                t0 = time.monotonic()
                await q.submit(lambda: order.append(("high", 0)), priority="high")
                high_elapsed = time.monotonic() - t0
                t_low1 = asyncio.create_task(
                    q.submit(chunk, 1, priority="low"))
                t_low2 = asyncio.create_task(
                    q.submit(chunk, 2, priority="low"))
                await asyncio.gather(t_low0, t_low1, t_low2)
                return high_elapsed

            high_elapsed = asyncio.run(run())
            # 最大等待 ≤ 1 块（0.2s）+ 调度余量：证明切块后 high 不被长任务饿死
            assert high_elapsed < 0.5, (
                f"high 等待 {high_elapsed:.3f}s，应 ≤ 1 块（0.2s）+ 余量"
            )
            # 块间排空：high 先于后续 low 块执行
            assert order == [("low", 0), ("high", 0), ("low", 1), ("low", 2)], (
                f"high 应插到后续 low 块前: {order}"
            )
        finally:
            q.shutdown()
