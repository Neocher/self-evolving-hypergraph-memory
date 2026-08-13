"""
P2-1 梦境写压力感知测试
======================
覆盖:
  · 持续写入 (30s 内 ≥ 阈值) → check_and_trigger 推迟返回 False
  · 推迟不丢候选 (累积计数保留)
  · 写入停止后梦境正常触发
  · 显式触发不受写压力影响
  · 【H2】写队列深度守卫: PERSIST 在队列过半满时跳过 (degraded, 零写)
运行: python -m pytest tests/test_dream_pressure.py -v
"""
import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock

from core.dream_scheduler import DreamScheduler, DreamSchedulerConfig


def run(coro):
    return asyncio.run(coro)


async def _fast_pipeline(*args, **kwargs):
    from core.dream_pipeline import DreamReport
    return DreamReport(
        dream_id="d1", trigger_mode="explicit", timestamp=time.time(),
        duration_seconds=0.01, stats={}, community_count=0, prune_count=0,
        conflict_count=0, audit_block_hash="",
    )


def _sched(**cfg):
    return DreamScheduler(
        config=DreamSchedulerConfig(min_interval_seconds=0, **cfg),
        pipeline_fn=_fast_pipeline,
    )


class TestWritePressureDefer:
    def test_deferred_under_sustained_writes(self):
        sched = _sched(write_pressure_window_seconds=30.0, write_pressure_threshold=15)
        sched._last_run_time = 0.0
        sched._new_node_count = 100  # 满足累积触发条件
        # 模拟 20 次写入发生在最近 30s 内
        for i in range(20):
            run(sched.on_node_created())

        triggered = run(sched.check_and_trigger())
        assert triggered is False, "写压力下应推迟梦境"
        assert sched.is_running is False
        assert sched._dream_run_count == 0

    def test_defer_keeps_candidates(self):
        sched = _sched()
        sched._last_run_time = 0.0
        sched._new_node_count = 100
        for _ in range(20):
            run(sched.on_node_created())
        before = sched.accumulated_count  # 100 + 20 次 on_node_created
        run(sched.check_and_trigger())
        # 推迟不丢候选: 累积计数保留 (未被重置为 0)
        assert sched.accumulated_count == before
        assert sched.accumulated_count > 0

    def test_runs_after_writes_settle(self):
        sched = _sched()
        sched._last_run_time = 0.0
        sched._new_node_count = 100
        for _ in range(20):
            run(sched.on_node_created())
        assert run(sched.check_and_trigger()) is False
        # 写入停止: 清空压力窗口
        sched._recent_write_times.clear()

        # 【FIX】触发 + 等待后台梦境完成必须在同一事件循环内：
        # check_and_trigger 内部 asyncio.create_task 创建的 _run_dream 任务，
        # 在 asyncio.run 返回时会被取消，导致 _dream_run_count 永不递增。
        async def _trigger_and_wait():
            triggered = await sched.check_and_trigger()
            assert triggered is True, "写入停止后应恢复触发"
            for _ in range(100):
                if not sched.is_running:
                    return
                await asyncio.sleep(0.01)
        run(_trigger_and_wait())
        assert sched._dream_run_count == 1

    def test_old_writes_no_pressure(self):
        """30s 之前的写入不构成压力。"""
        sched = _sched(write_pressure_window_seconds=30.0, write_pressure_threshold=15)
        sched._last_run_time = 0.0
        sched._new_node_count = 100
        old = time.time() - 120.0
        for i in range(20):
            run(sched.on_node_created())
        # 把时间戳改成 120s 前
        sched._recent_write_times.clear()
        for i in range(20):
            sched._recent_write_times.append(old + i)

        triggered = run(sched.check_and_trigger())
        assert triggered is True, "旧写入不构成压力, 应正常触发"

    def test_explicit_trigger_not_blocked_by_pressure(self):
        sched = _sched()
        for _ in range(20):
            run(sched.on_node_created())
        # 【FIX】同 test_runs_after_writes_settle：触发与等待须同一事件循环
        async def _trigger_and_wait():
            accepted = await sched.trigger_explicit()
            assert accepted is True, "显式触发不受写压力影响"
            for _ in range(100):
                if not sched.is_running:
                    return
                await asyncio.sleep(0.01)
        run(_trigger_and_wait())
        assert sched._dream_run_count == 1


class TestH2PersistDepthGuard:
    """H2：PERSIST 写队列深度守卫——队列过半满时跳过 PERSIST（degraded），零写。

    修复前：梦境直接模式在队列深度无关条件下无条件写回，加重写队列积压；
    修复后：pending_count > max_pending//2 → 跳过 4 步 PERSIST + warning，
    下次梦境按 H5 upsert 语义自愈。只减少写、不新增写路径。
    """

    def _pipe(self):
        from core.dream_pipeline import DreamPipeline
        pipe = DreamPipeline()
        pipe._write_queue = MagicMock()
        return pipe

    def test_busy_queue_skips_persist(self, caplog):
        """队列过半满 → PERSIST 零调用 + degraded=True + warning。"""
        pipe = self._pipe()
        pipe._write_queue.pending_count.return_value = 60
        pipe._write_queue.max_pending = 100
        # 统计真实 PERSIST 调用（经 _persist_async 的 submit）
        pipe._persist_async = AsyncMock()

        store = MagicMock()
        store.execute_cypher.return_value = []
        store.query_cypher.return_value = []

        with caplog.at_level(logging.WARNING, logger="core.dream_pipeline"):
            report = run(pipe.run(
                nodes=[{"id": "n1", "content": "alpha", "created_at": time.time()},
                       {"id": "n2", "content": "beta", "created_at": time.time()}],
                connections={},
                trigger_mode="explicit",
                graphlite_store=store,
                candidate_store=None,
            ))

        assert report.degraded is True, "队列过半满应 degraded=True"
        pipe._persist_async.assert_not_called(), \
            "PERSIST 应被整体跳过（4 步零调用）"
        assert any("PERSIST skipped" in r.getMessage()
                   for r in caplog.records), "应打 warning 说明跳过原因"

    def test_idle_queue_guard_decision(self):
        """H2 守卫条件单元测试: 空闲队列 (1/100) → 不触发跳过。"""
        q = MagicMock()
        q.pending_count.return_value = 1
        q.max_pending = 100
        assert not (q is not None and q.pending_count() > q.max_pending // 2)

    def test_no_queue_guard_inert(self):
        """H2 守卫条件: 无 write_queue (None) → 守卫不生效。"""
        q = None
        assert not (q is not None and q.pending_count() > q.max_pending // 2)
