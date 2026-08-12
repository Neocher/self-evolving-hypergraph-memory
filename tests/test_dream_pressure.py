"""
P2-1 梦境写压力感知测试
======================
覆盖:
  · 持续写入 (30s 内 ≥ 阈值) → check_and_trigger 推迟返回 False
  · 推迟不丢候选 (累积计数保留)
  · 写入停止后梦境正常触发
  · 显式触发不受写压力影响
运行: python -m pytest tests/test_dream_pressure.py -v
"""
import asyncio
import time

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

        triggered = run(sched.check_and_trigger())
        assert triggered is True, "写入停止后应恢复触发"
        # 等待后台梦境完成
        async def _wait():
            for _ in range(100):
                if not sched.is_running:
                    return
                await asyncio.sleep(0.01)
        run(_wait())
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
        accepted = run(sched.trigger_explicit())
        assert accepted is True, "显式触发不受写压力影响"
        async def _wait():
            for _ in range(100):
                if not sched.is_running:
                    return
                await asyncio.sleep(0.01)
        run(_wait())
        assert sched._dream_run_count == 1
