"""
梦境调度器 H3/H4/H5 修复测试
============================
覆盖:
  · H3 — max_dream_duration 超时强制 + 失败统计
  · H3 — 超时包装不破坏正常梦境
  · H4 — 状态在梦境完成后（finally）保存最新计数
  · H4 — 显式触发路径同样保存状态
  · H5 — 重启 reconcile 标记 interrupted 并允许下次触发
  · H5 — PERSIST 部分失败打 degraded 标记
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.dream_pipeline import DreamPipeline, DreamReport
from core.dream_scheduler import DreamScheduler, DreamSchedulerConfig, TriggerMode


def run(coro):
    """同步运行异步协程（测试用）。"""
    return asyncio.run(coro)


def _make_report(dream_id: str = "d1") -> DreamReport:
    return DreamReport(
        dream_id=dream_id,
        trigger_mode="explicit",
        timestamp=time.time(),
        duration_seconds=0.01,
        stats={"created": 1, "updated": 0, "deleted": 0},
        community_count=1,
        prune_count=0,
        conflict_count=0,
        audit_block_hash="",
    )


async def _fast_pipeline(*args, **kwargs):
    """快速成功的伪管道。"""
    return _make_report()


class TestH3Timeout:
    def test_timeout_enforced_and_counted_as_failure(self):
        """H3: 超过 max_dream_duration_seconds 的梦境被中止并计入失败统计。"""
        saved: list[dict] = []

        async def slow_pipeline(*args, **kwargs):
            await asyncio.sleep(5)
            return _make_report()

        sched = DreamScheduler(
            config=DreamSchedulerConfig(max_dream_duration_seconds=1),
            pipeline_fn=slow_pipeline,
            state_persist_fn=saved.append,
        )
        sched._is_running = True
        sched._current_dream_id = "dream-timeout"

        run(sched._run_dream(TriggerMode.EXPLICIT))

        assert sched._dream_fail_count == 1
        assert sched._dream_run_count == 0  # 超时不计入成功
        assert sched.is_running is False
        assert sched._current_dream_id is None
        # 超时后 finally 保存状态：is_running=false 已落盘
        assert saved, "状态应已持久化"
        assert saved[-1]["is_running"] is False

    def test_timeout_wrapper_does_not_break_normal_dream(self):
        """H3: wait_for 包装不破坏正常完成的梦境。"""
        saved: list[dict] = []
        sched = DreamScheduler(
            config=DreamSchedulerConfig(max_dream_duration_seconds=300),
            pipeline_fn=_fast_pipeline,
            state_persist_fn=saved.append,
        )
        sched._new_node_count = 100
        sched._is_running = True
        sched._current_dream_id = "dream-ok"

        run(sched._run_dream(TriggerMode.ACCUMULATED))

        assert sched._dream_fail_count == 0
        assert sched._dream_run_count == 1
        assert sched._new_node_count == 0  # 成功重置累积计数
        assert sched._last_run_time > 0
        assert sched.is_running is False


class TestH4SaveTiming:
    def test_state_saved_after_completion_with_latest_counters(self):
        """H4: 持久化的状态是梦境完成后的最新计数，而非运行前状态。"""
        saved: list[dict] = []
        sched = DreamScheduler(
            pipeline_fn=_fast_pipeline,
            state_persist_fn=saved.append,
        )
        sched._new_node_count = 100
        sched._is_running = True
        sched._current_dream_id = "dream-h4"

        run(sched._run_dream(TriggerMode.ACCUMULATED))

        assert saved
        latest = saved[-1]
        assert latest["new_node_count"] == 0  # 完成计数已落盘
        assert latest["dream_run_count"] == 1
        assert latest["last_run_time"] > 0
        assert latest["is_running"] is False

    def test_explicit_trigger_persists_state(self):
        """H4: 显式触发路径同样保存状态（触发时 is_running=true 落盘）。"""
        saved: list[dict] = []
        sched = DreamScheduler(
            pipeline_fn=_fast_pipeline,
            state_persist_fn=saved.append,
        )

        async def _trigger_and_wait():
            accepted = await sched.trigger_explicit()
            assert accepted is True
            # 等待 _run_dream 后台任务完成
            for _ in range(100):
                if not sched.is_running and saved and saved[-1]["is_running"] is False:
                    break
                await asyncio.sleep(0.01)

        run(_trigger_and_wait())

        assert saved
        assert saved[0]["is_running"] is True  # 触发即落盘运行中状态（H5 崩溃检测）
        assert saved[0]["current_dream_id"]
        assert saved[-1]["is_running"] is False  # 完成/失败后落盘最新状态
        assert saved[-1]["dream_run_count"] == 1


class TestH5Reconcile:
    def test_reconcile_marks_interrupted_dream_and_allows_next_trigger(self):
        """H5: 重启后 is_running=true（无完成记录）→ 标记 interrupted，允许下次触发。"""
        saved: list[dict] = []
        sched = DreamScheduler(
            pipeline_fn=_fast_pipeline,
            state_persist_fn=saved.append,
        )
        # 模拟从持久层恢复：上次进程在梦境中途崩溃
        sched.load_state({
            "is_running": True,
            "current_dream_id": "dream-crashed",
            "dream_run_count": 2,
            "new_node_count": 50,
            "last_run_time": 0.0,
        })

        assert sched.is_running is True
        assert sched.reconcile_after_restart() is True
        assert sched.is_running is False  # 已标记 interrupted
        assert sched._current_dream_id is None
        # 中断标记已持久化（is_running=false 落盘）
        assert saved and saved[-1]["is_running"] is False
        assert saved[-1]["current_dream_id"] is None

        # 允许下次触发：显式触发必须被接受，并实际完成一次梦境
        async def _trigger_after_reconcile():
            accepted = await sched.trigger_explicit()
            assert accepted is True
            # 等待 _run_dream 后台任务完成并落盘最新状态
            for _ in range(100):
                if not sched.is_running and saved[-1]["is_running"] is False:
                    break
                await asyncio.sleep(0.01)

        run(_trigger_after_reconcile())
        assert sched._dream_run_count == 3  # 恢复值 2 + 本次成功 1
        assert saved[-1]["dream_run_count"] == 3

    def test_reconcile_noop_when_not_running(self):
        """H5: 上次正常完成（is_running=false）时 reconcile 无操作。"""
        sched = DreamScheduler(pipeline_fn=_fast_pipeline)
        sched.load_state({"is_running": False, "dream_run_count": 5})
        assert sched.reconcile_after_restart() is False


class TestH5DegradedPersist:
    def test_persist_partial_failure_marks_report_degraded(self):
        """H5: PERSIST 步骤抛异常 → DreamReport.degraded=True（下次梦境可修复）。"""
        pipe = DreamPipeline()
        store = MagicMock()
        store.execute_cypher.return_value = []
        store.query_cypher.return_value = []

        def boom(*args, **kwargs):
            raise RuntimeError("graphlite persist exploded")

        # 模拟 PERSIST 中 hyperedge 步骤崩溃（其余步骤正常）
        pipe._persist_hyperedges = boom

        report = run(pipe.run(
            nodes=[{"id": "n1", "content": "alpha", "created_at": time.time()},
                   {"id": "n2", "content": "beta", "created_at": time.time()}],
            connections={},
            trigger_mode="explicit",
            graphlite_store=store,
            candidate_store=None,
        ))

        assert report.degraded is True

    def test_persist_success_not_degraded(self):
        """H5: PERSIST 全部成功时 degraded=False。"""
        pipe = DreamPipeline()
        store = MagicMock()
        store.execute_cypher.return_value = []
        store.query_cypher.return_value = []

        report = run(pipe.run(
            nodes=[{"id": "n1", "content": "alpha", "created_at": time.time()},
                   {"id": "n2", "content": "beta", "created_at": time.time()}],
            connections={},
            trigger_mode="explicit",
            graphlite_store=store,
            candidate_store=None,
        ))

        assert report.degraded is False
