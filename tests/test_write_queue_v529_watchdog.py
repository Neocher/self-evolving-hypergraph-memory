"""
v5.29 F6 看门狗增强修正测试（Codex 审核 MED-M1/M2 + 二轮复核 M1.x/M2.1）
============================================================
修复（2026-08-13）:
  · M1 — 长写不误报：单步长写（如梦境 PERSIST 单步 > _stuck_timeout）会让心跳
    短暂过期，但随写完成恢复——若按瞬时 _is_stuck() 计数，3 次并发 submit 超时
    即 logger.critical，会误导运营重启打断合法长写。新逻辑要求 stuck 状态
    **持续超过观察窗**（_stuck_observe_window = 2×stuck_timeout）才计数。
  · M2 — 成功路径归零：wait_for 正常返回时 _deadlock_suspect_count 归零
    （"连续 N 次超时" = 相邻超时之间无成功）；计数操作受 _stuck_lock 保护。
  · M2.1（二轮复核）— 并发 >=3 单批多次告警：计数归零在并发超时对齐时失效
    （归零后剩余并发又凑满 3 次 → 单批两条告警）。改为 _last_critical_at
    时间去抖（60s 去重），与并发到达顺序无关，单批必然只告警一次。
  · M1.3（二轮复核）— 成功路径同锁内一并清理 _stuck_since=None：上次 stuck
    起算时刻残留会导致新卡死沿用旧时刻被误判为"已持续超过观察窗"。
  · M1.4（二轮复核）— 告警文案 worker unresponsive (alive or dead)：心跳+积压
    无法区分线程存活但无响应与线程死亡，文案覆盖后者。

覆盖:
  · 长写不误报（观察窗内恢复，全程无 critical）
  · 真死锁（持续超过观察窗）仍触发 critical（F6 特性不被 M1 修复破坏）
  · 成功路径归零（含 超时累积 → 恢复 → 成功 全周期）
  · M2.1：并发 >=3 单批仅一条告警（时间去抖 vs 计数归零）
  · 边界：超窗长写 + 持续流量 → 已文档化残留误报恰一条 + 无流量单条长写不告警
    （检测依赖持续写流量）+ 长写完成后恢复不复发

运行: python -m pytest tests/test_write_queue_v529_watchdog.py -v
"""

import asyncio
import logging
import threading
import time

import pytest

from core.write_queue import WriteQueue


class TestM1LongWriteNoFalseCritical:
    """M1：观察窗过滤慢写——长写不误报 critical。"""

    def test_long_write_recovers_within_observe_window(self, caplog):
        """单步长写（心跳过期）期间多次 submit 超时，但 stuck 状态持续未超
        观察窗 → 不计数、无 critical；长写完成后队列恢复可用。"""
        q = WriteQueue(max_pending=10, wait_timeout=0.05)
        try:
            # 压缩时间尺度：stuck_timeout=0.2s，观察窗=2×=0.4s
            q._stuck_timeout = 0.2
            q._stuck_observe_window = 0.4
            entered = threading.Event()
            release = threading.Event()

            def long_write():
                entered.set()
                release.wait(3)  # 模拟梦境长 PERSIST：单步卡过 stuck_timeout

            async def run():
                t1 = asyncio.create_task(q.submit(long_write))
                await asyncio.to_thread(entered.wait, 2)
                # 长写已卡过 stuck_timeout → 心跳过期（瞬时 _is_stuck() 为 True，
                # 属慢写；旧逻辑从这里起 3 次超时就 critical 误报）
                q._last_activity = time.monotonic() - q._stuck_timeout - 0.01
                # 观察窗（0.4s）内连发 3 次超时 submit（3×0.05s=0.15s < 0.4s）
                for _ in range(3):
                    with pytest.raises(asyncio.TimeoutError):
                        await q.submit(lambda: 1)
                # 长写在观察窗内完成 → 队列恢复
                release.set()
                with pytest.raises(asyncio.TimeoutError):
                    await t1  # t1 自身也早超时（迟到完成语义）
                # 等积压的 lambda 被写线程消费完，验证队列恢复可用
                for _ in range(100):
                    if q.pending_count() == 0:
                        break
                    await asyncio.sleep(0.01)
                assert await q.submit(lambda: "ok") == "ok"

            with caplog.at_level(logging.CRITICAL, logger="core.write_queue"):
                asyncio.run(run())
            assert q._deadlock_suspect_count == 0, "长写不应累积疑似计数"
            crits = [r for r in caplog.records if r.levelname == "CRITICAL"]
            assert not crits, (
                f"长写期间不应触发 critical 误报: {[r.getMessage() for r in crits]}"
            )
        finally:
            release.set()
            q.shutdown()

    def test_true_deadlock_beyond_window_still_alerts(self, caplog):
        """真死锁：stuck 状态持续超过观察窗 → 连续 3 次超时仍触发 critical
        （M1 修复不得破坏 F6 原有告警能力）。"""
        q = WriteQueue(max_pending=10, wait_timeout=0.05)
        try:
            q._stuck_timeout = 0.2
            q._stuck_observe_window = 0.4
            entered = threading.Event()
            release = threading.Event()

            def blocked():
                entered.set()
                release.wait(30)  # 引擎级死锁：永不返回

            async def run():
                t1 = asyncio.create_task(q.submit(blocked))
                await asyncio.to_thread(entered.wait, 2)
                # 模拟 stuck 状态已持续超过观察窗（心跳过期 + 起算时刻在窗口外）
                q._last_activity = time.monotonic() - q._stuck_timeout - 1
                q._stuck_since = time.monotonic() - q._stuck_observe_window - 1
                for _ in range(3):
                    with pytest.raises(asyncio.TimeoutError):
                        await q.submit(lambda: 1)
                with pytest.raises(asyncio.TimeoutError):
                    await t1  # 取回 t1 自身超时异常（避免 asyncio 未取回告警）

            with caplog.at_level(logging.CRITICAL, logger="core.write_queue"):
                asyncio.run(run())
            crits = [r for r in caplog.records if r.levelname == "CRITICAL"]
            assert any(
                "manual restart required" in r.getMessage() for r in crits
            ), "真死锁应触发 critical 告警（M1 修复破坏了 F6？）"
        finally:
            release.set()
            q.shutdown()


class TestM2SuccessResetsCount:
    """M2：成功路径归零——wait_for 正常返回时清空疑似计数。"""

    def test_success_path_resets_suspect_count(self):
        """此前已累积疑似计数 → 一次成功 submit 即归零。"""
        q = WriteQueue(wait_timeout=0.5)
        try:
            q._deadlock_suspect_count = 2  # 模拟此前累积 2 次超时疑似（未达 critical）
            assert asyncio.run(q.submit(lambda: 42)) == 42
            assert q._deadlock_suspect_count == 0, "成功路径应归零（连续 N 次语义）"
        finally:
            q.shutdown()

    def test_timeouts_then_success_resets_full_cycle(self):
        """全周期：持续卡死窗口内超时计数 +1 → 队列恢复 → 成功 submit 归零。"""
        q = WriteQueue(max_pending=10, wait_timeout=0.05)
        try:
            q._stuck_timeout = 0.2
            q._stuck_observe_window = 0.4
            entered = threading.Event()
            release = threading.Event()

            def blocked():
                entered.set()
                release.wait(5)

            async def run():
                t1 = asyncio.create_task(q.submit(blocked))
                await asyncio.to_thread(entered.wait, 2)
                # 先等 t1 自身超时结束（此时心跳仍新鲜 → 不计数），再制造持续
                # 卡死状态，保证后续计数确定性地只来自显式 submit 的超时
                with pytest.raises(asyncio.TimeoutError):
                    await t1
                # 模拟 stuck 状态已持续超过观察窗（心跳过期 + 起算时刻在窗口外）
                q._last_activity = time.monotonic() - q._stuck_timeout - 1
                q._stuck_since = time.monotonic() - q._stuck_observe_window - 1
                with pytest.raises(asyncio.TimeoutError):
                    await q.submit(lambda: 1)
                assert q._deadlock_suspect_count == 1, "持续卡死窗口内超时应计数 +1"
                release.set()  # 恢复（模拟长写完成）
                for _ in range(100):
                    if q.pending_count() == 0:
                        break
                    await asyncio.sleep(0.01)
                assert await q.submit(lambda: "ok") == "ok"
                assert q._deadlock_suspect_count == 0, "成功路径应归零"

            asyncio.run(run())
        finally:
            release.set()
            q.shutdown()


class TestM21ConcurrentBatchSingleAlert:
    """M2.1（二轮复核）：并发超时对齐——单批多次告警的回归测试。

    旧逻辑：并发 >=3 的 submit 同一批超时，计数 1→2→3 触发 critical 后归零，
    剩余并发又凑满 3 次 → 单批两条 critical。新逻辑 _last_critical_at 时间去抖
    （60s）与并发到达顺序无关，同一批必然只告警一次。
    """

    def test_concurrent_batch_emits_single_critical(self, caplog):
        """并发 6 个 submit 同批超时对齐（含 1 个在途长写共 7 次超时事件）：
        只允许 1 条 critical 告警。"""
        q = WriteQueue(max_pending=50, wait_timeout=0.05)
        try:
            q._stuck_timeout = 0.2
            q._stuck_observe_window = 0.4
            entered = threading.Event()
            release = threading.Event()

            def blocked():
                entered.set()
                release.wait(30)  # 引擎级死锁：永不返回

            async def run():
                t1 = asyncio.create_task(q.submit(blocked))
                await asyncio.to_thread(entered.wait, 2)
                # 心跳过期 + stuck 起算时刻在观察窗外 → 每个超时都计为"持续疑似"
                q._last_activity = time.monotonic() - q._stuck_timeout - 1
                q._stuck_since = time.monotonic() - q._stuck_observe_window - 1
                # 并发 6 个超时对齐在同一批（6×0.05s ≈ 0.3s << 60s 去抖窗）
                tasks = [asyncio.create_task(q.submit(lambda: 1)) for _ in range(6)]
                for t in tasks:
                    with pytest.raises(asyncio.TimeoutError):
                        await t
                with pytest.raises(asyncio.TimeoutError):
                    await t1  # 取回 t1 自身超时异常（避免 asyncio 未取回告警）

            with caplog.at_level(logging.CRITICAL, logger="core.write_queue"):
                asyncio.run(run())
            crits = [r for r in caplog.records if r.levelname == "CRITICAL"]
            assert len(crits) == 1, (
                f"并发单批应仅 1 条 critical（时间去抖），实际 "
                f"{len(crits)}: {[r.getMessage() for r in crits]}"
            )
            assert q._deadlock_suspect_count < 3, "计数不应残留到可再次触发告警"
        finally:
            release.set()
            q.shutdown()


class TestResidualFpBoundary:
    """二轮复核：检测依赖持续写流量 + 超窗长写残留误报边界（已文档化）。"""

    def test_over_window_long_write_with_sustained_traffic(self, caplog):
        """① 无持续流量：超窗长写单条（无并发 submit 超时）不告警——检测依赖持续
        写流量（只有 submit 超时事件才累计计数）。
        ② 超窗长写 + 持续流量：心跳无法区分"超窗长写"与真死锁 → 已文档化残留误报，
        恰 1 条 critical（M1.4 文案 worker unresponsive (alive or dead)）。
        ③ 长写完成 → 队列恢复，成功路径清理计数与 _stuck_since → 不复发。
        """
        q = WriteQueue(max_pending=50, wait_timeout=0.05)
        try:
            q._stuck_timeout = 0.2
            q._stuck_observe_window = 0.4
            entered = threading.Event()
            release = threading.Event()

            def long_write():
                entered.set()
                release.wait(10)  # 长写：真实超过观察窗（0.4s），但终将完成

            async def run():
                t1 = asyncio.create_task(q.submit(long_write))
                await asyncio.to_thread(entered.wait, 2)
                # t1 自身超时（此刻心跳仍新鲜 → 不计数）
                with pytest.raises(asyncio.TimeoutError):
                    await t1
                # 模拟超窗长写：心跳过期 + stuck 起算时刻在观察窗外
                q._last_activity = time.monotonic() - q._stuck_timeout - 1
                q._stuck_since = time.monotonic() - q._stuck_observe_window - 1
                # ① 无持续流量：无 submit 超时事件 → 不告警
                await asyncio.sleep(0.15)
                crits = [r for r in caplog.records if r.levelname == "CRITICAL"]
                assert not crits, "无持续流量时不应告警（检测依赖持续写流量）"
                # ② 持续流量：并发 6 个超时 submit → 单批仅一条（残留误报，已文档化）
                tasks = [asyncio.create_task(q.submit(lambda: 1)) for _ in range(6)]
                for t in tasks:
                    with pytest.raises(asyncio.TimeoutError):
                        await t
                # ③ 长写完成 → 队列恢复；成功路径清理计数与 stuck 起算时刻
                release.set()
                for _ in range(200):
                    if q.pending_count() == 0:
                        break
                    await asyncio.sleep(0.01)
                assert await q.submit(lambda: "ok") == "ok"
                assert q._deadlock_suspect_count == 0, "恢复后计数应归零"
                assert q._stuck_since is None, "恢复后 stuck 起算时刻应清理（M1.3）"

            with caplog.at_level(logging.CRITICAL, logger="core.write_queue"):
                asyncio.run(run())
            crits = [r for r in caplog.records if r.levelname == "CRITICAL"]
            assert len(crits) == 1, (
                f"超窗长写+持续流量为已文档化残留误报：应恰好 1 条 critical，"
                f"实际 {len(crits)}: {[r.getMessage() for r in crits]}"
            )
            assert "unresponsive" in crits[0].getMessage(), "M1.4 文案应覆盖 alive or dead"
        finally:
            release.set()
            q.shutdown()
