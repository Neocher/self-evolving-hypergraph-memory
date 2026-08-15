"""
写串行化队列 — SHM v5.23
========================
背景（实测 2026-08-11）：单事件循环 + 写路径 GraphLite 同步直调 → 每个写请求
阻塞整个 loop（8 并发写 3.2s/条）。而 GraphLite 写操作跨线程实测挂起
（session 共享，但写/事务只能固定单线程），检索 to_thread 方案不适用于写。

设计：所有 GraphLite 写调用收敛到**专用写线程**串行执行，事件循环只负责
入队 + 等 Future，不再被同步写阻塞：

```
async handler → await submit(fn)          # 入队 + 等 future（不阻塞 loop）
    ↓ queue.Queue（maxsize 背压）
专用写线程（daemon）串行消费 fn(*args, **kwargs)
    ↓ future.set_result / set_exception
handler 恢复
```

关键实现点（对齐 .trio-plan-v523.md §2.2/§4）：
- 用 `queue.Queue`（线程安全）而非 asyncio.Queue（不能从线程 get）。
- 事件循环侧用 `asyncio.wrap_future(fut)` 直接桥接 concurrent future——
  **不占用任何 executor 线程**。旧实现用 `run_in_executor(专用单 worker executor,
  _await_future)` 阻塞等结果：写线程卡死时唯一 worker 被永久占死，后续所有请求
  排队等死（wait_for 超时只取消 asyncio 侧，executor 线程仍卡在 fut.result()）
  → 全 503 且重试永不恢复（2026-08-13 修复，见 test_write_queue_v528.py）。
- 超时：`asyncio.wait_for(..., wait_timeout)`。⚠️ 超时只放弃等待方，
  **不能取消正在写线程执行的 GraphLite 调用**——任务仍会落库（迟到完成
  语义，见 submit docstring）。concurrent future 不会被 wait_for 取消，
  写线程 set_result 永远合法，无 InvalidStateError 崩溃。
- 看门狗（F3）：写线程记录心跳 `_last_activity`；**仅线程死亡时** `_restart_worker()`
  重建（alive+慢写不重启——避免双消费者并发写 GraphLite；空闲队列不误判卡死）。
  能救回"线程意外死亡"，救不回 GraphLite 引擎级死锁（后者需进程重启）。
  【F6】引擎级死锁仅告警不自动重启：连续 3 次超时 + stuck 状态**持续超过观察窗**
  （2×stuck_timeout）才 critical（观察窗过滤单步长写的心跳短暂过期）；成功路径归零。
  【F6-M2.1】告警去抖：critical 后 60s 内不重复告警（`_last_critical_at` 时间戳去抖，
  比计数归零可靠——并发 >=3 的超时同批对齐时，计数归零后剩余并发又凑满 3 次会
  单批多次告警；时间去抖与并发顺序无关，同一批必然只告警一次）。
  【F6-M1.3/M1.4】检测依赖**持续写流量**：只有 submit 超时事件才累计计数，单条长写
  无并发流量不告警。⚠️ 残留误报边界：长写超过观察窗 + 持续写流量时仍会告警——
  心跳无法区分"超窗长写"与真死锁，属已知边界（成功路径清理计数 + stuck 起算时刻，
  恢复后不复发）。告警文案 worker unresponsive (alive or dead) 同时覆盖线程死亡场景。
- 重入：写线程内再 submit → 直接同步执行（防死锁）。
- 优雅关闭：sentinel 退出 + join(timeout) 兜底；shutdown 先 drain 在途写。

【v5.40 Write-Priority】优先级：
- 底层从 queue.Queue 升级为 queue.PriorityQueue（单队列天然无「双队列 notify
  死睡」问题——queue.Queue 的 notify 每队列私有，低队列 blocking get 时高队列
  put 无人唤醒）。入队元组 `(0 if priority=="high" else 1, seq, task)`，
  seq 用 itertools.count（保证同优先级 FIFO 且永不比较 _WriteTask）。
- 低准入闸：priority!=="high"（low/normal）且 qsize() >= low_max → 抛
  WriteQueueFullError（low_max < maxsize 为 high 预留容量，不破坏
  max_pending 语义）。high 只在 qsize==maxsize 时被拒（与现状一致）。
- 外部调用方默认 high（qsubmit setdefault），梦境 PERSIST 显式 low ——
  切块后写线程块间排空 high，外部写不再被 30-60s 梦境长任务饿到 503。

不引入锁/事务/重构；调用方按"写调用"语义选择 submit（读调用留事件循环）。
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import queue
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_SENTINEL = object()


class WriteQueueFullError(Exception):
    """入队积压超过 max_pending（背压拒绝）。由 API 层转 503。"""


class WriteQueueClosedError(Exception):
    """队列已关闭（shutdown 后拒绝新写）。"""


@dataclass
class _WriteTask:
    fn: Callable[..., Any]
    args: tuple
    kwargs: dict
    fut: Future  # concurrent.futures.Future（跨线程 set_result 合法）


class WriteQueue:
    """串行写队列：所有 GraphLite 写调用收敛到单写线程串行执行。

    Args:
        max_pending: 入队积压上限，满则 submit 抛 WriteQueueFullError（背压）。
        wait_timeout: 调用方等待单次写完成的最长时间（秒）。
    """

    def __init__(
        self,
        max_pending: int = 100,
        wait_timeout: float = 30.0,
        ping_fn: Optional[Callable[[], bool]] = None,
        low_max: Optional[int] = None,
    ):
        """
        Args:
            max_pending: 入队积压上限，满则 submit 抛 WriteQueueFullError（背压）。
            wait_timeout: 调用方等待单次写完成的最长时间（秒）。
            ping_fn: 【L3/M1】可选引擎存活探针（True=引擎存活，慢写而非死锁）。
                由 app 注入：独立 daemon 连接 + 1 条 trivial 查询，join(timeout) 兜底。
                探针通过 → critical 降级为 warning；失败/挂 → 仍 critical。
            low_max: 【v5.40】低优先级准入闸阈值。low/normal 入队时
                qsize() >= low_max 即拒（为 high 预留容量）；默认
                max_pending - max_pending//10（小队列退化为无闸，不破坏
                max_pending 语义与既有背压测试）。
        """
        self._q: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_pending)
        # 【v5.40】优先级元组序号：同优先级内 FIFO + 元组第三元素永不参与比较
        self._seq = itertools.count()
        self._low_max = (
            low_max if low_max is not None
            else max(1, max_pending - max_pending // 10)
        )
        self._wait_timeout = wait_timeout
        self._closed = False
        self._worker = threading.Thread(target=self._run, daemon=True, name="shm-writer-worker")
        self._worker_ident: Optional[int] = 0
        self._worker.start()
        self._worker_ident = self._worker.ident
        # 【F3】看门狗心跳：写线程每处理一个任务更新；submit 检测卡死
        self._last_activity: float = time.monotonic()
        self._stuck_timeout: float = max(wait_timeout * 2, 60.0)
        self._stuck_lock = threading.Lock()
        # 【F6】死锁疑似计数：连续 N 次 submit 超时且写线程疑似卡死（worker 存活
        # 但心跳超时）时告警建议人工重启。不自动重启——引擎级死锁需进程重启，
        # 由 systemd/人工决策。M2：计数操作全部受 _stuck_lock 保护，成功路径归零。
        self._deadlock_suspect_count = 0
        # 【F6-M2.1】告警去抖：critical 后记录时刻，_critical_debounce（60s）内即使
        # 计数再次达到阈值也不重复告警（防刷屏）。比"计数归零"可靠——并发 >=3 的
        # 超时在同一批内对齐时，计数归零后剩余并发超时又凑满 3 次 → 单批多次告警；
        # 时间去抖与并发到达顺序无关，同一批必然只告警一次。
        self._last_critical_at: Optional[float] = None
        self._critical_debounce: float = 60.0
        # 【F6-M1】持续观察窗：stuck 状态（心跳过期+积压）须**持续**超过该时长才
        # 认定疑似死锁。单步长写（如梦境 PERSIST 单步 > stuck_timeout）会让心跳
        # 短暂过期但随写完成恢复——若按瞬时 _is_stuck() 计数，3 次超时即 critical
        # 会误导运营重启打断合法长写。真死锁的 stuck 状态必然持续存在。
        self._stuck_observe_window: float = self._stuck_timeout * 2
        self._stuck_since: Optional[float] = None  # 首次检测到 stuck 的时刻
        # 【L3】完成计数：写线程每完成一个任务 +1（_stuck_lock 保护）。两次超时之间
        # 有完成 → 慢写非死锁，疑似计数归零不累计（"连续 N 次超时" = 相邻超时
        # 之间无成功，完成即证明队列在推进）。
        self._completed_tasks = 0
        self._completed_at_last_suspect = 0
        # 【L3】当前在途任务快照（写线程侧设置；critical 文案附 repr）
        self._current_task: Optional[_WriteTask] = None
        self._ping_fn = ping_fn

    # ─── 事件循环侧：唯一入口 ───────────────────────────

    async def submit(
        self,
        fn: Callable[..., Any],
        *args,
        priority: str = "normal",
        **kwargs,
    ) -> Any:
        """入队并等待写线程完成，返回 fn 的结果。

        - 队列满 → WriteQueueFullError（API 层转 503）
        - 【v5.40】低准入闸：priority!="high" 且 qsize() >= low_max →
          WriteQueueFullError（为 high 预留容量，梦境 low 块积压时降级）
        - 写线程执行异常 → 原样抛回
        - 超过 wait_timeout → asyncio.TimeoutError
          ⚠️ 超时只放弃等待：写任务仍在写线程继续执行并真实落库
          （迟到完成），调用方**不应安全重试**（写入用 uuid 主键天然幂等，
          但重复触发仍由调用方自行判断）。
        - 写线程内重入 submit → 直接同步执行（防死锁）。
        """
        if self._closed:
            raise WriteQueueClosedError("write queue closed")
        # 写线程内重入：同步执行，不排队（否则写线程等自己 → 死锁）
        if self._worker_ident == threading.get_ident():
            return fn(*args, **kwargs)
        # 【F3】看门狗：写线程卡死/死亡时尝试重启（尽力而为），避免永久 503
        if self._is_stuck():
            self._restart_worker()
        fut: Future = Future()
        task = _WriteTask(fn=fn, args=args, kwargs=kwargs, fut=fut)
        # 【v5.40】低准入闸：low/normal 积压达 low_max 即拒（high 不受限，
        # 仅在 qsize==maxsize 时经 put_nowait 背压）。准入闸在重入检查之后——
        # 写线程内重入直接同步执行，不入队。
        if priority != "high" and self._q.qsize() >= self._low_max:
            raise WriteQueueFullError(
                f"write queue low-priority gate ({self._q.qsize()}/{self._low_max} "
                f"pending, high reserved)"
            )
        try:
            self._q.put_nowait(
                (0 if priority == "high" else 1, next(self._seq), task)
            )
        except queue.Full:
            raise WriteQueueFullError(
                f"write queue full ({self._q.qsize()}/{self._q.maxsize} pending)"
            )
        # 【F1 修复】asyncio.wrap_future 直接桥接 concurrent future → 不占用任何
        # executor 线程。旧实现 run_in_executor(单 worker executor, fut.result())
        # 在写线程卡死时把唯一 worker 永久占死 → 后续所有请求排队等死 → 全 503
        # 且永不恢复（重试无效）。
        # shield 保护：wait_for 超时只取消 shield 外层，**不会取消底层 concurrent
        # future**（否则写线程迟到 set_result 抛 InvalidStateError，迟到完成语义
        # 被破坏）。超时只影响本请求，其他请求不受影响。
        try:
            result = await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(fut)),
                timeout=self._wait_timeout,
            )
        except asyncio.TimeoutError:
            # 【F6】看门狗增强：连续 3 次超时且**持续**疑似卡死（worker 无响应——
            # 心跳超时 + 队列积压，线程存活或已死亡均覆盖）→ critical 告警建议
            # 人工重启。不自动重启。M1 观察窗：须 stuck 状态持续超过
            # _stuck_observe_window（2×stuck_timeout）才计数——单步长写会短暂心跳
            # 过期但会恢复，不满足"持续" → 不误报。
            # 【L3】慢写非死锁：两次超时之间有任务完成（_completed_tasks 增加）
            # → 队列在推进，疑似计数归零不累计。
            if self._is_stuck_sustained():
                should_alert = False
                with self._stuck_lock:
                    if self._completed_tasks > self._completed_at_last_suspect:
                        self._deadlock_suspect_count = 0
                    self._completed_at_last_suspect = self._completed_tasks
                    self._deadlock_suspect_count += 1
                    if self._deadlock_suspect_count >= 3:
                        now = time.monotonic()
                        # M2.1 时间去抖：同一批并发超时对齐时只告警一次（计数
                        # 归零后剩余并发又凑满 3 次 → 旧逻辑单批多次告警）。
                        should_alert = (
                            self._last_critical_at is None
                            or now - self._last_critical_at >= self._critical_debounce
                        )
                        self._deadlock_suspect_count = 0  # 门控"连续 ≥3 次超时"
                if should_alert:
                    # 【L3③】M1 ping 联动：锁外 ping（有界调用）。引擎存活
                    # （ping 通过）→ 慢写而非死锁，降级 warning 不 critical。
                    ping_ok = False
                    if self._ping_fn is not None:
                        try:
                            ping_ok = bool(self._ping_fn())
                        except Exception:
                            ping_ok = False
                    if ping_ok:
                        logger.warning(
                            "write queue worker slow (ping ok, alive); "
                            "degraded, no restart"
                        )
                    else:
                        # 文案 L3②：附当前在途任务 repr + 心跳时长；M1.4 文案
                        # worker unresponsive (alive or dead) 同时覆盖线程死亡场景。
                        heartbeat_age = time.monotonic() - self._last_activity
                        current = self._current_task
                        task_repr = repr(current) if current is not None else "none"
                        logger.critical(
                            "write queue worker unresponsive (alive or dead); "
                            "manual restart required; "
                            "heartbeat_age=%.1fs in-flight=%s",
                            heartbeat_age, task_repr,
                        )
                        with self._stuck_lock:
                            self._last_critical_at = time.monotonic()
            raise
        # 【F6-M2】成功路径清理：本次请求在 wait_timeout 内真实完成 → 写线程在推进，
        # 此前累积的死锁疑似不成立（"连续 N 次超时" = 相邻超时之间无成功）。同锁内
        # 一并清掉 _stuck_since（M1.3）——上次 stuck 起算时刻不得残留到下一次卡死
        # 判定，否则新卡死会沿用旧起算时刻被误判为"已持续超过观察窗"。
        with self._stuck_lock:
            self._deadlock_suspect_count = 0
            self._stuck_since = None
        return result

    # ─── 写线程侧：串行消费 ────────────────────────────

    def _run(self) -> None:
        while True:
            item = self._q.get()
            # 【v5.40】PriorityQueue 元组：item = (priority, seq, task)；sentinel
            # 包装为 (2, seq, _SENTINEL)（最低优先级，drain 后排空才退出）。
            if item[2] is _SENTINEL:
                self._q.task_done()
                break
            task: _WriteTask = item[2]
            try:
                # 【F3】心跳：任务开始前打点（即便任务卡死也能检测）
                self._touch_activity()
                # 【L3】在途任务快照（看门狗 critical 文案附 repr）
                self._current_task = task
                result = task.fn(*task.args, **task.kwargs)
                task.fut.set_result(result)
            except BaseException as exc:  # 含 GraphLite 异常 / CircuitBreakerOpen
                task.fut.set_exception(exc)
            finally:
                self._current_task = None
                self._touch_activity()
                self._q.task_done()
                # 【L3】完成计数（_stuck_lock 保护）：submit 超时分支用它判定
                # "两次超时之间队列有完成 → 慢写非死锁"。
                with self._stuck_lock:
                    self._completed_tasks += 1

    def _touch_activity(self) -> None:
        with self._stuck_lock:
            self._last_activity = time.monotonic()

    def _is_stuck(self) -> bool:
        """诊断：线程死亡，或有在途任务但心跳超时（慢写/挂起）。

        空闲（unfinished_tasks == 0）不算卡死——心跳停滞只是无活可做，
        误判会导致空闲后首次 submit 触发 spurious 重启 + 僵尸线程泄漏。
        """
        if not self._worker.is_alive():
            return True
        if self._q.unfinished_tasks == 0:
            return False
        with self._stuck_lock:
            return time.monotonic() - self._last_activity > self._stuck_timeout

    def _is_stuck_sustained(self) -> bool:
        """【F6-M1】疑似引擎级卡死：stuck 状态**持续**超过观察窗。

        与 _is_stuck() 的区别：_is_stuck() 只看"此刻"心跳是否过期——单步长写
        （> _stuck_timeout，如梦境 PERSIST 单步）会让心跳短暂过期，随后随写完成
        恢复，属慢写而非死锁。真死锁的 stuck 状态不会自行恢复，故要求其持续
        超过 _stuck_observe_window（2× stuck_timeout）才认定疑似卡死，过滤慢写、
        避免 critical 误导运营重启打断合法长写。
        """
        now = time.monotonic()
        if not self._is_stuck():
            with self._stuck_lock:
                self._stuck_since = None
            return False
        with self._stuck_lock:
            if self._stuck_since is None:
                self._stuck_since = now
            return now - self._stuck_since >= self._stuck_observe_window

    def _restart_worker(self) -> None:
        """**仅在线程死亡时**重建写线程接管队列。

        ⚠️ alive+慢写场景**不重启**：旧线程仍活着，慢写返回后会回到 _q.get()
        继续消费，与新线程并存 → 双消费者并发写 GraphLite，违反单写线程约束
        （GraphLite 跨线程写实测挂起，只会更糟）。该场景由 F1 无线程占用 +
        wait_for 超时兜底；引擎级死锁重启也救不回（需进程重启）。
        线程死亡（daemon 被系统回收/代码 bug 致线程退出）时重建是有价值的恢复。
        """
        with self._stuck_lock:
            if self._worker.is_alive():
                return  # 线程活着绝不重启（避免双写并发）
            logger.warning("WriteQueue worker dead — restarting write thread (best-effort)")
            self._worker = threading.Thread(target=self._run, daemon=True, name="shm-writer-worker")
            self._worker.start()
            self._worker_ident = self._worker.ident
            self._last_activity = time.monotonic()
            # 【F6-M1/M2】新线程 = 新纪元：清掉旧 worker 时期的 stuck 起算时刻与
            # 疑似计数（它们针对已死亡的旧线程，不得带入新线程）。
            self._stuck_since = None
            self._deadlock_suspect_count = 0

    # ─── 背压 / 探测（P2-1 梦境错峰探测点）──────────────

    def pending_count(self) -> int:
        """已入队未完成的任务数（近似值）。"""
        return self._q.qsize()

    @property
    def max_pending(self) -> int:
        return self._q.maxsize

    # ─── M1 引擎级死锁探测（只读诊断，供 health 端点）──────────

    def diagnose(self) -> dict:
        """返回写队列/写线程诊断快照（只读，不触发任何写/重启）。

        【M1】只做探测、不做自动重建——自动重建 = 从另一线程关/重建 GraphLite，
        属 8/12 事故区禁止。重建仍由 systemd/人工决策。
        返回: {worker_alive, heartbeat_age, depth, current_task, stack}。
        """
        with self._stuck_lock:
            heartbeat_age = time.monotonic() - self._last_activity
            current = self._current_task
            task_repr = repr(current) if current is not None else "none"
        stack = ""
        if self._worker.is_alive():
            try:
                frames = sys._current_frames()
                fid = self._worker.ident
                if fid in frames:
                    import traceback
                    stack = "".join(
                        traceback.format_stack(frames[fid])
                    ).strip()
            except Exception:
                stack = "<unavailable>"
        return {
            "worker_alive": bool(self._worker.is_alive()),
            "heartbeat_age": round(heartbeat_age, 3),
            "depth": self._q.qsize(),
            "current_task": task_repr,
            "stack": stack,
        }

    # ─── 优雅关闭 ──────────────────────────────────────

    def shutdown(self, drain: bool = True, drain_timeout: float = 10.0) -> None:
        """停止写队列。drain=True 时先等已入队任务全部完成再退出（默认）。

        顺序必须：先 drain 写队列 → 再 close GraphLite（在 app lifespan 内）。

        【F4】drain 限时：写任务卡死（如 GraphLite 死锁）时以 drain_timeout
        为界的轮询超时退出，而非 `_q.join()` 无限阻塞——否则
        `_worker.join(timeout=5.0)` 兜底永远执行不到，uvicorn 关闭挂起。
        超时后剩余任务 future 保持未完成，在途 HTTP 请求由各自的 wait_for
        超时自然失败；shutdown 总耗时上界 = drain_timeout + worker.join 兜底。
        """
        if self._closed:
            return
        self._closed = True
        if drain:
            deadline = time.monotonic() + max(0.0, drain_timeout)
            while self._q.unfinished_tasks > 0:
                if time.monotonic() >= deadline:
                    logger.warning(
                        "WriteQueue drain timed out after %.1fs "
                        "(%d task(s) still pending, worker may be stuck)",
                        drain_timeout, self._q.unfinished_tasks,
                    )
                    break
                time.sleep(0.01)  # 轮询 task_done（限时, 不无限阻塞）
        try:
            self._q.put_nowait((2, next(self._seq), _SENTINEL))
        except queue.Full:
            # 【M3】worker 卡死 + 队列真满时阻塞 put 会永久等空位，join 兜底永远
            # 执行不到 → uvicorn 关闭挂起。改为 get_nowait 循环清空积压：对每个
            # 未完成的 _WriteTask 先 fut.set_exception(WriteQueueClosedError) 再
            # task_done()——等待方立即失败而非各自挂到 wait_for 超时；然后重放
            # sentinel（worker 若恢复则正常退出）。不新增消费者（仍是单写线程）。
            # 【v5.40】PriorityQueue 元组：item = (priority, seq, task)。
            failed = 0
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                try:
                    if (
                        isinstance(item, tuple)
                        and item[2] is not _SENTINEL
                        and not item[2].fut.done()
                    ):
                        item[2].fut.set_exception(
                            WriteQueueClosedError(
                                "write queue closed during shutdown with pending task"
                            )
                        )
                        failed += 1
                finally:
                    self._q.task_done()
            logger.warning(
                "WriteQueue shutdown: queue full, failed %d pending task(s) "
                "(worker stuck; daemon thread cleaned by process exit)",
                failed,
            )
            # 重放 sentinel：积压已清空，队列必有空位
            try:
                self._q.put_nowait((2, next(self._seq), _SENTINEL))
            except queue.Full:
                pass
        self._worker.join(timeout=5.0)
