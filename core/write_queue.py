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
- 重入：写线程内再 submit → 直接同步执行（防死锁）。
- 优雅关闭：sentinel 退出 + join(timeout) 兜底；shutdown 先 drain 在途写。

不引入锁/事务/重构；调用方按"写调用"语义选择 submit（读调用留事件循环）。
"""

from __future__ import annotations

import asyncio
import logging
import queue
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

    def __init__(self, max_pending: int = 100, wait_timeout: float = 30.0):
        self._q: queue.Queue = queue.Queue(maxsize=max_pending)
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

    # ─── 事件循环侧：唯一入口 ───────────────────────────

    async def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """入队并等待写线程完成，返回 fn 的结果。

        - 队列满 → WriteQueueFullError（API 层转 503）
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
        try:
            self._q.put_nowait(task)
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
        return await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(fut)),
            timeout=self._wait_timeout,
        )

    # ─── 写线程侧：串行消费 ────────────────────────────

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                self._q.task_done()
                break
            task: _WriteTask = item
            try:
                # 【F3】心跳：任务开始前打点（即便任务卡死也能检测）
                self._touch_activity()
                result = task.fn(*task.args, **task.kwargs)
                task.fut.set_result(result)
            except BaseException as exc:  # 含 GraphLite 异常 / CircuitBreakerOpen
                task.fut.set_exception(exc)
            finally:
                self._touch_activity()
                self._q.task_done()

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

    # ─── 背压 / 探测（P2-1 梦境错峰探测点）──────────────

    def pending_count(self) -> int:
        """已入队未完成的任务数（近似值）。"""
        return self._q.qsize()

    @property
    def max_pending(self) -> int:
        return self._q.maxsize

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
            self._q.put_nowait(_SENTINEL)
        except queue.Full:
            # MED-4：worker 卡死 + 积压达 maxsize 时阻塞 put 会永久等空位，
            # join 兜底永远执行不到 → uvicorn 关闭挂起。worker 卡死时 join
            # 本就会超时，daemon 线程随进程退出清理，跳过 sentinel 无害。
            logger.warning(
                "WriteQueue shutdown: queue full, skipping sentinel "
                "(worker stuck; daemon thread cleaned by process exit)"
            )
        self._worker.join(timeout=5.0)
