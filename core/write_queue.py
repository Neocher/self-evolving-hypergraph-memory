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
- 事件循环侧用 `loop.run_in_executor(self._executor, _await_future, fut)`
  桥接——`_executor` 是**专用单 worker** ThreadPoolExecutor，不借默认池，
  读路径 to_thread 完全不受写影响；`concurrent.futures.Future` 的
  set_result 可跨线程，asyncio 侧只阻塞等待。
- 超时：`asyncio.wait_for(..., wait_timeout)`。⚠️ 超时只放弃等待方，
  **不能取消正在写线程执行的 GraphLite 调用**——任务仍会落库（迟到完成
  语义，见 submit docstring）。concurrent future 不会被 wait_for 取消，
  写线程 set_result 永远合法，无 InvalidStateError 崩溃。
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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shm-writer")
        self._wait_timeout = wait_timeout
        self._closed = False
        self._worker = threading.Thread(target=self._run, daemon=True, name="shm-writer-worker")
        self._worker_ident: int = 0
        self._worker.start()
        self._worker_ident = self._worker.ident

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
        fut: Future = Future()
        task = _WriteTask(fn=fn, args=args, kwargs=kwargs, fut=fut)
        try:
            self._q.put_nowait(task)
        except queue.Full:
            raise WriteQueueFullError(
                f"write queue full ({self._q.qsize()}/{self._q.maxsize} pending)"
            )
        loop = asyncio.get_running_loop()
        # run_in_executor(专用单 worker executor)：阻塞等结果不占事件循环；
        # wait_for 超时只取消外层 wrapper，写线程的 task.fut 不受影响。
        return await asyncio.wait_for(
            loop.run_in_executor(self._executor, self._await_future, fut),
            timeout=self._wait_timeout,
        )

    def _await_future(self, fut: Future) -> Any:
        """在专用执行器线程里阻塞等写线程结果——不占事件循环。"""
        return fut.result()

    # ─── 写线程侧：串行消费 ────────────────────────────

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                self._q.task_done()
                break
            task: _WriteTask = item
            try:
                result = task.fn(*task.args, **task.kwargs)
                task.fut.set_result(result)
            except BaseException as exc:  # 含 GraphLite 异常 / CircuitBreakerOpen
                task.fut.set_exception(exc)
            finally:
                self._q.task_done()

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
        self._q.put(_SENTINEL)
        self._worker.join(timeout=5.0)
        self._executor.shutdown(wait=False)
