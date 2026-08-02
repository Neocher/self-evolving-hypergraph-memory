"""
[Harness Fix] 指数退避重试装饰器
===============================
所有外部调用（GraphLite 查询、FAISS 搜索、LLM 调用）使用 @with_retry() 装饰。

重试策略：
- 最大尝试次数：3 次
- 初始延迟：1 秒
- 退避因子：2.0（1s → 2s → 4s）
- 可重试异常：ConnectionError, TimeoutError（可扩展）
- max_total_timeout：全局超时保护（0.0 = 不限制）

[Fix] 双模式支持：装饰同步函数返回同步包装器（time.sleep / time.monotonic），
装饰异步函数返回异步包装器（asyncio.sleep / loop.time）。同步包装器无需
运行中的事件循环，可安全用于 GraphLiteStore 等同步 API 的重试。
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Callable, Optional, Tuple, Type


def _retry_delay(
    attempt: int,
    clock: Callable[[], float],
    start_time: Optional[float],
    max_total_timeout: float,
    base_delay: float,
    backoff: float,
) -> float:
    """计算第 attempt 次的退避延迟（秒）；超出全局预算则抛 TimeoutError。"""
    delay = base_delay * (backoff ** attempt)
    if start_time is not None:
        remaining = max_total_timeout - (clock() - start_time)
        delay = min(delay, max(0.1, remaining - 0.5))
        if delay <= 0:
            raise TimeoutError(
                f"with_retry total timeout {max_total_timeout}s exceeded"
            )
    return delay


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    max_total_timeout: float = 0.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError, TimeoutError
    ),
):
    """
    指数退避重试装饰器。

    [Fix] 新增 max_total_timeout 全局超时保护。
          0.0 表示不限制总超时。

    Args:
        max_attempts: 最大尝试次数（默认 3）
        base_delay: 初始延迟秒数（默认 1.0）
        backoff: 退避乘数（默认 2.0）
        max_total_timeout: 全局最大超时（秒），0.0=不限制
        retryable_exceptions: 可重试的异常类型元组

    Returns:
        装饰后的函数

    Example:
        @with_retry()
        def query_graphlite(...):
            ...

        @with_retry(max_attempts=5, base_delay=0.5, backoff=1.5)
        def call_llm(...):
            ...

        @with_retry(max_total_timeout=10.0)
        def timed_query(...):
            ...
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                loop_time = asyncio.get_running_loop().time
                start_time = loop_time() if max_total_timeout > 0 else None
                for attempt in range(max_attempts):
                    if start_time is not None:
                        elapsed = loop_time() - start_time
                        if elapsed >= max_total_timeout:
                            raise TimeoutError(
                                f"with_retry total timeout {max_total_timeout}s exceeded "
                                f"after {attempt} attempts ({elapsed:.1f}s)"
                            )
                    try:
                        return await func(*args, **kwargs)
                    except retryable_exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            delay = _retry_delay(
                                attempt, loop_time, start_time,
                                max_total_timeout, base_delay, backoff,
                            )
                            await asyncio.sleep(delay)
                raise last_exception  # type: ignore[misc]
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            clock = time.monotonic
            start_time = clock() if max_total_timeout > 0 else None
            for attempt in range(max_attempts):
                if start_time is not None:
                    elapsed = clock() - start_time
                    if elapsed >= max_total_timeout:
                        raise TimeoutError(
                            f"with_retry total timeout {max_total_timeout}s exceeded "
                            f"after {attempt} attempts ({elapsed:.1f}s)"
                        )
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = _retry_delay(
                            attempt, clock, start_time,
                            max_total_timeout, base_delay, backoff,
                        )
                        time.sleep(delay)
            raise last_exception  # type: ignore[misc]
        return sync_wrapper
    return decorator
