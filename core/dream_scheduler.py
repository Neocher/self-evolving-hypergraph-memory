"""
梦境调度器
=========
三种触发模式：
1. 空闲触发 — 对话结束空闲 5 分钟后自动启动
2. 累积触发 — 累积 100 个新节点时触发
3. 显式触发 — 通过 API 调用显式触发

优先级计算:
    P = (community_total_degree / community_node_count)
        × (1 + community_freshness_factor)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TriggerMode(Enum):
    IDLE = "idle"  # 空闲触发
    ACCUMULATED = "accum"  # 累积触发
    EXPLICIT = "explicit"  # 显式触发


@dataclass
class DreamSchedulerConfig:
    """梦境调度配置"""

    idle_timeout_seconds: int = 300  # 空闲 5 分钟后触发
    accum_threshold: int = 100  # 累积 100 个新节点触发
    min_interval_seconds: int = 60  # 最小间隔（防止频繁触发）
    max_dream_duration_seconds: int = 300  # 单次梦境最长 5 分钟
    cpu_affinity_low_priority: bool = True  # 低优先级 CPU 亲和性
    memory_limit_mb: int = 256  # 梦境线程内存限制


class DreamScheduler:
    """
    梦境调度器。

    监控系统状态，在合适的时机触发梦境整合管道。
    支持三种触发模式，带优先级计算和并发控制。
    """

    def __init__(
        self,
        config: Optional[DreamSchedulerConfig] = None,
        pipeline_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config or DreamSchedulerConfig()
        self.pipeline_fn = pipeline_fn
        self._last_run_time: float = 0.0
        self._is_running: bool = False
        self._new_node_count: int = 0
        self._last_activity_time: float = time.time()
        self._lock = asyncio.Lock()

    async def on_activity(self) -> None:
        """记录活动时间戳（每次有节点创建/更新时调用）。"""
        self._last_activity_time = time.time()

    async def on_node_created(self) -> None:
        """节点创建通知，增加累积计数。"""
        self._new_node_count += 1

    async def check_and_trigger(self) -> bool:
        """
        检查是否满足触发条件，是则启动梦境。

        Returns:
            True 如果梦境被触发
        """
        async with self._lock:
            if self._is_running:
                return False
            now = time.time()
            if now - self._last_run_time < self.config.min_interval_seconds:
                return False

            idle_time = now - self._last_activity_time
            should_run = False
            trigger_mode: Optional[TriggerMode] = None

            if idle_time > self.config.idle_timeout_seconds:
                should_run = True
                trigger_mode = TriggerMode.IDLE
            elif self._new_node_count >= self.config.accum_threshold:
                should_run = True
                trigger_mode = TriggerMode.ACCUMULATED

            if should_run:
                self._is_running = True
                asyncio.create_task(self._run_dream(trigger_mode))
                return True
        return False

    async def trigger_explicit(self) -> bool:
        """显式触发梦境（通过 API 调用）。"""
        async with self._lock:
            if self._is_running:
                return False
            self._is_running = True
            asyncio.create_task(self._run_dream(TriggerMode.EXPLICIT))
            return True

    async def _run_dream(self, trigger_mode: Optional[TriggerMode]) -> None:
        """执行梦境管道（内部协程）。"""
        logger.info("Dream triggered by %s mode", trigger_mode.value if trigger_mode else "unknown")
        try:
            if self.pipeline_fn:
                await self.pipeline_fn()
            self._new_node_count = 0
            self._last_run_time = time.time()
        except Exception:
            logger.exception("Dream pipeline failed")
        finally:
            self._is_running = False

    def compute_priority(
        self,
        community_total_degree: float,
        community_node_count: int,
        community_freshness: float,
    ) -> float:
        """
        计算社区梦境优先级。

        P = (total_degree / node_count) × (1 + freshness)

        Args:
            community_total_degree: 社区内节点连接度之和
            community_node_count: 社区内节点数量
            community_freshness: 社区新鲜度 (0~1, 1=全新)

        Returns:
            优先级分数（越高越优先处理）
        """
        if community_node_count == 0:
            return 0.0
        density = community_total_degree / community_node_count
        return density * (1.0 + community_freshness)

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def accumulated_count(self) -> int:
        return self._new_node_count
