"""
τ 时间常数衰减引擎
===================
每个记忆节点自创建起携带一个指数衰减权重：
    τ(t) = τ₀ · exp(-t / τ_decay)

默认 τ_decay = 1800s (30分钟)，当 t = τ_decay 时衰减到初始值的 ~37%。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TauDecayConfig:
    """τ 衰减配置"""

    tau_initial: float = 1.0  # 初始 τ 值
    tau_decay_seconds: float = 1800.0  # 衰减时间常数（秒）
    decay_threshold: float = 0.1  # 修剪阈值：低于此值且低连接度的节点被标记为候选
    refresh_on_access: bool = True  # 访问时是否刷新（模拟再巩固）

    def validate(self) -> None:
        """校验配置合理性"""
        assert 0 < self.tau_initial <= 1.0, "tau_initial must be in (0, 1]"
        assert self.tau_decay_seconds > 0, "tau_decay_seconds must be positive"
        assert 0 < self.decay_threshold < 1.0, "decay_threshold must be in (0, 1)"


class TauDecayEngine:
    """
    τ 衰减引擎
    管理所有节点的 τ 值计算、阈值检测和再巩固刷新。
    """

    def __init__(self, config: Optional[TauDecayConfig] = None) -> None:
        self.config = config or TauDecayConfig()
        self.config.validate()
        self._tau_cache: dict[str, tuple[float, float]] = {}

    def compute_tau(self, created_at: float, accessed_at: Optional[float] = None) -> float:
        """
        计算当前 τ 值。

        Args:
            created_at: 节点创建时间戳（Unix秒）
            accessed_at: 最近访问时间戳（None = 使用当前时间）

        Returns:
            当前 τ 值 (0, τ₀]
        """
        dt = (accessed_at or time.time()) - created_at
        if dt < 0:
            dt = 0
        return self.config.tau_initial * math.exp(-dt / self.config.tau_decay_seconds)

    def is_decay_candidate(
        self, created_at: float, accessed_at: Optional[float] = None
    ) -> bool:
        """判断节点是否低于衰减阈值，标记为修剪候选。"""
        return self.compute_tau(created_at, accessed_at) < self.config.decay_threshold

    def refresh_tau(self, node_id: str, created_at: float) -> float:
        """再巩固：访问节点时刷新其虚拟时间戳，提升 τ 值。"""
        return self.config.tau_initial

    def batch_compute(self, nodes: list[tuple[str, float, Optional[float]]]) -> dict[str, float]:
        """
        批量计算 τ 值。O(N) 时间复杂度。

        Args:
            nodes: [(node_id, created_at, accessed_at), ...]

        Returns:
            {node_id: tau_value, ...}
        """
        now = time.time()
        result: dict[str, float] = {}
        for node_id, created_at, accessed_at in nodes:
            result[node_id] = self.compute_tau(created_at, accessed_at or now)
        return result
