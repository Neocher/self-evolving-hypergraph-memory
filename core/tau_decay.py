"""
τ 时间常数衰减引擎 v2.0
===================
每个记忆节点自创建起携带一个指数衰减权重：
    τ(t) = τ₀ · exp(-t / τ_decay)

v2.0 新特性：
- [自适应τ衰减] 每个节点根据 importance 动态调整 τ_decay
- [重要性评分] 写入时记录 importance，影响衰减速度
- [再巩固增强] 多次访问形成渐进式强化
"""
from __future__ import annotations

# —— 重要性平滑参数 ——
IMP_SMOOTH_ALPHA = 0.7
IMP_BOOST_FACTOR = 2.0
HIGH_IMPORTANCE_THRESHOLD = 0.7
ACCESS_FREQ_DIVISOR = 50
ACCESS_COUNT_THRESHOLD = 5

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

# —— RoMem 时序相位旋转常量 ——
_UNIT_TO_SECONDS: dict[str, float] = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}

ROEM_ALPHA: float = float(os.environ.get("ROEM_ALPHA", "1.0"))

ROEM_PERIODS: list[float] = [
    30 * 60,      # 30 minutes
    3600,         # 1 hour
    7200,         # 2 hours
    14400,        # 4 hours
    28800,        # 8 hours
    86400,        # 24 hours
]


def _parse_roem_periods() -> list[float]:
    """从 ROEM_PERIODS 环境变量解析周期列表（逗号分隔，支持 s/m/h/d 后缀）。"""
    raw = os.environ.get("ROEM_PERIODS", "")
    if not raw:
        return ROEM_PERIODS
    periods: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token[-1] in _UNIT_TO_SECONDS:
            unit = token[-1]
            num = float(token[:-1])
            periods.append(num * _UNIT_TO_SECONDS[unit])
        else:
            periods.append(float(token))
    return periods


def _phase_similarity(age_seconds: float, periods: list[float]) -> float:
    """傅里叶相位相似度 — RoMem 时序相位旋转核心。

    对给定时间间隔，计算各周期通道的余弦响应均值。
    Returns: [-1, 1]
    """
    if not periods:
        return 0.0
    similarity = 0.0
    for period in periods:
        angle = 2.0 * math.pi * age_seconds / period
        similarity += math.cos(angle)
    return similarity / len(periods)


@dataclass
class TauDecayConfig:
    """τ 衰减配置"""

    tau_initial: float = 1.0  # 初始 τ 值
    tau_decay_seconds: float = 1800.0  # 默认衰减时间常数（秒）
    decay_threshold: float = 0.1  # 修剪阈值：低于此值且低连接度的节点被标记为候选
    refresh_on_access: bool = True  # 访问时是否刷新（模拟再巩固）
    
    # v2.0 自适应参数
    enable_adaptive: bool = True  # 是否启用自适应 τ_decay
    tau_decay_min: float = 300.0    # 最短衰减常数（5分钟，高重要性节点）
    tau_decay_max: float = 7200.0   # 最长衰减常数（2小时，低重要性节点）
    importance_decay_modulator: float = 0.5  # 重要性对衰减的调制强度
    # τ_decay_effective = τ_decay_base · (1 - I · m)
    # 其中 I ∈ [0,1] 是归一化重要性，m 是调制强度
    # I=1.0 → τ_decay = τ_decay_base · (1 - 0.5) = 0.5·τ_decay_base (记忆更持久)
    # I=0.0 → τ_decay = τ_decay_base (原始行为)
    
    refresh_boost: float = 0.3  # 每次再巩固的时间提升比例
    max_refresh_boost: float = 2.0  # 再巩固累积上限
    
    # v2.5 AdaMem 可学习遗忘 (P2)
    enable_learnable: bool = False
    decay_learning_rate: float = 0.01
    decay_target_window: int = 10
    
    def validate(self) -> None:
        """校验配置合理性"""
        assert 0 < self.tau_initial <= 1.0, "tau_initial must be in (0, 1]"
        assert self.tau_decay_seconds > 0, "tau_decay_seconds must be positive"
        assert 0 < self.decay_threshold < 1.0, "decay_threshold must be in (0, 1)"


@dataclass
class NodeMemoryInfo:
    """节点记忆信息（v2.0 扩展）"""
    node_id: str
    created_at: float
    accessed_at: Optional[float] = None
    importance: float = 0.5  # 重要性 [0, 1]，v2.0
    access_count: int = 0     # 访问次数，v2.0
    tau_decay_custom: Optional[float] = None  # 自定义衰减常数，v2.0


class AdaptiveDecayLearner:
    """
    AdaMem 可学习遗忘 — 在线 SGD 反馈学习器 (P2)

    通过外部反馈信号（访问频率、门控标记）累积窗口后，
    执行一步 SGD 更新对应节点的 tau_decay_custom。
    """

    def __init__(self, config: TauDecayConfig) -> None:
        self.config = config
        # {node_id: [(tau_desired, timestamp), ...]}
        self._feedback_buffer: dict[str, list[tuple[float, float]]] = {}

    def record_feedback(self, node_id: str, tau_desired: float) -> None:
        """记录一次反馈信号：访问↑→τ_desired↑, 遗忘↓→τ_desired↓"""
        if node_id not in self._feedback_buffer:
            self._feedback_buffer[node_id] = []
        self._feedback_buffer[node_id].append((tau_desired, time.time()))

    def update_decay(self, engine: TauDecayEngine) -> dict[str, float]:
        """
        积累反馈满窗口后 SGD 更新 tau_decay_custom。

        Returns:
            {node_id: new_tau_decay, ...} 被更新的节点映射
        """
        updates: dict[str, float] = {}
        for node_id, feedbacks in list(self._feedback_buffer.items()):
            if len(feedbacks) < self.config.decay_target_window:
                continue
            tau_target = sum(f[0] for f in feedbacks) / len(feedbacks)
            self._feedback_buffer[node_id] = []
            info = engine._node_info.get(node_id)
            if info is None:
                continue
            current = info.tau_decay_custom or self.config.tau_decay_seconds
            lr = self.config.decay_learning_rate
            updated = current + lr * (tau_target - current)
            engine.set_custom_decay(node_id, updated)
            updates[node_id] = updated
        return updates


class TauDecayEngine:
    """
    τ 衰减引擎 v2.0
    管理所有节点的 τ 值计算、自适应衰减、重要性调制和再巩固。
    """

    def __init__(self, config: Optional[TauDecayConfig] = None,
                 learner: Optional[AdaptiveDecayLearner] = None) -> None:
        self.config = config or TauDecayConfig()
        self.config.validate()
        self._tau_cache: dict[str, tuple[float, float]] = {}
        # v2.0: 节点记忆信息缓存
        self._node_info: dict[str, NodeMemoryInfo] = {}
        # v2.5 AdaMem 可学习遗忘
        self.learner: Optional[AdaptiveDecayLearner] = learner

    def register_node(self, node_id: str, created_at: Optional[float] = None,
                      importance: float = 0.5) -> None:
        """注册一个节点（v2.0）
        
        在创建节点时调用，记录重要性等信息。
        """
        if node_id not in self._node_info:
            self._node_info[node_id] = NodeMemoryInfo(
                node_id=node_id,
                created_at=created_at or time.time(),
                importance=max(0.0, min(1.0, importance)),
            )

    def update_importance(self, node_id: str, importance: float) -> None:
        """更新节点重要性（v2.0）"""
        if node_id in self._node_info:
            # 平滑更新，防止单次异常值剧烈波动
            old = self._node_info[node_id].importance
            self._node_info[node_id].importance = IMP_SMOOTH_ALPHA * old + (1 - IMP_SMOOTH_ALPHA) * max(0.0, min(1.0, importance))

    def set_custom_decay(self, node_id: str, tau_decay: float) -> None:
        """设置自定义衰减常数（v2.0）
        
        用于 EvolveMem 式的自优化衰减。
        """
        if node_id in self._node_info:
            self._node_info[node_id].tau_decay_custom = max(
                self.config.tau_decay_min, min(self.config.tau_decay_max, tau_decay)
            )

    def _get_effective_tau_decay(self, node_id: str) -> float:
        """计算有效衰减常数（v2.0 自适应）
        
        考虑三个因素：
        1. 自定义 τ_decay（如有，优先级最高）
        2. 基础 τ_decay × 重要性调制
        3. 访问次数增强
        """
        info = self._node_info.get(node_id)
        if not info:
            return self.config.tau_decay_seconds
        
        # 自定义衰减
        if info.tau_decay_custom is not None:
            return info.tau_decay_custom
        
        if not self.config.enable_adaptive:
            return self.config.tau_decay_seconds
        
        # 重要性调制：高重要性 → 衰减更慢（τ_decay 更大）
        I = info.importance  # [0, 1]
        m = self.config.importance_decay_modulator
        base = self.config.tau_decay_seconds
        # 高重要性 → 增加 τ_decay（衰减更慢，记忆更持久）
        boost = 1.0 + I * m * IMP_BOOST_FACTOR  # I=1.0 → 2x, I=0.5 → 1.5x, I=0 → 1x
        effective = base * boost
        
        # 访问次数增强：经常访问的节点衰减更慢
        if info.access_count > ACCESS_COUNT_THRESHOLD:
            freq_boost = 1.0 + min(1.0, info.access_count / ACCESS_FREQ_DIVISOR)
            effective *= freq_boost
        
        return max(self.config.tau_decay_min, 
                   min(self.config.tau_decay_max, effective))

    def compute_tau(self, node_id: str, created_at: Optional[float] = None,
                    force_now: Optional[float] = None) -> float:
        """计算当前 τ 值（v2.0 自适应版本）
        
        使用节点级别的自适应衰减常数。
        
        Args:
            node_id: 节点ID（用于查找自定义衰减）
            created_at: 节点创建时间戳
            force_now: 强制时间（用于测试）
        """
        info = self._node_info.get(node_id)
        created = created_at or (info.created_at if info else time.time())
        now = force_now or time.time()
        dt = max(0, now - created)
        tau_decay = self._get_effective_tau_decay(node_id)
        return self.config.tau_initial * math.exp(-dt / tau_decay)

    def compute_strength(self, created_at: float) -> float:
        """Compute RoMem temporal phase strength.

        结合 τ 衰减与傅里叶相位相似度，产生周期性记忆强度。
        S = alpha * tau(t) + (1-alpha) * (phase_sim + 1) / 2

        ROEM_ALPHA=1.0 时行为与 compute_tau() 完全一致（向后兼容）。
        """
        age = max(0.0, time.time() - created_at)
        tau_val = self.compute_tau(node_id="", created_at=created_at)
        phase_sim = _phase_similarity(age, ROEM_PERIODS)
        return ROEM_ALPHA * tau_val + (1.0 - ROEM_ALPHA) * (phase_sim + 1.0) / 2.0

    def is_decay_candidate(self, node_id: str, created_at: Optional[float] = None,
                          connections: Optional[dict] = None) -> bool:
        """判断节点是否低于衰减阈值，标记为修剪候选（v2.0）
        
        增强条件：τ < threshold 且（低重要性 或 无连接）
        """
        tau = self.compute_tau(node_id, created_at)
        if tau >= self.config.decay_threshold:
            return False
        
        # v2.0: 高重要性节点即使 τ 低也推迟修剪
        info = self._node_info.get(node_id)
        if info and info.importance > HIGH_IMPORTANCE_THRESHOLD and tau >= self.config.decay_threshold * 0.5:
            return False
        
        # 有连接且连接数 > 3 的节点保留
        if connections and len(connections) > 3:
            return False
        
        return True

    def refresh_tau(self, node_id: str, created_at: Optional[float] = None) -> float:
        """再巩固：访问节点时提升 τ 值（v2.0 渐进增强）
        
        每次访问不仅重置 τ，还累积 refresh_boost。
        多次访问形成渐进式强化——模拟记忆的"间隔重复"效应。
        """
        if not self.config.refresh_on_access:
            return self.compute_tau(node_id, created_at)
        
        info = self._node_info.get(node_id)
        if info:
            info.access_count += 1
        
        return self.config.tau_initial

    def batch_compute(self, nodes: list[tuple[str, float, Optional[float], float]]) -> dict[str, float]:
        """批量计算 τ 值（v2.0 扩展参数）
        
        Args:
            nodes: [(node_id, created_at, accessed_at, importance), ...]
            
        Returns:
            {node_id: tau_value, ...}
        """
        result: dict[str, float] = {}
        for item in nodes:
            if len(item) == 4:
                node_id, created_at, accessed_at, importance = item
                self.register_node(node_id, created_at, importance)
            else:
                node_id, created_at, accessed_at = item
            result[node_id] = self.compute_tau(node_id, created_at)
        return result
    
    def get_adaptation_stats(self) -> dict:
        """获取自适应衰减统计（v2.0）"""
        infos = list(self._node_info.values())
        if not infos:
            return {"node_count": 0}
        return {
            "node_count": len(infos),
            "avg_importance": sum(n.importance for n in infos) / len(infos),
            "avg_access_count": sum(n.access_count for n in infos) / len(infos),
            "adaptive_enabled": self.config.enable_adaptive,
            "tau_decay_range": [self.config.tau_decay_min, self.config.tau_decay_max],
        }
