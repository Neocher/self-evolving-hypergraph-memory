"""
SSM 选择性门控
=============
来自 HMTE 论文的 State Space Model 门控机制。
每个超边维护一个隐状态 h_t，决定该超边的保留/遗忘。

门控值: g_t = sigmoid(W_g · h_t + b_g)
状态转移: h_t = A · h_{t-1} + B · x_t

实现为 2 层 MLP（128 维隐层），CPU 微秒级推理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SSMGateConfig:
    """SSM 门控配置"""

    hidden_dim: int = 128  # 隐层维度
    input_dim: int = 8  # 输入特征维度
    gate_threshold: float = 0.5  # 保留/遗忘阈值
    state_decay: float = 0.9  # 状态衰减系数

    # 输入特征索引（可配置）
    feat_mean_activation: int = 0  # 超边内节点平均激活度
    feat_age_hours: int = 1  # 超边年龄（小时）
    feat_access_freq: int = 2  # 访问频率
    feat_member_count: int = 3  # 成员节点数
    feat_community_density: int = 4  # 社区密度
    feat_tau_mean: int = 5  # 平均 τ 值
    feat_tau_variance: int = 6  # τ 方差
    feat_connection_entropy: int = 7  # 连接熵


class SSMGate:
    """
    SSM 选择性门控。

    决定哪些超边值得保留，哪些应该被遗忘。
    每个超边有独立的隐状态，随时间步更新。
    """

    def __init__(self, config: Optional[SSMGateConfig] = None) -> None:
        self.config = config or SSMGateConfig()
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化 SSM 权重矩阵"""
        D = self.config.hidden_dim
        M = self.config.input_dim
        self.A = np.eye(D) * self.config.state_decay + np.random.randn(D, D) * 0.01
        self.B = np.random.randn(D, M) * 0.1
        self.W_g = np.random.randn(1, D) * 0.1
        self.b_g = np.zeros((1, 1))

    def step(
        self, hidden_state: np.ndarray, input_features: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        单步状态转移。

        Args:
            hidden_state: 上一时间步的隐状态 (hidden_dim,)
            input_features: 当前时间步的输入特征 (input_dim,)

        Returns:
            (new_hidden_state, gate_value)
            gate_value 在 (0, 1) 之间，> threshold 时保留超边
        """
        try:
            h_t = np.tanh(self.A @ hidden_state + self.B @ input_features)
            # 使用 np.clip 防止 sigmoid 溢出
            z = np.clip(self.W_g @ h_t + self.b_g, -100, 100)
            g_t = 1.0 / (1.0 + np.exp(-z))
            return h_t, float(g_t[0, 0])
        except Exception:
            # 门控计算失败时默认放行（gate_value=1.0）
            logger.warning("SSM gate step failed, defaulting to gate=1.0",
                         exc_info=True)
            return hidden_state, 1.0

    def should_keep(self, gate_value: float) -> bool:
        """根据门控值判断是否保留超边。"""
        return gate_value > self.config.gate_threshold

    def compute_input_features(self, hyperedge_data: dict) -> np.ndarray:
        """从超边数据计算输入特征向量。"""
        features = np.zeros(self.config.input_dim)
        features[self.config.feat_mean_activation] = hyperedge_data.get(
            "mean_activation", 0.0
        )
        features[self.config.feat_age_hours] = hyperedge_data.get("age_hours", 0.0)
        features[self.config.feat_access_freq] = hyperedge_data.get("access_freq", 0.0)
        features[self.config.feat_member_count] = hyperedge_data.get("member_count", 0)
        features[self.config.feat_community_density] = hyperedge_data.get(
            "community_density", 0.0
        )
        features[self.config.feat_tau_mean] = hyperedge_data.get("tau_mean", 0.0)
        features[self.config.feat_tau_variance] = hyperedge_data.get("tau_variance", 0.0)
        features[self.config.feat_connection_entropy] = hyperedge_data.get(
            "connection_entropy", 0.0
        )
        return features

    def reset_state(self) -> np.ndarray:
        """重置隐状态为零向量"""
        return np.zeros(self.config.hidden_dim)
