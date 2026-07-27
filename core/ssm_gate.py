"""
SSM 选择性门控 v2.0
=============
来自 HMTE 论文的 State Space Model 门控机制。
每个超边维护一个隐状态 h_t，决定该超边的保留/遗忘。

v2.0 新特性：
- [在线学习] 根据结果反馈微调 SSM 权重（简单 REINFORCE 风格更新）
- [重要性输入] 增加 importance 特征维度
- [自适应阈值] 阈值随系统负载动态调整

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
    """SSM 门控配置 v2.0"""

    hidden_dim: int = 128  # 隐层维度
    input_dim: int = 9  # v2.0: 增加 importance 维度
    gate_threshold: float = 0.5  # 保留/遗忘阈值
    state_decay: float = 0.9  # 状态衰减系数

    # 输入特征索引
    feat_mean_activation: int = 0
    feat_age_hours: int = 1
    feat_access_freq: int = 2
    feat_member_count: int = 3
    feat_community_density: int = 4
    feat_tau_mean: int = 5
    feat_tau_variance: int = 6
    feat_connection_entropy: int = 7
    feat_importance: int = 8  # v2.0: 超边平均重要性

    # v2.0: 在线学习参数
    learning_rate: float = 0.001  # SSM 权重学习率
    enable_online_learning: bool = True  # 是否启用在线学习
    reward_decay: float = 0.95  # 奖励折扣因子


class SSMGate:
    """
    SSM 选择性门控 v2.0
    决定哪些超边值得保留，哪些应该被遗忘。
    """

    def __init__(self, config: Optional[SSMGateConfig] = None) -> None:
        self.config = config or SSMGateConfig()
        self._init_weights()
        # v2.0: 学习状态
        self._step_count: int = 0
        self._total_reward: float = 0.0

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
            z = np.clip(self.W_g @ h_t + self.b_g, -100, 100)
            g_t = 1.0 / (1.0 + np.exp(-z))
            self._step_count += 1
            return h_t, float(g_t[0, 0])
        except Exception:
            logger.warning("SSM gate step failed, defaulting to gate=1.0",
                         exc_info=True)
            return hidden_state, 1.0

    def should_keep(self, gate_value: float) -> bool:
        """根据门控值判断是否保留超边。"""
        return gate_value > self.config.gate_threshold

    def compute_input_features(self, hyperedge_data: dict) -> np.ndarray:
        """从超边数据计算输入特征向量（v2.0: 增加 importance）"""
        features = np.zeros(self.config.input_dim)
        features[self.config.feat_mean_activation] = hyperedge_data.get("mean_activation", 0.0)
        features[self.config.feat_age_hours] = hyperedge_data.get("age_hours", 0.0)
        features[self.config.feat_access_freq] = hyperedge_data.get("access_freq", 0.0)
        features[self.config.feat_member_count] = hyperedge_data.get("member_count", 0)
        features[self.config.feat_community_density] = hyperedge_data.get("community_density", 0.0)
        features[self.config.feat_tau_mean] = hyperedge_data.get("tau_mean", 0.0)
        features[self.config.feat_tau_variance] = hyperedge_data.get("tau_variance", 0.0)
        features[self.config.feat_connection_entropy] = hyperedge_data.get("connection_entropy", 0.0)
        # v2.0: importance 特征
        features[self.config.feat_importance] = hyperedge_data.get("importance", 0.5)
        return features

    # ======================== v2.0 新增 ========================

    def learn(self, gate_value: float, outcome: float, prev_h: np.ndarray) -> float:
        """在线学习：根据实际结果调整门控权重
        
        简单 REINFORCE 风格更新：
        如果 gate_value > threshold（决定保留）但 outcome 差（不应保留），
        或者 gate_value < threshold（决定遗忘）但 outcome 好（应保留），
        则调整 W_g 使下次决策更准。
        
        Args:
            gate_value: 当时的门控值
            outcome: 实际结果 [0, 1]，1=保留正确，0=遗忘正确
            prev_h: 决策时的隐状态
            
        Returns:
            reward: 本次更新的奖励值
        """
        if not self.config.enable_online_learning:
            return 0.0
        
        # 计算奖励
        decision = 1.0 if gate_value > self.config.gate_threshold else 0.0
        # reward = +1 决策正确，-1 错误
        reward = 1.0 if abs(decision - outcome) < 0.5 else -1.0
        
        # 折扣累积奖励
        self._total_reward = self.config.reward_decay * self._total_reward + reward
        
        # REINFORCE: ∇J ∝ reward · ∇log π(a|s)
        # 简化：reward > 0 则强化当前决策方向，reward < 0 则反转
        lr = self.config.learning_rate
        grad_dir = 1.0 if reward > 0 else -1.0
        
        # 更新门控权重
        h_2d = prev_h.reshape(-1, 1)  # (D, 1)
        self.W_g += lr * grad_dir * h_2d.T  # (1, D)
        self.b_g += lr * grad_dir * 0.01
        
        # 限制权重范围防止发散
        np.clip(self.W_g, -5, 5, out=self.W_g)
        np.clip(self.b_g, -5, 5, out=self.b_g)
        
        return reward

    def adapt_threshold(self, system_load: float = 0.5) -> float:
        """自适应调整门控阈值
        
        系统负载高（节点多）时提高阈值，更激进地遗忘。
        系统负载低时降低阈值，更保守地保留。
        
        Args:
            system_load: [0, 1]，接近 1 表示高负载
            
        Returns:
            新的阈值
        """
        # 基础阈值 ± 20% 范围
        new_threshold = self.config.gate_threshold * (0.8 + 0.4 * system_load)
        self.config.gate_threshold = max(0.3, min(0.8, new_threshold))
        return self.config.gate_threshold

    def reset_state(self) -> np.ndarray:
        """重置隐状态为零向量"""
        return np.zeros(self.config.hidden_dim)

    def get_stats(self) -> dict:
        """获取门控统计（v2.0）"""
        return {
            "total_steps": self._step_count,
            "total_reward": round(self._total_reward, 3),
            "current_threshold": round(self.config.gate_threshold, 3),
            "online_learning": self.config.enable_online_learning,
        }
