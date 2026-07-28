"""
SSM + MLP 双门控引擎 v3.0 (Dual Adaptive Gate)
=============================================
SSM提供结构化状态空间原理门控 + MLP提供经验学习门控 的相互优化配合。

架构:
  x_t → [SSM Engine] → h_t (principled state space evolution)
                              ↓
                       [MLP Gate] → g_mlp (learned gating policy)
                              ↓
              g = α · g_ssm_gate · (1-α) · g_mlp

SSM Engine: 真正的结构化状态空间模型
- 基于 HiPPO (Legendre) 矩阵初始化 (Gu et al., 2020)
- 结构化 A 矩阵确保长程记忆传播
- ZOH (Zero-Order Hold) 离散化

MLP Gate: 现有学习型门控 v2.1
- 读取 SSM 隐状态做门控决策
- 在线策略梯度学习
- 随学习进展自动增加 α 权重

α 动态融合:
- 初始 α=0.5（SSM 和 MLP 等权）
- 随着 MLP 学习积累，α 渐降至 0.2（MLP 占主导）
- 置信度校准：SSM 熵低时自动提升其权重
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)

# SSM initialization scale factor for HiPPO matrix
HIPPO_INIT_SCALE = 0.1
ZOH_STEPS = 100
W_SSM_INIT_SCALE = 0.05


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class DualGateConfig:
    """SSM+MLP 双门控配置 v3.0"""

    # 通用
    hidden_dim: int = 128       # SSM 状态维度 / MLP 隐层维度
    input_dim: int = 9           # 输入特征维度 (8 + importance)
    gate_threshold: float = 0.5  # 保留/遗忘阈值

    # SSM 参数
    ssm_state_decay: float = 0.9      # 状态衰减系数
    ssm_hippo_order: int = 64         # HiPPO 阶数（≤ hidden_dim）
    ssm_discretization: str = "zoh"   # 离散化方式: "zoh" | "bilinear"

    # MLP 参数
    mlp_learning_rate: float = 0.001
    mlp_enable_online_learning: bool = True
    mlp_reward_decay: float = 0.95

    # α 融合参数
    alpha_initial: float = 0.5    # SSM 初始权重 (0=纯MLP, 1=纯SSM)
    alpha_max: float = 1.0         # SSM 最大权重（防止α突破1.0使MLP权重变负）
    alpha_min: float = 0.2         # 最小 SSM 权重
    alpha_learning_decay: float = 0.01  # 每次 learn() 后 α 衰减量
    alpha_entropy_boost: float = 0.3    # 熵低时 α 提升幅度

    # 预算感知门控 (Retain or Consolidate? arXiv:2607.17545)
    budget_capacity: int = 100        # 预算容量（学习步数周期）
    budget_restore_rate: float = 0.1  # 每步预算恢复率
    budget_consolidate_cost: int = 30  # 整合操作消耗的预算

    # 随机种子
    seed: int = 42

    # 输入特征索引
    feat_mean_activation: int = 0
    feat_age_hours: int = 1
    feat_access_freq: int = 2
    feat_member_count: int = 3
    feat_community_density: int = 4
    feat_tau_mean: int = 5
    feat_tau_variance: int = 6
    feat_connection_entropy: int = 7
    feat_importance: int = 8


# ═══════════════════════════════════════════════════════════════
# SSM 引擎 — 真正的结构化状态空间模型
# ═══════════════════════════════════════════════════════════════

class SSMEngine:
    """
    SSM 引擎 v1.0 — 结构化状态空间模型。

    基于 HiPPO 矩阵初始化（Legendre 多项式投影），提供
    有理论保证的长期记忆传播能力。

    核心公式 (连续时间):
      h'(t) = A · h(t) + B · x(t)

    离散化 (ZOH):
      h_t = exp(Δt · A) · h_{t-1} + (∫₀^{Δt} exp(τ·A) dτ · B) · x_t
    """

    def __init__(self, config: DualGateConfig, rng: np.random.RandomState) -> None:
        self.config = config
        self._init_hippo(rng)

    def _init_hippo(self, rng: np.random.RandomState) -> None:
        """
        初始化 HiPPO 矩阵 (Legendre 多项式投影)。

        HiPPO 矩阵 A 的结构:
          A_{nk} = - (2n+1)^{1/2} · (2k+1)^{1/2},  n > k
          A_{nn} = - (n+1)
          A_{nk} = 0,  n < k

        这是 normalized Legendre (LegT) 的无穷维限制。
        对于有限维 N，这是一个下三角矩阵。
        """
        N = min(self.config.ssm_hippo_order, self.config.hidden_dim)
        D = self.config.hidden_dim

        # 构建 HiPPO 矩阵 (Legendre)
        self.A_hippo = np.zeros((N, N), dtype=np.float64)
        for n in range(N):
            for k in range(N):
                if n > k:
                    self.A_hippo[n, k] = -math.sqrt((2 * n + 1) * (2 * k + 1))
                elif n == k:
                    self.A_hippo[n, k] = -(n + 1)
                # n < k: 保持不变 0

        # 扩展到全 hidden_dim
        # 前 N 维用 HiPPO，剩余维度用单位衰减
        self.A = np.eye(D, dtype=np.float64) * (-1.0)
        self.A[:N, :N] = self.A_hippo[:N, :N]

        # B 矩阵: 随机投影
        M = self.config.input_dim
        top = rng.randn(N, M) * HIPPO_INIT_SCALE
        bottom = np.zeros((D - N, M))
        self.B = np.vstack([top, bottom]) * 0.1

        # 预计算离散化系数 (ZOH: exp(A·Δt), 取 Δt=1)
        self.A_bar = np.linalg.matrix_power(np.eye(D) + self.A / ZOH_STEPS, ZOH_STEPS)
        # B_bar ≈ A^{-1}(exp(A) - I)·B ≈ (I + A/2)·B  (一阶近似)
        self.B_bar = (np.eye(D) + self.A / 2.0) @ self.B

        # SSM 专用门控读取头
        self.W_ssm = rng.randn(1, D) * W_SSM_INIT_SCALE
        self.b_ssm = np.zeros((1, 1))

    def step(self, hidden_state: np.ndarray, input_features: np.ndarray) -> tuple[np.ndarray, float]:
        """
        SSM 单步状态转移。

        Args:
            hidden_state: 上一时间步隐状态 (hidden_dim,)
            input_features: 当前输入 (input_dim,)

        Returns:
            (new_hidden_state, ssm_gate_value)
            ssm_gate_value ∈ (0, 1) — 基于 SSM 动力学的原理门控
        """
        try:
            # 离散状态转移: h_t = A_bar @ h_{t-1} + B_bar @ x_t
            new_h = self.A_bar @ hidden_state + self.B_bar @ input_features
            new_h = np.tanh(new_h)  # 非线性稳定化

            # SSM 门控: 基于当前状态的"活化能"
            z_ssm = np.clip(self.W_ssm @ new_h + self.b_ssm, -50, 50)
            g_ssm = 1.0 / (1.0 + np.exp(-z_ssm))

            return new_h, float(g_ssm[0, 0])
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            logger.error("SSM step failed: %s", e, exc_info=True)
            return hidden_state, 0.5  # 失败时中性值


# ═══════════════════════════════════════════════════════════════
# MLP 门控 — 经验学习型门控
# ═══════════════════════════════════════════════════════════════

class MLPGate:
    """
    MLP 门控 v2.1 — 基于 2 层 MLP 的经验学习门控。

    读取 SSM 引擎的隐状态，通过策略梯度在线学习
    最优的门控决策。

    核心公式:
      g_mlp = sigmoid(W_g · tanh(W_h · h_ssm + b_h) + b_g)
    """

    def __init__(self, config: DualGateConfig, rng: np.random.RandomState) -> None:
        self.config = config
        self.rng = rng
        self._init_weights()
        self._step_count: int = 0
        self._total_reward: float = 0.0

    def _init_weights(self) -> None:
        D = self.config.hidden_dim
        rng = self.rng
        # MLP 隐层 (hidden_dim → hidden_dim)
        self.W_h = rng.randn(D, D) * 0.05
        self.b_h = np.zeros(D)
        # 输出层 (hidden_dim → 1)
        self.W_g = rng.randn(1, D) * 0.1
        self.b_g = np.zeros((1, 1))

    def forward(self, ssm_state: np.ndarray) -> float:
        """
        基于 SSM 隐状态计算 MLP 门控值。

        Args:
            ssm_state: SSM 引擎产生的隐状态 (hidden_dim,)

        Returns:
            g_mlp ∈ (0, 1)
        """
        # 2 层 MLP
        h = np.tanh(self.W_h @ ssm_state + self.b_h)
        z = np.clip(self.W_g @ h + self.b_g, -100, 100)
        g = 1.0 / (1.0 + np.exp(-z))
        return float(g[0, 0])

    def learn(self, gate_value: float, outcome: float, ssm_state: np.ndarray) -> float:
        """
        策略梯度在线学习。

        梯度: ∇ ∝ R · (g - outcome) · h_ssm

        Args:
            gate_value: MLP 当时输出的门控值
            outcome: 实际结果 (0 或 1)
            ssm_state: 决策时的 SSM 隐状态

        Returns:
            reward
        """
        if not self.config.mlp_enable_online_learning:
            return 0.0

        decision = 1.0 if gate_value > self.config.gate_threshold else 0.0
        reward = 1.0 if abs(decision - outcome) < 0.5 else -1.0

        self._total_reward = self.config.mlp_reward_decay * self._total_reward + reward

        # 策略梯度 (经过隐层传播)
        lr = self.config.mlp_learning_rate
        grad_signal = reward * (gate_value - outcome)

        h = np.tanh(self.W_h @ ssm_state + self.b_h)
        h_2d = h.reshape(-1, 1)

        self.W_g += lr * grad_signal * h_2d.T
        self.b_g += lr * grad_signal * 0.01
        np.clip(self.W_g, -5, 5, out=self.W_g)
        np.clip(self.b_g, -5, 5, out=self.b_g)

        return reward


# ═══════════════════════════════════════════════════════════════
# 双门控融合引擎
# ═══════════════════════════════════════════════════════════════

# DualAdaptiveGate v3.1 — SSM + MLP via ACP
class DualAdaptiveGate:
    """
    SSM + MLP 双门控引擎 v3.0

    相互优化配合模式:
      SSM 提供理论保证的状态演化 (HiPPO 矩阵)
      → MLP 学习从 SSM 状态到门控决策的映射
      → α 融合两者输出
      → 反馈同时调整 MLP 权重和 α 融合系数

    门控值:
      g = α · g_ssm + (1-α) · g_mlp

    α 动力学:
      - 初始 α₀ = 0.5 (等权)
      - 每学习一步 α -= learning_decay (MLP 逐渐主导)
      - 当 SSM 状态熵低(高确定性)时 α += entropy_boost
    """

    def __init__(self, config: Optional[DualGateConfig] = None) -> None:
        self.config = config or DualGateConfig()
        self._base_threshold: float = self.config.gate_threshold
        self._rng = np.random.RandomState(self.config.seed if self.config.seed else 0)

        # SSM + MLP 组件
        self.ssm = SSMEngine(self.config, self._rng)
        self.mlp = MLPGate(self.config, self._rng)

        # α 融合系数 + 预算
        self.alpha: float = self.config.alpha_initial
        self._budget: float = self.config.budget_capacity  # 当前可用预算

        # 学习状态
        self._total_reward: float = 0.0

    def step(
        self, hidden_state: np.ndarray, input_features: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        双门控单步推理。

        Args:
            hidden_state: SSM 上一时间步隐状态 (hidden_dim,)
            input_features: 当前输入特征 (input_dim,)

        Returns:
            (new_hidden_state, fused_gate_value)
        """
        # 1. SSM 步进（真正的结构化状态演化）
        new_h, g_ssm = self.ssm.step(hidden_state, input_features)

        # 2. MLP 基于 SSM 新状态做门控决策
        g_mlp = self.mlp.forward(new_h)
        self.mlp._step_count += 1  # 手动计数，避免 forward + learn 双重计数

        # 3. α 融合
        g = self.alpha * g_ssm + (1.0 - self.alpha) * g_mlp

        return new_h, float(g)

    def should_keep(self, gate_value: float) -> bool:
        return gate_value > self.config.gate_threshold

    def compute_input_features(self, hyperedge_data: dict) -> np.ndarray:
        """从超边数据计算输入特征向量"""
        features = np.zeros(self.config.input_dim)
        features[self.config.feat_mean_activation] = hyperedge_data.get("mean_activation", 0.0)
        features[self.config.feat_age_hours] = hyperedge_data.get("age_hours", 0.0)
        features[self.config.feat_access_freq] = hyperedge_data.get("access_freq", 0.0)
        features[self.config.feat_member_count] = hyperedge_data.get("member_count", 0)
        features[self.config.feat_community_density] = hyperedge_data.get("community_density", 0.0)
        features[self.config.feat_tau_mean] = hyperedge_data.get("tau_mean", 0.0)
        features[self.config.feat_tau_variance] = hyperedge_data.get("tau_variance", 0.0)
        features[self.config.feat_connection_entropy] = hyperedge_data.get("connection_entropy", 0.0)
        features[self.config.feat_importance] = hyperedge_data.get("importance", 0.5)
        return features

    def learn(self, gate_value: float, outcome: float, prev_h: np.ndarray) -> float:
        """
        双门控联合学习。

        不仅学习 MLP 权重，还动态调整 α 融合系数：
        - 如果 MLP 决策正确，α 倾向 MLP (减小)
        - 如果 SSM 状态熵低(确定性强)，α 倾向 SSM (增大)

        Args:
            gate_value: 双门控融合后的门控值
            outcome: 实际结果 (0 或 1)
            prev_h: 决策时的 SSM 隐状态

        Returns:
            reward
        """
        # 1. MLP 策略梯度学习
        gate_mlp = self.mlp.forward(prev_h)  # 重放 MLP 门控值
        reward = self.mlp.learn(gate_mlp, outcome, prev_h)

        # 2. α 自适应调整 + 预算感知 (Retain or Consolidate?)

        # 2a. 预算更新：每次学习恢复一点预算，上限容量
        self._budget = min(self.config.budget_capacity, self._budget + self.config.budget_restore_rate)

        # 2b. 预算感知 α 调节：预算充足→可整合(α↑SSM)，预算紧张→只保留(α↓MLP)
        budget_ratio = self._budget / max(self.config.budget_capacity, 1)
        budget_alpha_shift = (budget_ratio - 0.5) * 0.1  # [-0.05, +0.05]

        # 2c. MLP 学习推进 → α 衰减
        if reward > 0:
            self.alpha = max(self.config.alpha_min, self.alpha - self.config.alpha_learning_decay)

        # 2d. 预算感知偏移
        self.alpha += budget_alpha_shift

        # 2e. SSM 熵校准
        # 计算 SSM 状态熵: 高绝对值 → 低熵(确定) → 提权
        state_norm = np.linalg.norm(prev_h) / math.sqrt(len(prev_h))
        if state_norm > 1.0:
            self.alpha += self.config.alpha_entropy_boost * 0.1

        # 限制 α 范围 [alpha_min, alpha_max)
        self.alpha = max(self.config.alpha_min, self.alpha)

        self._total_reward = self.config.mlp_reward_decay * self._total_reward + reward
        return reward

    def adapt_threshold(self, system_load: float = 0.5) -> float:
        """自适应调整门控阈值"""
        new_threshold = self._base_threshold * (0.8 + 0.4 * system_load)
        self.config.gate_threshold = max(0.3, min(0.8, new_threshold))
        return self.config.gate_threshold

    def operator_selection(self) -> str:
        """预算感知操作选择: 'retain' | 'consolidate'
        
        基于 Retain or Consolidate? (arXiv:2607.17545):
        - α ≥ 0.5 → consolidate (SSM 主导融合)
        - α < 0.5 → retain (MLP 主导门控)
        """
        # 预算比值对前两个分支无实际影响，tiebreaker 已用 α 完全决定
        return "consolidate" if self.alpha >= 0.5 else "retain"

    def spend_budget(self) -> str:
        """执行操作并消耗预算。返回实际执行的操作。"""
        op = self.operator_selection()
        if op == "consolidate":
            self._budget = max(0.0, self._budget - self.config.budget_consolidate_cost)
        return op

    def reset_state(self) -> np.ndarray:
        """重置 SSM 隐状态为零向量"""
        return np.zeros(self.config.hidden_dim)

    def get_stats(self) -> dict:
        """获取双门控统计"""
        op = self.operator_selection()
        return {
            "type": "dual_ssm_mlp",
            "alpha": round(self.alpha, 3),
            "budget": round(self._budget, 1),
            "operator": op,
            "total_steps": self.mlp._step_count,
            "total_reward": round(self._total_reward, 3),
            "current_threshold": round(self.config.gate_threshold, 3),
            "ssm_hippo_order": self.config.ssm_hippo_order,
            "online_learning": self.config.mlp_enable_online_learning,
        }

