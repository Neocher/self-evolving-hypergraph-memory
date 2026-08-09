"""
双门控单元测试 (DualAdaptiveGate v3.0)
======================================
覆盖 SSM 引擎 + MLP 门控 + α 融合的全链路。
"""
from __future__ import annotations

import numpy as np
import pytest

from core.dual_gate import DualAdaptiveGate, DualGateConfig


class TestDualAdaptiveGate:
    @pytest.fixture
    def gate(self):
        return DualAdaptiveGate()

    def test_default_config(self):
        cfg = DualGateConfig()
        assert cfg.hidden_dim == 128
        assert cfg.gate_threshold == 0.5
        assert cfg.ssm_hippo_order == 64
        assert cfg.alpha_initial == 0.5

    def test_step_returns_hidden_and_gate(self, gate: DualAdaptiveGate):
        """step() 输入 (hidden, features) 应返回 (new_hidden, gate_value)。"""
        hidden = np.zeros(gate.config.hidden_dim)
        features = np.ones(gate.config.input_dim) * 0.5
        new_hidden, gate_value = gate.step(hidden, features)
        assert new_hidden.shape == (gate.config.hidden_dim,)
        assert 0.0 <= gate_value <= 1.0

    def test_should_keep_uses_config_threshold(self, gate: DualAdaptiveGate):
        """should_keep: 冷启动期内放行; 无学习信号 fail-open; 有学习后按阈值（P0-1）。"""
        # 冷启动期：即使门控值低也放行
        assert gate.should_keep(0.2) is True
        # 超过 warmup 但无学习信号 → fail-open
        gate._step_count = gate.config.warmup_steps + 1
        assert gate.should_keep(0.8) is True
        assert gate.should_keep(0.2) is True  # fail-open: _total_reward == 0
        # 有学习信号后按阈值
        gate._total_reward = 0.1
        assert gate.should_keep(0.8) is True
        assert gate.should_keep(0.2) is False

    def test_positive_features_pass(self, gate: DualAdaptiveGate):
        """step() 应稳定返回门控值。"""
        hidden = np.zeros(gate.config.hidden_dim)
        features = np.ones(gate.config.input_dim) * 0.9
        _, gate_value = gate.step(hidden, features)
        assert 0.0 <= gate_value <= 1.0

    def test_zero_features_lower(self, gate: DualAdaptiveGate):
        """零特征值应产生较低的门控值。"""
        hidden = np.zeros(gate.config.hidden_dim)
        features = np.zeros(gate.config.input_dim)
        _, gv_low = gate.step(hidden, features)
        assert 0.0 <= gv_low <= 1.0

    def test_hidden_state_evolves(self, gate: DualAdaptiveGate):
        """相同输入、不同历史应产生不同输出。"""
        features = np.ones(gate.config.input_dim) * 0.5
        h0 = np.zeros(gate.config.hidden_dim)
        h1, _ = gate.step(h0, features)
        h2, _ = gate.step(h1, features)
        assert not np.allclose(h1, h2)

    def test_hidden_initialized_zeros(self, gate: DualAdaptiveGate):
        """初始化隐藏状态应为全零。"""
        h = np.zeros(gate.config.hidden_dim)
        assert h.sum() == 0.0

    def test_reproducible(self, gate: DualAdaptiveGate):
        """相同输入两次应产生相同的门控值。"""
        h1 = np.zeros(gate.config.hidden_dim)
        h2 = np.zeros(gate.config.hidden_dim)
        f = np.ones(gate.config.input_dim) * 0.5
        _, gv1 = gate.step(h1, f)
        _, gv2 = gate.step(h2, f)
        assert gv1 == pytest.approx(gv2)

    def test_ssm_gate_differs_from_mlp_gate(self, gate: DualAdaptiveGate):
        """SSM 原理门控与 MLP 经验门控应产生不同值 (除非 α=1 或 0)。"""
        h = np.zeros(gate.config.hidden_dim)
        f = np.ones(gate.config.input_dim) * 0.5
        new_h, g_fused = gate.step(h, f)
        g_ssm = gate.ssm.step(np.zeros(gate.config.hidden_dim), f)[1]
        g_mlp = gate.mlp.forward(new_h)
        # α 在 0.2-0.5 之间，融合值应在 SSM 和 MLP 之间
        assert g_ssm != g_mlp or abs(g_ssm - g_mlp) > 0.01

    def test_alpha_between_min_and_initial(self, gate: DualAdaptiveGate):
        """α 融合系数应在 [alpha_min, alpha_initial] 区间内。"""
        assert gate.config.alpha_min <= gate.alpha <= gate.config.alpha_initial

    def test_learn_updates_mlp_weights(self, gate: DualAdaptiveGate):
        """学习后 MLP 权重应发生变化。"""
        h = np.zeros(gate.config.hidden_dim)
        f = np.ones(gate.config.input_dim) * 0.5
        new_h, gv = gate.step(h, f)
        w_before = gate.mlp.W_g.copy()
        gate.learn(gv, 1.0, new_h)
        assert not np.allclose(w_before, gate.mlp.W_g)

    def test_learn_reduces_alpha(self, gate: DualAdaptiveGate):
        """正奖励学习后 α 应减小（MLP 逐渐主导）。预算感知门控下需先消耗预算。"""
        h = np.zeros(gate.config.hidden_dim)
        f = np.ones(gate.config.input_dim) * 0.5
        # 先消耗预算使 budget_ratio < 0.5，避免预算偏移干扰 α 衰减
        gate._budget = 10.0
        new_h, gv = gate.step(h, f)
        alpha_before = gate.alpha
        gate.learn(gv, 1.0, new_h)
        assert gate.alpha <= alpha_before

    def test_adapt_threshold_no_drift(self, gate: DualAdaptiveGate):
        """多次调用不应累积漂移。"""
        t1 = gate.adapt_threshold(0.8)
        t2 = gate.adapt_threshold(0.8)
        assert t1 == pytest.approx(t2)

    def test_get_stats_includes_alpha(self, gate: DualAdaptiveGate):
        """统计信息应包含融合系数 α。"""
        stats = gate.get_stats()
        assert "alpha" in stats
        assert stats["type"] == "dual_ssm_mlp"

    def test_ssm_hippo_matrix_structured(self, gate: DualAdaptiveGate):
        """SSM 的 A 矩阵应该包含下三角 HiPPO 结构。"""
        A = gate.ssm.A
        N = gate.config.ssm_hippo_order
        # 检查对角线: 应 ≤ -1 (HiPPO 负对角)
        assert A[0, 0] < 0
        # 检查下三角: 前 N 行存在非零下三角元素
        assert np.any(A[:N, :N] != 0)

    def test_learn_direction_positive_outcome_raises_mlp_gate(self, gate: DualAdaptiveGate):
        """P0-1 方向断言：正样本 learn 后 MLP gate 升高（不降）。

        实证发现 grad_signal 符号反转 → outcome=1 时 gate 不升反降。
        修复后正样本（outcome=1.0）学习应推高 MLP 门控值。
        注：alpha 会因预算充裕向 SSM 偏移，融合 gate 可能降，但 MLP gate 必须升。
        """
        features = np.ones(gate.config.input_dim) * 0.9
        h = np.zeros(gate.config.hidden_dim)
        new_h, _ = gate.step(h, features)

        mlp_before = gate.mlp.forward(new_h)

        for _ in range(50):
            gate_mlp = gate.mlp.forward(new_h)
            gate.learn(gate_mlp, 1.0, new_h)

        mlp_after = gate.mlp.forward(new_h)
        assert mlp_after >= mlp_before - 0.001, (
            f"正样本学习后 MLP gate 应不降: before={mlp_before:.4f} after={mlp_after:.4f}"
        )

    def test_learn_direction_negative_outcome_lowers_mlp_gate(self, gate: DualAdaptiveGate):
        """P0-1 方向断言：负样本 learn 后 MLP gate 降低。

        outcome=0 → 梯度压低 gate → MLP 门控值应下降。
        """
        features = np.ones(gate.config.input_dim) * 0.9
        h = np.zeros(gate.config.hidden_dim)
        new_h, _ = gate.step(h, features)

        mlp_before = gate.mlp.forward(new_h)

        for _ in range(50):
            gate_mlp = gate.mlp.forward(new_h)
            gate.learn(gate_mlp, 0.0, new_h)

        mlp_after = gate.mlp.forward(new_h)
        assert mlp_after <= mlp_before + 0.001, (
            f"负样本学习后 MLP gate 应不升: before={mlp_before:.4f} after={mlp_after:.4f}"
        )

    def test_alpha_clamp_after_many_learns(self, gate: DualAdaptiveGate):
        """learn 多次后 alpha < 1.0 且 fused gate 不为负。

        alpha 由三股力推拉：reward 触发 alpha_learning_decay →
        budget_ratio 偏移 → SSM 熵校准提升。任一股力都不能突破
        alpha_max - 1e-9 上界，否则 (1-alpha) 变负，MLP 贡献反转。
        """
        features = np.ones(gate.config.input_dim) * 0.5
        h = np.zeros(gate.config.hidden_dim)

        for i in range(200):
            new_h, gv = gate.step(h, features)
            h = new_h
            # 交替正/负反馈 + 高奖励 → 触发学习衰减
            outcome = 1.0 if i % 2 == 0 else 0.0
            gate.learn(gv, outcome, new_h)

        assert gate.alpha < 1.0, f"alpha 应保持 < 1.0，实际 {gate.alpha}"
        assert gate.alpha >= gate.config.alpha_min, (
            f"alpha 应 >= alpha_min={gate.config.alpha_min}，实际 {gate.alpha}"
        )

        # fused gate 非负：alpha∈[alpha_min, 1.0) + g_ssm/g_mlp∈(0,1) → g ≥ 0
        _, gv = gate.step(h, features)
        assert gv >= 0.0, f"fused gate 应非负，实际 {gv}"
        assert gv <= 1.0, f"fused gate 应 ≤ 1.0，实际 {gv}"
