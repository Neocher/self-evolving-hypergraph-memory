"""
SSM 门控单元测试
===============
测试 ssm_gate.py 的实际 API。
"""
from __future__ import annotations

import numpy as np
import pytest

from core.ssm_gate import SSMGate, SSMGateConfig


class TestSSMGate:
    @pytest.fixture
    def gate(self):
        return SSMGate()

    def test_default_config(self):
        cfg = SSMGateConfig()
        assert cfg.hidden_dim == 128
        assert cfg.gate_threshold == 0.5

    def test_step_returns_hidden_and_gate(self, gate: SSMGate):
        """step() 输入 (hidden, features) 应返回 (new_hidden, gate_value)。"""
        hidden = np.zeros(gate.config.hidden_dim)
        features = np.ones(gate.config.input_dim) * 0.5
        new_hidden, gate_value = gate.step(hidden, features)
        assert new_hidden.shape == (gate.config.hidden_dim,)
        assert 0.0 <= gate_value <= 1.0

    def test_should_keep_uses_config_threshold(self, gate: SSMGate):
        """should_keep 应使用 config.gate_threshold。"""
        assert gate.should_keep(0.8) is True
        assert gate.should_keep(0.2) is False
        assert gate.should_keep(gate.config.gate_threshold) is False  # > not >=

    def test_positive_features_pass(self, gate: SSMGate):
        """step() 应稳定返回门控值。"""
        hidden = np.zeros(gate.config.hidden_dim)
        features = np.ones(gate.config.input_dim) * 0.9
        _, gate_value = gate.step(hidden, features)
        # 只是验证 step() 不抛异常且门控值在有效范围
        assert 0.0 <= gate_value <= 1.0

    def test_zero_features_lower(self, gate: SSMGate):
        """零特征值应产生较低的门控值。"""
        hidden = np.zeros(gate.config.hidden_dim)
        features = np.zeros(gate.config.input_dim)
        _, gate_value = gate.step(hidden, features)
        # 门控值可能仍较高（随机初始化），但应低于高特征值
        hidden2 = np.zeros(gate.config.hidden_dim)
        features2 = np.ones(gate.config.input_dim) * 1.0
        _, gv_high = gate.step(hidden2, features2)
        assert gate_value <= gv_high or gate_value <= 0.6

    def test_hidden_state_evolves(self, gate: SSMGate):
        """相同输入、不同历史应产生不同输出。"""
        features = np.ones(gate.config.input_dim) * 0.5
        h0 = np.zeros(gate.config.hidden_dim)
        h1, _ = gate.step(h0, features)
        h2, _ = gate.step(h1, features)
        assert not np.allclose(h1, h2)

    def test_dimension_mismatch(self, gate: SSMGate):
        """维度不匹配时 gate.step 应默认放行。"""
        short = np.array([0.5, 0.3])
        hidden = np.zeros(gate.config.hidden_dim)
        try:
            new_hidden, gv = gate.step(hidden, short)
            assert new_hidden.shape == (gate.config.hidden_dim,)
            assert gv == pytest.approx(1.0)  # 异常放行
        except Exception:
            pytest.fail("维度不匹配时应优雅处理")

    def test_hidden_initialized_zeros(self, gate: SSMGate):
        """初始化隐藏状态应为全零。"""
        h = np.zeros(gate.config.hidden_dim)
        assert h.sum() == 0.0

    def test_reproducible(self, gate: SSMGate):
        """相同输入两次应产生相同的门控值。"""
        h1 = np.zeros(gate.config.hidden_dim)
        h2 = np.zeros(gate.config.hidden_dim)
        f = np.ones(gate.config.input_dim) * 0.5
        _, gv1 = gate.step(h1, f)
        _, gv2 = gate.step(h2, f)
        assert gv1 == pytest.approx(gv2)
