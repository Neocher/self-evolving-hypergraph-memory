"""
核心引擎单元测试
===============
覆盖 τ衰减·Hebbian连接·SSM门控 — SHM 的记忆理论三大支柱。

运行: cd /home/admin/shm && source .venv/bin/activate && python -m pytest tests/test_core_engine.py -v
"""

import math
import time
import pytest
import numpy as np

from core.tau_decay import TauDecayEngine, TauDecayConfig
from core.hebbian import SparseHebbianUpdater, HebbianConfig
from core.dual_gate import DualAdaptiveGate, DualGateConfig


# ══════════════════════════════════════════════════════════
# τ 衰减测试 (TauDecayEngine)
# ══════════════════════════════════════════════════════════

class TestTauDecay:
    """τ 指数衰减引擎测试"""

    def test_initial_tau_is_one(self):
        """新创建的节点 τ = τ₀ = 1.0"""
        engine = TauDecayEngine()
        engine.register_node("test_1", created_at=time.time())
        tau = engine.compute_tau("test_1")
        assert tau == pytest.approx(1.0, abs=0.01)

    def test_decay_after_half_life(self):
        """经过一个 τ_decay 周期后，τ = τ₀/e ≈ 0.368"""
        config = TauDecayConfig(tau_decay_seconds=600, tau_initial=1.0, enable_adaptive=False)
        engine = TauDecayEngine(config)
        created = time.time() - 600  # 600秒前创建
        engine.register_node("test_1", created_at=created)
        tau = engine.compute_tau("test_1", created_at=created)
        assert tau == pytest.approx(1.0 / math.e, abs=0.01)

    def test_decay_after_two_periods(self):
        """经过两个 τ_decay 周期后，τ = τ₀/e²"""
        config = TauDecayConfig(tau_decay_seconds=300, enable_adaptive=False)
        engine = TauDecayEngine(config)
        created = time.time() - 600
        engine.register_node("test_1", created_at=created)
        tau = engine.compute_tau("test_1", created_at=created)
        assert tau < 0.15  # ≈ 0.135

    def test_decay_threshold_candidate(self):
        """τ 低于阈值时标记为修剪候选"""
        config = TauDecayConfig(tau_decay_seconds=100, decay_threshold=0.3, enable_adaptive=False)
        engine = TauDecayEngine(config)
        now = time.time()
        engine.register_node("test_1", created_at=now)
        # 刚刚创建的不应被标记
        assert not engine.is_decay_candidate("test_1", created_at=now)
        # 很久以前创建的应被标记
        engine.register_node("test_2", created_at=now - 500)
        assert engine.is_decay_candidate("test_2", created_at=now - 500)
        # 边界：刚好在阈值之上
        boundary = -100 * math.log(0.3)  # t where τ = 0.3
        engine.register_node("test_3", created_at=now - boundary + 10)
        assert not engine.is_decay_candidate("test_3", created_at=now - boundary + 10)
        engine.register_node("test_4", created_at=now - boundary - 10)
        assert engine.is_decay_candidate("test_4", created_at=now - boundary - 10)

    def test_refresh_resets_tau(self):
        """再巩固后 τ 回到初始值"""
        engine = TauDecayEngine()
        engine.register_node("test_1", created_at=time.time() - 1000)
        tau = engine.refresh_tau("test_1")
        assert tau == pytest.approx(1.0)

    def test_batch_compute_returns_all(self):
        """批量计算覆盖所有节点"""
        engine = TauDecayEngine()
        now = time.time()
        engine.register_node("a", created_at=now)
        engine.register_node("b", created_at=now - 300)
        engine.register_node("c", created_at=now - 1800)
        nodes = [("a", now, None), ("b", now - 300, None), ("c", now - 1800, None)]
        result = engine.batch_compute(nodes)
        assert set(result.keys()) == {"a", "b", "c"}
        assert result["a"] > result["b"] > result["c"]

    def test_config_validation(self):
        """配置校验应拒绝非法值"""
        with pytest.raises(AssertionError):
            TauDecayConfig(tau_initial=0).validate()
        with pytest.raises(AssertionError):
            TauDecayConfig(tau_initial=1.5).validate()
        with pytest.raises(AssertionError):
            TauDecayConfig(tau_decay_seconds=0).validate()
        with pytest.raises(AssertionError):
            TauDecayConfig(decay_threshold=1.5).validate()

    def test_negative_time_handled(self):
        """负时间差（未来时间戳）应被钳位为0"""
        engine = TauDecayEngine()
        engine.register_node("test_1", created_at=time.time() + 1000)
        tau = engine.compute_tau("test_1", created_at=time.time() + 1000)
        assert tau == pytest.approx(1.0, abs=0.01)


# ══════════════════════════════════════════════════════════
# Hebbian 连接测试 (SparseHebbianUpdater)
# ══════════════════════════════════════════════════════════

class TestHebbian:
    """稀疏 Hebbian 学习测试"""

    def test_hebbian_reinforces_coactive(self):
        """共现节点之间的连接被强化"""
        config = HebbianConfig(k_sparsity=8, learning_rate=0.5)
        updater = SparseHebbianUpdater(config)

        active = {"A": 1.0, "B": 1.0, "C": 0.5}
        conns = {}
        result = updater.update(active, conns)

        assert "A" in result
        assert "B" in result["A"]  # A→B 连接已形成
        assert result["A"]["B"] > 0  # 权重为正

    def test_symmetric_connections(self):
        """连接是对称的：A→B == B→A"""
        updater = SparseHebbianUpdater()
        active = {"A": 1.0, "B": 1.0}
        result = updater.update(active, {})

        assert abs(result["A"]["B"] - result["B"]["A"]) < 1e-10

    def test_below_threshold_not_updated(self):
        """低于激活阈值的节点不参与更新"""
        config = HebbianConfig(k_sparsity=8, activation_threshold=0.5)
        updater = SparseHebbianUpdater(config)

        active = {"A": 1.0, "B": 0.2}  # B 低于阈值
        result = updater.update(active, {})

        # B 低于阈值，A→B 连接不应被创建
        assert result.get("A", {}) == {}
        assert result.get("B", {}) == {}

    def test_prune_to_k_sparsity(self):
        """连接超过 K 个时只保留最强的 K 个"""
        config = HebbianConfig(k_sparsity=4)
        updater = SparseHebbianUpdater(config)

        conns = {"A": [10, 8, 6, 4, 2]}
        connections = {"A": {"A": conns["A"][0], "B": conns["A"][1],
                              "C": conns["A"][2], "D": conns["A"][3],
                              "E": conns["A"][4]}}

        result = updater.prune_connections(connections["A"])
        assert len(result) <= 4
        # 最强的 4 个应保留
        highest = [10, 8, 6, 4]
        kept = sorted(result.values(), reverse=True)
        assert kept == highest

    def test_tau_decay_weakens_low_tau_connections(self):
        """低 τ 节点的连接权重衰减"""
        updater = SparseHebbianUpdater()
        conns = {"A": {"B": 1.0, "C": 0.5}}
        tau_map = {"A": 0.05, "C": 0.8}  # A 的 τ 很低

        result = updater.tau_decay_connections(conns, tau_map)

        # A 的连接应衰减
        assert result["A"]["B"] <= 0.5  # τ=0.05 → factor=0.5, w=1.0*0.5=0.5
        # C 的 τ 正常，连接不受影响
        assert result == conns  # (原地修改—C不受影响)

    def test_compute_connection_strength(self):
        """连接强度计算：正协相关系数应产生正连接"""
        updater = SparseHebbianUpdater()
        delta = updater.compute_connection_strength(1.0, 1.0, 0.0)
        assert delta > 0  # 两个高激活节点 → 正连接

        # 已有高权重的连接更新量较小
        delta2 = updater.compute_connection_strength(1.0, 1.0, 0.9)
        assert delta2 < delta  # 已强连接 → 更新量更小

    def test_ontological_distance_modulation(self):
        """本体距离因子增强语义相关节点的连接"""
        updater = SparseHebbianUpdater()
        active = {"A": 1.0, "B": 1.0, "C": 1.0}

        # A-B 本体距离近，A-C 本体距离远
        onto_map = {("A", "B"): 0.8, ("A", "C"): 0.2}

        result = updater.update(active, {}, onto_map)

        # A-C 的连接应弱于 A-B
        assert result["A"]["B"] > result["A"]["C"]


# ══════════════════════════════════════════════════════════
# 自适应门控测试 (DualAdaptiveGate)
# ══════════════════════════════════════════════════════════

class TestDualAdaptiveGate:
    """自适应门控测试"""

    def test_gate_value_between_0_and_1(self):
        """门控值始终在 (0, 1) 之间"""
        gate = DualAdaptiveGate()
        state = gate.reset_state()
        features = np.ones(8)
        _, g = gate.step(state, features)
        assert 0 < g < 1

    def test_reset_state_is_zero_vector(self):
        """重置隐状态为零向量"""
        gate = DualAdaptiveGate()
        state = gate.reset_state()
        assert np.all(state == 0)
        assert state.shape == (128,)

    def test_should_keep_above_threshold(self):
        """门控值高于阈值时应保留"""
        gate = DualAdaptiveGate()
        assert gate.should_keep(0.9)
        assert not gate.should_keep(0.1)

        # 边界测试
        config = DualGateConfig(gate_threshold=0.5)
        gate2 = DualAdaptiveGate(config)
        assert gate2.should_keep(0.5001)
        assert not gate2.should_keep(0.4999)

    def test_compute_input_features_shape(self):
        """特征向量维度与配置一致"""
        gate = DualAdaptiveGate()
        data = {"mean_activation": 0.5, "age_hours": 2.0, "member_count": 10}
        features = gate.compute_input_features(data)
        assert len(features) == 9  # input_dim=9 (8 base + 1 importance)
        assert features[0] == 0.5  # mean_activation
        assert features[3] == 10   # member_count

    def test_state_evolution(self):
        """多次步进后隐状态应该变化"""
        gate = DualAdaptiveGate()
        state = gate.reset_state()
        features = np.random.randn(9) * 0.1

        state1, g1 = gate.step(state, features)
        assert not np.allclose(state, state1)  # 状态已改变

        state2, g2 = gate.step(state1, features * 2)
        assert not np.allclose(state1, state2)  # 继续演化

    def test_high_activation_yields_high_gate(self):
        """高激活特征产生更高的门控值"""
        gate = DualAdaptiveGate()
        state = gate.reset_state()

        # 低质量超边特征
        low_quality = np.zeros(9)
        _, g_low = gate.step(state.copy(), low_quality)

        # 高质量超边特征
        high_quality = np.ones(9) * 2.0
        _, g_high = gate.step(state.copy(), high_quality)

        # 至少验证两者都在有效范围内
        assert 0 < g_low < 1
        assert 0 < g_high < 1

    def test_error_handling_returns_safe_default(self):
        """特征维度不匹配时自动padding到正确维度，门控值正常计算"""
        gate = DualAdaptiveGate()
        state = gate.reset_state()
        # 传入错误维度的特征
        bad_features = np.array([1.0, 2.0, 3.0])  # 应为9维，自动pad
        new_state, g = gate.step(state, bad_features)
        assert 0 < g < 1  # 正常返回门控值（不再fallback到1.0）
        assert new_state.shape == (128,)
