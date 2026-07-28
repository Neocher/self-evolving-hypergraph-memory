"""
τ 衰减引擎单元测试
=================
测试 tau_decay.py 的核心数学模型。
"""
from __future__ import annotations

import time
import math

import pytest

from core.tau_decay import TauDecayEngine, TauDecayConfig


class TestTauDecayConfig:
    def test_default_values(self):
        cfg = TauDecayConfig()
        assert cfg.tau_initial == 1.0
        assert cfg.tau_decay_seconds == 1800.0
        assert cfg.decay_threshold == 0.1
        assert cfg.refresh_on_access is True

    def test_custom_values(self):
        cfg = TauDecayConfig(
            tau_initial=0.8,
            tau_decay_seconds=3600,
            decay_threshold=0.05,
            refresh_on_access=False,
        )
        assert cfg.tau_initial == 0.8
        assert cfg.tau_decay_seconds == 3600
        assert cfg.decay_threshold == 0.05


class TestTauDecayEngine:
    def test_compute_tau_fresh(self):
        engine = TauDecayEngine()
        now = time.time()
        engine.register_node("test", created_at=now)
        tau = engine.compute_tau("test")
        assert tau == pytest.approx(1.0, rel=0.01)

    def test_compute_tau_decay(self):
        engine = TauDecayEngine(TauDecayConfig(enable_adaptive=False))
        past = time.time() - 900
        engine.register_node("test", created_at=past)
        tau = engine.compute_tau("test", created_at=past)
        expected = math.exp(-900 / 1800)
        assert tau == pytest.approx(expected, rel=0.1)
        assert tau < 1.0
        assert tau > 0.0

    def test_compute_tau_below_threshold(self):
        engine = TauDecayEngine()
        long_ago = time.time() - 7200
        engine.register_node("test", created_at=long_ago)
        tau = engine.compute_tau("test", created_at=long_ago)
        assert tau < engine.config.decay_threshold

    def test_refresh_tau(self):
        """refresh_tau() 应将 τ 值恢复到初始值。"""
        engine = TauDecayEngine()
        created = time.time() - 3600
        engine.register_node("dummy", created_at=created)
        old_tau = engine.compute_tau("dummy")
        refreshed = engine.refresh_tau("dummy")
        assert refreshed == engine.config.tau_initial
        assert refreshed > old_tau

    def test_custom_decay_rate(self):
        fast = TauDecayEngine(TauDecayConfig(tau_decay_seconds=600, enable_adaptive=False))
        slow = TauDecayEngine(TauDecayConfig(tau_decay_seconds=3600, enable_adaptive=False))
        past = time.time() - 1800
        fast.register_node("t", created_at=past)
        slow.register_node("t", created_at=past)
        tau_fast = fast.compute_tau("t", created_at=past)
        tau_slow = slow.compute_tau("t", created_at=past)
        assert tau_fast < tau_slow

    def test_zero_age(self):
        engine = TauDecayEngine()
        engine.register_node("test", created_at=time.time())
        tau = engine.compute_tau("test")
        assert tau == pytest.approx(1.0, rel=0.01)

    def test_negative_age(self):
        """未来时间戳应被截断为初始 τ 值。"""
        engine = TauDecayEngine()
        future = time.time() + 3600
        engine.register_node("test", created_at=future)
        tau = engine.compute_tau("test", created_at=future)
        assert tau == pytest.approx(1.0, abs=0.01)

    def test_threshold_boundary(self):
        """τ 值与 decay_threshold 比较。"""
        engine = TauDecayEngine()
        engine.register_node("test", created_at=time.time())
        assert engine.compute_tau("test") > engine.config.decay_threshold
        very_old = time.time() - 86400 * 30
        engine.register_node("old", created_at=very_old)
        assert engine.compute_tau("old", created_at=very_old) < engine.config.decay_threshold

    def test_long_term_stability(self):
        engine = TauDecayEngine()
        very_old = time.time() - 86400 * 30
        engine.register_node("old", created_at=very_old)
        tau = engine.compute_tau("old", created_at=very_old)
        assert tau >= 0.0

    def test_consistency(self):
        engine = TauDecayEngine()
        now = time.time()
        engine.register_node("test", created_at=now)
        tau1 = engine.compute_tau("test")
        tau2 = engine.compute_tau("test")
        assert tau1 == pytest.approx(tau2)
