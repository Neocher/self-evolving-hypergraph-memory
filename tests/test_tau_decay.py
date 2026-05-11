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
        tau = engine.compute_tau(now)
        assert tau == pytest.approx(1.0, rel=0.01)

    def test_compute_tau_decay(self):
        engine = TauDecayEngine()
        past = time.time() - 900
        tau = engine.compute_tau(past)
        expected = math.exp(-900 / 1800)
        assert tau == pytest.approx(expected, rel=0.01)
        assert tau < 1.0
        assert tau > 0.0

    def test_compute_tau_below_threshold(self):
        engine = TauDecayEngine()
        long_ago = time.time() - 7200
        tau = engine.compute_tau(long_ago)
        assert tau < engine.config.decay_threshold

    def test_refresh_tau(self):
        """refresh_tau() 应将 τ 值恢复到初始值。"""
        engine = TauDecayEngine()
        created = time.time() - 3600
        old_tau = engine.compute_tau(created)
        refreshed = engine.refresh_tau("dummy", created)
        assert refreshed == engine.config.tau_initial
        assert refreshed > old_tau

    def test_custom_decay_rate(self):
        fast = TauDecayEngine(TauDecayConfig(tau_decay_seconds=600))
        slow = TauDecayEngine(TauDecayConfig(tau_decay_seconds=3600))
        past = time.time() - 1800
        tau_fast = fast.compute_tau(past)
        tau_slow = slow.compute_tau(past)
        assert tau_fast < tau_slow

    def test_zero_age(self):
        engine = TauDecayEngine()
        tau = engine.compute_tau(time.time())
        assert tau == pytest.approx(1.0, rel=0.01)

    def test_negative_age(self):
        """未来时间戳应被截断为初始 τ 值。"""
        engine = TauDecayEngine()
        future = time.time() + 3600
        tau = engine.compute_tau(future)
        assert tau == pytest.approx(1.0, abs=0.01)

    def test_threshold_boundary(self):
        """τ 值与 decay_threshold 比较。"""
        engine = TauDecayEngine()
        assert engine.compute_tau(time.time()) > engine.config.decay_threshold
        very_old = time.time() - 86400 * 30
        assert engine.compute_tau(very_old) < engine.config.decay_threshold

    def test_long_term_stability(self):
        engine = TauDecayEngine()
        very_old = time.time() - 86400 * 30
        tau = engine.compute_tau(very_old)
        assert tau >= 0.0

    def test_consistency(self):
        engine = TauDecayEngine()
        now = time.time()
        tau1 = engine.compute_tau(now)
        tau2 = engine.compute_tau(now)
        assert tau1 == pytest.approx(tau2)
