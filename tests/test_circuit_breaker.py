"""
断路器模式单元测试
================
测试 kuzu_store.py 的 CircuitBreaker 逻辑。
"""
from __future__ import annotations

import time

import pytest

from graph.ryu_store import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerConfig as CBC,
    CircuitState,
    CircuitBreakerOpen,
)


class TestCircuitBreaker:
    """断路器核心逻辑。"""

    def test_initial_state_closed(self):
        """新建断路器应为 CLOSED 状态。"""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available() is True

    def test_record_success(self):
        """成功记录不应跳闸。"""
        cb = CircuitBreaker(CBC(window_size=5))
        for _ in range(5):
            cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_record_failure_trips_open(self):
        """连续失败超过阈值应跳闸。"""
        cb = CircuitBreaker(CBC(failure_threshold=0.5, window_size=4))
        # 3 次失败 + 1 次成功 = 75% 错误率 > 50%
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_requests(self):
        """OPEN 状态的断路器应拒绝请求。"""
        cb = CircuitBreaker(CBC(failure_threshold=0.5, window_size=2))
        cb.record_failure()
        cb.record_failure()  # 100% 错误率
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitBreakerOpen):
            cb.is_available()

    def test_open_transitions_to_half_open_after_timeout(self):
        """OPEN 状态在 recovery_timeout 后应转为 HALF_OPEN。"""
        cb = CircuitBreaker(CBC(
            failure_threshold=0.5,
            window_size=2,
            recovery_timeout=0.01,  # 10ms
        ))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.is_available() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        """HALF_OPEN 下成功请求应回到 CLOSED。"""
        cb = CircuitBreaker(CBC(
            failure_threshold=0.5,
            window_size=2,
            recovery_timeout=0.01,
        ))
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        cb.is_available()  # → HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        """HALF_OPEN 下失败请求应回到 OPEN。"""
        cb = CircuitBreaker(CBC(
            failure_threshold=0.5,
            window_size=2,
            recovery_timeout=0.01,
        ))
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        cb.is_available()  # → HALF_OPEN
        cb.record_failure()  # 探测失败
        assert cb.state == CircuitState.OPEN

    def test_error_rate_below_threshold(self):
        """错误率不超过阈值应保持 CLOSED。
        
        [注意] 要先让窗口填满再检查，避免中途跳闸。
        """
        cb = CircuitBreaker(CBC(failure_threshold=0.5, window_size=10))
        # 先用成功填满窗口
        for _ in range(6):
            cb.record_success()
        # 再加入失败，使错误率 4/10 = 40% < 50%
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_error_rate_at_threshold_boundary(self):
        """错误率超过阈值（>50%）应跳闸。"""
        cb = CircuitBreaker(CBC(failure_threshold=0.5, window_size=4))
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()  # 3/4 = 75% > 50%
        cb.record_success()
        assert cb.state == CircuitState.OPEN

    def test_window_sliding_behavior(self):
        """超过 window_size 的旧记录应被遗忘。"""
        cb = CircuitBreaker(CBC(failure_threshold=0.5, window_size=4))
        # 先成功填满窗口，再加入失败滑过
        for _ in range(4):
            cb.record_success()  # 窗口: [T,T,T,T]
        # 失败 4 次 → 窗口滑动：F→F→F→F = 100% > 50%
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_insufficient_samples(self):
        """样本不足 2 个时不应跳闸。"""
        cb = CircuitBreaker(CBC(failure_threshold=0.1, window_size=4))
        cb.record_failure()  # 只有 1 个样本
        assert cb.state == CircuitState.CLOSED
