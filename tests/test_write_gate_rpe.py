"""D-MEM RPE 写入门控测试（v6.4.0）"""
import pytest

from config.settings import WriteGateConfig, Settings


class TestWriteGateConfig:
    def test_defaults_disabled(self):
        c = WriteGateConfig()
        assert c.enabled is False, "默认关零回归"
        assert c.surprise_deep == 0.45
        assert c.surprise_cache == 0.25

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            WriteGateConfig(surprise_deep=0.2, surprise_cache=0.5)  # deep < cache 非法
        with pytest.raises(ValueError):
            WriteGateConfig(cache_tau=1.5)  # > 1 非法
        with pytest.raises(ValueError):
            WriteGateConfig(cache_tau=0.0)  # <= 0 非法

    def test_settings_exposes_write_gate(self):
        s = Settings()
        assert s.write_gate.enabled is False


class TestRpeRouteLogic:
    """路由决策纯函数验证（对齐 write.py 内联逻辑）"""

    @staticmethod
    def _route(surprise, utility, c=None):
        c = c or WriteGateConfig()
        if surprise >= c.surprise_deep and utility >= c.utility_min:
            return "deep"
        if surprise >= c.surprise_cache and utility >= c.utility_min * 0.7:
            return "cache"
        return "ignore"

    def test_high_surprise_high_utility_deep(self):
        assert self._route(0.8, 0.9) == "deep"

    def test_mid_surprise_cache(self):
        assert self._route(0.3, 0.8) == "cache"

    def test_low_surprise_ignore(self):
        assert self._route(0.1, 0.9) == "ignore"

    def test_high_surprise_low_utility_ignore(self):
        # 高惊奇但低效用 → 忽略（D-MEM：长期效用是必要门控）
        assert self._route(0.9, 0.1) == "ignore"

    def test_deep_threshold_boundary(self):
        assert self._route(0.45, 0.5) == "deep"
        assert self._route(0.44, 0.5) == "cache"
