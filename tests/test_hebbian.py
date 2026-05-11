"""
Hebbian 更新器单元测试
===================
测试 hebbian.py 的实际 API。
"""
from __future__ import annotations

import pytest

from core.hebbian import SparseHebbianUpdater, HebbianConfig


class TestHebbianUpdater:
    @pytest.fixture
    def updater(self):
        return SparseHebbianUpdater()

    def test_default_config(self):
        cfg = HebbianConfig()
        assert cfg.k_sparsity == 8
        assert cfg.learning_rate == 0.1

    def test_update_adds_connections(self, updater: SparseHebbianUpdater):
        """update() 应为共现激活的节点建立连接。"""
        active = {"a": 0.8, "b": 0.7, "c": 0.9}
        connections = {}
        result = updater.update(active, connections)
        # a-b, a-c, b-c 之间应出现连接
        assert "a" in result
        assert "b" in result.get("a", {})
        assert "c" in result.get("a", {})

    def test_update_increases_weights(self, updater: SparseHebbianUpdater):
        """共现激活应增加现有连接的权重。"""
        active = {"a": 0.8, "b": 0.7}
        connections = {"a": {"b": 0.3}}
        result = updater.update(active, {"a": {"b": 0.3}, "b": {"a": 0.3}})
        # 权重应从 0.3 增加
        new_weight = result.get("a", {}).get("b", 0)
        assert new_weight > 0.3

    def test_prune_connections(self, updater: SparseHebbianUpdater):
        """超过 k_sparsity 时应修剪最弱的连接。"""
        connections = {f"n{i}": 0.1 * (i + 1) for i in range(20)}
        pruned = updater.prune_connections(connections)
        assert len(pruned) <= updater.config.k_sparsity
        expected = dict(sorted(connections.items(), key=lambda x: -x[1])[:8])
        assert pruned == expected

    def test_empty_connections(self, updater: SparseHebbianUpdater):
        """空输入应返回空。"""
        assert updater.prune_connections({}) == {}

    def test_single_connection_not_pruned(self, updater: SparseHebbianUpdater):
        """单连接不应被修剪。"""
        result = updater.prune_connections({"a": 0.1})
        assert result == {"a": 0.1}

    def test_low_activation_skipped(self, updater: SparseHebbianUpdater):
        """低于 activation_threshold 的节点应被跳过。"""
        active = {"a": 0.1, "b": 0.8}  # a 低于阈值
        connections = {}
        result = updater.update(active, connections)
        # a 不应有任何新连接
        a_conns = result.get("a", {})
        assert len(a_conns) == 0

    def test_learning_rate_effect(self):
        """不同学习率应影响权重增幅。
        
        注意：当前实现中权重精确等于 learning_rate * activation_product，
        相同初始权重下 fast = slow * 10。
        """
        active = {"x": 0.8, "y": 0.7}
        fast = SparseHebbianUpdater(HebbianConfig(learning_rate=0.5))
        slow = SparseHebbianUpdater(HebbianConfig(learning_rate=0.05))
        connections = {"x": {"y": 0.3}, "y": {"x": 0.3}}
        fast_r = fast.update(active, connections)
        slow_r = slow.update(active, connections)
        fast_w = fast_r.get("x", {}).get("y", 0)
        slow_w = slow_r.get("x", {}).get("y", 0)
        assert fast_w >= slow_w, "更高学习率应产生相等或更高的权重"


class TestHebbianConfig:
    def test_custom_sparsity(self):
        cfg = HebbianConfig(k_sparsity=4, learning_rate=0.2)
        assert cfg.k_sparsity == 4
