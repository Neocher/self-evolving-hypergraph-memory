"""
稀疏 Hebbian 更新
=================
"Neurons that fire together, wire together."

受 DNC/SDNC 启发，不维护全连接矩阵（O(N²)），
而是维护每个节点的 K=8 个最强输出连接。

复杂度: O(N log N + K)，其中 K=8（经验值，可配置）。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class HebbianConfig:
    """Hebbian 学习配置"""

    k_sparsity: int = 8  # 每个节点保留的最强连接数 (DNC 经验值)
    learning_rate: float = 0.1  # 学习率 η
    decay_constant: float = 0.01  # 权重衰减常数，防止无限增长
    activation_threshold: float = 0.3  # 激活度阈值，低于此的不更新
    max_connections_per_node: int = 64  # 绝对上限，防止异常增长


class SparseHebbianUpdater:
    """
    稀疏 Hebbian 连接更新器。

    只维护和更新 K 个最强连接 + 当前共现激活的边。
    """

    def __init__(self, config: Optional[HebbianConfig] = None) -> None:
        self.config = config or HebbianConfig()

    def update(
        self,
        active_nodes: dict[str, float],
        all_connections: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """
        执行一次稀疏 Hebbian 更新。

        Args:
            active_nodes: {node_id: activation_value, ...}
                当前共现激活的节点及其激活度 (0~1)
            all_connections: {node_id: {neighbor_id: weight, ...}, ...}
                全部节点的连接矩阵（稀疏存储）

        Returns:
            更新后的连接矩阵（原地修改 + 返回引用）
        """
        K = self.config.k_sparsity
        eta = self.config.learning_rate
        decay = self.config.decay_constant

        active_ids = list(active_nodes.keys())

        # Step 1: 预计算每个激活节点的 top-K 连接
        top_k_map: dict[str, set[str]] = {}
        for nid in active_ids:
            conns = all_connections.get(nid, {})
            top_k = heapq.nlargest(K, conns.items(), key=lambda x: x[1])
            top_k_map[nid] = {k for k, _ in top_k}

        # Step 2: 更新激活节点之间的连接
        for i in range(len(active_ids)):
            ni = active_ids[i]
            ai = active_nodes[ni]
            if ai < self.config.activation_threshold:
                continue
            for j in range(i + 1, len(active_ids)):
                nj = active_ids[j]
                aj = active_nodes[nj]
                if aj < self.config.activation_threshold:
                    continue
                delta = eta * (ai * aj - all_connections.get(ni, {}).get(nj, 0) * decay)
                all_connections.setdefault(ni, {})[nj] = (
                    all_connections.get(ni, {}).get(nj, 0) + delta
                )
                all_connections.setdefault(nj, {})[ni] = (
                    all_connections.get(nj, {}).get(ni, 0) + delta
                )

        # Step 3: 对所有已修改的节点剪枝到 K 个
        for nid in active_ids:
            conns = all_connections.get(nid, {})
            if len(conns) > K:
                pruned = heapq.nlargest(K, conns.items(), key=lambda x: x[1])
                all_connections[nid] = dict(pruned)

        return all_connections

    def compute_connection_strength(
        self,
        node_i_activation: float,
        node_j_activation: float,
        current_weight: float,
    ) -> float:
        """
        计算单条连接的 Hebbian 更新量。
        Δw_ij = η · (a_i · a_j - w_ij · τ_decay)
        """
        eta = self.config.learning_rate
        decay = self.config.decay_constant
        return eta * (node_i_activation * node_j_activation - current_weight * decay)

    def prune_connections(self, connections: dict[str, float]) -> dict[str, float]:
        """将连接数剪枝到 K 个最强。"""
        K = self.config.k_sparsity
        if len(connections) <= K:
            return connections
        pruned = heapq.nlargest(K, connections.items(), key=lambda x: x[1])
        return dict(pruned)
