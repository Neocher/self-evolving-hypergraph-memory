"""
稀疏 Hebbian 更新 v2.0
=================
"Neurons that fire together, wire together."

受 DNC/SDNC 启发，不维护全连接矩阵（O(N²)），
而是维护每个节点的 K=8 个最强输出连接。

v2.0: 新增 RyuGraph 持久化支持，Hebbian 连接不再仅存内存。
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class HebbianConfig:
    """Hebbian 学习配置"""

    k_sparsity: int = 8  # 每个节点保留的最强连接数 (DNC 经验值)
    learning_rate: float = 0.1  # 学习率 η
    decay_constant: float = 0.01  # 权重衰减常数，防止无限增长
    activation_threshold: float = 0.3  # 激活度阈值，低于此的不更新
    max_connections_per_node: int = 64  # 绝对上限，防止异常增长
    # v2.0: 持久化配置
    persist_to_graph: bool = True  # 是否写回 Kuzu
    persist_every_n_updates: int = 1  # 每次更新都持久化，防止崩溃后短期记忆丢失


class SparseHebbianUpdater:
    """
    稀疏 Hebbian 连接更新器 v2.0。

    只维护和更新 K 个最强连接 + 当前共现激活的边。
    可选写入 KuzuStore 持久化。
    """

    def __init__(self, config: Optional[HebbianConfig] = None,
                 kuzu_store=None) -> None:
        self.config = config or HebbianConfig()
        self._kuzu_store = kuzu_store
        self._update_counter = 0

    def set_kuzu_store(self, store) -> None:
        """运行时注入 GraphLiteStore（用于启动后注入）。"""
        self._kuzu_store = store

    def update(
        self,
        active_nodes: dict[str, float],
        all_connections: dict[str, dict[str, float]],
        ontological_distance_map: Optional[dict[tuple[str, str], float]] = None,
    ) -> dict[str, dict[str, float]]:
        """
        执行一次稀疏 Hebbian 更新。
        
        v2.0: 可选持久化结果到 Kuzu。

        [P1] 加入本体层次距离调制：
            Δw = η · (a_i · a_j · d_ij - w · τ_decay)
            其中 d_ij ∈ [0.3, 1.0] 是本体距离因子

        Args:
            active_nodes: {node_id: activation_value, ...}
            all_connections: {node_id: {neighbor_id: weight, ...}, ...}
            ontological_distance_map: {(node_a, node_b): distance_factor, ...}

        Returns:
            更新后的连接矩阵（原地修改 + 返回引用）
        """
        K = self.config.k_sparsity
        eta = self.config.learning_rate
        decay = self.config.decay_constant

        active_ids = list(active_nodes.keys())

        def _get_ont_dist(ni: str, nj: str) -> float:
            if ontological_distance_map is None:
                return 1.0
            d = ontological_distance_map.get((ni, nj))
            if d is not None:
                return d
            d = ontological_distance_map.get((nj, ni))
            return d if d is not None else 1.0

        top_k_map: dict[str, set[str]] = {}
        for nid in active_ids:
            conns = all_connections.get(nid, {})
            top_k = heapq.nlargest(K, conns.items(), key=lambda x: x[1])
            top_k_map[nid] = {k for k, _ in top_k}

        # 收集所有需要持久化的更新
        updates: list[tuple[str, str, float]] = []

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
                d_ij = _get_ont_dist(ni, nj)
                current_w = all_connections.get(ni, {}).get(nj, 0)
                delta = eta * (ai * aj * d_ij - current_w * decay)
                new_weight = current_w + delta
                all_connections.setdefault(ni, {})[nj] = new_weight
                all_connections.setdefault(nj, {})[ni] = new_weight
                updates.append((ni, nj, new_weight))

        for nid in active_ids:
            conns = all_connections.get(nid, {})
            if len(conns) > K:
                pruned = heapq.nlargest(K, conns.items(), key=lambda x: x[1])
                all_connections[nid] = dict(pruned)

        # v2.0: 批量持久化到 Kuzu
        self._update_counter += 1
        if self._kuzu_store is not None and self.config.persist_to_graph:
            if self._update_counter % self.config.persist_every_n_updates == 0:
                self._persist_batch(updates)

        return all_connections

    def _persist_batch(self, updates: list[tuple[str, str, float]]) -> None:
        """批量持久化 Hebbian 更新到 Kuzu。

        GraphLite 不支持 UNWIND/MERGE：改为循环逐条处理，
        边存在则 SET 权重，不存在则 INSERT（UPDATE 语义）。
        """
        if not self._kuzu_store or not updates:
            return
        try:
            for src, dst, weight in updates:
                if self._kuzu_store.execute_cypher(
                    "MATCH (a {id: $src})-[r:HEBBIAN_CONNECTION]->(b {id: $dst}) RETURN r",
                    {"src": src, "dst": dst},
                ):
                    self._kuzu_store.execute_cypher(
                        "MATCH (a {id: $src})-[r:HEBBIAN_CONNECTION]->(b {id: $dst}) "
                        "SET r.weight = $weight",
                        {"src": src, "dst": dst, "weight": weight},
                    )
                else:
                    self._kuzu_store.execute_cypher(
                        "MATCH (a {id: $src}), (b {id: $dst}) "
                        "INSERT (a)-[:HEBBIAN_CONNECTION {weight: $weight}]->(b)",
                        {"src": src, "dst": dst, "weight": weight},
                    )
        except Exception:
            logger.exception("Hebbian batch persist failed for %d updates", len(updates))

    def compute_connection_strength(
        self,
        node_i_activation: float,
        node_j_activation: float,
        current_weight: float,
    ) -> float:
        """计算单条连接的 Hebbian 更新量。"""
        eta = self.config.learning_rate
        decay = self.config.decay_constant
        return eta * (node_i_activation * node_j_activation - current_weight * decay)

    def prune_connections(self, connections: dict[str, float]) -> dict[str, float]:
        """保留 top-K 最强连接，其余删除。"""
        K = self.config.k_sparsity
        if len(connections) <= K:
            return connections
        top_k = heapq.nlargest(K, connections.items(), key=lambda x: x[1])
        return dict(top_k)

    def tau_decay_connections(
        self,
        all_connections: dict[str, dict[str, float]],
        tau_map: dict[str, float],
        decay_threshold: float = 0.1,
    ) -> dict[str, dict[str, float]]:
        """【P4】τ-Hebbian 联动：低 τ 节点的连接权重衰减。"""
        if not tau_map:
            return all_connections
        for nid, conns in all_connections.items():
            tau = tau_map.get(nid, 0.5)
            if tau < decay_threshold:
                factor = tau / decay_threshold
                all_connections[nid] = {
                    neighbor: w * factor
                    for neighbor, w in conns.items()
                }
        return all_connections
