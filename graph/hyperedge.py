"""
超边类型定义
===========
三种超边:
- EpisodeHyperedge:   连接同一主题的多个情节节点
- SemanticHyperedge:  连接多个概念节点指向抽象结论
- TemporalHyperedge:  连接时间临近的多个节点

Kuzu 原生不支持超边，编码为辅助节点 + Cypher 边连接。

[Harness Fix] 增加 member_ids 业务规则校验：至少 2 个成员节点。
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class HyperedgeType(str, Enum):
    EPISODE = "episode"     # 情节超边
    SEMANTIC = "semantic"   # 语义超边
    TEMPORAL = "temporal"   # 时态超边


@dataclass
class Hyperedge:
    """自适应门控数据模型"""
    id: str
    type: HyperedgeType
    member_ids: List[str]       # [Harness Fix] 调用前需 ensure_member_ids_valid()
    created_at: float
    metadata: dict = field(default_factory=dict)
    # 以下由 DualAdaptiveGate 维护 (SSM+MLP双门控)
    hidden_state: Optional[list] = None
    gate_value: float = 1.0

    @staticmethod
    def ensure_member_ids_valid(member_ids: List[str]) -> None:
        """
        [Harness Fix] 校验 member_ids 至少包含 2 个节点。

        Raises:
            ValueError: 如果 member_ids 长度 < 2
        """
        if len(member_ids) < 2:
            raise ValueError(
                f"Hyperedge must have at least 2 member nodes, got {len(member_ids)}"
            )


class HyperedgeManager:
    """
    超边管理器。

    负责超边的 CRUD、查询和 SSM 门控状态维护。
    """

    def __init__(self, kuzu_store) -> None:
        self.store = kuzu_store

    def create_episode_hyperedge(self,
                                  member_ids: List[str],
                                  topic: Optional[str] = None) -> Hyperedge:
        """
        创建情节超边，连接同一主题的多个 episode 节点。

        Raises:
            ValueError: 如果 member_ids 少于 2 个
        """
        Hyperedge.ensure_member_ids_valid(member_ids)
        edge = Hyperedge(
            id=str(uuid.uuid4()),
            type=HyperedgeType.EPISODE,
            member_ids=list(member_ids),
            created_at=time.time(),
            metadata={"topic": topic} if topic else {},
        )
        self._persist_hyperedge(edge)
        return edge

    def create_semantic_hyperedge(self,
                                   member_ids: List[str],
                                   conclusion: str) -> Hyperedge:
        """
        创建语义超边，连接多个概念节点指向一个抽象结论。

        Raises:
            ValueError: 如果 member_ids 少于 2 个
        """
        Hyperedge.ensure_member_ids_valid(member_ids)
        edge = Hyperedge(
            id=str(uuid.uuid4()),
            type=HyperedgeType.SEMANTIC,
            member_ids=list(member_ids),
            created_at=time.time(),
            metadata={"conclusion": conclusion},
        )
        self._persist_hyperedge(edge)
        return edge

    def create_temporal_hyperedge(self,
                                   member_ids: List[str],
                                   start_time: float,
                                   end_time: float) -> Hyperedge:
        """
        创建时态超边，连接时间临近的节点。

        Raises:
            ValueError: 如果 member_ids 少于 2 个
        """
        Hyperedge.ensure_member_ids_valid(member_ids)
        edge = Hyperedge(
            id=str(uuid.uuid4()),
            type=HyperedgeType.TEMPORAL,
            member_ids=list(member_ids),
            created_at=time.time(),
            metadata={"start_time": start_time, "end_time": end_time},
        )
        self._persist_hyperedge(edge)
        return edge

    def _persist_hyperedge(self, edge: Hyperedge) -> None:
        """将超边持久化到 Kuzu（辅助节点 + HYPEREDGE_MEMBER 边）。"""
        import json
        self.store.create_hyperedge_node({
            "id": edge.id,
            "type": edge.type.value,
            "created_at": edge.created_at,
            "gate_value": edge.gate_value,
            "metadata": json.dumps(edge.metadata, ensure_ascii=False),
        })
        for member_id in edge.member_ids:
            self.store.link_hyperedge_member(edge.id, member_id)

    def get_hyperedges_by_node(self, node_id: str) -> List[Hyperedge]:
        """查询包含指定节点的所有超边。"""
        import json
        rows = self.store.get_hyperedges_by_node(node_id)
        result: List[Hyperedge] = []
        for row in rows:
            try:
                metadata = json.loads(row.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            result.append(Hyperedge(
                id=row["id"],
                type=HyperedgeType(row["type"]),
                member_ids=self._resolve_member_ids(row["id"]),
                created_at=row.get("created_at", 0.0),
                metadata=metadata,
                gate_value=row.get("gate_value", 1.0),
            ))
        return result

    def _resolve_member_ids(self, hyperedge_id: str) -> List[str]:
        """从 Kuzu 解析超边的成员节点 ID 列表。"""
        members = self.store.get_hyperedge_members(hyperedge_id)
        return [m["id"] for m in members]

    def get_hyperedge(self, hyperedge_id: str) -> Optional[Hyperedge]:
        """按 ID 查询单个超边。"""
        rows = self.store.query_cypher(
            "MATCH (h:HyperedgeNode) WHERE h.id = $id RETURN h.*",
            {"id": hyperedge_id}
        )
        if not rows:
            return None
        import json
        row = dict(rows[0])
        try:
            metadata = json.loads(row.get("metadata", "{}"))
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return Hyperedge(
            id=row["id"],
            type=HyperedgeType(row["type"]),
            member_ids=self._resolve_member_ids(row["id"]),
            created_at=row.get("created_at", 0.0),
            metadata=metadata,
            gate_value=row.get("gate_value", 1.0),
        )
