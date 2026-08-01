"""
超边类型定义
===========
三种超边:
- EpisodeHyperedge:   连接同一主题的多个情节节点
- SemanticHyperedge:  连接多个概念节点指向抽象结论
- TemporalHyperedge:  连接时间临近的多个节点

GraphLite 原生不支持超边，编码为辅助节点 + Cypher 边连接。

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
    # P2 MemClaw 多Agent共享记忆
    agent_scope: str | list[str] = 'global'
    source_agent_id: str = 'system'
    source_timestamp: float = 0.0
    supersession_of: Optional[str] = None

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

    def __init__(self, graphlite_store) -> None:
        self.store = graphlite_store

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
                                    conclusion: str,
                                    agent_id: str = 'system',
                                    agent_scope: str | list[str] = 'global',
                                    supersession_of: Optional[str] = None) -> Hyperedge:
        """
        创建语义超边，连接多个概念节点指向一个抽象结论。

        Args:
            agent_id: 创建该超边的 agent 标识
            agent_scope: 可见范围 ('global' / agent_id / list of agent_ids)
            supersession_of: 被此超边取代的旧超边 ID（超车机制）

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
            agent_scope=agent_scope,
            source_agent_id=agent_id,
            source_timestamp=time.time(),
            supersession_of=supersession_of,
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
        """将超边持久化到 GraphLite（辅助节点 + 批量 HYPEREDGE_MEMBER 边）。"""
        import json
        self.store.create_hyperedge_node({
            "id": edge.id,
            "type": edge.type.value,
            "created_at": edge.created_at,
            "gate_value": edge.gate_value,
            "metadata": json.dumps(edge.metadata, ensure_ascii=False),
            "agent_scope": json.dumps(edge.agent_scope, ensure_ascii=False),
            "source_agent_id": edge.source_agent_id,
            "source_timestamp": edge.source_timestamp,
            "supersession_of": edge.supersession_of,
        })
        self.store.query_cypher(
            "UNWIND $member_ids AS mid "
            "MATCH (h:HyperedgeNode {id: $hid}), (e:EpisodeNode {id: mid}) "
            "CREATE (h)-[:HYPEREDGE_MEMBER]->(e)",
            {"hid": edge.id, "member_ids": edge.member_ids},
        )

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
                agent_scope=self._deserialize_agent_scope(row.get("agent_scope")),
                source_agent_id=row.get("source_agent_id", "system"),
                source_timestamp=row.get("source_timestamp", 0.0),
                supersession_of=row.get("supersession_of", None),
            ))
        return result

    def _resolve_member_ids(self, hyperedge_id: str) -> List[str]:
        """从 GraphLite 解析超边的成员节点 ID 列表。"""
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
            agent_scope=self._deserialize_agent_scope(row.get("agent_scope")),
            source_agent_id=row.get("source_agent_id", "system"),
            source_timestamp=row.get("source_timestamp", 0.0),
            supersession_of=row.get("supersession_of", None),
        )

    # ─── P2 MemClaw: 多Agent共享记忆 ─────────────────────────

    @staticmethod
    def _deserialize_agent_scope(raw) -> str | list[str]:
        if raw is None:
            return 'global'
        if isinstance(raw, str):
            try:
                import json
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, (str, list)) else 'global'
            except (json.JSONDecodeError, TypeError):
                return raw
        if isinstance(raw, list):
            return raw
        return 'global'

    def get_visible_hyperedges(self, agent_id: str, limit: int = 1000) -> List[Hyperedge]:
        """获取该 agent 可见的所有超边（global 或 scope 包含该 agent）。
        
        添加 LIMIT 防止全表扫描 OOM，同时优先返回 gate_value 高的超边。
        """
        all_rows = self.store.query_cypher(
            "MATCH (h:HyperedgeNode) RETURN h.* ORDER BY h.gate_value DESC LIMIT $limit",
            {"limit": limit},
        )
        import json
        result: List[Hyperedge] = []
        for row in all_rows:
            try:
                md = json.loads(row.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                md = {}
            scope = self._deserialize_agent_scope(row.get("agent_scope"))
            if scope != 'global':
                if isinstance(scope, list) and agent_id not in scope:
                    continue
                elif isinstance(scope, str) and scope != agent_id:
                    continue
            result.append(Hyperedge(
                id=row["id"],
                type=HyperedgeType(row["type"]),
                member_ids=self._resolve_member_ids(row["id"]),
                created_at=row.get("created_at", 0.0),
                metadata=md,
                gate_value=row.get("gate_value", 1.0),
                agent_scope=scope,
                source_agent_id=row.get("source_agent_id", "system"),
                source_timestamp=row.get("source_timestamp", 0.0),
                supersession_of=row.get("supersession_of", None),
            ))
        return result

    def create_multi_agent_hyperedge(self,
                                      member_ids: List[str],
                                      agent_ids: List[str],
                                      conclusion: str = '',
                                      topic: Optional[str] = None) -> Hyperedge:
        """
        创建多Agent共享超边。

        agent_ids 中的每个 agent 对该超边可见。
        """
        Hyperedge.ensure_member_ids_valid(member_ids)
        edge = Hyperedge(
            id=str(uuid.uuid4()),
            type=HyperedgeType.SEMANTIC,
            member_ids=list(member_ids),
            created_at=time.time(),
            metadata={"conclusion": conclusion, "topic": topic or ""},
            agent_scope=list(agent_ids),
            source_agent_id='system',
            source_timestamp=time.time(),
        )
        self._persist_hyperedge(edge)
        return edge

    def delete_hyperedge(self, hyperedge_id: str) -> bool:
        """删除超边及其所有关联边（DETACH DELETE）。

        使用 GraphLite 的 DETACH DELETE 确保超边节点和 HYPEREDGE_MEMBER 边被级联清理，
        避免删除超边后孤立边残留在图数据库中。
        """
        try:
            self.store.query_cypher(
                "MATCH (h:HyperedgeNode {id: $id}) DETACH DELETE h",
                {"id": hyperedge_id}
            )
            logger.info("Deleted hyperedge: %s", hyperedge_id[:8])
            return True
        except Exception:
            logger.exception("Failed to DETACH DELETE hyperedge: %s", hyperedge_id)
            return False

    def purge_orphaned_hyperedges(self) -> int:
        """删除没有任何成员节点的孤立超边。"""
        try:
            result = self.store.query_cypher(
                "MATCH (h:HyperedgeNode) "
                "WHERE NOT EXISTS { (h)-[:HYPEREDGE_MEMBER]->() } "
                "RETURN count(h) AS cnt"
            )
            orphan_count = 0
            if result:
                row = result[0]
                orphan_count = row[0] if isinstance(row, (list, tuple)) else row.get("cnt", 0)
            if orphan_count > 0:
                self.store.query_cypher(
                    "MATCH (h:HyperedgeNode) "
                    "WHERE NOT EXISTS { (h)-[:HYPEREDGE_MEMBER]->() } "
                    "DETACH DELETE h"
                )
                logger.info("Purged %d orphaned hyperedges", orphan_count)
            return orphan_count
        except Exception:
            logger.exception("Failed to purge orphaned hyperedges")
            return 0
