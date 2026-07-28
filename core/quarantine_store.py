"""
隔离存储
========
管理 Kuzu 图中记忆节点的隔离状态。

隔离方案: SET e.quarantine = true, e.quarantine_reason = reason
（使用 Kuzu 节点属性标记，无需额外表结构）

隔离节点的行为:
- 不加入 FAISS 索引
- 梦境 _gather_step 跳过
- 检索结果默认排除
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class QuarantineStore:
    """
    隔离存储 —— 标记/查询/恢复隔离节点。

    维护内存中的隔离 ID 集合，避免每次检查都查询 Kuzu。
    """

    def __init__(self, kuzu_store=None):
        self._kuzu_store = kuzu_store
        self._quarantined_ids: set[str] = set()

    # ── Kuzu 访问 ─────────────────────────────────────────

    @property
    def kuzu_store(self):
        return self._kuzu_store

    @kuzu_store.setter
    def kuzu_store(self, store):
        self._kuzu_store = store

    # ── 核心操作 ─────────────────────────────────────────

    def quarantine(self, episode_id: str, reason: str, source: str = "defense") -> bool:
        """标记节点为隔离状态。"""
        if self._kuzu_store is None:
            logger.warning("QuarantineStore: no Kuzu store available; using in-memory only")
            self._quarantined_ids.add(episode_id)
            return True
        try:
            self._kuzu_store.query_cypher(
                "MATCH (e:EpisodeNode {id: $id}) "
                "SET e.quarantine = true, "
                "e.quarantine_reason = $reason, "
                "e.quarantine_source = $source, "
                "e.quarantined_at = $ts",
                {"id": episode_id, "reason": reason, "source": source, "ts": time.time()},
            )
            self._quarantined_ids.add(episode_id)
            logger.warning("Quarantined node %s: %s", episode_id[:12], reason[:80])
            return True
        except Exception:
            logger.exception("Failed to quarantine node %s", episode_id[:12])
            return False

    def promote(self, episode_id: str) -> bool:
        """解除节点的隔离状态。"""
        self._quarantined_ids.discard(episode_id)
        if self._kuzu_store is None:
            return True
        try:
            self._kuzu_store.query_cypher(
                "MATCH (e:EpisodeNode {id: $id}) "
                "SET e.quarantine = false, "
                "e.quarantine_reason = null, "
                "e.quarantine_source = null, "
                "e.quarantined_at = null",
                {"id": episode_id},
            )
            logger.info("Promoted node %s out of quarantine", episode_id[:12])
            return True
        except Exception:
            logger.exception("Failed to promote node %s", episode_id[:12])
            return False

    # ── 查询 ─────────────────────────────────────────────

    def is_quarantined(self, episode_id: str) -> bool:
        """检查节点是否被隔离（基于内存集合，O(1)）。"""
        return episode_id in self._quarantined_ids

    def get_quarantined_ids(self) -> set[str]:
        """返回所有隔离节点 ID 的副本。"""
        return set(self._quarantined_ids)

    def list_quarantined(self, limit: int = 100) -> list[dict]:
        """列出隔离节点详情。"""
        if self._kuzu_store is None:
            return [
                {"id": eid, "reason": "", "source": "", "quarantined_at": 0.0}
                for eid in list(self._quarantined_ids)[:limit]
            ]
        try:
            rows = self._kuzu_store.query_cypher(
                "MATCH (e:EpisodeNode) WHERE e.quarantine = true "
                "RETURN e.id, e.content, e.quarantine_reason, "
                "e.quarantine_source, e.quarantined_at "
                "ORDER BY e.quarantined_at DESC LIMIT $limit",
                {"limit": limit},
            )
            result = []
            for row in rows:
                if isinstance(row, (list, tuple)):
                    result.append({
                        "id": str(row[0]) if len(row) > 0 else "",
                        "content": str(row[1]) if len(row) > 1 else "",
                        "reason": str(row[2]) if len(row) > 2 else "",
                        "source": str(row[3]) if len(row) > 3 else "",
                        "quarantined_at": float(row[4]) if len(row) > 4 else 0.0,
                    })
                elif isinstance(row, dict):
                    result.append({
                        "id": str(row.get("e.id", row.get("id", ""))),
                        "content": str(row.get("e.content", row.get("content", ""))),
                        "reason": str(row.get("e.quarantine_reason", row.get("quarantine_reason", ""))),
                        "source": str(row.get("e.quarantine_source", row.get("quarantine_source", ""))),
                        "quarantined_at": float(row.get("e.quarantined_at", row.get("quarantined_at", 0.0))),
                    })
            return result
        except Exception:
            logger.exception("Failed to list quarantined nodes")
            return []

    def count(self) -> int:
        """统计隔离节点数量。"""
        return len(self._quarantined_ids)

    def refresh(self) -> int:
        """从 Kuzu 同步隔离节点 ID 到内存集合。

        在服务启动或怀疑内存集合不同步时调用。
        """
        if self._kuzu_store is None:
            return len(self._quarantined_ids)
        try:
            rows = self._kuzu_store.query_cypher(
                "MATCH (e:EpisodeNode) WHERE e.quarantine = true RETURN e.id",
            )
            fresh_ids: set[str] = set()
            for row in rows:
                if isinstance(row, (list, tuple)) and len(row) > 0:
                    fresh_ids.add(str(row[0]))
                elif isinstance(row, dict):
                    fresh_ids.add(str(row.get("e.id", row.get("id", ""))))
            self._quarantined_ids = fresh_ids
            logger.info("QuarantineStore refreshed: %d quarantined nodes", len(fresh_ids))
            return len(fresh_ids)
        except Exception:
            logger.exception("QuarantineStore refresh failed")
            return len(self._quarantined_ids)
