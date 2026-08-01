"""
事务性记忆写入 (Transactional Memory Writes)
=============================================
基于 MemTX: Transactional Belief Commit for Stateful Agent Memory
(arXiv:2607.13157, Jul 2026)

核心思想：「写入 ≠ 信念」— 记忆写入采用两阶段提交：
  Phase 1 (WRITE):  数据写入临时暂存区，对外不可见
  Phase 2 (COMMIT): 确认后正式持久化，对外可见
  Phase 2 (ROLLBACK): 废弃暂存区数据

使用方式：
  with tx_manager.transaction() as tx:
      store.add_node(tx, data)   → 写入暂存
      store.add_edge(tx, data)   → 写入暂存
      if ok: tx.commit()         → 批量持久化
      else:  tx.rollback()       → 丢弃暂存
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OpType(Enum):
    CREATE_NODE = "create_node"
    UPDATE_NODE = "update_node"
    DELETE_NODE = "delete_node"
    CREATE_EDGE = "create_edge"
    DELETE_EDGE = "delete_edge"


@dataclass
class StagedOperation:
    """暂存的操作"""
    op: OpType
    node_type: str  # EpisodeNode | CommunityNode | HyperedgeNode | ConceptualNode
    data: dict
    timestamp: float = 0.0
    op_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.op_id:
            self.op_id = str(uuid.uuid4())[:8]


class TransactionStatus(Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MemoryTransaction:
    """一次记忆事务"""
    tx_id: str
    operations: list[StagedOperation] = field(default_factory=list)
    status: TransactionStatus = TransactionStatus.PENDING
    created_at: float = 0.0
    metadata: dict = field(default_factory=dict)
    _mgr: Optional["TransactionManager"] = None  # 父管理器引用
    _explicitly_closed: bool = False  # 显式标记是否已由调用方 commit/rollback

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

    def add(self, op: StagedOperation):
        self.operations.append(op)

    def commit(self) -> dict:
        """提交 — 两阶段 OCC 版本检查（Phase 1 全量检查，Phase 2 全量写入）

        Phase 1: 对所有 UPDATE/DELETE 操作执行版本检查，任一失败则整体中止
        Phase 2: 全部通过后，统一自增版本号
        """
        if self._mgr is not None:
            # Phase 1: 全量 version check（无副作用）
            for op in self.operations:
                node_id = op.data.get("id") or op.data.get("node_id", "")
                if node_id and op.op in (OpType.UPDATE_NODE, OpType.DELETE_NODE):
                    expected_ver = op.data.get("_expected_version")
                    if expected_ver is not None:
                        if not self._mgr.version_check(expected_ver, node_id):
                            current_ver = -1
                            if self._mgr._graph_store:
                                node = self._mgr._graph_store.get_episode(node_id)
                                current_ver = node.get("version", 1) if node else -1
                            conflict = self._mgr.record_conflict(
                                node_id=node_id,
                                expected_version=expected_ver,
                                current_version=current_ver,
                                strategy="occ_abort",
                                resolved=False,
                                detail=f"TX {self.tx_id[:8]} commit aborted by OCC"
                            )
                            raise RuntimeError(
                                f"OCC conflict on {node_id[:12]}: "
                                f"expected v{expected_ver}, aborting TX {self.tx_id[:8]}"
                            )

            # Phase 2: 全部检查通过后，统一自增版本号（全量写入）
            for op in self.operations:
                node_id = op.data.get("id") or op.data.get("node_id", "")
                if node_id and op.op in (OpType.UPDATE_NODE, OpType.DELETE_NODE):
                    if self._mgr._graph_store is not None:
                        # GraphLite 不支持 COALESCE: 两步法 (查 version → 存在+1 / 不存在置 1)
                        cur = self._mgr._graph_store.query_cypher(
                            "MATCH (e:EpisodeNode {id: $id}) RETURN e.version AS v",
                            {"id": node_id}
                        )
                        if cur and cur[0].get("v") is not None:
                            self._mgr._graph_store.query_cypher(
                                "MATCH (e:EpisodeNode {id: $id}) SET e.version = $v",
                                {"id": node_id, "v": int(cur[0]["v"]) + 1}
                            )
                        else:
                            self._mgr._graph_store.query_cypher(
                                "MATCH (e:EpisodeNode {id: $id}) SET e.version = 1",
                                {"id": node_id}
                            )

        self.status = TransactionStatus.COMMITTED
        self._explicitly_closed = True
        ops = [{"op": o.op.value, "type": o.node_type, "id": o.op_id}
               for o in self.operations]
        ops_str = "; ".join(f"{o['op']}({o['type']})" for o in ops)
        logger.info("TX %s committed: %d ops [%s]", self.tx_id[:8],
                    len(self.operations), ops_str)
        if self._mgr is not None:
            self._mgr._record(self)
        return {"tx_id": self.tx_id, "status": "committed",
                "ops_count": len(self.operations)}

    def rollback(self) -> dict:
        """回滚 — 从 GraphLite 清除暂存数据"""
        self.status = TransactionStatus.ROLLED_BACK
        self._explicitly_closed = True
        if self._mgr is not None and self._mgr._graph_store is not None:
            try:
                tx_tag = f"tx_{self.tx_id[:8]}"
                for op in reversed(self.operations):
                    node_id = op.data.get("id") or op.data.get("node_id", "")
                    if node_id and op.op in (OpType.CREATE_NODE, OpType.UPDATE_NODE):
                        self._mgr._graph_store.query_cypher(
                            "MATCH (n {id: $id}) WHERE n.tx_tag = $tag "
                            "REMOVE n.tx_tag",
                            {"id": node_id, "tag": tx_tag}
                        )
                    elif node_id and op.op == OpType.DELETE_NODE:
                        self._mgr._graph_store.query_cypher(
                            "MATCH (n {id: $id}) WHERE n.tx_tag = $tag "
                            "REMOVE n.tx_tag",
                            {"id": node_id, "tag": tx_tag}
                        )
                logger.info("TX %s rollback: GraphLite cleanup complete for %d ops",
                           self.tx_id[:8], len(self.operations))
            except Exception as e:
                logger.warning("TX %s rollback GraphLite cleanup failed: %s",
                              self.tx_id[:8], e)
        logger.info("TX %s rolled back: %d ops discarded",
                    self.tx_id[:8], len(self.operations))
        if self._mgr is not None:
            self._mgr._record(self)
        return {"tx_id": self.tx_id, "status": "rolled_back",
                "ops_count": len(self.operations)}

    def is_active(self) -> bool:
        return self.status == TransactionStatus.PENDING


class TransactionManager:
    """事务管理器 — 管理所有活跃事务 + OCC 版本检查 + 冲突日志"""

    def __init__(self, graph_store=None):
        self._active: dict[str, MemoryTransaction] = {}
        self._history: list[MemoryTransaction] = []  # 已完成的事务日志（ring buffer, max=1000）
        self._conflict_log: deque[dict] = deque(maxlen=1000)  # 写入冲突日志
        self._graph_store = graph_store

    def begin(self, metadata: Optional[dict] = None) -> MemoryTransaction:
        """开启新事务"""
        tx = MemoryTransaction(
            tx_id=str(uuid.uuid4()),
            metadata=metadata or {},
            _mgr=self,
        )
        self._active[tx.tx_id] = tx
        logger.debug("TX %s began", tx.tx_id[:8])
        return tx

    def get(self, tx_id: str) -> Optional[MemoryTransaction]:
        return self._active.get(tx_id)

    def _record(self, tx: MemoryTransaction):
        """由 MemoryTransaction.commit/rollback 回调记录（ring buffer, max 1000）"""
        self._history.append(tx)
        if len(self._history) > 1000:
            self._history = self._history[-1000:]
        self._active.pop(tx.tx_id, None)

    def commit(self, tx: MemoryTransaction) -> dict:
        """提交事务 — 调用 tx.commit() 自动记录"""
        return tx.commit()

    def rollback(self, tx: MemoryTransaction) -> dict:
        """回滚事务 — 调用 tx.rollback() 自动记录"""
        return tx.rollback()

    def transaction(self, metadata: Optional[dict] = None):
        """上下文管理器 — with tx_manager.transaction() as tx:"""
        return _TransactionContext(self, metadata)

    def version_check(
        self,
        expected_version: int,
        node_id: str,
    ) -> bool:
        """
        OCC 版本检查 — 在 prepare 阶段调用。
        内部从 GraphLite 获取实际版本，避免调用方传参不一致。

        Args:
            expected_version: 事务开始时读取的版本号。
            node_id: 节点 ID。

        Returns:
            True 如果版本一致，允许提交。
            False 如果版本不一致，需要消解或回滚。
        """
        if self._graph_store is None:
            return True  # 无数据库可用时跳过
        node = self._graph_store.get_episode(node_id)
        current_version = node.get("version", 1) if node else None
        ok = expected_version == current_version
        if not ok:
            logger.warning(
                "Version mismatch on %s: expected v%d, current v%d",
                node_id[:12] if node_id else "?", expected_version, current_version,
            )
        return ok

    def record_conflict(
        self,
        node_id: str,
        expected_version: int,
        current_version: int,
        strategy: str = "",
        resolved: bool = False,
        detail: str = "",
    ) -> dict:
        """
        记录写入冲突到 conflict_log ring buffer。

        Args:
            node_id: 冲突涉及的节点 ID。
            expected_version: 写入方预期版本。
            current_version: 数据库当前版本。
            strategy: 使用的消解策略（lww/merge/additive）。
            resolved: 是否已消解。
            detail: 补充说明。

        Returns:
            记录的冲突条目。
        """
        entry = {
            "node_id": node_id,
            "expected_version": expected_version,
            "current_version": current_version,
            "strategy": strategy,
            "resolved": resolved,
            "timestamp": time.time(),
            "detail": detail,
        }
        self._conflict_log.append(entry)
        logger.info(
            "Conflict recorded: %s v%d->v%d strategy=%s resolved=%s",
            node_id[:12], expected_version, current_version,
            strategy, resolved,
        )
        return entry

    def get_conflict_log(self, limit: int = 50) -> list[dict]:
        """查询最近的冲突日志（按时间倒序）。"""
        return list(reversed(self._conflict_log))[:limit]

    def state(self) -> dict:
        return {
            "active_count": len(self._active),
            "history_count": len(self._history),
            "conflict_count": len(self._conflict_log),
            "unresolved_conflicts": sum(
                1 for c in self._conflict_log if not c["resolved"]
            ),
            "committed": sum(1 for t in self._history
                             if t.status == TransactionStatus.COMMITTED),
            "rolled_back": sum(1 for t in self._history
                               if t.status == TransactionStatus.ROLLED_BACK),
            "recent_ops": sum(len(t.operations) for t in self._history[-5:]),
        }


class _TransactionContext:
    """事务上下文管理器"""

    def __init__(self, mgr: TransactionManager, metadata: Optional[dict]):
        self._mgr = mgr
        self._metadata = metadata
        self.transaction: Optional[MemoryTransaction] = None

    def __enter__(self) -> MemoryTransaction:
        self.transaction = self._mgr.begin(self._metadata)
        return self.transaction

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.transaction is None:
            return
        if exc_type is not None:
            # 异常退出 → 自动回滚（仅当事务仍活跃且未显式关闭）
            if self.transaction.is_active() and not self.transaction._explicitly_closed:
                self._mgr.rollback(self.transaction)
        elif self.transaction.is_active() and not self.transaction._explicitly_closed:
            # 正常退出但未显式 commit/rollback → 自动提交
            self._mgr.commit(self.transaction)
        # 已显式 commit/rollback → 无操作
