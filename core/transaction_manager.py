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
    _mgr: Any = None  # 父管理器引用

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

    def add(self, op: StagedOperation):
        self.operations.append(op)

    def commit(self) -> dict:
        """提交 — 通过管理器记录"""
        self.status = TransactionStatus.COMMITTED
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
        """回滚 — 通过管理器记录"""
        self.status = TransactionStatus.ROLLED_BACK
        logger.info("TX %s rolled back: %d ops discarded",
                    self.tx_id[:8], len(self.operations))
        if self._mgr is not None:
            self._mgr._record(self)
        return {"tx_id": self.tx_id, "status": "rolled_back",
                "ops_count": len(self.operations)}

    def is_active(self) -> bool:
        return self.status == TransactionStatus.PENDING


class TransactionManager:
    """事务管理器 — 管理所有活跃事务"""

    def __init__(self):
        self._active: dict[str, MemoryTransaction] = {}
        self._history: list[MemoryTransaction] = []  # 已完成的事务日志

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
        """由 MemoryTransaction.commit/rollback 回调记录"""
        self._history.append(tx)
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

    def state(self) -> dict:
        return {
            "active_count": len(self._active),
            "history_count": len(self._history),
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
        if exc_type is not None or self.transaction.status != TransactionStatus.PENDING:
            # 异常退出 → 自动回滚
            if self.transaction.is_active():
                self._mgr.rollback(self.transaction)
        # 正常退出但未显式 commit/rollback → 自动提交
        elif self.transaction.is_active():
            self._mgr.commit(self.transaction)
