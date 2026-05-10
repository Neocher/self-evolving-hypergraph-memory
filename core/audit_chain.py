"""
BLAKE3 溯源链
============
每个梦境周期产生一个溯源区块，所有区块构成不可篡改的哈希链。

BLAKE3 相比 SHA-256 快 5-10 倍，支持 AVX2 硬件加速。

区块结构:
    prev_hash: str           ← 上一个区块的哈希
    operations: list[dict]   ← 本周期所有 CRUD 操作
    timestamp: str           ← ISO 8601
    stats: dict              ← 统计信息
    hash: str                ← blake3(prev_hash + blake3(operations) + timestamp)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional


@dataclass
class AuditOperation:
    """单次溯源操作记录"""

    op_type: str  # 'create' | 'update' | 'delete'
    node_id: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    reason: str = ""  # 'tau_decay' | 'hebbian_prune' | 'ssm_gate' | 'community_merge' | 'explicit'


@dataclass
class AuditBlock:
    """溯源区块"""

    prev_hash: str
    operations: list[AuditOperation]
    timestamp: str
    stats: dict  # {created, updated, deleted, before_size, after_size}
    hash: str = ""


class AuditChain:
    """
    BLAKE3 溯源链。

    维护一条不可篡改的操作哈希链。
    支持追加区块、回溯查询、状态回滚。
    """

    def __init__(self, storage_backend: Optional[Callable] = None) -> None:
        self.storage = storage_backend
        self._chain: list[AuditBlock] = []
        self._persist_path: Optional[str] = None  # 【FIX】文件持久化路径
        if storage_backend is None:
            # 默认使用文件持久化
            import os
            self._persist_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "data", "audit_chain.json"
            )
        self._load_chain()

    def _load_chain(self) -> None:
        """从存储加载已有链"""
        if self.storage:
            data = self.storage("load")
            if data:
                self._chain = data
                return
        # 【FIX】从文件加载兜底
        if self._persist_path:
            try:
                import json
                with open(self._persist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._chain = []
                for b in data:
                    # 【FIX】将operations从dict还原为AuditOperation对象，保证hash计算正确
                    ops = [AuditOperation(**op) if isinstance(op, dict) else op
                           for op in b.get("operations", [])]
                    b["operations"] = ops
                    self._chain.append(AuditBlock(**b))
            except (FileNotFoundError, json.JSONDecodeError, TypeError):
                self._chain = []

    def _compute_hash(self, block: AuditBlock) -> str:
        """
        计算区块哈希。
        hash = blake3(prev_hash + blake3(operations) + timestamp)
        """
        import blake3

        ops_json = json.dumps(
            [asdict(op) for op in block.operations],
            sort_keys=True,
            ensure_ascii=False,
        )
        ops_hash = blake3.blake3(ops_json.encode()).hexdigest()
        data = block.prev_hash + ops_hash + block.timestamp
        return blake3.blake3(data.encode()).hexdigest()

    def append_block(
        self, operations: list[AuditOperation], stats: dict
    ) -> AuditBlock:
        """
        追加一个溯源区块。

        Args:
            operations: 本周期所有操作
            stats: 统计信息 {created, updated, deleted, before_size, after_size}

        Returns:
            新创建的 AuditBlock（含 hash）
        """
        prev_hash = self._chain[-1].hash if self._chain else "0" * 64
        block = AuditBlock(
            prev_hash=prev_hash,
            operations=operations,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            stats=stats,
        )
        block.hash = self._compute_hash(block)
        self._chain.append(block)
        self._save_chain()
        return block

    def trace_node(self, node_id: str) -> list[AuditOperation]:
        """回溯指定节点的所有变更历史。"""
        result: list[AuditOperation] = []
        for block in self._chain:
            for op in block.operations:
                if op.node_id == node_id:
                    result.append(op)
        return result

    def verify_chain(self) -> bool:
        """验证整个溯源链的完整性。"""
        for i, block in enumerate(self._chain):
            expected_hash = self._compute_hash(block)
            if block.hash != expected_hash:
                return False
            if i > 0 and block.prev_hash != self._chain[i - 1].hash:
                return False
        return True

    def rollback_to(self, block_hash: str) -> int:
        """
        回滚到指定区块之后的状态。
        返回回滚的操作数。
        """
        for i, block in enumerate(self._chain):
            if block.hash == block_hash:
                rolled_back = sum(len(b.operations) for b in self._chain[i + 1 :])
                self._chain = self._chain[: i + 1]
                self._save_chain()
                return rolled_back
        return 0

    def _save_chain(self) -> None:
        """保存链到存储"""
        if self.storage:
            self.storage("save", self._chain)
            return
        # 【FIX】写入文件兜底
        if self._persist_path:
            import json
            import os
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            with open(self._persist_path, 'w', encoding='utf-8') as f:
                json.dump(
                    [asdict(b) for b in self._chain],
                    f, ensure_ascii=False, default=str
                )

    @property
    def chain_length(self) -> int:
        return len(self._chain)

    @property
    def last_block_hash(self) -> str:
        return self._chain[-1].hash if self._chain else "0" * 64
