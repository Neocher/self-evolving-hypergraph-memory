"""
写入消解系统 (Write Reconciliation)
====================================
基于 OCC 乐观锁的写入冲突检测与消解，支持三策略:
  - LWW (Last-Write-Wins):  简单时间戳比较，保留最新
  - Merge:                  属性级合并（同名属性取最新，不同属性叠加）
  - Additive:               实体级叠加（不覆盖，保留所有版本）

核心流程：
  1. ConflictDetector.check(version, expected) → 检测版本不一致
  2. StrategyResolver.resolve(strategy, current, incoming) → 执行消解
  3. ConflictLogger.record(...) → 记录冲突到 ring buffer

使用方式：
  reconciler = WriteReconciler(kuzu_store)
  result = reconciler.resolve(node_id, incoming_data, expected_version, strategy=Strategy.MERGE)

默认不启用现有写入路径，需手动调用 resolve()。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ConflictError(Exception):
    """版本冲突异常 — 当 OCC 检测到 version 不一致时抛出。"""

    def __init__(self, node_id: str, expected_version: int, current_version: int):
        self.node_id = node_id
        self.expected_version = expected_version
        self.current_version = current_version
        super().__init__(
            f"Version conflict on {node_id}: expected v{expected_version}, "
            f"current v{current_version}"
        )


class Strategy(Enum):
    """写入消解策略"""
    LWW = "lww"          # Last-Write-Wins — 时间戳比较，覆盖旧版本
    MERGE = "merge"      # 属性级合并—同名属性取最新，不同属性叠加
    ADDITIVE = "additive"  # 实体级叠加—保留所有版本，追加存储


@dataclass
class ConflictRecord:
    """单条冲突记录日志"""
    node_id: str
    expected_version: int
    current_version: int
    strategy: Strategy
    resolved: bool
    timestamp: float
    detail: str = ""


# ─── 冲突检测 ───────────────────────────────────────────────


class ConflictDetector:
    """
    OCC 乐观锁冲突检测器。

    比较写入时携带的 expected_version 与数据库中当前 version。
    一致 → 允许写入（乐观通过）；不一致 → 冲突，需要消解。
    """

    @staticmethod
    def check(current_version: int, expected_version: int) -> bool:
        """
        OCC 版本检查。

        Args:
            current_version: 数据库中节点的当前版本号。
            expected_version: 写入方携带的预期版本号。

        Returns:
            True 如果版本一致（无冲突）；False 如果版本不一致（有冲突）。
        """
        return current_version == expected_version

    @staticmethod
    def detect(
        kuzu_store: Any,
        node_id: str,
        expected_version: int,
    ) -> tuple[bool, Optional[dict], Optional[int]]:
        """
        从 Kuzu 读取节点当前版本并检测冲突。

        Args:
            kuzu_store: RyuStore 实例。
            node_id: 目标节点 ID。
            expected_version: 写入方预期的版本号。

        Returns:
            (has_conflict, current_node, current_version)
            has_conflict=True 表示版本不一致。
        """
        node = kuzu_store.get_episode(node_id)
        if node is None:
            # 节点不存在，无法检查版本 — 视为「新建」场景，不冲突
            return False, None, None

        current_version = node.get("version", 1)
        has_conflict = not ConflictDetector.check(current_version, expected_version)
        return has_conflict, node, current_version


# ─── 消解策略 ───────────────────────────────────────────────


class StrategyResolver:
    """
    三策略消解执行器。

    在 ConflictDetector 检测到冲突后，根据选定策略生成消解后的数据。
    不直接写入数据库，只返回消解后的 dict，由调用方决定如何写入。
    """

    @staticmethod
    def resolve_lww(current_node: dict, incoming_data: dict) -> dict:
        """
        LWW (Last-Write-Wins)：简单时间戳比较。
        - 如果 incoming 时间戳更新 → incoming 完全覆盖 current
        - 否则保留 current（丢弃 incoming）

        Args:
            current_node: 数据库中已有的节点数据。
            incoming_data: 待写入的新数据（含 timestamp）。

        Returns:
            消解后的节点数据 dict。
        """
        current_ts = current_node.get("created_at", 0.0)
        incoming_ts = incoming_data.get("created_at", 0.0)

        if incoming_ts >= current_ts:
            # incoming 更新 — 覆盖，但保留 id 和 version（version 会在写入时递增）
            resolved = dict(incoming_data)
            resolved["id"] = current_node.get("id", incoming_data.get("id", ""))
            return resolved
        else:
            # current 更新 — 保留现有数据
            return dict(current_node)

    @staticmethod
    def resolve_merge(current_node: dict, incoming_data: dict) -> dict:
        """
        Merge：属性级合并。
        - 同名属性取时间戳更新的值
        - 不同属性叠加
        - 特殊字段 (id, version, created_at) 按规则保留

        Args:
            current_node: 数据库中已有的节点数据。
            incoming_data: 待写入的新数据。

        Returns:
            合并后的节点数据 dict。
        """
        merged = dict(current_node)

        # 这些字段有特殊语义，不参与属性级覆盖
        _meta_fields = {"id", "version", "created_at"}

        for key, value in incoming_data.items():
            if key in _meta_fields:
                continue
            if key not in merged:
                # 新属性直接叠加
                merged[key] = value
            else:
                # 同名属性：取较新的值（时间戳靠 created_at 判断）
                # 如果 incoming_data 中有同名字段且时间戳更新则覆盖
                cur_ts = current_node.get("created_at", 0.0)
                inc_ts = incoming_data.get("created_at", 0.0)
                if inc_ts >= cur_ts:
                    merged[key] = value
                # else: 保留 current 的值（不覆盖）

        # created_at 取较早的（保留创建时间）
        merged["created_at"] = min(
            current_node.get("created_at", 0.0),
            incoming_data.get("created_at", 0.0),
        )
        return merged

    @staticmethod
    def resolve_additive(current_node: dict, incoming_data: dict) -> dict:
        """
        Additive：实体级叠加。
        - 不覆盖任何现有数据
        - 将 incoming 的所有字段追加到 current 的 _additive_versions 列表中
        - 保留 current 数据不变

        Args:
            current_node: 数据库中已有的节点数据。
            incoming_data: 待写入的新数据。

        Returns:
            保留了所有版本的节点数据，_additive_versions 列表包含历史。
        """
        resolved = dict(current_node)

        # 取出或初始化版本历史列表
        versions: list[dict] = resolved.get("_additive_versions", [])
        # 将当前版本也记录进去（如果尚未记录）
        if not versions:
            snapshot = {}
            for k, v in current_node.items():
                if k != "_additive_versions":
                    snapshot[k] = v
            versions.append(snapshot)

        # 追加 incoming 版本
        versions.append(dict(incoming_data))
        resolved["_additive_versions"] = versions

        # 标记为 additive 模式
        resolved["_additive"] = True
        return resolved


# ─── 冲突日志 — Ring Buffer ──────────────────────────────────


class ConflictLogger:
    """
    环形缓冲区冲突日志。

    记录所有检测到的冲突及其消解结果，支持按时间倒序查询。
    maxlen=1000，自动丢弃最旧记录。
    """

    def __init__(self, maxlen: int = 1000):
        self._log: deque[ConflictRecord] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def record(
        self,
        node_id: str,
        expected_version: int,
        current_version: int,
        strategy: Strategy,
        resolved: bool,
        detail: str = "",
    ) -> ConflictRecord:
        """记录一条冲突日志。"""
        record = ConflictRecord(
            node_id=node_id,
            expected_version=expected_version,
            current_version=current_version,
            strategy=strategy,
            resolved=resolved,
            timestamp=time.time(),
            detail=detail,
        )
        self._log.append(record)
        logger.info(
            "Conflict [%s] node=%s v%d→v%d resolved=%s strategy=%s",
            record.timestamp, node_id, expected_version,
            current_version, resolved, strategy.value,
        )
        return record

    def query(self, limit: int = 50) -> list[ConflictRecord]:
        """查询最近的冲突日志（按时间倒序）。"""
        return list(reversed(self._log))[:limit]

    def stats(self) -> dict:
        """冲突日志统计。"""
        total = len(self._log)
        resolved = sum(1 for r in self._log if r.resolved)
        unresolved = total - resolved
        strategy_counts: dict[str, int] = {}
        for r in self._log:
            strategy_counts[r.strategy.value] = strategy_counts.get(r.strategy.value, 0) + 1
        return {
            "total": total,
            "resolved": resolved,
            "unresolved": unresolved,
            "maxlen": self._maxlen,
            "strategy_breakdown": strategy_counts,
        }

    def clear(self) -> int:
        """清空日志，返回被清空的记录数。"""
        count = len(self._log)
        self._log.clear()
        return count


# ─── 写入消解器 ─────────────────────────────────────────────


class WriteReconciler:
    """
    写入消解主入口。

    编排冲突检测 → 策略消解 → 日志记录 三步骤。
    调用方需传入 kuzu_store 用于读取节点当前状态，resolve() 返回消解后的数据。

    典型用法：
        reconciler = WriteReconciler(kuzu_store, conflict_logger)
        resolved = reconciler.resolve(
            node_id="ep_xxx",
            incoming_data={"content": "...", "created_at": 1234567890.0},
            expected_version=2,
            strategy=Strategy.MERGE,
        )
        if resolved["conflict"]:
            # 版本冲突，resolved["data"] 是消解后的数据
            kuzu_store.update_with_version(node_id, resolved["data"], version=None)
        else:
            # 无冲突，正常写入
            ...
    """

    def __init__(
        self,
        kuzu_store: Any,
        conflict_logger: Optional[ConflictLogger] = None,
    ):
        self._kuzu_store = kuzu_store
        self._detector = ConflictDetector()
        self._resolver = StrategyResolver()
        self._logger = conflict_logger or ConflictLogger()

    @property
    def conflict_logger(self) -> ConflictLogger:
        return self._logger

    def resolve(
        self,
        node_id: str,
        incoming_data: dict,
        expected_version: int,
        strategy: Strategy = Strategy.LWW,
        force: bool = False,
    ) -> dict:
        """
        执行完整的冲突检测 + 消解流程。

        Args:
            node_id: 目标节点 ID。
            incoming_data: 待写入的新数据。
            expected_version: 写入方预期的版本号。
            strategy: 消解策略（默认 LWW）。
            force: 如果 True，跳过版本检查直接强行写入。

        Returns:
            {
                "conflict": bool,        # 是否发生了版本冲突
                "resolved": bool,        # 是否成功消解
                "strategy": str,         # 实际使用的策略
                "data": dict,            # 消解后的数据（可直接用于写入）
                "expected_version": int,  # 写入方版本
                "current_version": Optional[int],  # 数据库当前版本
            }
        """
        # 步骤 1: 版本检测
        has_conflict, current_node, current_version = self._detector.detect(
            self._kuzu_store, node_id, expected_version,
        )

        if not has_conflict or force:
            # 无冲突 或 强制写入 — 直接放行
            data = dict(incoming_data)
            data["id"] = node_id
            self._logger.record(
                node_id=node_id,
                expected_version=expected_version,
                current_version=current_version or 1,
                strategy=strategy,
                resolved=True,
                detail="no_conflict" if not has_conflict else "force_write",
            )
            return {
                "conflict": has_conflict,
                "resolved": True,
                "strategy": strategy.value,
                "data": data,
                "expected_version": expected_version,
                "current_version": current_version,
            }

        # 步骤 2: 策略消解
        resolved_data: dict = {}
        resolved_ok = True
        detail = ""

        try:
            if strategy == Strategy.LWW:
                resolved_data = self._resolver.resolve_lww(current_node, incoming_data)
                detail = f"lww: ts_cur={current_node.get('created_at',0)} ts_inc={incoming_data.get('created_at',0)}"
            elif strategy == Strategy.MERGE:
                resolved_data = self._resolver.resolve_merge(current_node, incoming_data)
                detail = f"merge: {len(incoming_data)} fields"
            elif strategy == Strategy.ADDITIVE:
                resolved_data = self._resolver.resolve_additive(current_node, incoming_data)
                detail = f"additive: appended version to history"
            else:
                resolved_data = dict(incoming_data)
                detail = f"unknown_strategy_fallback_lww"
        except Exception as e:
            resolved_ok = False
            resolved_data = dict(incoming_data)
            detail = f"resolve_error: {e}"
            logger.exception("Strategy resolution failed for node %s", node_id)

        # 确保 id 始终存在
        resolved_data["id"] = node_id

        # 步骤 3: 记录冲突日志
        self._logger.record(
            node_id=node_id,
            expected_version=expected_version,
            current_version=current_version or 1,
            strategy=strategy,
            resolved=resolved_ok,
            detail=detail,
        )

        return {
            "conflict": True,
            "resolved": resolved_ok,
            "strategy": strategy.value,
            "data": resolved_data,
            "expected_version": expected_version,
            "current_version": current_version,
        }

    def resolve_batch(
        self,
        items: list[tuple[str, dict, int, Strategy]],
    ) -> list[dict]:
        """
        批量消解 — 每个元素为 (node_id, incoming_data, expected_version, strategy)。

        Returns:
            与输入顺序对应的 resolve() 结果列表。
        """
        return [
            self.resolve(node_id=node_id, incoming_data=data,
                         expected_version=ver, strategy=s)
            for node_id, data, ver, s in items
        ]
