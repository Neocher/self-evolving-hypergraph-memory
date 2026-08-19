"""策略反馈环（阶段4-2，v6.0.0）。

FeedbackEngine.apply(rewards) 以评测 harness 的命中反馈驱动记忆策略升级：
节点成功计数 → 达阈值 → 升级 fact_track='core'（经 store.update_with_version 落库）。

- **不碰边权重/τ**（防答案泄漏）：仅升级节点事实轨标签
- **不进生产在线路径**：仅评测 harness 显式调用；检索侧 core ×1.1 boost 是
  既有机制（query_router._deduplicate_and_sort，v5.35.0），本引擎只负责把
  节点升级到 core 轨，生产 retrieve() 零改动
- **幂等**：已升级节点记 upgraded 集合，重复 apply 不重复升级；升级落库为
  force 写入（update_with_version expected_version=None），重复 SET 无副作用
"""
from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger("shm.feedback")


class FeedbackEngine:
    """节点成功计数 → 阈值升级 fact_track='core' 的策略反馈引擎。"""

    UPGRADE_THRESHOLD = 2  # 成功计数阈值：1 不升，2 升（设计 AC）

    def __init__(self, store, threshold: int = UPGRADE_THRESHOLD):
        if threshold < 1:
            raise ValueError(f"FeedbackEngine threshold={threshold} 必须 >= 1")
        self._store = store
        self.threshold = int(threshold)
        self._counts: dict[str, int] = {}
        self._upgraded: set[str] = set()

    @property
    def counts(self) -> dict[str, int]:
        """成功计数快照（测试断言用）。"""
        return dict(self._counts)

    @property
    def upgraded(self) -> set[str]:
        """已升级节点集合（幂等标记）。"""
        return set(self._upgraded)

    def apply(self, rewards: Iterable[tuple]) -> list[str]:
        """应用一轮评测反馈 → 返回本次升级的 node_id 列表。

        rewards: [(query, hit_node_ids, correct)]；仅 correct=True 计入成功计数。
        """
        for query, hit_node_ids, correct in rewards:
            if not correct:
                continue
            for nid in hit_node_ids or []:
                if nid is None or str(nid) in self._upgraded:
                    continue
                key = str(nid)
                self._counts[key] = self._counts.get(key, 0) + 1
        return self._flush_upgrades()

    def _flush_upgrades(self) -> list[str]:
        """达阈值的节点升级 fact_track='core'（force 写，幂等）。"""
        upgraded: list[str] = []
        for nid, count in list(self._counts.items()):
            if nid in self._upgraded or count < self.threshold:
                continue
            if self._store is None or not hasattr(self._store, "update_with_version"):
                continue
            try:
                ok = self._store.update_with_version(
                    nid, {"fact_track": "core"}, None
                )
            except Exception:
                logger.debug("feedback upgrade failed for %s", nid[:12], exc_info=True)
                continue
            if ok:
                self._upgraded.add(nid)
                upgraded.append(nid)
                logger.info("Feedback upgraded %s to core (count=%d)", nid[:12], count)
        return upgraded
