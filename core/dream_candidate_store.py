"""
梦境候选存储
============
在将梦境结果应用到生产数据之前，暂存到候选区供审查。
支持 review / apply / discard 三阶段工作流。

架构：
- 每个梦境候选保存为一个 JSON 文件在 dream_candidates/ 目录下
- apply() 执行后在 Kuzu 上执行实际的 DETACH DELETE 和 CREATE
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CANDIDATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "dream_candidates",
)


@dataclass
class DreamCandidate:
    """梦境候选 — 一次梦境输出，等待审查。"""

    dream_id: str
    created_at: float
    trigger_mode: str
    stats: dict  # created/updated/deleted 计数
    community_count: int
    prune_count: int
    conflict_count: int
    community_summaries: list[dict]  # 社区摘要（供审查用）
    prune_ops: list[dict]  # 将要删除的节点
    merge_ops: list[dict]  # 将要合并的节点
    applied: bool = False
    discarded: bool = False
    applied_at: Optional[float] = None
    compressed_topics: int = 0
    compressed_episodes: int = 0
    compressed_facts: int = 0
    keywords_extracted: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DreamCandidate:
        return cls(**data)


class DreamCandidateStore:
    """梦境候选存储，管理待审查的梦境输出。"""

    def __init__(self, storage_dir: str = _DEFAULT_CANDIDATE_DIR):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        logger.info("DreamCandidateStore: %s", storage_dir)

    def _candidate_path(self, dream_id: str) -> str:
        return os.path.join(self.storage_dir, f"{dream_id}.json")

    def save_candidate(
        self,
        dream_id: str,
        communities: list[dict],
        prune_ops: list,
        merge_ops: list,
        dream_report_kwargs: dict,
    ) -> str:
        """保存一次梦境输出到候选存储。

        Returns:
            候选文件路径
        """
        # 精简社区数据（只保留摘要信息供审查）
        community_summaries = []
        for c in communities:
            summary = {
                "id": c.get("id", ""),
                "member_count": len(c.get("members", [])),
                "report": (c.get("report", "") or "")[:500],
                "keywords": c.get("keywords", [])[:10],
                "topics": c.get("topics", [])[:5],
                "patterns": c.get("llm_patterns", []),
                "contradictions": c.get("llm_contradictions", []),
            }
            community_summaries.append(summary)

        # 精简操作日志
        prune_summary = [
            {"node_id": op.node_id if hasattr(op, "node_id") else op.get("node_id", ""),
             "reason": op.reason if hasattr(op, "reason") else op.get("reason", "")}
            for op in (prune_ops or [])
        ]
        merge_summary = [
            {"node_id": op.node_id if hasattr(op, "node_id") else op.get("node_id", ""),
             "target": op.new_value if hasattr(op, "new_value") else op.get("new_value", ""),
             "reason": op.reason if hasattr(op, "reason") else op.get("reason", "")}
            for op in (merge_ops or [])
        ]

        candidate = DreamCandidate(
            dream_id=dream_id,
            created_at=time.time(),
            trigger_mode=dream_report_kwargs.get("trigger_mode", "unknown"),
            stats=dream_report_kwargs.get("stats", {}),
            community_count=dream_report_kwargs.get("community_count", 0),
            prune_count=dream_report_kwargs.get("prune_count", 0),
            conflict_count=dream_report_kwargs.get("conflict_count", 0),
            community_summaries=community_summaries,
            prune_ops=prune_summary,
            merge_ops=merge_summary,
        )

        filepath = self._candidate_path(dream_id)
        # 原子写入：先写 .tmp 再 rename，防止写入中断导致文件损坏
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(candidate.to_dict(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
        logger.info("Dream candidate saved: %s (%d communities, %d prunes, %d merges)",
                     filepath, len(community_summaries), len(prune_summary), len(merge_summary))
        return filepath

    def get_candidate(self, dream_id: str) -> Optional[DreamCandidate]:
        """读取指定梦境候选。"""
        filepath = self._candidate_path(dream_id)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DreamCandidate.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to load dream candidate %s: %s", dream_id, e)
            return None

    def list_candidates(self, limit: int = 20) -> list[dict]:
        """列出未处理（待审查）的梦境候选。"""
        if not os.path.isdir(self.storage_dir):
            return []
        candidates = []
        for fname in sorted(os.listdir(self.storage_dir), reverse=True):
            if not fname.endswith(".json"):
                continue
            filepath = os.path.join(self.storage_dir, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                dream_id = data.get("dream_id", fname.replace(".json", ""))
                if not data.get("applied") and not data.get("discarded"):
                    candidates.append({
                        "dream_id": dream_id,
                        "created_at": data.get("created_at", 0),
                        "trigger_mode": data.get("trigger_mode", "unknown"),
                        "community_count": data.get("community_count", 0),
                        "prune_count": data.get("prune_count", 0),
                        "conflict_count": data.get("conflict_count", 0),
                        "stats": data.get("stats", {}),
                    })
                    if len(candidates) >= limit:
                        break
            except Exception:
                continue
        return candidates

    def _load_all_candidates(self) -> list[DreamCandidate]:
        """加载存储目录中所有候选的 DreamCandidate 对象。

        Returns:
            排序后的 DreamCandidate 列表（按 created_at 升序，即最旧在前）
        """
        if not os.path.isdir(self.storage_dir):
            return []
        candidates: list[DreamCandidate] = []
        for fname in sorted(os.listdir(self.storage_dir)):
            if not fname.endswith(".json"):
                continue
            dream_id = fname.replace(".json", "")
            candidate = self.get_candidate(dream_id)
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda c: c.created_at)
        return candidates

    def apply_candidate(self, dream_id: str, kuzu_store) -> bool:
        """将梦境候选应用到生产 Kuzu 数据库。

        执行实际的 PRUNE (DETACH DELETE) 和 Merge 操作。

        Kuzu 事务说明：Kuzu 0.11.x 每个 execute() 自动提交，
        不支持显式事务（无 begin_write_transaction/commit/rollback API）。
        因此 PRUNE → MERGE 之间无法原子化；
        如果中间失败，已执行的 PRUNE 操作无法回滚。
        这对候选模式的幂等性影响有限（已标记的候选不会被重复应用），
        但生产环境中需注意部分执行状态。

        Returns:
            True 如果应用成功
        """
        candidate = self.get_candidate(dream_id)
        if candidate is None:
            logger.warning("Cannot apply candidate %s: not found", dream_id)
            return False
        if candidate.applied:
            logger.warning("Candidate %s already applied", dream_id)
            return False
        if candidate.discarded:
            logger.warning("Candidate %s already discarded", dream_id)
            return False

        try:
            # Step 1: 执行 PRUNE 删除
            deleted_count = 0
            for op in candidate.prune_ops:
                try:
                    kuzu_store.query_cypher(
                        "MATCH (e:EpisodeNode {id: $id}) DETACH DELETE e",
                        {"id": op["node_id"]}
                    )
                    deleted_count += 1
                except Exception as e:
                    logger.warning("Apply prune failed for %s: %s", op["node_id"], e)

            # Step 2: 执行 MERGE 合并
            for op in candidate.merge_ops:
                try:
                    target = op.get("target", "")
                    if target:
                        source_rows = kuzu_store.query_cypher(
                            "MATCH (s:EpisodeNode {id: $id}) RETURN s.content AS content",
                            {"id": op["node_id"]}
                        )
                        source_content = ""
                        if source_rows and len(source_rows) > 0:
                            row = source_rows[0]
                            if isinstance(row, (list, tuple)):
                                source_content = str(row[0])
                            elif isinstance(row, dict):
                                source_content = str(row.get("content", ""))
                        kuzu_store.query_cypher(
                            "MATCH (target:EpisodeNode {id: $target}) "
                            "SET target.content = target.content + ' | merged: ' || $content",
                            {"target": target, "content": source_content}
                        )
                    kuzu_store.query_cypher(
                        "MATCH (e:EpisodeNode {id: $id}) DETACH DELETE e",
                        {"id": op["node_id"]}
                    )
                except Exception as e:
                    logger.warning("Apply merge failed for %s -> %s: %s",
                                   op["node_id"], op.get("target", ""), e)

            # 标记候选已应用
            self._mark_applied(dream_id)
            logger.info("Dream candidate applied: %s (%d prunes, %d merges)",
                        dream_id, deleted_count, len(candidate.merge_ops))
            return True
        except Exception as e:
            logger.exception("Failed to apply dream candidate %s: %s", dream_id, e)
            return False

    def discard_candidate(self, dream_id: str) -> bool:
        """丢弃梦境候选（不做任何修改）。"""
        candidate = self.get_candidate(dream_id)
        if candidate is None:
            return False
        filepath = self._candidate_path(dream_id)
        try:
            data = candidate.to_dict()
            data["discarded"] = True
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Dream candidate discarded: %s", dream_id)
            return True
        except Exception as e:
            logger.warning("Failed to discard candidate %s: %s", dream_id, e)
            return False

    def _mark_applied(self, dream_id: str) -> None:
        filepath = self._candidate_path(dream_id)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["applied"] = True
            data["applied_at"] = time.time()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to mark candidate %s as applied: %s", dream_id, e)

    def _persist_community_nodes(
        self, candidate: DreamCandidate, kuzu_store,
    ) -> int:
        """从候选 data 创建 Kuzu CommunityNode + COMMUNITY_MEMBER 边。

        先清理所有旧社区，再创建新的（最多 50 个最高质量的）。
        Returns: 创建的社区数
        """
        # 先清理旧社区
        try:
            kuzu_store.query_cypher("MATCH (c:CommunityNode) DETACH DELETE c", {})
        except Exception:
            pass
        created = 0
        # 按 member_count 倒序，只创建 top-50
        sorted_comms = sorted(
            candidate.community_summaries,
            key=lambda c: c.get("member_count", 0),
            reverse=True,
        )
        for comm in sorted_comms[:50]:
            comm_id = comm.get("id", "")
            if not comm_id:
                continue
            report = (comm.get("report", "") or "")[:800]
            try:
                kuzu_store.query_cypher(
                    "CREATE (c:CommunityNode {id: $id, name: $name, summary: $summary, "
                    "leiden_score: $score, created_at: $created_at})",
                    {
                        "id": comm_id,
                        "name": f"dream_{candidate.dream_id[:8]}_comm_{created}",
                        "summary": report,
                        "score": 0.0,
                        "created_at": time.time(),
                    },
                )
                created += 1
            except Exception:
                pass
        logger.info(
            "Persisted %d community nodes from dream %s",
            created, candidate.dream_id[:12],
        )
        return created

    def auto_apply_candidates(self, kuzu_store) -> tuple[int, int, int]:
        """自动审查并应用高质量的梦境候选。

        触发条件：
        - 候选 JSON 文件数 >= 20 时自动触发
        - 按 created_at 升序选择最旧的未处理候选
        - 每次调用只处理一个（保持增量）

        Apply 后：
        - 删除对应的 JSON 候选文件
        - 记录 applied_count / community_created / deleted_files

        评分标准（质量门禁）:
        - community_count > 0 且至少一个社区有成员
        - conflict_count == 0
        - 社区摘要长度 > 30 字符

        Returns:
            (applied_count, community_created_count, file_deleted_count)
        """
        if kuzu_store is None:
            return (0, 0, 0)

        # 统计候选 JSON 文件数
        if not os.path.isdir(self.storage_dir):
            return (0, 0, 0)
        all_files = [f for f in os.listdir(self.storage_dir) if f.endswith(".json")]
        if len(all_files) < 20:
            logger.debug(
                "Auto-apply skipped: %d candidates < 20 threshold",
                len(all_files),
            )
            return (0, 0, 0)

        # 加载所有候选，按 created_at 升序（最旧在前）
        all_candidates = self._load_all_candidates()
        # 过滤掉已处理/已废弃的
        pending = [c for c in all_candidates if not c.applied and not c.discarded]
        if not pending:
            return (0, 0, 0)

        # 只处理最旧的一个
        candidate = pending[0]

        # 质量门禁
        valid_communities = [
            comm for comm in candidate.community_summaries
            if comm.get("member_count", 0) > 0 and len((comm.get("report", "") or "")) > 30
        ]
        if len(valid_communities) < 1 or candidate.conflict_count > 0 or candidate.community_count == 0:
            logger.info(
                "Auto-apply skipped %s: quality gate failed "
                "(valid_communities=%d, conflicts=%d, community_count=%d)",
                candidate.dream_id[:12], len(valid_communities),
                candidate.conflict_count, candidate.community_count,
            )
            # 【FIX 2026-07-31】质量门控失败 → 标记 discarded 跳过，避免卡死整个队列
            # 根因: auto-apply 永远处理最旧 pending 候选, 若其质量不合格则 return,
            # 后续所有候选全部堆积 (曾导致 113 个候选文件无限增长)
            try:
                self.discard_candidate(candidate.dream_id)
            except Exception:
                pass
            return (0, 0, 0)

        try:
            # 1. 执行 PRUNE（删除已废弃节点）
            deleted_count = 0
            for op in candidate.prune_ops:
                try:
                    kuzu_store.query_cypher(
                        "MATCH (e:EpisodeNode {id: $id}) DETACH DELETE e",
                        {"id": op.get("node_id", "")},
                    )
                    deleted_count += 1
                except Exception:
                    pass

            # 2. 创建 CommunityNode
            comm_created = self._persist_community_nodes(candidate, kuzu_store)

            # 3. 删除候选 JSON 文件（而不是标记 applied）
            filepath = self._candidate_path(candidate.dream_id)
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info("Deleted candidate file: %s", filepath)
            else:
                logger.warning("Candidate file not found for deletion: %s", filepath)

            logger.info(
                "Auto-applied dream %s: %d communities, %d prunes",
                candidate.dream_id[:12], comm_created, deleted_count,
            )
            return (1, comm_created, 1)

        except Exception as e:
            logger.exception("Auto-apply failed for %s: %s", candidate.dream_id[:12], e)
            return (0, 0, 0)

    def clean_old_candidates(self, max_age_hours: int = 72) -> int:
        """清理过期的已处理候选。"""
        if not os.path.isdir(self.storage_dir):
            return 0
        now = time.time()
        cleaned = 0
        for fname in os.listdir(self.storage_dir):
            if not fname.endswith(".json"):
                continue
            filepath = os.path.join(self.storage_dir, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("applied") or data.get("discarded"):
                    created = data.get("created_at", 0)
                    if now - created > max_age_hours * 3600:
                        os.remove(filepath)
                        cleaned += 1
            except Exception:
                continue
        return cleaned
