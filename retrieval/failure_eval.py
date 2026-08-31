"""
失败查询评估集构建器 (Failed-Query Eval Builder) — Phase 2 失败驱动闭环
======================================================================
把 FailureLogger 的失败快照转化为配对 held-out 评估集，激活 Phase 1
core/validation_gate.held_out_paired_gate（Recuris「模型提议，算术裁决」）：

  - 提取失败查询集：quality() < 阈值 或 degraded 的快照，按 query 去重，
    排除不可重放的合成/占位快照（source=="probe"、query 为空或占位符）
  - 重放：在给定 EvolvableParams 下对每个失败查询调 QueryRouter.retrieve()
    N 次（默认 N=3，seed/temperature 变化语义），每次复用
    RetrievalSnapshot.quality() 作为分数口径（0~1 连续分数）
  - build_heldout_scores(base_params, cand_params) -> {"base": {qid: [scores]},
    "cand": {qid: [scores]}} —— base/cand 同一批 held-out item 配对比较

纯逻辑 + 只读检索调用：不修改底层 QueryRouter；重放前后恢复原 cfg（try/finally）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Optional

from retrieval.self_evolving import RetrievalSnapshot

logger = logging.getLogger(__name__)

# 与 self_evolving._sync_params 一致的 EvolvableParams → QueryRouterConfig 字段映射
_PARAM_FIELDS = [
    "weight_fusion_vector", "weight_fusion_bm25", "weight_fusion_entity",
    "tau_weight", "vector_weight",
    "top_k_l1", "top_k_fusion", "top_k_keyword", "top_k_vector",
    "bm25_k1", "bm25_b", "mesa_boost",
]

# 失败判定阈值（与 FailureLogger.quality_threshold 默认一致）
_QUALITY_THRESHOLD = 0.4

# 不可重放的查询（梦境探针合成快照的占位 query 等）
_PLACEHOLDER_QUERIES = {"", "<health-probe>"}


def query_id(query: str) -> str:
    """稳定 item id：query 文本的 sha1 前缀（去重键，跨快照稳定）。"""
    return hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]


class FailedQueryEval:
    """失败查询评估集构建器。

    Args:
        logger: FailureLogger（失败快照源，snapshots deque）
        query_router: QueryRouter（只读重放目标；重放时临时改 config，完后恢复）
        n_seeds: 每个查询重放次数（per-seed 分数），默认 3
        quality_threshold: quality() < 阈值或 degraded 视为失败查询
        persist_path: data/failure_queries.json 持久化路径（None → 仓库 data/ 下）
    """

    def __init__(self, logger, query_router, n_seeds: int = 3,
                 quality_threshold: float = _QUALITY_THRESHOLD,
                 persist_path: Optional[str] = None):
        self._logger = logger
        self._qr = query_router
        self.n_seeds = max(1, int(n_seeds))
        self._quality_threshold = quality_threshold
        self._persist_path = persist_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "failure_queries.json",
        )

    # ─── 失败查询集提取 ───

    def _is_failure(self, snap: RetrievalSnapshot) -> bool:
        """失败判定：quality() < 阈值 或 degraded（硬失败信号不被质量门槛吞掉）。"""
        return snap.degraded or snap.quality() < self._quality_threshold

    def extract_failed_queries(self) -> dict:
        """从 FailureLogger 快照提取失败查询集（去重 + 排除不可重放）。

        Returns: {qid: {"query", "num_results", "avg_score", "quality",
                        "source", "first_failed_at"}}
        """
        items: dict = {}
        for snap in list(self._logger.snapshots):
            if not self._is_failure(snap):
                continue
            # 合成/占位快照无法作为真实查询重放（probe 快照 query="<health-probe>"）
            if snap.source == "probe" or snap.query in _PLACEHOLDER_QUERIES:
                continue
            qid = query_id(snap.query)
            if qid in items:
                continue  # 去重：同一查询只保留首个失败快照
            items[qid] = {
                "query": snap.query,
                "num_results": snap.num_results,
                "avg_score": snap.avg_score,
                "quality": round(snap.quality(), 4),
                "source": snap.source,
                "first_failed_at": snap.timestamp,
            }
        return items

    def persist_failed_queries(self, items: Optional[dict] = None) -> bool:
        """原子写失败查询集到 data/failure_queries.json（mkstemp + os.replace，同 save_state 模式）。"""
        items = items if items is not None else self.extract_failed_queries()
        try:
            os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=os.path.dirname(self._persist_path) or ".", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self._persist_path)
                return True
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning("Failure queries persist failed: %s", e)
            return False

    # ─── 重放 ───

    def _sync_config(self, params) -> None:
        """把 EvolvableParams 临时写入 QueryRouter.config（与 _sync_params 同字段映射）。"""
        cfg = self._qr.config
        for f in _PARAM_FIELDS:
            if hasattr(cfg, f):
                setattr(cfg, f, getattr(params, f))

    def _snapshot_from_result(self, query: str, results) -> RetrievalSnapshot:
        """把一次检索结果转成临时快照，复用 RetrievalSnapshot.quality() 口径。"""
        raw = results if isinstance(results, list) else results.get("results", [])
        scores = [r.get("score", 0.5) for r in raw[:10]] if raw else [0.0]
        contents = [r.get("content", "")[:100] for r in raw[:10]]
        return RetrievalSnapshot(
            timestamp=time.time(),
            query=query,
            params_before={},
            num_results=len(raw),
            top_scores=scores[:5],
            avg_score=sum(scores) / max(1, len(scores)),
            top_distinct=len(set(contents)),
            latency_ms=0.0,
            degraded=len(raw) == 0,
        )

    def replay_queries(self, items: dict, params) -> dict:
        """在给定参数下重放 item 集，返回 {qid: [per-seed 分数]}。

        只读检索调用；重放前后恢复原 cfg（try/finally）；单次重放异常
        记 0 分（degraded 语义），不中断整批。
        """
        if not items:
            return {}
        cfg = self._qr.config
        saved = {f: getattr(cfg, f) for f in _PARAM_FIELDS if hasattr(cfg, f)}
        try:
            self._sync_config(params)
            out: dict = {}
            for qid, meta in items.items():
                scores = []
                for _ in range(self.n_seeds):
                    try:
                        res = self._qr.retrieve(meta["query"])
                    except Exception:
                        res = []  # 重放失败 → degraded 语义（quality 0），不中断
                    snap = self._snapshot_from_result(meta["query"], res)
                    scores.append(round(snap.quality(), 4))
                out[qid] = scores
            return out
        finally:
            for f, v in saved.items():
                setattr(cfg, f, v)

    def replay(self, params) -> dict:
        """提取失败查询集并在给定参数下重放（便捷入口）。"""
        return self.replay_queries(self.extract_failed_queries(), params)

    # ─── 配对评估集 ───

    def build_heldout_scores(self, base_params, cand_params) -> dict:
        """构建配对 held-out 分数集：{"base": {qid: [scores]}, "cand": {qid: [scores]}}。

        同一批失败查询（held-out item）分别在 base/cand 两套参数下重放
        N 次取 per-seed 分数；失败查询集同步持久化到 data/failure_queries.json
        （持久化失败仅告警，不阻断判定）。
        """
        items = self.extract_failed_queries()
        if items:
            self.persist_failed_queries(items)
        return {
            "base": self.replay_queries(items, base_params),
            "cand": self.replay_queries(items, cand_params),
        }
