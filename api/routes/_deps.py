"""
共享依赖 — 服务容器、Router、全局状态、辅助函数
=============================================
所有路由模块从本文件导入共享依赖。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shm._version import __version__, __version_name__

import base64
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from api.models import (
    AuditOperation,
    AuditTrace,
    CommunityInfo,
    CommunityListResponse,
    DreamReport,
    DreamTriggerResponse,
    EpisodeCreate,
    EpisodeResponse,
    EpisodicResult,
    HealthStatus,
    HyperedgeCreate,
    HyperedgeListResponse,
    HyperedgeResponse,
    HyperedgeType as APIHyperedgeType,
    MultimodalRecord,
    MultimodalResponse,
    PromoteRequest,
    PromoteResponse,
    RetrieveRequest,
    RetrieveResponse,
    SearchVectorRequest,
    SearchVectorResult,
    SearchVectorResponse,
    SensoryRecord,
    SensoryResponse,
    # Ontology v2
    EntityTypeDefModel,
    EntityTypeListResponse,
    AttributeDefModel,
    EdgeTypeDefModel,
    EdgeTypeListResponse,
    OntologyStatsResponse,
    # v2.0: 工作记忆
    SessionMemoryCreate,
    SessionMemoryItem,
    SessionMemoryListResponse,
)
from graph.hyperedge import HyperedgeManager, HyperedgeType as CoreHyperedgeType
from observability.health import HealthChecker, HealthCheckResult
from observability.logger import get_logger, set_trace_id
from core.defense import MemoryDefenseEngine, DefenseConfig, MemoryDefenseVerdict
from graph.graphlite_store import EpisodeCache
from core.quarantine_store import QuarantineStore
from core.write_reconciler import WriteReconciler, ConflictLogger, Strategy
from core.write_queue import WriteQueueClosedError, WriteQueueFullError
from observability.metrics import (
    get_metrics,
    record_circuit_breaker,
    record_request,
)

from pydantic import BaseModel, Field

logger = get_logger(__name__)
router = APIRouter()


# ─── 服务容器 ──────────────────────────────────────────────


@dataclass
class Services:
    """依赖注入服务容器，由 app.py 在启动时构造。"""

    graphlite_store: Any = None
    faiss_index: Any = None
    faiss_dim: int = 512
    faiss_index_type: str = "IVFFlat"
    faiss_nlist: int = 100
    faiss_id_map: dict = field(default_factory=dict)
    tau_engine: Any = None
    hebbian_updater: Any = None
    ssm_gate: Any = None
    dream_scheduler: Any = None
    dream_pipeline: Any = None
    audit_chain: Any = None
    query_router: Any = None
    encoder: Any = None
    hyperedge_manager: Any = None
    ontology_validator: Any = None
    # 本体 v2（动态类型系统）
    ontology_v2: Any = None
    # 置信度追踪（Step 2）
    evidence_tracker: Any = None
    # FAISS 批量写入缓冲区
    _faiss_buffer: list[tuple] = field(default_factory=list)
    _faiss_buffer_lock: Any = None
    # v2.0: 工作记忆存储（会话级临时上下文）
    _session_memory: dict = field(default_factory=dict)
    _session_memory_lock: Any = None
    # v2.0: 自适应τ衰减配置
    _tau_adaptive_config: dict = field(default_factory=lambda: {
        "enable_adaptive": True,
        "importance_decay_modulator": 0.5,
        "tau_decay_min": 300,
        "tau_decay_max": 7200,
    })
    # 写入消解系统
    write_reconciler: Any = None
    # 事务管理器的引用（由 app.py 注入）
    tx_manager: Any = None
    # 记忆投毒防御系统
    defense_engine: Any = None
    quarantine_store: Any = None
    # v5.23: 写串行化队列（由 app.py 注入；None 时写路径回退同步直调）
    write_queue: Any = None
    # 【M5】检索侧 episode 内容缓存（EpisodeCache: OrderedDict LRU + TTL）。
    # 由 app.py 注入 query_router 共享引用；flush_faiss_buffer 是本缓存唯一写入方。
    _episode_cache: Any = None


_services: Optional[Services] = None


def init_services(svc: Services) -> None:
    """由 app.py 在启动时调用，注入服务容器。"""
    global _services
    svc._faiss_buffer_lock = threading.Lock()
    svc._session_memory_lock = threading.Lock()
    # 【M5】episode 内容缓存（LRU + TTL）与 query_router 共享引用
    if svc._episode_cache is None:
        svc._episode_cache = EpisodeCache()
    # 初始化记忆投毒防御引擎 + 隔离存储
    try:
        svc.defense_engine = MemoryDefenseEngine(config=DefenseConfig(), encoder=svc.encoder)
    except Exception:
        logger.warning("DefenseEngine init failed (non-fatal)")
    try:
        svc.quarantine_store = QuarantineStore(graph_store=svc.graphlite_store)
    except Exception:
        logger.warning("QuarantineStore init failed (non-fatal)")
    _services = svc


async def get_services() -> Services:
    """FastAPI Depends 注入点：返回服务容器。"""
    if _services is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    return _services


async def qsubmit(deps: Services, fn, *args, **kwargs) -> Any:
    """【v5.23】经写串行化队列提交同步 GraphLite 写调用。

    - 队列存在 → await WriteQueue.submit（写线程串行执行，不阻塞事件循环）
    - 队列不存在（测试/降级）→ 同步直调，行为与改造前一致
    - 队列满 / 已关闭 / 等待超时 → HTTPException 503（背压拒绝；超时后任务仍会落库，
      调用方不应安全重试——写入均以 uuid 主键天然幂等）
    """
    q = getattr(deps, "write_queue", None)
    if q is None:
        return fn(*args, **kwargs)
    try:
        return await q.submit(fn, *args, **kwargs)
    except (WriteQueueFullError, WriteQueueClosedError, asyncio.TimeoutError) as e:
        logger.warning("Write queue rejected (status=503): %s", e)
        raise HTTPException(status_code=503, detail=f"write queue busy: {e}") from e


# ─── 辅助函数 ──────────────────────────────────────────────


def _ok(data: dict) -> dict:
    return {"status": "ok", **data}


def _now() -> float:
    return time.time()


_FAISS_BATCH_SIZE = 50  # 攒够 50 条后批量写入 FAISS（避免每次写入都 flush）

# 【P6】异步 embedding 队列
_embed_queue: list[tuple[str, str, float]] = []  # (episode_id, content, created_at)
_embed_queue_lock: threading.Lock = threading.Lock()

# 【Perf】查询嵌入缓存（LRU, 避免重复编码）
_embed_cache: dict[str, Any] = {}  # query_text → numpy vector
_embed_cache_max: int = 256

# 【Perf】检索结果缓存（相同query+top_k直接返回）
_result_cache: dict[str, Any] = {}  # f"{query}:{top_k}" → RetrieveResponse
_result_cache_max: int = 128
_result_cache_lock: threading.Lock = threading.Lock()


def flush_faiss_buffer(deps: Services) -> int:
    """
    将 FAISS 缓冲区中的待写入项批量写入索引。

    Returns:
        实际写入的数量
    """
    if not deps._faiss_buffer or deps.faiss_index is None:
        return 0
    with deps._faiss_buffer_lock:
        batch = deps._faiss_buffer[:]
        deps._faiss_buffer.clear()
    if not batch:
        return 0
    try:
        ids = np.array([item[0] for item in batch], dtype=np.int64)
        vecs = np.array([item[1] for item in batch], dtype=np.float32)
        deps.faiss_index.add_with_ids(vecs, ids)
        # 更新 id_map
        if hasattr(deps, "faiss_id_map") and deps.faiss_id_map is not None:
            for faiss_id, _emb, ep_id in batch:
                deps.faiss_id_map[int(faiss_id)] = ep_id
        # 【M5】episode 内容缓存填充：faiss_id_map 与 _episode_cache 均为
        # query_router 共享引用，本批写入的节点内容在此预填，检索零回查。
        # 【Core-Boost】填充时带上 fact_track（一次性批量回查），否则 L1 缓存
        # 命中路径 core 节点 fact_track 恒 "active" 丢失 boost。
        cache = getattr(deps, "_episode_cache", None)
        if cache is None:
            # 未显式初始化（测试/降级路径）→ 惰性创建，保证 flush 恒写入
            cache = EpisodeCache()
            deps._episode_cache = cache
        fact_tracks: dict[str, str] = {}
        store = getattr(deps, "graphlite_store", None)
        if store is not None and hasattr(store, "get_episodes_batch"):
            ep_ids = [ep_id for _faiss_id, _emb, ep_id in batch]
            try:
                fact_tracks = {
                    str(ep.get("id", "")): ep.get("fact_track", "active")
                    for ep in store.get_episodes_batch(ep_ids)
                    if isinstance(ep, dict)
                }
            except Exception:
                fact_tracks = {}  # 回查失败 → 缺省 active，不阻断 flush
        for _faiss_id, _emb, ep_id in batch:
            try:
                cache[ep_id] = {"id": ep_id, "fact_track": fact_tracks.get(ep_id, "active")}
            except Exception:
                break
        logger.debug("FAISS batch flush: %d vectors added", len(batch))
        return len(batch)
    except Exception:
        logger.exception("FAISS batch flush failed, %d vectors pending", len(batch))
        # 重新放回缓冲区
        with deps._faiss_buffer_lock:
            deps._faiss_buffer.extend(batch)
        return 0


def incremental_faiss_update(deps: Services, removed_node_ids: list[str]) -> int:
    """
    梦境后增量更新 FAISS 索引：删除被 PRUNE/RESOLVE 移除的节点向量。

    Args:
        deps: 服务容器
        removed_node_ids: 从 GraphLite 删除的 EpisodeNode ID 列表

    Returns:
        实际从 FAISS 删除的向量数
    """
    if not removed_node_ids or deps.faiss_index is None:
        return 0
    try:

        removed = [int(uuid.uuid5(uuid.NAMESPACE_OID, str(nid)).int & ((1 << 63) - 1))
                   for nid in removed_node_ids]
        id_selector = np.array(removed, dtype=np.int64)
        removed_count = deps.faiss_index.remove_ids(id_selector)

        # 同步 faiss_id_map
        if hasattr(deps, "faiss_id_map") and deps.faiss_id_map is not None:
            remove_set = set(removed)
            deps.faiss_id_map = {
                k: v for k, v in deps.faiss_id_map.items()
                if k not in remove_set
            }

        logger.info("FAISS incremental update: %d vectors removed (of %d requested)",
                     removed_count, len(removed_node_ids))
        return removed_count
    except Exception:
        logger.exception("FAISS incremental update failed, %d nodes pending cleanup",
                         len(removed_node_ids))
        return 0


# ─── Cypher 查询代理 ──────────────────────────────────────


class CypherQueryRequest(BaseModel):
    query: str = Field(..., description="Cypher 查询语句")
    params: dict = Field(default_factory=dict, description="查询参数")
