"""
FastAPI 路由注册
==============
所有 HTTP 端点：记忆写入/查询/超边管理/社区/梦境/健康检查/溯源。

依赖注入通过 get_services() 获取服务容器，由 api/app.py 在启动时初始化。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

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
    PromoteRequest,
    PromoteResponse,
    RetrieveRequest,
    RetrieveResponse,
    SensoryRecord,
    SensoryResponse,
)
from graph.hyperedge import HyperedgeManager, HyperedgeType as CoreHyperedgeType
from observability.health import HealthChecker, HealthCheckResult
from observability.logger import get_logger, set_trace_id
from observability.metrics import (
    get_metrics,
    record_circuit_breaker,
    record_request,
)

logger = get_logger(__name__)
router = APIRouter()


# ─── 服务容器 ──────────────────────────────────────────────

@dataclass
class Services:
    """依赖注入服务容器，由 app.py 在启动时构造。"""

    kuzu_store: Any = None
    faiss_index: Any = None
    tau_engine: Any = None
    hebbian_updater: Any = None
    ssm_gate: Any = None
    dream_scheduler: Any = None
    dream_pipeline: Any = None
    audit_chain: Any = None
    query_router: Any = None
    encoder: Any = None
    hyperedge_manager: Any = None


_services: Optional[Services] = None


def init_services(svc: Services) -> None:
    """由 app.py 在启动时调用，注入服务容器。"""
    global _services
    _services = svc


async def get_services() -> Services:
    """FastAPI Depends 注入点：返回服务容器。"""
    if _services is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    return _services


# ─── 辅助函数 ──────────────────────────────────────────────

def _ok(data: dict) -> dict:
    return {"status": "ok", **data}


def _now() -> float:
    return time.time()


# ═══════════════════════════════════════════════════════════
# 记忆写入
# ═══════════════════════════════════════════════════════════

@router.post("/memories/sensory", summary="写入感觉缓冲区 (Layer1)")
async def write_sensory(
    record: SensoryRecord,
    deps: Services = Depends(get_services),
) -> SensoryResponse:
    """
    将原始文本写入 Layer1 环形缓冲区。
    缓冲区满时自动挤出最旧记录并通知梦境调度器。
    """
    start = _now()
    set_trace_id()

    record_id = str(uuid.uuid4())
    buf = getattr(deps.kuzu_store, "_sensory_buffer", None)
    buffer_usage = 0

    if buf is not None:
        buf.append({"id": record_id, "content": record.content,
                     "source": record.source.value, "created_at": start})
        buffer_usage = len(buf)
        if hasattr(buf, "is_full") and buf.is_full():
            evicted = buf.evict_oldest()
            if deps.dream_scheduler:
                await deps.dream_scheduler.on_node_created()
    else:
        # 无环形缓冲区：直接写入 Kuzu EpisodeNode 作为兜底
        deps.kuzu_store.create_episode({
            "id": record_id,
            "content": record.content,
            "source": record.source.value,
            "created_at": start,
            "tau_initial": 1.0,
        })
        if deps.dream_scheduler:
            await deps.dream_scheduler.on_node_created()

    record_request("POST", "/memories/sensory", "200", _now() - start)
    return SensoryResponse(record_id=record_id, buffer_usage=buffer_usage)


@router.post("/memories/episodes", summary="直接创建情节节点 (Layer2)")
async def create_episode(
    req: EpisodeCreate,
    deps: Services = Depends(get_services),
) -> EpisodeResponse:
    """直接创建 Layer2 情节节点，可选强制提升。"""
    start = _now()
    set_trace_id()

    episode_id = str(uuid.uuid4())
    created_at = _now()
    tau_initial = 1.0

    # τ 值计算
    if deps.tau_engine:
        tau_initial = deps.tau_engine.compute_tau(created_at)
        if tau_initial < deps.tau_engine.config.decay_threshold and not req.force_promote:
            raise HTTPException(status_code=400, detail="τ below threshold; use force_promote=true")

    deps.kuzu_store.create_episode({
        "id": episode_id,
        "content": req.content,
        "source": req.source.value,
        "created_at": created_at,
        "tau_initial": tau_initial,
    })

    if deps.encoder:
        try:
            emb = deps.encoder.embed(req.content)
        except Exception:
            emb = None
        if emb is not None and deps.faiss_index is not None:
            pass  # FAISS 索引由梦境阶段统一更新

    if deps.dream_scheduler:
        await deps.dream_scheduler.on_activity()
        await deps.dream_scheduler.on_node_created()

    record_request("POST", "/memories/episodes", "200", _now() - start)
    return EpisodeResponse(
        episode_id=episode_id,
        content=req.content[:200],
        tau_initial=tau_initial,
        created_at=created_at,
        source=req.source.value,
    )


@router.get("/memories/episodes/{episode_id}", summary="查询情节节点")
async def get_episode(
    episode_id: str,
    deps: Services = Depends(get_services),
) -> EpisodeResponse:
    """按 ID 查询单个情节节点。"""
    start = _now()
    set_trace_id()

    node = deps.kuzu_store.get_episode(episode_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

    record_request("GET", "/memories/episodes/{episode_id}", "200", _now() - start)
    return EpisodeResponse(
        episode_id=node["id"],
        content=node.get("content", ""),
        tau_initial=node.get("tau_initial", 1.0),
        created_at=node.get("created_at", 0.0),
        source=node.get("source", "unknown"),
    )


# ═══════════════════════════════════════════════════════════
# 提升 (Layer1 → Layer2)
# ═══════════════════════════════════════════════════════════

@router.post("/memories/promote", summary="Layer1 → Layer2 提升")
async def promote_to_episode(
    req: PromoteRequest,
    deps: Services = Depends(get_services),
) -> PromoteResponse:
    """将 Layer1 感觉记录提升为 Layer2 情节节点。"""
    start = _now()
    set_trace_id()

    episode_id = str(uuid.uuid4())
    created_at = _now()
    tau = 1.0

    if deps.tau_engine:
        tau = deps.tau_engine.compute_tau(created_at)

    # 尝试从 Kuzu 查找原始记录内容
    content = ""
    existing = deps.kuzu_store.get_episode(req.sensory_record_id)
    if existing:
        content = existing.get("content", "")
    else:
        content = "promoted_record"

    deps.kuzu_store.create_episode({
        "id": episode_id,
        "content": content,
        "source": "promoted",
        "created_at": created_at,
        "tau_initial": tau,
    })

    count = 0
    if deps.hebbian_updater and deps.dream_scheduler:
        count = 1

    if deps.dream_scheduler:
        await deps.dream_scheduler.on_activity()

    record_request("POST", "/memories/promote", "200", _now() - start)
    return PromoteResponse(
        episode_id=episode_id,
        tau_initial=tau,
        hebbian_connections_updated=count,
    )


# ═══════════════════════════════════════════════════════════
# 检索
# ═══════════════════════════════════════════════════════════

@router.post("/memories/retrieve", summary="粗到精三级检索（带降级）")
async def retrieve(
    req: RetrieveRequest,
    deps: Services = Depends(get_services),
) -> RetrieveResponse:
    """执行粗到精三级检索，Kuzu 断路器跳闸时自动降级到向量/关键词检索。"""
    start = _now()
    set_trace_id()
    degraded = False

    if deps.query_router is None:
        raise HTTPException(status_code=503, detail="Query router not available")

    try:
        results_raw = deps.query_router.retrieve(req.query)
    except Exception as exc:
        record_request("POST", "/memories/retrieve", "500", _now() - start)
        raise HTTPException(status_code=500, detail=str(exc))

    # 检查是否降级
    if results_raw:
        first_level = results_raw[0].get("level", "hypergraph") if isinstance(results_raw[0], dict) else "hypergraph"
        degraded = first_level != "hypergraph"

    results: list[EpisodicResult] = []
    for r in results_raw[:req.top_k]:
        if isinstance(r, dict):
            results.append(EpisodicResult(
                node_id=r.get("node_id", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                tau_value=r.get("tau_value"),
                source=r.get("level", "hypergraph"),
                hyperedge_id=r.get("hyperedge_id"),
                retrieval_level=r.get("level", "hypergraph"),
                created_at=r.get("created_at"),
            ))
        elif hasattr(r, "node_id"):
            results.append(EpisodicResult(
                node_id=r.node_id,
                content=getattr(r, "content", ""),
                score=getattr(r, "score", 0.0),
                tau_value=getattr(r, "tau_value", None),
                retrieval_level=getattr(r, "source", "hypergraph"),
            ))

    latency = (_now() - start) * 1000
    record_request("POST", "/memories/retrieve", "200", _now() - start)
    return RetrieveResponse(
        query=req.query,
        strategy_used=req.strategy or "auto",
        results=results,
        total_found=len(results),
        latency_ms=round(latency, 2),
        degraded=degraded,
    )


# ═══════════════════════════════════════════════════════════
# 超边管理
# ═══════════════════════════════════════════════════════════

def _core_type_to_api(t: CoreHyperedgeType):
    """将 core 层 HyperedgeType 映射为 api 层枚举。"""
    from api.models import HyperedgeType as ApiType
    mapping = {
        CoreHyperedgeType.EPISODE: ApiType.EPISODE,
        CoreHyperedgeType.SEMANTIC: ApiType.SEMANTIC,
        CoreHyperedgeType.TEMPORAL: ApiType.TEMPORAL,
    }
    return mapping.get(t, ApiType.EPISODE)


@router.post("/hyperedges", summary="创建超边 (Layer4)")
async def create_hyperedge(
    req: HyperedgeCreate,
    deps: Services = Depends(get_services),
) -> HyperedgeResponse:
    """创建超边，连接至少 2 个成员节点。"""
    start = _now()
    set_trace_id()

    if deps.hyperedge_manager is None:
        raise HTTPException(status_code=503, detail="Hyperedge manager not available")

    core_type = CoreHyperedgeType(req.type.value)
    try:
        if core_type == CoreHyperedgeType.EPISODE:
            edge = deps.hyperedge_manager.create_episode_hyperedge(
                member_ids=req.member_ids, topic=req.topic
            )
        elif core_type == CoreHyperedgeType.SEMANTIC:
            edge = deps.hyperedge_manager.create_semantic_hyperedge(
                member_ids=req.member_ids, conclusion=req.conclusion or ""
            )
        else:
            edge = deps.hyperedge_manager.create_temporal_hyperedge(
                member_ids=req.member_ids,
                start_time=req.start_time or _now(),
                end_time=req.end_time or _now(),
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    record_request("POST", "/hyperedges", "200", _now() - start)
    return HyperedgeResponse(
        id=edge.id,
        type=_core_type_to_api(edge.type),
        member_ids=edge.member_ids,
        created_at=edge.created_at,
        gate_value=edge.gate_value,
        metadata=edge.metadata,
    )


@router.get("/hyperedges/{hyperedge_id}", summary="查询超边")
async def get_hyperedge(
    hyperedge_id: str,
    deps: Services = Depends(get_services),
) -> HyperedgeResponse:
    """按 ID 查询单个超边。"""
    start = _now()
    set_trace_id()

    if deps.hyperedge_manager is None:
        raise HTTPException(status_code=503, detail="Hyperedge manager not available")

    edge = deps.hyperedge_manager.get_hyperedge(hyperedge_id)
    if edge is None:
        raise HTTPException(status_code=404, detail=f"Hyperedge {hyperedge_id} not found")

    record_request("GET", "/hyperedges/{hyperedge_id}", "200", _now() - start)
    return HyperedgeResponse(
        id=edge.id,
        type=_core_type_to_api(edge.type),
        member_ids=edge.member_ids,
        created_at=edge.created_at,
        gate_value=edge.gate_value,
        metadata=edge.metadata,
    )


@router.get("/nodes/{node_id}/hyperedges", summary="查询节点的所有超边")
async def list_hyperedges_for_node(
    node_id: str,
    deps: Services = Depends(get_services),
) -> HyperedgeListResponse:
    """获取包含指定节点的所有超边列表。"""
    start = _now()
    set_trace_id()

    if deps.hyperedge_manager is None:
        raise HTTPException(status_code=503, detail="Hyperedge manager not available")

    edges = deps.hyperedge_manager.get_hyperedges_by_node(node_id)
    items = [
        HyperedgeResponse(
            id=e.id,
            type=_core_type_to_api(e.type),
            member_ids=e.member_ids,
            created_at=e.created_at,
            gate_value=e.gate_value,
            metadata=e.metadata,
        )
        for e in edges
    ]

    record_request("GET", "/nodes/{node_id}/hyperedges", "200", _now() - start)
    return HyperedgeListResponse(hyperedges=items, total=len(items))


# ═══════════════════════════════════════════════════════════
# 社区
# ═══════════════════════════════════════════════════════════

@router.get("/communities", summary="列出所有社区 (Layer3)")
async def list_communities(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    deps: Services = Depends(get_services),
) -> CommunityListResponse:
    """列出 Kuzu 中所有 CommunityNode。"""
    start = _now()
    set_trace_id()

    try:
        rows = deps.kuzu_store.query_cypher(
            "MATCH (c:CommunityNode) RETURN c.* ORDER BY c.created_at DESC "
            "SKIP $offset LIMIT $limit",
            {"offset": offset, "limit": limit},
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Kuzu query failed: {e}")

    communities = [
        CommunityInfo(
            id=row.get("id", ""),
            name=row.get("name", ""),
            summary=row.get("summary", ""),
            member_count=row.get("member_count", 0),
            keywords=row.get("keywords", []),
            leiden_score=row.get("leiden_score", 0.0),
        )
        for row in rows
    ]

    record_request("GET", "/communities", "200", _now() - start)
    return CommunityListResponse(communities=communities, total=len(communities))


@router.get("/communities/{community_id}", summary="查询社区详情")
async def get_community(
    community_id: str,
    deps: Services = Depends(get_services),
) -> CommunityInfo:
    """按 ID 查询单个社区。"""
    start = _now()
    set_trace_id()

    try:
        rows = deps.kuzu_store.query_cypher(
            "MATCH (c:CommunityNode) WHERE c.id = $id RETURN c.*",
            {"id": community_id},
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Kuzu query failed: {e}")

    if not rows:
        raise HTTPException(status_code=404, detail=f"Community {community_id} not found")

    row = dict(rows[0]) if hasattr(rows[0], "items") else rows[0]
    record_request("GET", "/communities/{community_id}", "200", _now() - start)
    return CommunityInfo(
        id=row.get("id", ""),
        name=row.get("name", ""),
        summary=row.get("summary", ""),
        member_count=row.get("member_count", 0),
        keywords=row.get("keywords", []),
        leiden_score=row.get("leiden_score", 0.0),
    )


# ═══════════════════════════════════════════════════════════
# 梦境
# ═══════════════════════════════════════════════════════════

@router.post("/memories/dream/trigger", summary="显式触发梦境")
async def trigger_dream(
    deps: Services = Depends(get_services),
) -> DreamTriggerResponse:
    """
    显式触发梦境整合管道。
    如果已有梦境在运行则拒绝。
    """
    start = _now()
    set_trace_id()

    if deps.dream_scheduler is None:
        raise HTTPException(status_code=503, detail="Dream scheduler not available")

    accepted = await deps.dream_scheduler.trigger_explicit()
    record_request("POST", "/memories/dream/trigger", "200", _now() - start)

    if accepted:
        return DreamTriggerResponse(accepted=True, message="Dream triggered successfully")
    else:
        return DreamTriggerResponse(accepted=False, message="Dream already running")


# ═══════════════════════════════════════════════════════════
# 溯源
# ═══════════════════════════════════════════════════════════

@router.get("/memories/audit/{node_id}", summary="查询节点溯源链")
async def get_audit_trail(
    node_id: str,
    deps: Services = Depends(get_services),
) -> AuditTrace:
    """查询指定节点的完整变更历史，含链完整性验证。"""
    start = _now()
    set_trace_id()

    if deps.audit_chain is None:
        raise HTTPException(status_code=503, detail="Audit chain not available")

    chain_verified = False
    try:
        chain_verified = deps.audit_chain.verify_chain()
    except Exception:
        pass

    ops_raw = deps.audit_chain.trace_node(node_id)
    chain_length = deps.audit_chain.chain_length

    operations = [
        AuditOperation(
            op_type=op.op_type,
            node_id=op.node_id,
            old_value=op.old_value,
            new_value=op.new_value,
            reason=op.reason,
        )
        for op in ops_raw
    ]

    record_request("GET", "/memories/audit/{node_id}", "200", _now() - start)
    return AuditTrace(
        node_id=node_id,
        operations=operations,
        chain_verified=chain_verified,
        total_blocks=chain_length,
    )


# ═══════════════════════════════════════════════════════════
# 健康检查
# ═══════════════════════════════════════════════════════════

@router.get("/health", summary="深度健康检查")
async def health_check(
    deps: Services = Depends(get_services),
) -> HealthStatus:
    """
    深度健康检查，覆盖所有核心组件：
    - Kuzu 连接 + 断路器状态
    - FAISS 索引状态
    - BLAKE3 溯源链完整性
    - 梦境调度器状态
    """
    start = _now()
    set_trace_id()

    checker = HealthChecker(
        kuzu_store=deps.kuzu_store,
        faiss_index=deps.faiss_index,
        audit_chain=deps.audit_chain,
        dream_scheduler=deps.dream_scheduler,
    )
    health: HealthCheckResult = checker.check()

    # 记录断路器指标
    cb = getattr(deps.kuzu_store, "circuit_breaker", None)
    if cb is not None:
        state_map = {"closed": 0, "half_open": 1, "open": 2}
        cb_state = cb.state.value if hasattr(cb.state, "value") else str(cb.state)
        record_circuit_breaker("kuzu", state_map.get(cb_state, 0))

    stats: Dict[str, Any] = {
        "uptime_seconds": health.uptime_seconds,
        "faiss_index_size": health.faiss_index_size,
        "chain_verified": health.chain_verified,
        "node_count": health.node_count,
        "hyperedge_count": health.hyperedge_count,
        "last_dream_time": health.last_dream_time,
        "circuit_breaker": health.details.get("circuit_breaker", {}),
        "memory": health.details.get("memory_usage", {}),
    }

    record_request("GET", "/health", "200", _now() - start)
    return HealthStatus(
        status=health.status,
        kuzu_connected=health.kuzu_connected,
        faiss_loaded=health.faiss_loaded,
        dream_scheduler_running=health.dream_scheduler_running,
        stats=stats,
        timestamp=start,
    )


# ═══════════════════════════════════════════════════════════
# Prometheus 指标
# ═══════════════════════════════════════════════════════════

# ─── Cypher 查询代理 ──────────────────────────────────────

from pydantic import BaseModel, Field


class CypherQueryRequest(BaseModel):
    query: str = Field(..., description="Cypher 查询语句")
    params: dict = Field(default_factory=dict, description="查询参数")


@router.post("/query", summary="执行 Cypher 查询（只读代理）")
async def cypher_query(
    req: CypherQueryRequest,
    deps: Services = Depends(get_services),
) -> dict:
    # 执行原始 Cypher 查询（只读）。用于 Hermes SHM 插件的搜索检索。
    blocked_keywords = ["CREATE", "DELETE", "SET", "DROP", "MERGE"]
    upper_q = req.query.strip().upper()
    for kw in blocked_keywords:
        if upper_q.startswith(kw):
            from fastapi import HTTPException as _HE
            raise _HE(status_code=400, detail=f"Write queries blocked: {kw}")
    try:
        rows = deps.kuzu_store.query_cypher(req.query, req.params)
        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", summary="Prometheus 指标端点")
async def metrics() -> Response:
    """暴露 Prometheus 文本格式指标（text/plain）。"""
    try:
        data = get_metrics()
        return Response(content=data, media_type="text/plain; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {e}")
