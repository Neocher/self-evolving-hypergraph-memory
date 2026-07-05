"""
FastAPI 路由注册
==============
所有 HTTP 端点：记忆写入/查询/超边管理/社区/梦境/健康检查/溯源。

依赖注入通过 get_services() 获取服务容器，由 api/app.py 在启动时初始化。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shm._version import __version__, __version_name__

import numpy as np

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
    SearchVectorRequest,
    SearchVectorResult,
    SearchVectorResponse,
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
    faiss_dim: int = 384
    faiss_index_type: str = "IVFFlat"
    faiss_nlist: int = 100
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
    # FAISS 批量写入缓冲区
    _faiss_buffer: list[tuple] = field(default_factory=list)
    _faiss_buffer_lock: Any = None


_services: Optional[Services] = None


def init_services(svc: Services) -> None:
    """由 app.py 在启动时调用，注入服务容器。"""
    global _services
    import threading
    svc._faiss_buffer_lock = threading.Lock()
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


_FAISS_BATCH_SIZE = 10  # 攒够 10 条后批量写入 FAISS


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
        removed_node_ids: 从 Kuzu 删除的 EpisodeNode ID 列表

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

    # SSM门控过滤：低价值内容跳过持久化
    if deps.ssm_gate is not None and deps.tau_engine is not None:
        features = np.array([
            float(len(req.content)),            # 内容长度
            float(created_at - time.time()),    # 时间衰减信号
        ], dtype=np.float32)
        if features.shape[0] != deps.ssm_gate.config.input_dim:
            features = np.pad(features, (0, max(0, deps.ssm_gate.config.input_dim - features.shape[0])),
                             mode="constant")[:deps.ssm_gate.config.input_dim]
        # 懒初始化 hidden_state
        try:
            deps.ssm_gate.hidden_state
        except AttributeError:
            deps.ssm_gate.hidden_state = deps.ssm_gate.reset_state()
        hidden, gate_value = deps.ssm_gate.step(features, deps.ssm_gate.hidden_state)
        deps.ssm_gate.hidden_state = hidden
        if not deps.ssm_gate.should_keep(gate_value):
            logger.debug("SSM gate filtered episode", content_len=len(req.content), gate=float(gate_value))
            return EpisodeResponse(episode_id=episode_id, status="filtered", tau_initial=0.0,
                                   content=req.content, source=req.source)

    # [Ontology] 写时验证
    ontology_note = None
    if deps.ontology_validator is not None:
        val_result = deps.ontology_validator.write_validate(req.content, episode_id)
        if not val_result.passed:
            logger.warning("Ontology write_validate rejected", content=req.content[:50],
                          reason=f"confidence={val_result.confidence:.2f}, conflicts={val_result.conflict_count}")
        if val_result.conflict_count > 0:
            ontology_note = f"[本体警告] 与 {val_result.conflict_count} 条已有事实存在矛盾"
            for c in val_result.contradictions:
                try:
                    deps.kuzu_store.execute_cypher(
                        "MERGE (:ConflictNode {id: $id, episode_a: $a, episode_b: $b, "
                        "rule_id: $rule, detected_at: $t, resolved: false})",
                        {"id": f"conflict_{episode_id}_{c.get(conflict_id,)}",
                         "a": episode_id, "b": c.get("conflict_id", ""),
                         "rule": "write_validate", "t": 0.0})
                except Exception:
                    pass
    deps.kuzu_store.create_episode({
        "id": episode_id,
        "content": req.content,
        "source": req.source.value,
        "created_at": created_at,
        "tau_initial": tau_initial,
    })

    if deps.encoder:
        try:
            # asyncio-compatible embed timeout
            emb = deps.encoder.embed(req.content)
        except Exception:
            emb = None
        if emb is not None and deps.faiss_index is not None:
            faiss_id = int(uuid.uuid5(uuid.NAMESPACE_OID, str(episode_id)).int & ((1 << 63) - 1))
            emb_array = emb.reshape(1, -1).astype(np.float32)

            # 缓冲写入，攒够批量再 flush
            with deps._faiss_buffer_lock:
                deps._faiss_buffer.append((faiss_id, emb_array.flatten(), episode_id))
                buf_size = len(deps._faiss_buffer)

            # 攒够批量 → 立即 flush
            if buf_size >= _FAISS_BATCH_SIZE:
                flush_faiss_buffer(deps)

            # 【FIX】Hebbian 连接：立即 flush 以确保新向量在索引中可搜索
            try:
                flush_faiss_buffer(deps)
                if deps.faiss_index.ntotal > 1:
                    distances, indices = deps.faiss_index.search(emb_array, 6)
                    conn_count = 0
                    for rank in range(1, len(indices[0])):
                        nb_id = int(indices[0][rank])
                        if nb_id < 0 or nb_id == faiss_id:
                            continue
                        similarity = max(0.0, 1.0 - float(distances[0][rank]) / 2.0)
                        if similarity < 0.3:
                            continue
                        nb_ep_id = None
                        if hasattr(deps, "faiss_id_map") and deps.faiss_id_map is not None:
                            nb_ep_id = deps.faiss_id_map.get(nb_id)
                        if not nb_ep_id:
                            continue
                        deps.kuzu_store.query_cypher(
                            "MATCH (a:EpisodeNode {id: $aid}), (b:EpisodeNode {id: $bid}) "
                            "CREATE (a)-[:HEBBIAN_CONNECTION {weight: $w}]->(b)",
                            {"aid": episode_id, "bid": nb_ep_id, "w": round(similarity, 4)}
                        )
                        deps.kuzu_store.query_cypher(
                            "MATCH (a:EpisodeNode {id: $aid}), (b:EpisodeNode {id: $bid}) "
                            "CREATE (a)-[:HEBBIAN_CONNECTION {weight: $w}]->(b)",
                            {"aid": nb_ep_id, "bid": episode_id, "w": round(similarity, 4)}
                        )
                        conn_count += 1
                    if conn_count > 0:
                        logger.debug("Hebbian connections created: %d for episode %s",
                                     conn_count, episode_id)
            except Exception as he:
                logger.warning("Hebbian connection creation failed: %s", he)

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

    # 当所有上游检索都返回空时，直接 Cypher 兜底
    if not results_raw and deps.kuzu_store is not None:
        try:
            words = [w.strip().lower() for w in req.query.split() if len(w.strip()) > 1]
            if words:
                params = {f"w{i}": w for i, w in enumerate(words[:5])}
                conditions = " OR ".join(f"toLower(e.content) CONTAINS $w{i}" for i in range(len(words[:5])))
                cypher = f"MATCH (e:EpisodeNode) WHERE {conditions} RETURN e.id AS node_id, e.content AS content LIMIT 10"
                fallback_rows = deps.kuzu_store.query_cypher(cypher, params)
                degraded = True
                for row in fallback_rows:
                    if isinstance(row, (list, tuple)):
                        nid, content = row[0], row[1] if len(row) > 1 else ""
                    elif isinstance(row, dict):
                        nid, content = row.get("node_id", ""), row.get("content", "")
                    else:
                        continue
                    results_raw.append({
                        "node_id": str(nid),
                        "content": str(content),
                        "score": 0.5,
                        "level": "kuzu_fallback",
                    })
                logger.info("Cypher fallback provided %d results", len(results_raw))
        except Exception:
            logger.exception("Cypher fallback failed")

    # 检查是否降级
    if results_raw:
        first_level = results_raw[0].get("level", "hypergraph") if isinstance(results_raw[0], dict) else "hypergraph"
        degraded = first_level != "hypergraph"

    # [Ontology] 读时验证：一致性交叉检查 + 置信度修正
    # [Ontology] 读时验证：一致性交叉检查 + 置信度修正
    if deps.ontology_validator is not None and results_raw:
        try:
            validated = deps.ontology_validator.read_validate(
                [{
                    "id": r.get("node_id", ""),
                    "score": r.get("score", 0.0),
                    "tau_value": r.get("tau_value", r.get("tau", 0.5)),
                    "trust_score": r.get("trust_score", 0.5),
                    "content": r.get("content", ""),
                } for r in results_raw[:req.top_k]],
                req.query,
            )
            # 用调整后的分数覆盖原始分数
            v_map = {v.episode_id: v for v in validated}
            for r in results_raw[:req.top_k]:
                rid = r.get("node_id", "")
                if rid in v_map:
                    v = v_map[rid]
                    r["score"] = v.adjusted_score if v.adjusted_score is not None else 0.0
                    if v.conflict_note:
                        r["conflict_note"] = v.conflict_note
        except Exception as val_err:
            logger.warning("Ontology validation failed, using raw scores", error=str(val_err))
        except Exception as val_err:
            logger.warning("Ontology validation failed, using raw scores", error=str(val_err))

    results: list[EpisodicResult] = []
    for r in results_raw[:req.top_k]:
        if isinstance(r, dict):
            results.append(EpisodicResult(
                node_id=r.get("node_id", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0) or 0.0,
                tau_value=r.get("tau_value") or 0.0,
                source=r.get("level", "hypergraph") or "hypergraph",
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


@router.post("/search/vector", summary="纯向量检索（直通 FAISS）")
async def search_vector(
    req: SearchVectorRequest,
    deps: Services = Depends(get_services),
) -> SearchVectorResponse:
    """使用 FAISS 向量索引执行纯向量检索，回查 Kuzu 获取节点详情。

    当 encoder 或 FAISS 索引不可用时自动降级返回空结果。
    """
    start = _now()
    set_trace_id()
    degraded = False

    # 降级检查
    if deps.encoder is None or deps.faiss_index is None:
        degraded = True
        logger.warning("Vector search degraded: encoder=%s, faiss_index=%s",
                       deps.encoder is not None, deps.faiss_index is not None)
        latency = (_now() - start) * 1000
        record_request("POST", "/search/vector", "200", _now() - start)
        return SearchVectorResponse(
            query=req.query,
            results=[],
            total_found=0,
            latency_ms=round(latency, 2),
            degraded=degraded,
        )

    try:
        # 1. 编码查询文本
        emb = deps.encoder.embed(req.query)
        if emb is None:
            raise ValueError("Encoder returned None")


        emb_array = emb.reshape(1, -1).astype(np.float32)

        # 2. FAISS 向量检索
        k = min(req.limit, deps.faiss_index.ntotal) if deps.faiss_index.ntotal > 0 else req.limit
        if k == 0:
            latency = (_now() - start) * 1000
            record_request("POST", "/search/vector", "200", _now() - start)
            return SearchVectorResponse(
                query=req.query,
                results=[],
                total_found=0,
                latency_ms=round(latency, 2),
                degraded=False,
            )

        distances, indices = deps.faiss_index.search(emb_array, k)

        # 3. 回查 Kuzu 获取节点详情
        results: list[SearchVectorResult] = []
        faiss_id_map = getattr(deps, "faiss_id_map", {}) or {}
        for rank in range(len(indices[0])):
            faiss_id = int(indices[0][rank])
            if faiss_id < 0:
                continue

            # 距离转得分 (L2 → 相似度)
            l2_dist = float(distances[0][rank])
            score = max(0.0, 1.0 - l2_dist / 2.0)

            # 通过 faiss_id_map 回查 episode ID
            episode_id = faiss_id_map.get(faiss_id)
            if not episode_id:
                continue

            # 从 Kuzu 获取节点详情
            try:
                node = deps.kuzu_store.get_episode(episode_id) if deps.kuzu_store else None
                content = node.get("content", "") if node else ""
            except Exception:
                content = ""

            results.append(SearchVectorResult(
                node_id=episode_id,
                content=content,
                score=round(score, 4),
                faiss_id=faiss_id,
            ))

    except Exception as exc:
        logger.exception("Vector search failed")
        record_request("POST", "/search/vector", "500", _now() - start)
        raise HTTPException(status_code=500, detail=str(exc))

    latency = (_now() - start) * 1000
    record_request("POST", "/search/vector", "200", _now() - start)
    return SearchVectorResponse(
        query=req.query,
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
            "LIMIT $limit",
            {"offset": offset, "limit": limit},
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Kuzu query failed: {e}")

    def _to_dict(row):
        """将 query_cypher 返回的 tuple 转为 dict（列名已知）。"""
        keys = ["id", "name", "summary", "leiden_score", "created_at",
                "member_count", "keywords"]
        if isinstance(row, dict):
            return row
        if isinstance(row, (list, tuple)):
            return {keys[i] if i < len(keys) else f"col_{i}": row[i]
                    for i in range(len(row))}
        return {}

    row_dicts = [_to_dict(r) for r in rows]
    communities = [
        CommunityInfo(
            id=row.get("id", ""),
            name=row.get("name", ""),
            summary=row.get("summary", ""),
            member_count=row.get("member_count", 0),
            keywords=row.get("keywords", []),
            leiden_score=row.get("leiden_score", 0.0),
        )
        for row in row_dicts
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


@router.post("/dream/notify", summary="通知调度器有新节点创建")
async def dream_notify(
    deps: Services = Depends(get_services),
) -> dict:
    """
    通知梦境调度器有新节点创建。
    供 Hermes SHM 插件在写入记忆后调用，加速梦境触发。
    """
    if deps.dream_scheduler is not None and hasattr(deps.dream_scheduler, "on_node_created"):
        await deps.dream_scheduler.on_node_created()
        logger.debug("Dream scheduler notified of new node")
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════
# 梦境候选（非破坏性模式）
# ═══════════════════════════════════════════════════════════


@router.get("/dream/candidates", summary="列出梦境候选（待审查）")
async def list_dream_candidates(
    deps: Services = Depends(get_services),
):
    """列出所有待审查的梦境候选。"""
    from api.models import DreamCandidateListResponse, DreamCandidateSummary

    store = getattr(deps, "dream_candidate_store", None)
    if store is None:
        return DreamCandidateListResponse(candidates=[], total=0)

    raw = store.list_candidates(limit=50)
    candidates = [
        DreamCandidateSummary(**r) for r in raw
    ]
    return DreamCandidateListResponse(candidates=candidates, total=len(candidates))


@router.get("/dream/candidates/{dream_id}", summary="审查梦境候选详情")
async def review_dream_candidate(
    dream_id: str,
    deps: Services = Depends(get_services),
):
    """查看梦境候选的详细内容，审查后再决定 apply 或 discard。"""
    from api.models import DreamCandidateDetail

    store = getattr(deps, "dream_candidate_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Dream candidate store not available")

    candidate = store.get_candidate(dream_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Dream candidate {dream_id} not found")

    return DreamCandidateDetail(
        dream_id=candidate.dream_id,
        created_at=candidate.created_at,
        trigger_mode=candidate.trigger_mode,
        stats=candidate.stats,
        community_count=candidate.community_count,
        prune_count=candidate.prune_count,
        conflict_count=candidate.conflict_count,
        community_summaries=candidate.community_summaries,
        prune_ops=candidate.prune_ops,
        merge_ops=candidate.merge_ops,
        applied=candidate.applied,
        discarded=candidate.discarded,
    )


@router.post("/dream/candidates/{dream_id}/apply", summary="应用梦境候选到生产库")
async def apply_dream_candidate(
    dream_id: str,
    deps: Services = Depends(get_services),
):
    """
    将梦境候选中的 PRUNE 和 MERGE 操作应用到生产 Kuzu 数据库。
    操作不可逆，请先 review。
    """
    from api.models import DreamApplyResponse

    store = getattr(deps, "dream_candidate_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Dream candidate store not available")

    if deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Kuzu store not available")

    success = store.apply_candidate(dream_id, deps.kuzu_store)
    if success:
        return DreamApplyResponse(success=True, dream_id=dream_id, message="Dream applied to production")
    else:
        return DreamApplyResponse(success=False, dream_id=dream_id, message="Apply failed (see logs)")


@router.post("/dream/candidates/{dream_id}/discard", summary="丢弃梦境候选")
async def discard_dream_candidate(
    dream_id: str,
    deps: Services = Depends(get_services),
) -> dict:
    """丢弃梦境候选，不做任何修改。"""
    store = getattr(deps, "dream_candidate_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Dream candidate store not available")

    success = store.discard_candidate(dream_id)
    if success:
        return {"status": "ok", "dream_id": dream_id, "message": "Dream candidate discarded"}
    else:
        raise HTTPException(status_code=404, detail=f"Dream candidate {dream_id} not found")


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

    logger = get_logger()
    chain_verified = False
    try:
        chain_verified = deps.audit_chain.verify_chain()
    except Exception:
        logger.warning("Health check: audit chain verification failed, defaulting to False")

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
        "version": __version__,
        "version_name": __version_name__,
        "uptime_seconds": health.uptime_seconds,
        "faiss_index_size": health.faiss_index_size,
        "chain_verified": health.chain_verified,
        "node_count": health.node_count,
        "hyperedge_count": health.hyperedge_count,
        "last_dream_time": health.last_dream_time,
        "dream_run_count": health.dream_run_count,  # 【FIX】梦境运行次数
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
    import re
    blocked_pattern = re.compile(
        r'\b(?:CREATE|DELETE|SET|DROP|MERGE|REMOVE|DETACH)\b',
        re.IGNORECASE
    )
    if blocked_pattern.search(req.query):
        from fastapi import HTTPException as _HE
        raise _HE(status_code=400, detail=f"Write queries blocked: contains CREATE/DELETE/SET/DROP/MERGE/REMOVE/DETACH")
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


# ─── 【FIX】FAISS 索引重建 ─────────────────────────────────


@router.post("/index/rebuild", summary="重建 FAISS 索引")
async def rebuild_index(
    deps: Services = Depends(get_services),
) -> dict:
    """
    重建 FAISS 向量索引：
    1. 从 Kuzu 读取所有 EpisodeNode
    2. 对每个节点生成 embedding
    3. 重建 FAISS IndexIDMap(FlatL2)
    4. 返回重建结果统计
    """
    start = _now()
    set_trace_id()

    if deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Kuzu store not available")
    if deps.encoder is None:
        raise HTTPException(status_code=503, detail="Text encoder not available")

    import numpy as np
    import faiss

    # 获取所有节点
    rows = deps.kuzu_store.query_cypher(
        "MATCH (e:EpisodeNode) RETURN e.id, e.content LIMIT 10000"
    )
    if not rows:
        return {"status": "ok", "indexed_count": 0, "message": "No episodes found"}

    node_ids = []
    contents = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            nid, content = str(row[0]), str(row[1]) if len(row) > 1 else ""
        elif isinstance(row, dict):
            nid, content = str(row.get("id", "")), str(row.get("content", ""))
        else:
            continue
        if content.strip():
            node_ids.append(nid)
            contents.append(content)

    if not contents:
        return {"status": "ok", "indexed_count": 0, "message": "No episodes with content"}

    # 批量编码
    logger.info("Rebuilding FAISS index: encoding %d episodes", len(contents))
    embeddings = deps.encoder.embed_batch(contents)

    # 重建索引
    dim = deps.faiss_dim
    index_type = deps.faiss_index_type

    if index_type == "IVFFlat" and len(embeddings) >= max(deps.faiss_nlist * 2, 2000):
        nlist = min(deps.faiss_nlist, len(embeddings) // 2)
        quantizer = faiss.IndexFlatL2(dim)
        base_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        base_index.train(embeddings.astype(np.float32))
        base_index.nprobe = min(deps.faiss_nlist // 10, 10)  # 搜索探测数
        new_index = faiss.IndexIDMap(base_index)
        logger.info("FAISS rebuilt with IVFFlat", dim=dim, nlist=nlist, nprobe=10, vectors=len(embeddings))
    else:
        # FlatL2 — 小数据集精确检索，无量化损失
        new_index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        logger.info("FAISS rebuilt with FlatL2", dim=dim, vectors=len(embeddings))

    faiss_ids = np.array([
        uuid.uuid5(uuid.NAMESPACE_OID, str(nid)).int & ((1 << 63) - 1)
        for nid in node_ids
    ], dtype=np.int64)
    new_index.add_with_ids(embeddings.astype(np.float32), faiss_ids)

    # 替换现有索引
    deps.faiss_index = new_index
    if hasattr(deps, "faiss_id_map"):
        deps.faiss_id_map = dict(zip(faiss_ids.tolist(), node_ids))
    # 同步更新查询路由的索引引用
    if deps.query_router is not None:
        deps.query_router.faiss_index = new_index
        deps.query_router.faiss_id_map = deps.faiss_id_map
        # 同步 TF-IDF 索引
        tfidf = getattr(deps, "tfidf_index", None)
        if tfidf is not None:
            deps.query_router.tfidf_index = tfidf
    # 同步更新查询路由的索引引用
    if deps.query_router is not None:
        deps.query_router.faiss_index = new_index
        deps.query_router.faiss_id_map = deps.faiss_id_map
        # 同步 TF-IDF 索引
        tfidf = getattr(deps, "tfidf_index", None)
        if tfidf is not None:
            deps.query_router.tfidf_index = tfidf

    # 【FIX】同时重建Hebbian连接
    logger.info("Rebuilding Hebbian connections...")
    hebbian_count = 0
    # 先清空旧连接
    try:
        deps.kuzu_store.query_cypher("MATCH ()-[r:HEBBIAN_CONNECTION]->() DELETE r")
    except Exception:
        pass
    # 为每个节点找5个最相似邻居建连接
    for i in range(len(node_ids)):
        query_vec = embeddings[i:i+1].astype(np.float32)
        try:
            distances, indices = new_index.search(query_vec, 6)
            for rank in range(1, len(indices[0])):
                nb_idx = int(indices[0][rank])
                if nb_idx < 0 or nb_idx == faiss_ids[i]:
                    continue
                similarity = max(0.0, 1.0 - float(distances[0][rank]) / 2.0)
                if similarity < 0.3:
                    continue
                nb_node_id = node_ids[np.where(faiss_ids == nb_idx)[0][0]]
                try:
                    deps.kuzu_store.query_cypher(
                        "MATCH (a:EpisodeNode {id: $aid}), (b:EpisodeNode {id: $bid}) "
                        "CREATE (a)-[:HEBBIAN_CONNECTION {weight: $w}]->(b)",
                        {"aid": node_ids[i], "bid": nb_node_id, "w": round(similarity, 4)}
                    )
                    hebbian_count += 1
                except Exception:
                    pass
        except Exception:
            pass

    record_request("POST", "/index/rebuild", "200", _now() - start)
    logger.info("FAISS index rebuilt: %d vectors, %d Hebbian connections",
                new_index.ntotal, hebbian_count)

    # 拟合 TF-IDF 索引（提升 L3 关键词检索质量）
    try:
        tfidf = getattr(deps, "tfidf_index", None)
        if tfidf is not None and hasattr(tfidf, "fit") and contents:
            tfidf.fit(contents)
            logger.info("TF-IDF index fitted with %d texts", len(contents))
    except Exception:
        logger.exception("TF-IDF fit failed (non-fatal)")

    return {
        "status": "ok",
        "indexed_count": new_index.ntotal,
        "total_nodes": len(node_ids),
        "dimension": dim,
        "hebbian_connections": hebbian_count,
    }
