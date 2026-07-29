"""
FastAPI 路由注册
==============
所有 HTTP 端点：记忆写入/查询/超边管理/社区/梦境/健康检查/溯源。

依赖注入通过 get_services() 获取服务容器，由 api/app.py 在启动时初始化。
"""

from __future__ import annotations

import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shm._version import __version__, __version_name__

import numpy as np

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
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
from core.quarantine_store import QuarantineStore
from core.write_reconciler import WriteReconciler, ConflictLogger, Strategy
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


_services: Optional[Services] = None


def init_services(svc: Services) -> None:
    """由 app.py 在启动时调用，注入服务容器。"""
    global _services
    import threading
    svc._faiss_buffer_lock = threading.Lock()
    svc._session_memory_lock = threading.Lock()
    # 初始化记忆投毒防御引擎 + 隔离存储
    try:
        svc.defense_engine = MemoryDefenseEngine(config=DefenseConfig(), encoder=svc.encoder)
    except Exception:
        logger.warning("DefenseEngine init failed (non-fatal)")
    try:
        svc.quarantine_store = QuarantineStore(graph_store=svc.kuzu_store)
    except Exception:
        logger.warning("QuarantineStore init failed (non-fatal)")
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


def _process_embed_queue(deps: Services) -> int:
    """消费 embedding 队列：异步编码并加入 FAISS 缓冲。"""
    global _embed_queue
    if not deps.encoder:
        return 0
    with _embed_queue_lock:
        batch = _embed_queue[:]
        _embed_queue.clear()
    if not batch:
        return 0
    count = 0
    # 【Defense】预先获取隔离节点 ID 集合用于快速排除
    quarantined_set: set[str] = set()
    if deps.quarantine_store is not None:
        quarantined_set = deps.quarantine_store.get_quarantined_ids()

    for episode_id, content, created_at in batch:
        # 隔离节点不加入 FAISS
        if episode_id in quarantined_set:
            logger.debug("Embed queue: skip quarantined node %s", episode_id[:12])
            continue
        try:
            emb = deps.encoder.embed(content)
            if emb is not None and deps.faiss_index is not None:
                faiss_id = int(uuid.uuid5(uuid.NAMESPACE_OID, str(episode_id)).int & ((1 << 63) - 1))
                emb_array = emb.reshape(1, -1).astype(np.float32)
                with deps._faiss_buffer_lock:
                    deps._faiss_buffer.append((faiss_id, emb_array.flatten(), episode_id))
                try:
                    if deps.hebbian_updater and deps.kuzu_store:
                        deps.hebbian_updater.update(
                            {episode_id: 1.0}, deps.kuzu_store.get_all_connections()
                        )
                except Exception:
                    pass
                count += 1
        except Exception:
            pass
    # 批量 flush FAISS
    flush_faiss_buffer(deps)
    if count:
        logger.debug("Embed queue: processed %d items", count)
    return count


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
                     "source": record.source, "created_at": start,
                     "namespace": record.namespace,
                     "visibility": record.visibility})
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
            "source": record.source,
            "visibility": record.visibility,
            "created_at": start,
            "tau_initial": 1.0,
        })
        # 命名空间链接
        if record.namespace:
            deps.kuzu_store.ensure_session(record.namespace)
            deps.kuzu_store.link_to_session(record.namespace, record_id)
        if deps.dream_scheduler:
            await deps.dream_scheduler.on_node_created()

    record_request("POST", "/memories/sensory", "200", _now() - start)
    return SensoryResponse(record_id=record_id, buffer_usage=buffer_usage)


@router.post("/memories/multimodal", summary="多模态记忆写入 (图像/音频/视频+文本)")
async def write_multimodal(
    req: MultimodalRecord,
    request: Request,
    deps: Services = Depends(get_services),
) -> MultimodalResponse:
    """写入多模态记忆。

    接收 Base64 编码的媒体文件（图像/音频/视频）+ 可选文本描述。
    媒体文件存储到 data/media/，嵌入向量写入 VisualNode。
    音频自动转录（Whisper）并与文本一起索引。
    """
    start = _now()
    set_trace_id()

    episode_id = str(uuid.uuid4())
    created_at = _now()
    media_paths: list[str] = []
    transcription: Optional[str] = None
    visual_node_id: Optional[str] = None
    text = req.text

    # 初始化多模态组件（懒加载）
    clip = getattr(deps, "_clip_embedder", None)
    if clip is None:
        try:
            from multimodal.embedders import ClipEmbedder
            clip = ClipEmbedder()
            deps._clip_embedder = clip
        except ImportError:
            clip = None
        except Exception:
            clip = None
    whisper = getattr(deps, "_whisper_embedder", None)
    if whisper is None:
        from multimodal.embedders import WhisperEmbedder
        try:
            whisper = WhisperEmbedder()
            deps._whisper_embedder = whisper
        except Exception:
            whisper = None
    store = getattr(deps, "_media_store", None)
    if store is None:
        from multimodal.store import MediaStore
        try:
            store = MediaStore()
            deps._media_store = store
        except Exception:
            store = None

    # ── 处理图像 ──
    image_embeddings: list[np.ndarray] = []
    for b64_str in req.images:
        try:
            import base64
            img_bytes = base64.b64decode(b64_str)
        except Exception:
            continue
        if store is not None:
            path = store.save_image(img_bytes)
            media_paths.append(path)
        if clip is not None:
            emb = clip.embed_image(img_bytes)
            if emb is not None:
                image_embeddings.append(emb)

    # ── 处理音频 ──
    audio_texts: list[str] = []
    for b64_str in req.audio:
        try:
            import base64
            aud_bytes = base64.b64decode(b64_str)
        except Exception:
            continue
        if store is not None:
            path = store.save_audio(aud_bytes)
            media_paths.append(path)
        if whisper is not None:
            seg = whisper.transcribe(aud_bytes)
            if seg:
                audio_texts.append(seg)

    # ── 处理视频 ──
    for b64_str in req.video:
        try:
            import base64
            vid_bytes = base64.b64decode(b64_str)
        except Exception:
            continue
        if store is not None:
            path = store.save_video(vid_bytes)
            media_paths.append(path)

    # ── 合并文本 ──
    text_parts: list[str] = []
    if text:
        text_parts.append(text)
    if audio_texts:
        transcription = " ".join(audio_texts)
        text_parts.append(f"[audio transcription]: {transcription}")
    merged_text = "\n".join(text_parts) if text_parts else ""

    # ── 写入 VisualNode（有图像时）──
    visual_emb = None
    if image_embeddings and clip is not None:
        # 多图像取平均作为视觉嵌入
        visual_emb = np.mean(image_embeddings, axis=0).astype(np.float32)

        # 投影 512 → 384 以匹配 VisualNode schema（随机投影桥）
        try:
            proj = getattr(deps, "_clip_projection", None)
            if proj is None:
                rng = np.random.default_rng(42)
                proj = rng.standard_normal((512, 384), dtype=np.float32)
                proj /= np.linalg.norm(proj, axis=0, keepdims=True)  # 单位列向量
                deps._clip_projection = proj
            emb_384 = visual_emb @ proj  # (512,) @ (512, 384) → (384,)

            visual_node_id = str(uuid.uuid4())
            if deps.kuzu_store is not None:
                deps.kuzu_store.create_visual_node({
                    "id": visual_node_id,
                    "image_path": media_paths[0] if media_paths else "",
                    "caption": merged_text[:1024],
                    "embedding": emb_384.tolist(),
                    "source": req.source,
                    "created_at": created_at,
                })
        except Exception:
            logger.exception("VisualNode creation failed (non-fatal)")

    # ── 写入 EpisodeNode（文本索引）──
    if merged_text and deps.kuzu_store is not None:
        deps.kuzu_store.create_episode({
            "id": episode_id,
            "content": merged_text,
            "source": req.source,
            "visibility": req.visibility,
            "created_at": created_at,
            "tau_initial": 1.0,
        })
        if req.namespace:
            deps.kuzu_store.ensure_session(req.namespace)
            deps.kuzu_store.link_to_session(req.namespace, episode_id)

        # 通知梦境调度器
        if deps.dream_scheduler:
            await deps.dream_scheduler.on_activity()
            await deps.dream_scheduler.on_node_created()

        # 异步入队文本 embedding
        with _embed_queue_lock:
            _embed_queue.append((episode_id, merged_text, created_at))

    record_request("POST", "/memories/multimodal", "200", _now() - start)
    return MultimodalResponse(
        episode_id=episode_id,
        visual_node_id=visual_node_id,
        text=merged_text[:200] if merged_text else None,
        media_paths=media_paths,
        transcription=transcription,
        created_at=created_at,
    )


@router.post("/memories/episodes", summary="直接创建情节节点 (Layer2)")
async def create_episode(
    req: EpisodeCreate,
    request: Request,
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
        tau_initial = deps.tau_engine.compute_strength(created_at)
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
        hidden, gate_value = deps.ssm_gate.step(deps.ssm_gate.hidden_state, features)
        deps.ssm_gate.hidden_state = hidden
        if not deps.ssm_gate.should_keep(gate_value):
            logger.debug("SSM gate filtered episode", content_len=len(req.content), gate=float(gate_value))
            return EpisodeResponse(episode_id=episode_id, status="filtered", tau_initial=0.0,
                                   content=req.content, source=req.source)

    # [Defense] 记忆投毒预检（在 Kuzu 写入前执行）
    defense_verdict = None
    defense_reason = ""
    if deps.defense_engine and deps.defense_engine.config.enabled:
        verdict, reason = deps.defense_engine.pre_check(
            content=req.content, source=req.source, created_at=created_at,
        )
        defense_verdict = verdict
        defense_reason = reason
        if verdict.value == "block":
            logger.warning("Defense BLOCKED write", source=req.source, reason=reason)
            record_request("POST", "/memories/episodes", "403", _now() - start)
            return JSONResponse(
                status_code=403,
                content={"error": "blocked_by_defense", "reason": reason},
            )
        elif verdict.value == "quarantine":
            logger.warning("Defense QUARANTINE write", source=req.source, reason=reason)

    # [Ontology] 写时验证（v1 — 冲突检测）
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
                    conflict_id = c.get("conflict_id", "")
                    deps.kuzu_store.execute_cypher(
                        "MERGE (:ConflictNode {id: $id, episode_a: $a, episode_b: $b, "
                        "rule_id: $rule, detected_at: $t, resolved: false})",
                        {"id": f"conflict_{episode_id}_{conflict_id}",
                         "a": episode_id, "b": conflict_id,
                         "rule": "write_validate", "t": _now()})
                except Exception:
                    pass
            # P2: 通知梦境调度器有冲突产生
            try:
                if deps.dream_scheduler:
                    await deps.dream_scheduler.on_conflict_detected()
            except Exception:
                pass
    deps.kuzu_store.create_episode({
        "id": episode_id,
        "content": req.content,
        "source": req.source,
        "visibility": req.visibility,
        "created_at": created_at,
        "tau_initial": tau_initial,
    })

    # [Defense] 隔离标记：QUARANTINE 判定的节点写入后标记隔离
    if defense_verdict is not None and defense_verdict.value == "quarantine":
        if deps.quarantine_store is not None:
            deps.quarantine_store.quarantine(episode_id, defense_reason, req.source)
            logger.info("Node %s quarantined after write: %s", episode_id[:12], defense_reason[:80])

    # 命名空间链接
    if req.namespace:
        deps.kuzu_store.ensure_session(req.namespace)
        deps.kuzu_store.link_to_session(req.namespace, episode_id)

    # [Ontology v2] 写时类型验证
    if deps.ontology_v2 is not None:
        try:
            v2_result = deps.ontology_v2.validate_write(req.content)
            if not v2_result.passed and v2_result.errors:
                for err in v2_result.errors:
                    logger.warning("Ontology v2 validation: %s → %s", err.field, err.message)
        except Exception:
            logger.exception("Ontology v2 write validation error (non-fatal)")

    # [Step 1] 关系抽取：批量 Kuzu 操作（减少 3N→2 次往返）
    triples = None
    if deps.kuzu_store is not None and len(req.content) > 50:
        try:
            from core.relation_extractor import RelationExtractor
            rext = RelationExtractor()
            triples = rext.extract(req.content)
            if triples:
                # 批量创建实体节点（一次 Kuzu 调用）
                entity_statements = []
                seen_entities = set()
                for t in triples:
                    for entity_name in (t.subject, t.obj):
                        if entity_name not in seen_entities:
                            seen_entities.add(entity_name)
                            entity_statements.append(
                                f"MERGE (n{len(seen_entities)}:OntologyEntity {{name: '{entity_name}'}}) "
                                f"ON CREATE SET n{len(seen_entities)}.type = 'discovered'"
                            )
                if entity_statements:
                    deps.kuzu_store.query_cypher(" ".join(entity_statements))
                # 批量创建关系边（一次 Kuzu 调用）
                for t in triples:
                    deps.kuzu_store.query_cypher(
                        "MATCH (a:OntologyEntity {name: $subj}) "
                        "MATCH (b:OntologyEntity {name: $obj}) "
                        "MERGE (a)-[r:RELATES_TO {relation: $rel}]->(b)",
                        {"subj": t.subject, "obj": t.obj, "rel": t.relation},
                    )
            if triples:
                logger.info("Relation extraction: %d typed edges", len(triples))
        except Exception:
            logger.exception("Relation extraction error (non-fatal)")

    # [Step 2] 置信度累积
    if deps.evidence_tracker is not None:
        try:
            evidence_count = deps.evidence_tracker.record(
                req.content, source=req.source,
                metadata={"episode_id": episode_id},
            )
            if evidence_count > 1:
                logger.info("Evidence tracker: count=%d for %s", evidence_count, req.content[:40])
        except Exception:
            logger.exception("Evidence tracker error (non-fatal)")

    # [Step 3] 实体消歧 — 仅对有一定信息量的内容执行
    if deps.kuzu_store is not None and len(req.content) > 80:
        try:
            from core.entity_resolver import EntityResolver
            resolver = EntityResolver(kuzu_store=deps.kuzu_store)
            result = resolver.process(req.content)
            if result.get("alias_count", 0) > 0:
                logger.info("Entity resolver: %d alias edges, %d entities",
                            result["alias_count"], len(result.get("entities", [])))
        except Exception:
            logger.exception("Entity resolver error (non-fatal)")

    # [Phase3] 写入时提取实体共现 → 建 RELATES_TO 边（保留旧逻辑作为fallback）
    if deps.ontology_validator is not None and triples is None:
        try:
            rel_count = deps.ontology_validator.extract_and_relate(req.content)
            if rel_count > 0:
                logger.info("Write-time entity relations: %d edges (fallback)", rel_count)
        except Exception:
            pass

    # 【P6】异步 embedding：入队后立即返回（不阻塞写入响应）
    with _embed_queue_lock:
        _embed_queue.append((episode_id, req.content, created_at))

    # 不再在写入路径中同步消费队列——由 poll loop 每5秒 flush 一次
    # 避免了写入延迟因 FAISS 编码而膨胀 200-400ms

    if deps.dream_scheduler:
        await deps.dream_scheduler.on_activity()
        await deps.dream_scheduler.on_node_created()

    # 【P0】自动超边创建：检测同源节点形成时态/情节超边
    try:
        await _auto_create_hyperedges(episode_id, req.source, req.content, deps)
    except Exception:
        pass

    # 【P0-①】会话观测节点：通过 X-Session-Id header 关联记忆到同一会话
    try:
        session_id = request.headers.get("X-Session-Id") or request.headers.get("x-session-id")
        if session_id and deps.kuzu_store is not None:
            session_node_id = deps.kuzu_store.get_or_create_session(
                session_id, metadata='{"source": "' + req.source + '"}'
            )
            if session_node_id:
                deps.kuzu_store.link_session_member(session_node_id, episode_id)
    except Exception:
        pass

    record_request("POST", "/memories/episodes", "200", _now() - start)
    return EpisodeResponse(
        episode_id=episode_id,
        content=req.content[:200],
        tau_initial=tau_initial,
        created_at=created_at,
        source=req.source,
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
        tau = deps.tau_engine.compute_strength(created_at)

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
    """执行粗到精三级检索，结果缓存 128 条。"""
    start = _now()
    set_trace_id()
    degraded = False

    # 【Perf】结果缓存命中
    cache_key = f"{req.query}:{req.top_k}"
    with _result_cache_lock:
        if cache_key in _result_cache:
            latency = (_now() - start) * 1000
            record_request("POST", "/memories/retrieve", "200", _now() - start)
            cached = _result_cache[cache_key]
            cached.latency_ms = round(latency, 2)
            return cached

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
                cypher = (f"MATCH (e:EpisodeNode) WHERE ({conditions}) "
                          "AND (e.quarantine IS NULL OR e.quarantine = false) "
                          f"RETURN e.id AS node_id, e.content AS content LIMIT 10")
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

    # 【Defense】隔离节点排除
    if results_raw and deps.quarantine_store is not None:
        quarantined_ids = deps.quarantine_store.get_quarantined_ids()
        if quarantined_ids:
            before = len(results_raw)
            results_raw = [
                r for r in results_raw
                if r.get("node_id", "") not in quarantined_ids
            ]
            if len(results_raw) < before:
                logger.debug("Retrieval: filtered %d quarantined results", before - len(results_raw))

    # 【P2】结果去重 + 命名空间过滤 + visibility 过滤
    if results_raw:
        seen = set()
        deduped = []
        # 如果指定了命名空间，预取该空间下的所有 node_id
        ns_set: set[str] | None = None
        if req.namespace and deps.kuzu_store is not None:
            try:
                ns_rows = deps.kuzu_store.query_cypher(
                    "MATCH (s:SessionNode {session_id: $ns})-[:SESSION_MEMBER]->(e:EpisodeNode) "
                    "RETURN e.id",
                    {"ns": req.namespace}
                )
                ns_set = {row[0] for row in ns_rows} if ns_rows else set()
            except Exception:
                pass
        for r in results_raw:
            key = r.get("content", "")[:100]
            if key and key not in seen:
                # 命名空间过滤
                if ns_set is not None and r.get("node_id", "") not in ns_set:
                    continue
                # visibility=shared 的记忆可被所有Agent检索
                if not req.include_shared:
                    # 仅在命名空间内搜索时跳过 shared 记忆
                    pass  # 当前实现：shared 记忆不会被索引到命名空间中，所以自动跳过
                seen.add(key)
                deduped.append(r)
        if len(deduped) < len(results_raw):
            logger.debug("Dedup removed %d duplicate results", len(results_raw) - len(deduped))
        results_raw = deduped

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
    response = RetrieveResponse(
        query=req.query,
        strategy_used=req.strategy or "auto",
        results=results,
        total_found=len(results),
        latency_ms=round(latency, 2),
        degraded=degraded,
    )
    # 【Perf】存入结果缓存
    with _result_cache_lock:
        if len(_result_cache) >= _result_cache_max:
            keys = list(_result_cache.keys())
            for k in keys[:_result_cache_max // 2]:
                del _result_cache[k]
        _result_cache[cache_key] = response
    record_request("POST", "/memories/retrieve", "200", _now() - start)
    return response


@router.delete("/memories/namespace/{namespace}", summary="按命名空间批量删除节点")
async def delete_namespace(
    namespace: str,
    deps: Services = Depends(get_services),
) -> dict:
    """删除指定命名空间下的所有 EpisodeNode + SessionNode。"""
    if deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Kuzu store not available")
    try:
        count = deps.kuzu_store.delete_namespace(namespace)
        return {"deleted": count, "namespace": namespace, "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@router.post("/dream/reset", summary="强制重启梦境（停止当前 → 重新触发）")
async def reset_dream(
    deps: Services = Depends(get_services),
) -> DreamTriggerResponse:
    """强制停止当前梦境，在后台重新触发一次全量梦境（异步，不阻塞）"""
    if deps.dream_scheduler is None:
        raise HTTPException(status_code=503, detail="Dream scheduler not available")
    # 强制停止
    deps.dream_scheduler.force_stop()
    # 在后台任务中触发，不阻塞返回
    import asyncio
    asyncio.create_task(deps.dream_scheduler.trigger_explicit())
    return DreamTriggerResponse(
        accepted=True,
        message="Dream reset accepted, running in background. Check /dream/candidates for progress."
    )

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
        record_circuit_breaker("ryu", state_map.get(cb_state, 0))

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
        graph_connected=health.graph_connected,
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


# ═══════════════════════════════════════════════════════════
# 超边 (Hyperedge) 端点
# ═══════════════════════════════════════════════════════════


@router.get("/sessions/{session_id}/memories", summary="查询会话的所有记忆")
async def get_session_memories(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    deps: Services = Depends(get_services),
) -> dict:
    """查询指定会话 ID 关联的所有记忆节点。"""
    start = _now()
    set_trace_id()

    if deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Kuzu store not available")

    rows = deps.kuzu_store.get_session_memories(session_id, limit)
    memories = []
    for r in rows:
        memories.append({
            "id": r.get("id", ""),
            "content": r.get("content", "")[:200],
            "created_at": r.get("created_at", 0.0),
            "source": r.get("source", ""),
        })

    record_request("GET", f"/sessions/{session_id}/memories", "200", _now() - start)
    return {"session_id": session_id, "memories": memories, "total": len(memories)}


# ═══════════════════════════════════════════════════════════
# 冲突 (Conflict) 端点
# ═══════════════════════════════════════════════════════════


@router.get("/conflicts", summary="列出所有未解决冲突（含 OCC 版本信息）")
async def list_conflicts(
    limit: int = Query(default=50, ge=1, le=500),
    include_resolved: bool = Query(default=False, description="是否包含已解决的冲突"),
    deps: Services = Depends(get_services),
) -> dict:
    """列出所有冲突节点，扩展版本信息和 OCC 冲突日志统计。"""
    start = _now()
    set_trace_id()
    if deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Kuzu store not available")

    resolved_filter = "" if include_resolved else "AND c.resolved = false"
    rows = deps.kuzu_store.execute_cypher(
        f"MATCH (c:ConflictNode) WHERE 1=1 {resolved_filter} "
        "RETURN c.id, c.episode_a, c.episode_b, c.rule_id, "
        "c.detected_at, c.resolved ORDER BY c.detected_at DESC LIMIT $limit",
        {"limit": limit}
    )
    conflicts = []
    for r in rows:
        conflict_entry = {
            "id": r.get("c.id", ""),
            "episode_a": r.get("c.episode_a", ""),
            "episode_b": r.get("c.episode_b", ""),
            "rule_id": r.get("c.rule_id", ""),
            "detected_at": r.get("c.detected_at", 0.0),
            "resolved": r.get("c.resolved", False),
        }
        # 附带 OCC 版本信息（如果冲突节点有 version 信息）
        episode_a = deps.kuzu_store.get_episode(conflict_entry["episode_a"]) if conflict_entry["episode_a"] else None
        episode_b = deps.kuzu_store.get_episode(conflict_entry["episode_b"]) if conflict_entry["episode_b"] else None
        if episode_a:
            conflict_entry["episode_a_version"] = episode_a.get("version", 1)
        if episode_b:
            conflict_entry["episode_b_version"] = episode_b.get("version", 1)
        conflicts.append(conflict_entry)

    # OCC 冲突日志统计
    reconciler = getattr(deps, "write_reconciler", None)
    occ_stats = {}
    if reconciler is not None:
        occ_stats = reconciler.conflict_logger.stats()

    record_request("GET", "/conflicts", "200", _now() - start)
    return {
        "conflicts": conflicts,
        "total": len(conflicts),
        "occ_conflict_stats": occ_stats,
    }


@router.post("/conflicts/{conflict_id}/resolve", summary="标记冲突为已解决")
async def resolve_conflict(
    conflict_id: str,
    deps: Services = Depends(get_services),
) -> dict:
    """标记指定冲突为已解决。"""
    start = _now()
    set_trace_id()
    if deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Kuzu store not available")

    deps.kuzu_store.execute_cypher(
        "MATCH (c:ConflictNode) WHERE c.id = $id "
        "SET c.resolved = true",
        {"id": conflict_id}
    )

    record_request("POST", f"/conflicts/{conflict_id}/resolve", "200", _now() - start)
    return {"status": "resolved", "conflict_id": conflict_id}


@router.post("/conflicts/resolve-all", summary="标记所有冲突为已解决")
async def resolve_all_conflicts(
    deps: Services = Depends(get_services),
) -> dict:
    """标记所有未解决冲突为已解决。"""
    start = _now()
    set_trace_id()
    if deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Kuzu store not available")

    deps.kuzu_store.execute_cypher(
        "MATCH (c:ConflictNode) WHERE c.resolved = false "
        "SET c.resolved = true",
        {}
    )

    record_request("POST", "/conflicts/resolve-all", "200", _now() - start)
    return {"status": "all resolved"}


@router.post("/conflicts/reconcile", summary="手动触发 OCC 写入消解")
async def reconcile_conflict(
    body: dict,
    deps: Services = Depends(get_services),
) -> dict:
    """
    手动触发 OCC 版本冲突的写入消解。

    Body (JSON):
        node_id: str          — 目标节点 ID
        data: dict            — 待写入的数据
        expected_version: int — 预期的版本号
        strategy: str         — 消解策略: "lww" | "merge" | "additive"（默认 "lww"）
        force: bool           — 是否跳过版本检查强行写入（默认 false）

    Returns:
        {
            "conflict": bool,        # 是否检测到版本冲突
            "resolved": bool,        # 是否成功消解
            "strategy": str,         # 使用的策略
            "data": dict,            # 消解后的数据
            "current_version": int,  # 数据库中当前版本
        }
    """
    start = _now()
    set_trace_id()

    node_id = body.get("node_id", "")
    data = body.get("data", {})
    expected_version = body.get("expected_version", 1)
    strategy_name = body.get("strategy", "lww")
    force = body.get("force", False)

    if not node_id or not data:
        raise HTTPException(status_code=400, detail="node_id and data are required")

    # 将策略名映射到枚举
    strategy_map = {
        "lww": Strategy.LWW,
        "merge": Strategy.MERGE,
        "additive": Strategy.ADDITIVE,
    }
    strategy = strategy_map.get(strategy_name)
    if strategy is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{strategy_name}'. Use one of: lww, merge, additive",
        )

    # 获取或初始化 WriteReconciler
    reconciler: WriteReconciler = getattr(deps, "write_reconciler", None)
    if reconciler is None:
        # 如果 app 未注入，就地创建（使用共享的 conflict_logger）
        try:
            from core.transaction_manager import TransactionManager
            tx_mgr = getattr(deps, "tx_manager", None)
            conflict_logger = ConflictLogger(maxlen=1000)
            if tx_mgr is not None and hasattr(tx_mgr, "_conflict_log"):
                # 桥接 tx_manager 的 conflict_log 到 logger
                pass  # WriteReconciler 有自己的 ConflictLogger
            reconciler = WriteReconciler(
                kuzu_store=deps.kuzu_store,
                conflict_logger=conflict_logger,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to init reconciler: {e}")

    # 执行消解
    result = reconciler.resolve(
        node_id=node_id,
        incoming_data=data,
        expected_version=expected_version,
        strategy=strategy,
        force=force,
    )

    # 如果消解成功且无冲突（或 force），尝试写入
    if result["resolved"] and (not result["conflict"] or force):
        try:
            write_data = {k: v for k, v in result["data"].items()
                         if k in ("content", "source", "visibility")}
            write_data["id"] = node_id
            # 使用 update_with_version 确保写入原子性
            deps.kuzu_store.update_with_version(
                node_id=node_id,
                data=write_data,
                expected_version=result["current_version"] if not force else None,
            )
        except Exception as e:
            logger.exception("Reconcile write failed for node %s", node_id)
            result["write_error"] = str(e)

    # 同步记录到 tx_manager 的 conflict_log
    tx_mgr = getattr(deps, "tx_manager", None)
    if tx_mgr is not None and hasattr(tx_mgr, "record_conflict"):
        tx_mgr.record_conflict(
            node_id=node_id,
            expected_version=expected_version,
            current_version=result.get("current_version", 1),
            strategy=strategy.value,
            resolved=result["resolved"],
            detail=f"reconcile: conflict={result['conflict']} force={force}",
        )

    record_request("POST", "/conflicts/reconcile", "200", _now() - start)
    return {
        "conflict": result["conflict"],
        "resolved": result["resolved"],
        "strategy": result["strategy"],
        "data": result["data"],
        "expected_version": result["expected_version"],
        "current_version": result["current_version"],
    }


@router.get("/conflicts/reconcile/log", summary="查询 OCC 冲突消解日志")
async def get_reconcile_log(
    limit: int = Query(default=50, ge=1, le=1000),
    deps: Services = Depends(get_services),
) -> dict:
    """
    查询 WriteReconciler 的冲突消解日志（ring buffer）。
    按时间倒序返回最近 N 条。
    """
    start = _now()
    set_trace_id()

    reconciler: WriteReconciler = getattr(deps, "write_reconciler", None)
    logs = []
    occ_stats = {}
    if reconciler is not None:
        raw = reconciler.conflict_logger.query(limit=limit)
        logs = [
            {
                "node_id": r.node_id,
                "expected_version": r.expected_version,
                "current_version": r.current_version,
                "strategy": r.strategy.value,
                "resolved": r.resolved,
                "timestamp": r.timestamp,
                "detail": r.detail,
            }
            for r in raw
        ]
        occ_stats = reconciler.conflict_logger.stats()

    # 也合并 tx_manager 的 conflict_log
    tx_mgr = getattr(deps, "tx_manager", None)
    tx_logs: list[dict] = []
    if tx_mgr is not None and hasattr(tx_mgr, "get_conflict_log"):
        tx_logs = tx_mgr.get_conflict_log(limit=limit)

    record_request("GET", "/conflicts/reconcile/log", "200", _now() - start)
    return {
        "reconciler_log": logs,
        "reconciler_stats": occ_stats,
        "tx_manager_log": tx_logs,
        "total_reconciler": len(logs),
        "total_tx_manager": len(tx_logs),
    }


# ═══════════════════════════════════════════════════════════
# 多模态视觉 (Visual) 端点
# ═══════════════════════════════════════════════════════════

import os
import base64

VISUALS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "visuals")


@router.post("/memories/visual", summary="创建视觉记忆节点")
async def create_visual_memory(
    req: dict,
    deps: Services = Depends(get_services),
) -> dict:
    """创建视觉记忆节点。

    Body:
        image_base64: str — base64 编码的图像
        caption: str — 图像的文字描述（用于检索）
        source: str = "user" — 来源
    """
    start = _now()
    set_trace_id()

    image_b64 = req.get("image_base64", "")
    caption = req.get("caption", "")
    source = req.get("source", "user")
    if not image_b64 or not caption:
        raise HTTPException(status_code=400, detail="image_base64 and caption required")
    if deps.encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not available")
    if deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Kuzu store not available")

    visual_id = str(uuid.uuid4())
    created_at = _now()

    # 1. 保存图像到磁盘
    os.makedirs(VISUALS_DIR, exist_ok=True)
    image_path = os.path.join(VISUALS_DIR, f"{visual_id}.png")
    try:
        image_data = base64.b64decode(image_b64)
        with open(image_path, "wb") as f:
            f.write(image_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    # 2. 用 caption 文本编码（现有 all-MiniLM）
    emb = deps.encoder.embed(caption)
    if emb is None:
        raise HTTPException(status_code=500, detail="Embedding failed")
    emb_array = emb.reshape(-1).astype(np.float32)

    # 3. 存储到 Kuzu
    deps.kuzu_store.create_visual_node({
        "id": visual_id,
        "image_path": image_path,
        "caption": caption,
        "embedding": emb_array.tolist(),
        "source": source,
        "created_at": created_at,
    })

    record_request("POST", "/memories/visual", "200", _now() - start)
    return {
        "visual_id": visual_id,
        "caption": caption,
        "image_path": image_path,
        "created_at": created_at,
    }


@router.get("/memories/visual", summary="列出视觉记忆")
async def list_visual_memories(
    limit: int = Query(default=50, ge=1, le=500),
    deps: Services = Depends(get_services),
) -> dict:
    """列出所有视觉记忆节点。"""
    if deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Kuzu store not available")

    rows = deps.kuzu_store.get_visual_nodes(limit)
    items = []
    for r in rows:
        items.append({
            "id": r.get("id", ""),
            "caption": r.get("caption", ""),
            "image_path": r.get("image_path", ""),
            "source": r.get("source", ""),
            "created_at": r.get("created_at", 0.0),
        })
    return {"visuals": items, "total": len(items)}


@router.get("/memories/visual/{visual_id}", summary="查询视觉记忆详情")
async def get_visual_memory(
    visual_id: str,
    deps: Services = Depends(get_services),
) -> dict:
    """查询单个视觉记忆节点详情，含 base64 image。"""
    if deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Kuzu store not available")

    node = deps.kuzu_store.get_visual_node(visual_id)
    if not node:
        raise HTTPException(status_code=404, detail="Visual node not found")

    # 读取图像并转 base64
    image_data = ""
    image_path = node.get("image_path", "")
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    return {
        "id": node.get("id", ""),
        "caption": node.get("caption", ""),
        "image_base64": image_data,
        "source": node.get("source", ""),
        "created_at": node.get("created_at", 0.0),
    }


@router.get("/memories/visual/{visual_id}/heatmap", summary="生成注意力热图")
async def visualize_attention(
    visual_id: str,
    deps: Services = Depends(get_services),
) -> dict:
    """生成视觉记忆的注意力热图（基于 caption 关键词注意力模拟）。

    当真实 vision encoder 就绪后，此端点将替换为 VLM 注意力软图。
    当前版本：基于 caption 分词 + 关键词 TF-IDF 权重生成合成热图区域。
    """
    if deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Kuzu store not available")

    node = deps.kuzu_store.get_visual_node(visual_id)
    if not node:
        raise HTTPException(status_code=404, detail="Visual node not found")

    caption = node.get("caption", "")
    # 模拟热图：分词 → 每个词分配注意力权重
    words = caption.strip().split()
    total = max(len(words), 1)
    heat_regions = []
    for i, w in enumerate(words):
        # 权重从中心向边缘衰减
        pos_ratio = i / total
        weight = 1.0 - 0.5 * abs(pos_ratio - 0.5) * 2  # 中心词权重高
        heat_regions.append({
            "word": w,
            "weight": round(weight, 3),
            "position": round(pos_ratio, 3),
        })

    return {
        "visual_id": visual_id,
        "caption": caption,
        "heat_regions": heat_regions,
        "note": "Synthetic attention (real VLM attention when vision encoder is available)",
    }


@router.get("/hyperedges", summary="列出所有超边")
async def list_hyperedges(
    limit: int = Query(default=50, ge=1, le=500),
    deps: Services = Depends(get_services),
) -> HyperedgeListResponse:
    """查询所有超边（按创建时间倒序）。"""
    start = _now()
    set_trace_id()

    if deps.hyperedge_manager is None or deps.kuzu_store is None:
        raise HTTPException(status_code=503, detail="Hyperedge system not available")

    try:
        rows = deps.kuzu_store.query_cypher(
            "MATCH (h:HyperedgeNode) "
            "OPTIONAL MATCH (h)-[:HYPEREDGE_MEMBER]->(e:EpisodeNode) "
            "WITH h, collect(e.id) AS member_ids "
            "RETURN h.*, member_ids ORDER BY h.created_at DESC LIMIT $limit",
            {"limit": limit},
        )
        results = []
        import json as _j
        for row in rows:
            if isinstance(row, (list, tuple)):
                h = {"id": row[0], "type": row[1], "created_at": row[2], "gate_value": row[3], "metadata": row[4]}
                member_ids = list(row[5]) if len(row) > 5 else []
            elif isinstance(row, dict):
                h = {k.split(".")[-1]: v for k, v in row.items()}
                member_ids = list(h.pop("member_ids", []) or [])
            else:
                continue
            try:
                metadata = _j.loads(h.get("metadata", "{}")) if isinstance(h.get("metadata"), str) else h.get("metadata", {})
            except Exception:
                metadata = {}
            results.append(HyperedgeResponse(
                id=h["id"],
                type=APIHyperedgeType(h["type"]),
                member_ids=member_ids,
                created_at=h.get("created_at", 0.0),
                gate_value=h.get("gate_value", 1.0),
                metadata=metadata or {},
            ))

        record_request("GET", "/hyperedges", "200", _now() - start)
        return HyperedgeListResponse(hyperedges=results, total=len(results))
    except Exception as e:
        logger.exception("Failed to list hyperedges")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hyperedges/by-node/{node_id}", summary="查询节点所属的所有超边")
async def get_node_hyperedges(
    node_id: str,
    deps: Services = Depends(get_services),
) -> HyperedgeListResponse:
    """查询包含指定节点的所有超边。"""
    start = _now()
    set_trace_id()

    if deps.hyperedge_manager is None:
        raise HTTPException(status_code=503, detail="HyperedgeManager not available")

    edges = deps.hyperedge_manager.get_hyperedges_by_node(node_id)
    record_request("GET", f"/hyperedges/by-node/{node_id}", "200", _now() - start)
    return HyperedgeListResponse(
        hyperedges=[
            HyperedgeResponse(
                id=e.id,
                type=APIHyperedgeType(e.type.value),
                member_ids=e.member_ids,
                created_at=e.created_at,
                gate_value=e.gate_value,
                metadata=e.metadata,
            )
            for e in edges
        ],
        total=len(edges),
    )


# ─── 【P0】写入时自动创建超边 ────────────────────────────


async def _auto_create_hyperedges(episode_id: str, source: str, content: str, deps: Services) -> int:
    """
    写入新情节节点后自动检测并创建超边：
    - 时态超边：同一 source 在 300s 内写入的节点
    - 情节超边：同一 source 在 3600s 内连续写入的节点
    """
    if deps.hyperedge_manager is None or deps.kuzu_store is None:
        return 0

    try:
        created = 0
        now = _now()

        # 时态超边：最近 300s 内的同源节点
        recent_rows = deps.kuzu_store.query_cypher(
            "MATCH (e:EpisodeNode) WHERE e.id <> $id AND e.source = $src "
            "AND e.created_at >= $cutoff "
            "RETURN e.id ORDER BY e.created_at DESC LIMIT 5",
            {"id": episode_id, "src": source, "cutoff": now - 300},
        )
        recent_ids = []
        for row in recent_rows:
            if isinstance(row, (list, tuple)):
                recent_ids.append(str(row[0]))
            elif isinstance(row, dict):
                recent_ids.append(str(row.get("id", "")))
        if len(recent_ids) >= 2:
            # 用所有最近的 + 新节点一起创建时态超边
            member_ids = [episode_id] + recent_ids[:4]
            deps.hyperedge_manager.create_temporal_hyperedge(
                member_ids=member_ids,
                start_time=now - 300,
                end_time=now,
            )
            created += 1
            logger.info("Auto-created TEMPORAL hyperedge: %d members (source=%s)", len(member_ids), source)
        elif len(recent_ids) == 1:
            member_ids = [episode_id, recent_ids[0]]
            deps.hyperedge_manager.create_temporal_hyperedge(
                member_ids=member_ids, start_time=now - 300, end_time=now,
            )
            created += 1
            logger.info("Auto-created TEMPORAL hyperedge (pair): source=%s", source)

        # 情节超边：同一 source 在 3600s 内的节点池
        window_rows = deps.kuzu_store.query_cypher(
            "MATCH (e:EpisodeNode) WHERE e.id <> $id AND e.source = $src "
            "AND e.created_at >= $cutoff_window "
            "RETURN e.id ORDER BY e.created_at DESC LIMIT 20",
            {"id": episode_id, "src": source, "cutoff_window": now - 3600},
        )
        window_ids = []
        for row in window_rows:
            if isinstance(row, (list, tuple)):
                window_ids.append(str(row[0]))
            elif isinstance(row, dict):
                window_ids.append(str(row.get("id", "")))
        # 如果同一 source 在 1h 内有 5+ 个节点 → 创建情节超边
        if len(window_ids) >= 4:
            member_ids = [episode_id] + window_ids[:7]
            deps.hyperedge_manager.create_episode_hyperedge(
                member_ids=member_ids,
                topic=f"batch_{source}_{int(now)}",
            )
            created += 1
            logger.info("Auto-created EPISODE hyperedge: %d members (source=%s)", len(member_ids), source)

        return created
    except Exception as e:
        logger.warning("Auto-hyperedge creation failed (non-fatal): %s", e)
        return 0


# ─── Ontology v2 CRUD ────────────────────────────────────────


@router.post("/ontology/types", summary="注册实体类型")
async def register_entity_type(
    req: EntityTypeDefModel,
    deps: Services = Depends(get_services),
) -> EntityTypeDefModel:
    """注册新的实体类型定义"""
    from core.ontology_v2 import OntologyService, EntityTypeDef, AttributeDef, AttrType
    svc: OntologyService = deps.ontology_v2
    attr_defs = [AttributeDef(
        name=a.name,
        type=AttrType(a.type) if a.type in AttrType._value2member_map_ else AttrType.STRING,
        required=a.required,
        indexed=a.indexed,
        description=a.description,
        default=a.default,
        min_value=a.min_value,
        max_value=a.max_value,
        enum_values=a.enum_values,
    ) for a in req.attributes]
    edef = EntityTypeDef(name=req.name, description=req.description,
                         parent=req.parent, attributes=attr_defs)
    result = svc.register_entity_type(edef)
    return EntityTypeDefModel(name=result.name, description=result.description,
                               parent=result.parent, attributes=req.attributes)


@router.get("/ontology/types", summary="列出所有实体类型")
async def list_entity_types(
    request: Request,
    deps: Services = Depends(get_services),
) -> EntityTypeListResponse:
    """列出所有已注册的实体类型"""
    from core.ontology_v2 import OntologyService
    try:
        svc: OntologyService = deps.ontology_v2
        types = svc.list_entity_types()
        items = []
        for t in types:
            attrs = [
                AttributeDefModel(
                    name=a.name,
                    type=a.type.value if hasattr(a.type, 'value') else str(a.type),
                    required=a.required,
                    indexed=a.indexed,
                    description=a.description,
                )
                for a in (t.attributes or [])
            ]
            items.append(EntityTypeDefModel(
                name=t.name,
                description=t.description or "",
                parent=t.parent,
                attributes=attrs,
            ))
        return EntityTypeListResponse(entity_types=items, total=len(items))
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ontology/types/{name}", summary="查询实体类型详情")
async def get_entity_type(
    name: str,
    deps: Services = Depends(get_services),
) -> EntityTypeDefModel:
    from core.ontology_v2 import OntologyService
    svc: OntologyService = deps.ontology_v2
    t = svc.get_entity_type(name)
    if t is None:
        raise HTTPException(status_code=404, detail=f"Entity type '{name}' not found")
    return EntityTypeDefModel(
        name=t.name, description=t.description, parent=t.parent,
        attributes=[AttributeDefModel(name=a.name, type=a.type.value, required=a.required,
                                       indexed=a.indexed, description=a.description)
                    for a in t.attributes],
    )


@router.delete("/ontology/types/{name}", summary="删除实体类型")
async def delete_entity_type(
    name: str,
    deps: Services = Depends(get_services),
) -> dict:
    from core.ontology_v2 import OntologyService
    svc: OntologyService = deps.ontology_v2
    try:
        deleted = svc.delete_entity_type(name)
        return {"deleted": deleted, "name": name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ontology/edges", summary="注册边类型")
async def register_edge_type(
    req: EdgeTypeDefModel,
    deps: Services = Depends(get_services),
) -> EdgeTypeDefModel:
    from core.ontology_v2 import OntologyService, EdgeTypeDef, EdgeAttributeDef, AttrType
    svc: OntologyService = deps.ontology_v2
    attrs = [EdgeAttributeDef(name=a.name, type=AttrType(a.type), required=a.required, description=a.description)
             for a in req.attributes]
    edef = EdgeTypeDef(name=req.name, description=req.description,
                       source_types=req.source_types, target_types=req.target_types,
                       attributes=attrs, symmetry=req.symmetry)
    result = svc.register_edge_type(edef)
    return EdgeTypeDefModel(name=result.name, description=result.description,
                             source_types=result.source_types, target_types=result.target_types,
                             symmetry=result.symmetry)


@router.get("/ontology/edges", summary="列出所有边类型")
async def list_edge_types(
    deps: Services = Depends(get_services),
) -> EdgeTypeListResponse:
    from core.ontology_v2 import OntologyService
    svc: OntologyService = deps.ontology_v2
    types = svc.list_edge_types()
    items = [EdgeTypeDefModel(name=t.name, description=t.description,
                               source_types=t.source_types, target_types=t.target_types,
                               symmetry=t.symmetry)
             for t in types]
    return EdgeTypeListResponse(edge_types=items, total=len(items))


@router.get("/ontology/edges/{name}", summary="查询边类型详情")
async def get_edge_type(
    name: str,
    deps: Services = Depends(get_services),
) -> EdgeTypeDefModel:
    from core.ontology_v2 import OntologyService
    svc: OntologyService = deps.ontology_v2
    t = svc.get_edge_type(name)
    if t is None:
        raise HTTPException(status_code=404, detail=f"Edge type '{name}' not found")
    return EdgeTypeDefModel(name=t.name, description=t.description,
                             source_types=t.source_types, target_types=t.target_types,
                             symmetry=t.symmetry)


@router.delete("/ontology/edges/{name}", summary="删除边类型")
async def delete_edge_type(
    name: str,
    deps: Services = Depends(get_services),
) -> dict:
    from core.ontology_v2 import OntologyService
    svc: OntologyService = deps.ontology_v2
    deleted = svc.delete_edge_type(name)
    return {"deleted": deleted, "name": name}


@router.get("/ontology/stats", summary="本体系统统计")
async def ontology_stats(
    deps: Services = Depends(get_services),
) -> OntologyStatsResponse:
    from core.ontology_v2 import OntologyService
    svc: OntologyService = deps.ontology_v2
    return OntologyStatsResponse(
        entity_type_count=len(svc.entity_types),
        edge_type_count=len(svc.edge_types),
        baseline_loaded=True,
    )


@router.get("/evidence/stats", summary="置信度统计")
async def evidence_stats(
    deps: Services = Depends(get_services),
) -> Dict[str, Any]:
    """返回置信度追踪器的统计信息"""
    if deps.evidence_tracker is None:
        return {"status": "disabled", "message": "Evidence tracker not initialized"}
    return deps.evidence_tracker.stats()


# ─── 实体自动发现 ─────────────────────────────────────────────


class DiscoverResponse(BaseModel):
    status: str = ""
    total_nodes_scanned: int = 0
    candidate_count: int = 0
    proposed_types: List[Dict[str, Any]] = []
    scan_time_ms: float = 0.0
    entities: List[Dict[str, Any]] = []


class DiscoverApplyResponse(BaseModel):
    status: str = ""
    types_registered: int = 0
    skipped_existing: int = 0
    total_candidates: int = 0


@router.post("/ontology/discover", summary="自动发现候选实体和类型")
async def ontology_discover(
    req: Request,
    deps: Services = Depends(get_services),
) -> DiscoverResponse:
    """扫描已有数据，自动发现候选实体和类型定义"""
    from core.entity_discovery import EntityDiscoveryEngine
    from core.ontology_v2 import OntologyService

    engine = EntityDiscoveryEngine(ontology=deps.ontology_v2)

    # 从 Kuzu 获取内容样本
    contents = []
    if deps.kuzu_store:
        try:
            rows = deps.kuzu_store.query_cypher(
                "MATCH (e:EpisodeNode) RETURN e.content LIMIT 2000"
            )
            for r in rows:
                if isinstance(r, (list, tuple)) and len(r) > 0:
                    contents.append(str(r[0]))
                elif isinstance(r, dict):
                    contents.append(str(r.get("e.content", r.get("content", ""))))
        except Exception:
            pass

    result = engine.scan(contents, min_occurrences=2, max_candidates=50)

    entities_out = []
    for e in result.candidate_entities[:20]:
        entities_out.append({
            "name": e.canonical_name,
            "type": e.inferred_type,
            "occurrences": e.occurrences,
            "confidence": round(e.confidence, 2),
            "aliases": e.aliases[:5],
        })

    types_out = []
    for t in result.proposed_types:
        types_out.append({
            "name": t.name,
            "description": t.description,
            "entity_count": t.entity_count,
            "sample_entities": t.sample_entities[:5],
        })

    return DiscoverResponse(
        status="ok",
        total_nodes_scanned=result.total_nodes_scanned,
        candidate_count=len(result.candidate_entities),
        proposed_types=types_out,
        scan_time_ms=result.scan_time_ms,
        entities=entities_out,
    )


@router.post("/ontology/discover/apply", summary="应用发现的候选到本体系统")
async def ontology_discover_apply(
    req: Request,
    deps: Services = Depends(get_services),
) -> DiscoverApplyResponse:
    """将自动发现的候选类型注册到 Ontology v2"""
    from core.entity_discovery import EntityDiscoveryEngine
    from core.ontology_v2 import OntologyService

    engine = EntityDiscoveryEngine(ontology=deps.ontology_v2)

    # 与 discover 同样的扫描逻辑
    contents = []
    if deps.kuzu_store:
        try:
            rows = deps.kuzu_store.query_cypher(
                "MATCH (e:EpisodeNode) RETURN e.content LIMIT 2000"
            )
            for r in rows:
                if isinstance(r, (list, tuple)) and len(r) > 0:
                    contents.append(str(r[0]))
                elif isinstance(r, dict):
                    contents.append(str(r.get("e.content", r.get("content", ""))))
        except Exception:
            pass

    result = engine.scan(contents, min_occurrences=2, max_candidates=50)
    apply_result = engine.apply_to_ontology(result, auto_register=True)

    return DiscoverApplyResponse(
        status=apply_result.get("status", "ok"),
        types_registered=apply_result.get("types_registered", 0),
        skipped_existing=apply_result.get("skipped_existing", 0),
        total_candidates=apply_result.get("total_candidates", 0),
    )

from pydantic import BaseModel, Field
from typing import Optional, List as PyList


class BatchRelationInput(BaseModel):
    """批量写入关系边的输入"""
    relations: PyList[dict] = Field(..., description="关系三元组列表")
    source: str = Field("system", description="数据来源")


@router.post("/batch/relations", summary="批量写入关系边")
async def batch_relations(
    input_data: BatchRelationInput,
    deps: Services = Depends(get_services),
) -> dict:
    """批量将抽取的三元组写入 RELATES_TO 边（不创建 EpisodeNode）"""
    results = {"total": len(input_data.relations), "created": 0, "errors": 0, "error_details": []}
    start = _now()

    for item in input_data.relations:
        subj = item.get("subject", "").strip()
        rel = item.get("relation", "").strip()
        obj = item.get("object", "").strip()
        if not subj or not rel or not obj:
            results["errors"] += 1
            continue

        try:
            # 确保实体存在
            deps.kuzu_store.query_cypher(
                "MERGE (a:OntologyEntity {name: $name})",
                {"name": subj}
            )
            deps.kuzu_store.query_cypher(
                "MERGE (a:OntologyEntity {name: $name})",
                {"name": obj}
            )

            # 尝试语义化边类型
            rel_type = item.get("edge_type", "RELATES_TO")
            deps.kuzu_store.query_cypher(
                f"MATCH (a:OntologyEntity {{name: $subj}}), "
                f"(b:OntologyEntity {{name: $obj}}) "
                f"MERGE (a)-[r:{rel_type} {{relation: $rel}}]->(b)",
                {"subj": subj, "obj": obj, "rel": rel}
            )
            results["created"] += 1
        except Exception as e:
            results["errors"] += 1
            if len(results["error_details"]) < 3:
                results["error_details"].append(str(e)[:80])

    set_trace_id()
    record_request("POST", "/batch/relations", "200", _now() - start)
    return results


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


# ====================================================================
# v2.0: 工作记忆（Session Memory）路由
# ====================================================================

import uuid as _uuid


@router.post("/sessions/{session_id}/working-memory",
             summary="写入工作记忆（会话级临时上下文，不持久化到Kuzu）")
async def write_session_memory(
    session_id: str,
    body: SessionMemoryCreate,
    svc: Services = Depends(get_services),
):
    """写入一条工作记忆。工作记忆是会话级临时上下文，不持久化到Kuzu图数据库。
    
    用于追踪当前任务中的Agent状态、正在处理的上下文。
    类比 Human-Inspired Memory Architecture 的「工作记忆」层。
    """
    if svc._session_memory_lock is None:
        svc._session_memory_lock = __import__("threading").Lock()
    
    item = {
        "id": str(_uuid.uuid4()),
        "session_id": session_id,
        "content": body.content,
        "metadata": body.metadata or {},
        "created_at": _now(),
    }
    
    with svc._session_memory_lock:
        if session_id not in svc._session_memory:
            svc._session_memory[session_id] = []
        svc._session_memory[session_id].append(item)
        # 最多保留100条工作记忆
        if len(svc._session_memory[session_id]) > 100:
            svc._session_memory[session_id] = svc._session_memory[session_id][-100:]
    
    return {"id": item["id"], "session_id": session_id, "status": "created"}


@router.get("/sessions/{session_id}/working-memory",
            summary="查询工作记忆（最近N条，按时间倒序）")
async def read_session_memory(
    session_id: str,
    limit: int = 20,
    svc: Services = Depends(get_services),
):
    """查询工作记忆。返回最近N条记忆，按时间倒序。"""
    with svc._session_memory_lock:
        memories = svc._session_memory.get(session_id, [])
        recent = list(reversed(memories))[:limit]
    
    return SessionMemoryListResponse(
        session_id=session_id,
        results=[SessionMemoryItem(**m) for m in recent],
        total=len(memories),
    )


@router.delete("/sessions/{session_id}/working-memory",
               summary="清除工作记忆（指定memory_id则清除单条）")
async def delete_session_memory(
    session_id: str,
    memory_id: str = None,
    svc: Services = Depends(get_services),
):
    """清除工作记忆。指定memory_id只清除单条，否则清除整个会话。"""
    with svc._session_memory_lock:
        if memory_id:
            if session_id in svc._session_memory:
                svc._session_memory[session_id] = [
                    m for m in svc._session_memory[session_id]
                    if m["id"] != memory_id
                ]
            return {"status": "deleted", "session_id": session_id, "memory_id": memory_id}
        else:
            svc._session_memory.pop(session_id, None)
            return {"status": "cleared", "session_id": session_id}


# ====================================================================
# Phase 2: 程序记忆路由
# ====================================================================

@router.get("/procedural/patterns",
            summary="查询程序模式（重复行动模式）")
async def list_procedural_patterns(
    min_confidence: float = 0.3,
    svc: Services = Depends(get_services),
):
    """查询高频重复行动模式——程序记忆层。"""
    from core.procedural_memory import ProceduralMemoryEngine
    engine = ProceduralMemoryEngine(kuzu_store=svc.kuzu_store)
    patterns = engine.query_patterns(min_confidence=min_confidence)
    return {"patterns": patterns, "total": len(patterns)}


# ====================================================================
# Phase 2: 概念记忆路由
# ====================================================================

@router.get("/conceptual/concepts",
            summary="查询跨社区概念（最高抽象层）")
async def list_concepts(
    abstraction_level: str = None,
    svc: Services = Depends(get_services),
):
    """查询跨社区的抽象概念——概念记忆层。"""
    from core.conceptual_memory import ConceptualMemoryEngine
    engine = ConceptualMemoryEngine(kuzu_store=svc.kuzu_store)
    concepts = engine.get_concepts(level=abstraction_level)
    return {"concepts": concepts, "total": len(concepts)}


@router.post("/conceptual/analyze",
             summary="分析社区并生成概念")
async def analyze_concepts(
    svc: Services = Depends(get_services),
):
    """分析现有社区，自动发现跨社区概念。"""
    from core.conceptual_memory import ConceptualMemoryEngine
    # 获取所有社区
    result = svc.kuzu_store.conn.execute(
        "MATCH (c:CommunityNode) RETURN c.* ORDER BY c.created_at DESC"
    )
    rows = result.get_as_pl()
    dicts = rows.to_dicts() if rows else []
    communities = []
    for r in dicts:
        r = _clean_kuzu_row(r)
        communities.append({
            "id": r.get("id", ""),
            "name": r.get("name", ""),
            "summary": r.get("summary", ""),
            "keywords": json.loads(r.get("keywords", "[]")) if isinstance(r.get("keywords"), str) else [],
            "nodes": [],
        })
    
    engine = ConceptualMemoryEngine(kuzu_store=svc.kuzu_store)
    concepts = engine.analyze_communities(communities)
    return {"concepts_found": len(concepts), "concepts": concepts}
