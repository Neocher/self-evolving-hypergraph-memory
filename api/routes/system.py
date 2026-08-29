"""
系统路由 (health, metrics, audit, index, procedural, conceptual)
"""

import threading

from api.routes._deps import (
    router, Services, get_services, _now, logger,
    set_trace_id, record_request, get_metrics, record_circuit_breaker,
    Depends, HTTPException, Query, Request, Response,
    uuid, np, json,
    AuditTrace, AuditOperation, HealthStatus,
    CypherQueryRequest,
    HealthChecker, HealthCheckResult,
    __version__, __version_name__,
    Dict, Any,
)
from graph.common import _gql_value

# TTL 缓存：graph_store 相关统计（node_count/hyperedge_count），避免每次 /health 全扫描
_HEALTH_STATS_CACHE: Dict[str, int] = {"node_count": 0, "hyperedge_count": 0}
_HEALTH_STATS_CACHE_TIME: float = 0.0
_HEALTH_STATS_CACHE_LOCK = threading.Lock()
_HEALTH_STATS_TTL: float = 5.0

# Hebbian 批量建边：GraphLite 单查询上限（实测 20 对 OK，25 对静默丢弃）。
# 分号拼接（"; ".join）在 GraphLite 会静默截断只执行第一条 / QUERY_ERROR，
# 启动时多批失败会打满熔断窗口 → 全库假死（v5.31.1 修复）。
HEBBIAN_BATCH = 20


def _flush_hebbian_batch(store, pairs: list) -> bool:
    """GraphLite 批量建边：单条多模式 MATCH + 多边 INSERT（逗号分隔）。

    Args:
        store: GraphLiteStore 实例
        pairs: [(src_id, dst_id, weight), ...]，长度 ≤ HEBBIAN_BATCH
    Returns:
        bool: execute_cypher 有返回行视为成功（不吞异常契约——失败 raise，
            由调用方 try/except 处理；写路径熔断中立，不 record_success/failure）
    """
    if not pairs:
        return True
    match_parts: list[str] = []
    insert_parts: list[str] = []
    for i, (src, dst, w) in enumerate(pairs):
        # 【H1】id 经 _gql_value 转义（含 ' / \ 的 id 不再裸插注入 GQL）
        match_parts.append(f"(a{i}:EpisodeNode {{id: {_gql_value(str(src))}}})")
        match_parts.append(f"(b{i}:EpisodeNode {{id: {_gql_value(str(dst))}}})")
        insert_parts.append(f"(a{i})-[:HEBBIAN_CONNECTION {{weight: {w}}}]->(b{i})")
    gql = f"MATCH {', '.join(match_parts)} INSERT {', '.join(insert_parts)}"
    rows = store.execute_cypher(gql)
    return bool(rows)


def _recompute_status(result: HealthCheckResult) -> str:
    if not result.graph_connected:
        return "error"
    if not result.chain_verified or not result.faiss_loaded:
        return "degraded"
    return "ok"


def _decode_b64(s: str) -> str:
    """GraphLite 对 UTF-8 内容做 {b64}<base64> 透明编解码，此处解码回明文。"""
    if s.startswith("{b64}"):
        try:
            import base64
            return base64.b64decode(s[5:]).decode("utf-8", errors="replace")
        except Exception:
            return s
    return s


def _memory_status(rss_mb, warning_mb: int, critical_mb: int) -> str:
    """根据 RSS 阈值返回内存状态 (ok/warning/critical/unknown)。"""
    if rss_mb is None:
        return "unknown"
    if rss_mb > critical_mb:
        return "critical"
    if rss_mb > warning_mb:
        return "warning"
    return "ok"


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


@router.get("/health", summary="深度健康检查")
async def health_check(
    deps: Services = Depends(get_services),
) -> HealthStatus:
    """
    深度健康检查，覆盖所有核心组件：
    - GraphLite 连接 + 断路器状态
    - FAISS 索引状态
    - BLAKE3 溯源链完整性
    - 梦境调度器状态

    node_count / hyperedge_count 使用 5s TTL 缓存，避免每次全扫描 COUNT(*)。
    """
    global _HEALTH_STATS_CACHE, _HEALTH_STATS_CACHE_TIME

    start = _now()
    set_trace_id()

    checker = HealthChecker(
        graph_store=deps.graphlite_store,
        faiss_index=deps.faiss_index,
        audit_chain=deps.audit_chain,
        dream_scheduler=deps.dream_scheduler,
    )

    # 检查 TTL 缓存：5s 内跳过昂贵的 COUNT(*) 查询
    with _HEALTH_STATS_CACHE_LOCK:
        cache_hit = (start - _HEALTH_STATS_CACHE_TIME) < _HEALTH_STATS_TTL

    if cache_hit:
        # 临时置空 graph_store → check() 跳过 COUNT(*) 全扫描
        checker.graph_store = None
        health: HealthCheckResult = checker.check()

        # 恢复缓存的统计值
        with _HEALTH_STATS_CACHE_LOCK:
            health.node_count = _HEALTH_STATS_CACHE["node_count"]
            health.hyperedge_count = _HEALTH_STATS_CACHE["hyperedge_count"]

        # 用快速 RETURN 1 验证图连接（COUNT(*) 全扫描的 1/10 以内延迟）
        # P2 fix: query_cypher 永不抛异常，必须检查返回值
        if deps.graphlite_store is not None:
            try:
                rows = deps.graphlite_store.query_cypher("RETURN 1 AS test")
                health.graph_connected = bool(rows)
            except Exception:
                health.graph_connected = False

        # P1 fix: checker.graph_store=None 导致 _check_circuit_breaker()
        # 返回 {"state":"unknown"}。此处从 deps.graphlite_store 重建。
        cb = getattr(deps.graphlite_store, "circuit_breaker", None)
        if cb is not None:
            window = getattr(cb, "_window", [])
            ws = len(window)
            rf = sum(1 for r in window if not r) if ws > 0 else 0
            health.details["circuit_breaker"] = {
                "state": cb.state.value if hasattr(cb.state, "value") else str(cb.state),
                "window_size": ws,
                "recent_failures": rf,
                "success_rate": ((ws - rf) / ws * 100) if ws > 0 else 100.0,
            }
        elif deps.graphlite_store is not None:
            health.details["circuit_breaker"] = {"state": "not_configured"}

        # 重算整体状态（check() 在 graph_store=None 时将 status 误判为 error）
        health.status = _recompute_status(health)
    else:
        health: HealthCheckResult = checker.check()
        # 刷新缓存
        with _HEALTH_STATS_CACHE_LOCK:
            _HEALTH_STATS_CACHE["node_count"] = health.node_count
            _HEALTH_STATS_CACHE["hyperedge_count"] = health.hyperedge_count
            _HEALTH_STATS_CACHE_TIME = start

    cb = getattr(deps.graphlite_store, "circuit_breaker", None)
    if cb is not None:
        state_map = {"closed": 0, "half_open": 1, "open": 2}
        cb_state = cb.state.value if hasattr(cb.state, "value") else str(cb.state)
        record_circuit_breaker("ryu", state_map.get(cb_state, 0))

    memory: Dict[str, Any] = dict(health.details.get("memory_usage", {}))
    try:
        from config.settings import get_settings
        hcfg = get_settings().health
        memory["memory_status"] = _memory_status(
            memory.get("rss_mb"), hcfg.memory_warning_mb, hcfg.memory_critical_mb
        )
    except Exception:
        memory["memory_status"] = "unknown"

    stats: Dict[str, Any] = {
        "version": __version__,
        "version_name": __version_name__,
        "uptime_seconds": health.uptime_seconds,
        # 【2026-08-23】命名修正：向量索引已由 OverGraph HNSW 承担（v6.0.0 起
        # GraphLite/FAISS 均弃用），faiss_index_size 为历史字段名——新增
        # vector_index_size 主字段，旧字段保留兼容（监控脚本迁移后移除）。
        "vector_index_size": health.faiss_index_size,
        "faiss_index_size": health.faiss_index_size,  # deprecated: 兼容旧监控
        "chain_verified": health.chain_verified,
        "node_count": health.node_count,
        "hyperedge_count": health.hyperedge_count,
        "last_dream_time": health.last_dream_time,
        "dream_run_count": health.dream_run_count,
        "circuit_breaker": health.details.get("circuit_breaker", {}),
        "memory": memory,
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


@router.post("/query", summary="执行 Cypher 查询（只读代理）")
async def cypher_query(
    req: CypherQueryRequest,
    deps: Services = Depends(get_services),
) -> dict:
    import re
    stripped = re.sub(r'"[^"]*"|\'[^\']*\'|`[^`]*`', '', req.query)
    blocked_pattern = re.compile(
        r'\b(?:CREATE|DELETE|SET|DROP|MERGE|REMOVE|DETACH|INSERT|LOAD\s+CSV)\b',
        re.IGNORECASE
    )
    if blocked_pattern.search(stripped):
        from fastapi import HTTPException as _HE
        raise _HE(status_code=400, detail=f"Write queries blocked: contains CREATE/DELETE/SET/DROP/MERGE/REMOVE/DETACH/INSERT/LOAD CSV")
    try:
        rows = deps.graphlite_store.query_cypher(req.query, req.params)
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


@router.get("/evidence/stats", summary="置信度统计")
async def evidence_stats(
    deps: Services = Depends(get_services),
) -> Dict[str, Any]:
    """返回置信度追踪器的统计信息"""
    if deps.evidence_tracker is None:
        return {"status": "disabled", "message": "Evidence tracker not initialized"}
    return deps.evidence_tracker.stats()


def _rebuild_index_overgraph(deps: Services, adapter) -> dict:
    """v6.0.0 OverGraph 后端索引重建（D8）：dense_vector 批量写 + Hebbian 近邻。

    - 向量写入 EpisodeNode.dense_vector（adapter.rebuild 覆盖式）
    - Hebbian 边经 store.vector_search_dense（802×1 次，替代 FAISS search）：
      similarity = cosine s 直接用作边权重（R1 定标 d=1/s-1 的原始得分），
      s < 0.3 丢弃（与原 1-dist/2 阈值的保守对齐）
    - adapter 为共享引用，无需替换 deps.faiss_index / query_router 引用
    """
    store = deps.graphlite_store
    start = _now()
    rows = store.query_cypher(
        "MATCH (e:EpisodeNode) RETURN e.id AS id, e.content AS content LIMIT 10000"
    )
    if not rows:
        return {"status": "ok", "indexed_count": 0, "message": "No episodes found"}

    node_ids = []
    contents = []
    for row in rows:
        if isinstance(row, dict):
            nid = str(row.get("id", "") or "")
            content = str(row.get("content", "") or "")
        else:
            nid, content = "", ""
        if nid and content.strip():
            node_ids.append(nid)
            contents.append(content)

    if not contents:
        return {"status": "ok", "indexed_count": 0, "message": "No episodes with content"}

    logger.info("Rebuilding OverGraph vectors: encoding %d episodes", len(contents))
    embeddings = deps.encoder.embed_batch(contents)
    nodes = [{"node_id": nid, "embedding": vec}
             for nid, vec in zip(node_ids, embeddings)]
    indexed = adapter.rebuild(nodes)

    # Hebbian 批量建边（HEBBIAN_BATCH 上限与 graphlite 路径一致）
    hebbian_count = 0
    try:
        store.query_cypher("MATCH ()-[r:HEBBIAN_CONNECTION]->() DELETE r")
    except Exception:
        logger.exception("Failed to clear Hebbian connections")
    pending: list[tuple[str, str, float]] = []
    for i, qv in enumerate(embeddings):
        try:
            hits = store.vector_search_dense(6, qv)
            for ep_id, s in hits:
                if ep_id == node_ids[i] or s < 0.3:
                    continue
                pending.append((node_ids[i], ep_id, round(float(s), 4)))
                hebbian_count += 1
                if len(pending) >= HEBBIAN_BATCH:
                    try:
                        ok = _flush_hebbian_batch(store, pending)
                    except Exception as e:
                        logger.warning("Hebbian batch creation failed (%d pairs): %s",
                                       len(pending), e)
                        ok = False
                    if not ok:
                        logger.warning("Hebbian batch creation failed (%d pairs)",
                                       len(pending))
                    pending = []
        except Exception:
            logger.exception("OverGraph vector search failed for Hebbian at index %d", i)
    if pending:
        try:
            ok = _flush_hebbian_batch(store, pending)
        except Exception as e:
            logger.warning("Hebbian batch creation failed (%d pairs): %s",
                           len(pending), e)
            ok = False
        if not ok:
            logger.warning("Hebbian batch creation failed (%d pairs)", len(pending))

    record_request("POST", "/index/rebuild", "200", _now() - start)
    logger.info("OverGraph index rebuilt: %d vectors, %d Hebbian connections",
                indexed, hebbian_count)

    try:
        tfidf = getattr(deps, "tfidf_index", None)
        if tfidf is not None and hasattr(tfidf, "fit") and contents:
            tfidf.fit(contents)
            logger.info("TF-IDF index fitted with %d texts", len(contents))
    except Exception:
        logger.exception("TF-IDF fit failed (non-fatal)")

    return {
        "status": "ok",
        "indexed_count": indexed,
        "total_nodes": len(node_ids),
        "dimension": getattr(adapter, "dimension", 0),
        "hebbian_connections": hebbian_count,
    }


@router.post("/index/rebuild", summary="重建 FAISS 索引")
async def rebuild_index(
    deps: Services = Depends(get_services),
) -> dict:
    """
    重建 FAISS 向量索引：
    1. 从 GraphLite 读取所有 EpisodeNode
    2. 对每个节点生成 embedding
    3. 重建 FAISS IndexIDMap(FlatL2)
    4. 返回重建结果统计
    """
    start = _now()
    set_trace_id()

    if deps.graphlite_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")
    if deps.encoder is None:
        raise HTTPException(status_code=503, detail="Text encoder not available")

    # v6.0.0 OverGraph 分支（D8）：batch_upsert dense_vector + Hebbian 改 vector_search
    from retrieval.vector_index import VectorIndexAdapter
    if isinstance(getattr(deps, "faiss_index", None), VectorIndexAdapter):
        return _rebuild_index_overgraph(deps, deps.faiss_index)

    # graphlite 回滚分支（真 FAISS 重建：IVFFlat/FlatL2 + Hebbian 循环）已随
    # faiss-cpu 移除（报告 3.2 / 4.1-2）。backend 恒为 overgraph → 此处不可达；
    # 保留显式报错以防未来后端回退时静默空索引。
    raise HTTPException(
        status_code=503,
        detail="FAISS index rebuild requires OverGraph backend (graphlite/faiss removed)",
    )


@router.get("/procedural/patterns",
            summary="查询程序模式（重复行动模式）")
async def list_procedural_patterns(
    min_confidence: float = 0.3,
    svc: Services = Depends(get_services),
):
    """查询高频重复行动模式——程序记忆层。"""
    from core.procedural_memory import ProceduralMemoryEngine
    engine = ProceduralMemoryEngine(graphlite_store=svc.graphlite_store)
    patterns = engine.query_patterns(min_confidence=min_confidence)
    return {"patterns": patterns, "total": len(patterns)}


@router.get("/conceptual/concepts",
            summary="查询跨社区概念（最高抽象层）")
async def list_concepts(
    abstraction_level: str = None,
    svc: Services = Depends(get_services),
):
    """查询跨社区的抽象概念——概念记忆层。"""
    from core.conceptual_memory import ConceptualMemoryEngine
    engine = ConceptualMemoryEngine(graphlite_store=svc.graphlite_store)
    concepts = engine.get_concepts(level=abstraction_level)
    return {"concepts": concepts, "total": len(concepts)}


@router.post("/conceptual/analyze",
             summary="分析社区并生成概念")
async def analyze_concepts(
    svc: Services = Depends(get_services),
):
    """分析现有社区，自动发现跨社区概念。"""
    from core.conceptual_memory import ConceptualMemoryEngine
    rows = svc.graphlite_store.query_cypher(
        "MATCH (c:CommunityNode) RETURN c.* ORDER BY c.created_at DESC"
    )
    communities = []
    for r in rows:
        if isinstance(r, (list, tuple)):
            r = {"id": str(r[0]), "name": str(r[1]) if len(r) > 1 else "",
                 "summary": str(r[2]) if len(r) > 2 else "",
                 "keywords": r[3] if len(r) > 3 else []}
        elif isinstance(r, dict):
            r = {k.split(".")[-1]: v for k, v in r.items()}
        communities.append({
            "id": r.get("id", ""),
            "name": r.get("name", ""),
            "summary": r.get("summary", ""),
            "keywords": json.loads(r.get("keywords", "[]")) if isinstance(r.get("keywords"), str) else [],
            "nodes": [],
        })

    engine = ConceptualMemoryEngine(graphlite_store=svc.graphlite_store)
    concepts = engine.analyze_communities(communities)
    return {"concepts_found": len(concepts), "concepts": concepts}
