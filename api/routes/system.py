"""
系统路由 (health, metrics, audit, index, procedural, conceptual)
"""

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
    - Kuzu 连接 + 断路器状态
    - FAISS 索引状态
    - BLAKE3 溯源链完整性
    - 梦境调度器状态
    """
    start = _now()
    set_trace_id()

    checker = HealthChecker(
        graph_store=deps.kuzu_store,
        faiss_index=deps.faiss_index,
        audit_chain=deps.audit_chain,
        dream_scheduler=deps.dream_scheduler,
    )
    health: HealthCheckResult = checker.check()

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
        "dream_run_count": health.dream_run_count,
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


@router.get("/evidence/stats", summary="置信度统计")
async def evidence_stats(
    deps: Services = Depends(get_services),
) -> Dict[str, Any]:
    """返回置信度追踪器的统计信息"""
    if deps.evidence_tracker is None:
        return {"status": "disabled", "message": "Evidence tracker not initialized"}
    return deps.evidence_tracker.stats()


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

    rows = deps.kuzu_store.query_cypher(
        "MATCH (e:EpisodeNode) RETURN e.id, e.content LIMIT 10000"
    )
    if not rows:
        return {"status": "ok", "indexed_count": 0, "message": "No episodes found"}

    node_ids = []
    contents = []
    for row in rows:
        # GraphLite 返回深层嵌套 {"e": {"Node": {"properties": {...}}}}，
        # RyuStore 返回 [[id, content]] —— 两种格式都兼容
        if isinstance(row, dict) and "e" in row:
            try:
                from graph.graphlite_store import GraphLiteStore
                flat = GraphLiteStore._flatten_row(row, "e")
                nid, content = str(flat.get("id", "")), str(flat.get("content", ""))
            except Exception:
                nid, content = str(row.get("id", "")), str(row.get("content", ""))
        elif isinstance(row, (list, tuple)):
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

    logger.info("Rebuilding FAISS index: encoding %d episodes", len(contents))
    embeddings = deps.encoder.embed_batch(contents)

    dim = deps.faiss_dim
    index_type = deps.faiss_index_type

    if index_type == "IVFFlat" and len(embeddings) >= max(deps.faiss_nlist * 2, 2000):
        nlist = min(deps.faiss_nlist, len(embeddings) // 2)
        quantizer = faiss.IndexFlatL2(dim)
        base_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        base_index.train(embeddings.astype(np.float32))
        base_index.nprobe = min(deps.faiss_nlist // 10, 10)
        new_index = faiss.IndexIDMap(base_index)
        logger.info("FAISS rebuilt with IVFFlat", dim=dim, nlist=nlist, nprobe=10, vectors=len(embeddings))
    else:
        new_index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        logger.info("FAISS rebuilt with FlatL2", dim=dim, vectors=len(embeddings))

    faiss_ids = np.array([
        uuid.uuid5(uuid.NAMESPACE_OID, str(nid)).int & ((1 << 63) - 1)
        for nid in node_ids
    ], dtype=np.int64)
    new_index.add_with_ids(embeddings.astype(np.float32), faiss_ids)

    deps.faiss_index = new_index
    if hasattr(deps, "faiss_id_map"):
        deps.faiss_id_map = dict(zip(faiss_ids.tolist(), node_ids))
    if deps.query_router is not None:
        deps.query_router.faiss_index = new_index
        deps.query_router.faiss_id_map = deps.faiss_id_map
        if hasattr(deps.query_router, '_qr'):
            deps.query_router._qr.faiss_index = new_index
            deps.query_router._qr.faiss_id_map = deps.faiss_id_map
        tfidf = getattr(deps, "tfidf_index", None)
        if tfidf is not None:
            deps.query_router.tfidf_index = tfidf

    logger.info("Rebuilding Hebbian connections...")
    hebbian_count = 0
    try:
        deps.kuzu_store.query_cypher("MATCH ()-[r:HEBBIAN_CONNECTION]->() DELETE r")
    except Exception:
        logger.exception("Failed to clear Hebbian connections")
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
                    logger.exception("Hebbian connection creation failed for node %s", node_ids[i])
        except Exception:
            logger.exception("FAISS search failed for Hebbian rebuild at index %d", i)

    record_request("POST", "/index/rebuild", "200", _now() - start)
    logger.info("FAISS index rebuilt: %d vectors, %d Hebbian connections",
                new_index.ntotal, hebbian_count)

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
    rows = svc.kuzu_store.query_cypher(
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

    engine = ConceptualMemoryEngine(kuzu_store=svc.kuzu_store)
    concepts = engine.analyze_communities(communities)
    return {"concepts_found": len(concepts), "concepts": concepts}
