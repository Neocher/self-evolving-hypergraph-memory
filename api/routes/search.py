"""
检索路由 (retrieve, vector, cypher, namespace, sessions, working-memory)
"""

from api.routes._deps import (
    router, Services, get_services, _now, logger,
    _result_cache, _result_cache_lock, _result_cache_max,
    set_trace_id, record_request,
    Depends, Query, HTTPException,
    uuid, np, time, threading,
    RetrieveRequest, RetrieveResponse, EpisodicResult,
    SearchVectorRequest, SearchVectorResult, SearchVectorResponse,
    SessionMemoryCreate, SessionMemoryItem, SessionMemoryListResponse,
)


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

    # 【Defense】类型守卫：上游若返回非 list，置空避免下游 AttributeError
    if not isinstance(results_raw, list):
        logger.warning("Retrieval returned non-list type %s, resetting to []",
                       type(results_raw).__name__)
        results_raw = []

    # 当所有上游检索都返回空时，直接 Cypher 兜底
    if not results_raw and deps.graphlite_store is not None:
        try:
            words = [w.strip().lower() for w in req.query.split() if len(w.strip()) > 1]
            if words:
                params = {f"w{i}": w for i, w in enumerate(words[:5])}
                conditions = " OR ".join(f"toLower(e.content) CONTAINS $w{i}" for i in range(len(words[:5])))
                cypher = (f"MATCH (e:EpisodeNode) WHERE ({conditions}) "
                          "AND (e.quarantine IS NULL OR e.quarantine = false) "
                          f"RETURN e.id AS node_id, e.content AS content LIMIT 10")
                fallback_rows = deps.graphlite_store.query_cypher(cypher, params)
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
                        "level": "graphlite_fallback",
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
        if req.namespace and deps.graphlite_store is not None:
            try:
                ns_rows = deps.graphlite_store.query_cypher(
                    "MATCH (s:SessionNode {session_id: $ns})-[:SESSION_MEMBER]->(e:EpisodeNode) "
                    "RETURN e.id",
                    {"ns": req.namespace}
                )
                ns_set = {row[0] for row in ns_rows} if ns_rows else set()
            except Exception:
                logger.warning("Namespace query failed, skipping namespace filter")
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
    if deps.graphlite_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")
    try:
        count = deps.graphlite_store.delete_namespace(namespace)
        return {"deleted": count, "namespace": namespace, "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/vector", summary="纯向量检索（直通 FAISS）")
async def search_vector(
    req: SearchVectorRequest,
    deps: Services = Depends(get_services),
) -> SearchVectorResponse:
    """使用 FAISS 向量索引执行纯向量检索，回查 GraphLite 获取节点详情。

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

        # 3. 回查 GraphLite 获取节点详情
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

            # 从 GraphLite 获取节点详情
            try:
                node = deps.graphlite_store.get_episode(episode_id) if deps.graphlite_store else None
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


@router.get("/sessions/{session_id}/memories", summary="查询会话的所有记忆")
async def get_session_memories(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    deps: Services = Depends(get_services),
) -> dict:
    """查询指定会话 ID 关联的所有记忆节点。"""
    start = _now()
    set_trace_id()

    if deps.graphlite_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")

    rows = deps.graphlite_store.get_session_memories(session_id, limit)
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


@router.post("/sessions/{session_id}/working-memory",
             summary="写入工作记忆（会话级临时上下文，不持久化到GraphLite）")
async def write_session_memory(
    session_id: str,
    body: SessionMemoryCreate,
    svc: Services = Depends(get_services),
):
    """写入一条工作记忆。工作记忆是会话级临时上下文，不持久化到GraphLite图数据库。
    
    用于追踪当前任务中的Agent状态、正在处理的上下文。
    类比 Human-Inspired Memory Architecture 的「工作记忆」层。
    """
    if svc._session_memory_lock is None:
        svc._session_memory_lock = threading.Lock()
    
    item = {
        "id": str(uuid.uuid4()),
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
