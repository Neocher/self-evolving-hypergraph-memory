"""
检索路由 (retrieve, vector, cypher, namespace, sessions, working-memory)
"""

import asyncio

from api.routes._deps import (
    router, Services, get_services, _now, logger,
    _result_cache, _result_cache_lock, _result_cache_max,
    set_trace_id, record_request,
    Depends, Query, HTTPException,
    uuid, np, time, threading,
    RetrieveRequest, RetrieveResponse, EpisodicResult,
    SearchVectorRequest, SearchVectorResult, SearchVectorResponse,
    SessionMemoryCreate, SessionMemoryItem, SessionMemoryListResponse,
    qsubmit,
)

from graph.common import CircuitBreakerOpen

from core.content_guard import scan_content

from retrieval.query_router import RetrievalLevel

# 【H2】外部检索超时（秒）：QueryRouter.retrieve 挂起（GraphLite/FAISS 卡死）时
# 超时返回空结果而非无限挂起（与写路径超时对称）。注：超时只解绑 await，不终止底层线程。
# 【P3b R1 P0-2】基线 3.0s；FUSION+HyDE 开启时 _retrieve_timeout 放宽到 5.0s
# （LLM 生成假设段落预留预算）。【PERF 2026-08-07】15s→3s: 大库下实体匹配/超边
# 遍历超时立即降级返回向量结果, 不空等
_RETRIEVE_TIMEOUT = 3.0

# 【H2-a】降级分支超时（秒）：Cypher 兜底 / 隔离 ID 拉取 / 命名空间预取
# 同样套 wait_for —— 若挂的是 GraphLite，主检索超时后走兜底会再次无限挂起，
# 超时即跳过该降级分支。注意：10.0 实际长于主检索超时（3.0/5.0s）——主检索
# 超时返回空后，兜底仍有 10s 窗口尽力补充（非"短于"语义，2026-08-19 R2 修正注释）。
_DEGRADE_TIMEOUT = 10.0


def _level_from_strategy(strategy, config=None) -> "RetrievalLevel":
    """策略字符串 → 检索级别（CC 方案 A：hybrid → FUSION；auto/空/None 且
    rerank/hyDE 任一开启 → FUSION；其余回落 HYPERGRAPH 零回归）。

    strategy 为任意 Optional[str]（api/models.py 未收窄）：
      - "hybrid" → FUSION（显式覆盖，不变）
      - "auto"/""/None 且 config.rerank_enabled or config.hyde_enabled 为真 → FUSION
        （补齐 P3a 宣称但从未落地的生产三路融合：rerank_enabled=true 默认 → 生产
        auto 请求进入 FUSION 接通 bge-reranker 重排；config 缺省/不可得（None）时
        保持旧行为回落 HYPERGRAPH）
      - 其余（tau_first/vector_first/未知）→ HYPERGRAPH（降级链不变）
    """
    s = strategy.strip().lower() if isinstance(strategy, str) else None
    if s == "hybrid":
        return RetrievalLevel.FUSION
    if s in (None, "", "auto") and config is not None and (
        getattr(config, "rerank_enabled", False) or getattr(config, "hyde_enabled", False)
    ):
        return RetrievalLevel.FUSION
    return RetrievalLevel.HYPERGRAPH


def _query_router_config(qr):
    """取 QueryRouter.config（SelfEvolvingRetrieval 包装时解 _qr）。

    用 vars() 检查实例 __dict__ 而非 getattr(qr, "_qr", qr) 缺省——后者对
    MagicMock 测试替身会触发属性自动创建（假 _qr 子 mock），误解包后 config
    取到空 mock 恒真 → auto 误判 FUSION（P0-1 生产接线回归测试暴露）。
    """
    if qr is None:
        return None
    qr_vars = getattr(qr, "__dict__", None)
    if isinstance(qr_vars, dict) and "_qr" in qr_vars:
        qr = qr_vars["_qr"]
        qr_vars = getattr(qr, "__dict__", None)
    # 解包后优先读实例 __dict__：getattr 对 MagicMock/鸭子对象会返回真值
    # 子 mock（恒真 → auto 误判 FUSION），vars() 只返回真实赋值过的键。
    if isinstance(qr_vars, dict):
        return qr_vars.get("config")
    return getattr(qr, "config", None)


def _retrieve_timeout(level, config) -> float:
    """检索超时预算（秒）【P3b R1 P0-2 方案 B】：FUSION 且 HyDE 开启 → 5.0
    （LLM 生成假设段落预留预算），否则 3.0（原 _RETRIEVE_TIMEOUT——HyDE 默认关，
    关闭路径零回归）。config 缺省（None）按 hyde_enabled=False 处理。
    """
    if level == RetrievalLevel.FUSION and getattr(config, "hyde_enabled", False):
        return 5.0
    return _RETRIEVE_TIMEOUT


@router.post("/memories/retrieve", summary="粗到精三级检索（带降级）")
async def retrieve(
    req: RetrieveRequest,
    deps: Services = Depends(get_services),
) -> RetrieveResponse:
    """执行粗到精三级检索，结果缓存 128 条。"""
    start = _now()
    set_trace_id()
    degraded = False

    # 【Perf】结果缓存命中（键含 include_archived/session_ts/namespace/strategy，避免互相污染缓存）
    # 【P3a R3 P1-2/P2-1】ns 维度：路由按 namespace 过滤，同 query 不同 ns 不能串缓存；
    # strategy_raw 维度：auto/tau_first/vector_first 大多映射 HYPERGRAPH（auto 在
    # rerank/hyDE 开启时映射 FUSION）但 strategy_used 按原始值回显，不能共用缓存
    # （否则命中时 strategy_used 回显错误）。
    # 【P3b R1 P0-1】config 感知映射：rerank/hyDE 任一开启时 auto 进 FUSION——
    # 取 QueryRouter.config（SelfEvolvingRetrieval 包装时解 _qr），与 _retrieve_timeout 共用。
    qr = deps.query_router
    qr_config = _query_router_config(qr)
    level = _level_from_strategy(req.strategy, qr_config)
    cache_key = f"{req.query}:{req.top_k}:archived:{req.include_archived}:shared:{req.include_shared}:ts:{req.session_ts}:ns:{req.namespace or ''}:strategy:{level.value}:strategy_raw:{req.strategy or ''}"
    with _result_cache_lock:
        if cache_key in _result_cache:
            latency = (_now() - start) * 1000
            record_request("POST", "/memories/retrieve", "200", _now() - start)
            cached = _result_cache[cache_key]
            cached.latency_ms = round(latency, 2)
            return cached

    if qr is None:
        raise HTTPException(status_code=503, detail="Query router not available")

    retrieve_timeout = _retrieve_timeout(level, qr_config)
    try:
        # 【P1-3】全同步检索链路（FAISS/sklearn/GraphLite）移到线程池，避免阻塞事件循环
        # 【H2】外层 wait_for：GraphLite/FAISS 挂起时超时返回空结果（降级），不无限挂起
        results_raw = await asyncio.wait_for(
            asyncio.to_thread(
                deps.query_router.retrieve, req.query,
                include_archived=req.include_archived,
                session_ts=req.session_ts,
                level=level,
                rerank=None,
            ),
            timeout=retrieve_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("Retrieval timed out after %.1fs, returning empty results",
                       retrieve_timeout)
        # 【R2 P2-3】超时监控上下文：标注 level/strategy/HyDE 状态，
        # 便于生产 p95 调整 5.0s 启发式预算（HyDE 开启时超时是重点观察对象）。
        logger.warning(
            "Retrieval timeout ctx: level=%s strategy=%s hyde_enabled=%s",
            level.value if hasattr(level, "value") else level,
            req.strategy or "auto",
            getattr(qr_config, "hyde_enabled", False),
        )
        results_raw = []
        degraded = True
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
                # 【P0-3】中文 CONTAINS 不可用：GraphLite b64 编码无子串保持性，
                # 中文查询的 Cypher 兜底不保证命中；依赖向量/BM25 主通道。
                conditions = " OR ".join(f"e.content CONTAINS $w{i}" for i in range(len(words[:5])))
                cypher = (f"MATCH (e:EpisodeNode) WHERE ({conditions}) "
                          "AND (e.quarantine IS NULL OR e.quarantine = false) "
                          "AND (e.archived IS NULL OR e.archived = false) "
                          f"RETURN e.id AS node_id, e.content AS content LIMIT 10")
                # 【H2】【H2-a】Cypher 兜底移入线程池 + 套 wait_for：
                # GraphLite 卡死时超时即跳过兜底，不再无限挂起
                # 【P1-2】degraded 置位于 wait_for 之前（对齐 gateway_api.py:491）：
                # query_cypher 抛非超时异常（QueryError/ConnectionError）跳外层
                # except 时该行已执行，确保兜底失败 → 空结果 + degraded=True。
                degraded = True
                try:
                    fallback_rows = await asyncio.wait_for(
                        asyncio.to_thread(deps.graphlite_store.query_cypher, cypher, params),
                        timeout=_DEGRADE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Cypher fallback timed out after %.1fs, skipping",
                                   _DEGRADE_TIMEOUT)
                    fallback_rows = []
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

    # 检查是否降级
    # 【P2-2】在去重/命名空间/隔离过滤前基于 results_raw 原始结果判断：过滤后
    # results_raw 变空会丢失 _degradation_level 降级信号。Cypher 兜底 / 检索超时
    # 已在上方自行置 degraded=True，这里用 or 保留不覆盖。
    if results_raw:
        degraded = degraded or any(
            isinstance(r, dict) and "_degradation_level" in r
            for r in results_raw
        )

    # 【Defense】隔离节点排除
    if results_raw and deps.quarantine_store is not None:
        # 【H2】【H2-a】隔离 ID 拉取移入线程池 + 套 wait_for：GraphLite 卡死时
        # 超时跳过隔离过滤（结果可能多带隔离节点，但不挂起读路径）
        try:
            quarantined_ids = await asyncio.wait_for(
                asyncio.to_thread(deps.quarantine_store.get_quarantined_ids),
                timeout=_DEGRADE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Quarantine ID fetch timed out after %.1fs, skipping filter",
                           _DEGRADE_TIMEOUT)
            quarantined_ids = set()
        except Exception:
            logger.exception("Quarantine ID fetch failed")
            quarantined_ids = set()
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
                # 【H2】【H2-a】命名空间预取移入线程池 + 套 wait_for：
                # GraphLite 卡死时超时跳过命名空间过滤，不挂起
                ns_rows = await asyncio.wait_for(
                    asyncio.to_thread(
                        deps.graphlite_store.query_cypher,
                        "MATCH (s:SessionNode {id: $ns})-[:SESSION_MEMBER]->(e:EpisodeNode) "
                        "RETURN e.id",
                        {"ns": req.namespace},
                    ),
                    timeout=_DEGRADE_TIMEOUT,
                )
                ns_set = {
                    str(r.get("e.id", "") or r.get("id", "")) if isinstance(r, dict) else str(r[0])
                    for r in ns_rows
                    if isinstance(r, dict) or (isinstance(r, (list, tuple)) and r)
                } if ns_rows else set()
            except asyncio.TimeoutError:
                logger.warning("Namespace prefetch timed out after %.1fs, skipping filter",
                               _DEGRADE_TIMEOUT)
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

    # [Ontology] 读时验证：一致性交叉检查 + 置信度修正
    if deps.ontology_validator is not None and results_raw:
        try:
            # 【v5.24】首次检索触发的 lazy 同步（sync_entity_types + 关系边写）
            # 预同步经写队列——不在 loop 线程同步写（一次性，_ontology_synced 置位
            # 后检索纯读）。读验证本身留 loop。
            if not deps.ontology_validator._ontology_synced:
                try:
                    await qsubmit(deps, deps.ontology_validator.sync_entity_types_to_graphlite)
                    await qsubmit(deps, deps.ontology_validator._populate_relationships)
                    deps.ontology_validator._ontology_synced = True
                except HTTPException:
                    logger.warning("Write queue busy, ontology sync deferred (validation continues)")
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

    # 【P2-b】R6 内容级投毒标记: 纯正则扫描 (无 LLM/embedding/网络), fail-open → None
    # 不改 score、不删结果、不打补丁 content —— 只附加 risk_level 元数据 (读路径元数据)
    for r in results_raw[:req.top_k]:
        if isinstance(r, dict):
            try:
                r["risk_level"] = scan_content(r.get("content", "")).risk_level
            except Exception:
                r["risk_level"] = None

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
                risk_level=r.get("risk_level"),
                modality=r.get("modality"),
            ))
        elif hasattr(r, "node_id"):
            results.append(EpisodicResult(
                node_id=r.node_id,
                content=getattr(r, "content", ""),
                score=getattr(r, "score", 0.0),
                tau_value=getattr(r, "tau_value", None),
                retrieval_level=getattr(r, "source", "hypergraph"),
                risk_level=getattr(r, "risk_level", None),
                modality=getattr(r, "modality", None),
            ))

    # 【User-Profile】旁路上下文：画像命中 → profile_context 注入响应
    # （消费方 prepend 到 prompt；SelfEvolvingRetrieval 包装时取内层 QueryRouter）
    profile_context = None
    qr = deps.query_router
    if qr is not None:
        try:
            inner_qr = getattr(qr, "_qr", qr)
            search_profile = getattr(inner_qr, "search_profile", None)
            if callable(search_profile):
                sp = search_profile(req.query)
                if isinstance(sp, dict) and sp.get("matched"):
                    profile_context = sp.get("context") or None
        except Exception:
            profile_context = None

    latency = (_now() - start) * 1000
    response = RetrieveResponse(
        query=req.query,
        strategy_used=req.strategy or "auto",
        results=results,
        total_found=len(results),
        latency_ms=round(latency, 2),
        degraded=degraded,
        profile_context=profile_context,
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

        # 3. 回查 GraphLite 获取节点详情（批量，一次 GraphLite 查询）
        results: list[SearchVectorResult] = []
        faiss_id_map = getattr(deps, "faiss_id_map", {}) or {}

        # 先收集有效的 (faiss_id, episode_id, score) 三元组（保持 FAISS 排名顺序）
        hits: list[tuple[int, str, float]] = []
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

            hits.append((faiss_id, episode_id, round(score, 4)))

        # 批量回查 GraphLite（与 query_router 批量回查同模式，一次 GraphLite 查询）
        episodes_dict: dict[str, dict] = {}
        if hits and deps.graphlite_store is not None:
            try:
                episodes_dict = {
                    ep["id"]: ep
                    for ep in deps.graphlite_store.get_episodes_batch(
                        [eid for _, eid, _ in hits]
                    )
                }
            except CircuitBreakerOpen:
                episodes_dict = {}  # 熔断跳闸：静默跳过回查（content 为空）
            except Exception:
                logger.exception("search_vector: GraphLite batch lookup failed, results will have empty content")

        for faiss_id, episode_id, score in hits:
            node = episodes_dict.get(episode_id)
            if node and node.get("archived") in (True, "true", 1):
                continue  # 排除已归档节点
            content = node.get("content", "") if node else ""
            results.append(SearchVectorResult(
                node_id=episode_id,
                content=content,
                score=score,
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
