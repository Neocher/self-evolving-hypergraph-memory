"""
超边管理路由 (CRUD)
"""

from api.routes._deps import (
    router, Services, get_services, _now, logger,
    set_trace_id, record_request,
    Depends, Query, HTTPException,
    json,
    CoreHyperedgeType, APIHyperedgeType,
    HyperedgeCreate, HyperedgeResponse, HyperedgeListResponse,
)


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
