"""
本体系统路由 (types, edges, discover, batch)
"""

from api.routes._deps import (
    router, Services, get_services, _now, logger,
    set_trace_id, record_request,
    Depends, HTTPException, Request,
    EntityTypeDefModel, EntityTypeListResponse,
    AttributeDefModel, EdgeTypeDefModel,
    EdgeTypeListResponse, OntologyStatsResponse,
    Dict, Any, List, BaseModel, Field,
)


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


@router.post("/ontology/types", summary="注册实体类型")
async def register_entity_type(
    req: EntityTypeDefModel,
    deps: Services = Depends(get_services),
) -> EntityTypeDefModel:
    """注册新的实体类型定义"""
    from core.ontology_v2 import OntologyService, EntityTypeDef, AttributeDef, AttrType
    svc: "OntologyService" = deps.ontology_v2
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
        svc: "OntologyService" = deps.ontology_v2
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
    svc: "OntologyService" = deps.ontology_v2
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
    svc: "OntologyService" = deps.ontology_v2
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
    svc: "OntologyService" = deps.ontology_v2
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
    svc: "OntologyService" = deps.ontology_v2
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
    svc: "OntologyService" = deps.ontology_v2
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
    svc: "OntologyService" = deps.ontology_v2
    deleted = svc.delete_edge_type(name)
    return {"deleted": deleted, "name": name}


@router.get("/ontology/stats", summary="本体系统统计")
async def ontology_stats(
    deps: Services = Depends(get_services),
) -> OntologyStatsResponse:
    from core.ontology_v2 import OntologyService
    svc: "OntologyService" = deps.ontology_v2
    return OntologyStatsResponse(
        entity_type_count=len(svc.entity_types),
        edge_type_count=len(svc.edge_types),
        baseline_loaded=True,
    )


@router.post("/ontology/discover", summary="自动发现候选实体和类型")
async def ontology_discover(
    req: Request,
    deps: Services = Depends(get_services),
) -> DiscoverResponse:
    """扫描已有数据，自动发现候选实体和类型定义"""
    from core.entity_discovery import EntityDiscoveryEngine
    from core.ontology_v2 import OntologyService

    engine = EntityDiscoveryEngine(ontology=deps.ontology_v2)

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
            logger.warning("Episode content query failed for ontology discover")

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
            logger.warning("Episode content query failed for ontology discover apply")

    result = engine.scan(contents, min_occurrences=2, max_candidates=50)
    apply_result = engine.apply_to_ontology(result, auto_register=True)

    return DiscoverApplyResponse(
        status=apply_result.get("status", "ok"),
        types_registered=apply_result.get("types_registered", 0),
        skipped_existing=apply_result.get("skipped_existing", 0),
        total_candidates=apply_result.get("total_candidates", 0),
    )


class BatchRelationInput(BaseModel):
    """批量写入关系边的输入"""
    relations: List[dict] = Field(..., description="关系三元组列表")
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
            deps.kuzu_store.query_cypher(
                "MERGE (a:OntologyEntity {name: $name})",
                {"name": subj}
            )
            deps.kuzu_store.query_cypher(
                "MERGE (a:OntologyEntity {name: $name})",
                {"name": obj}
            )

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
