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
    Dict, Any, List, Optional, BaseModel, Field,
)
from fastapi import Query, Response


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


# ─── v5.19: OWL 导出 / 本体匹配 / LLM 关系抽取 ───────────────


class OntologyMatchAttribute(BaseModel):
    """序列化的属性定义（匹配请求体用）"""
    name: str
    type: str = "string"


class OntologyMatchEntityType(BaseModel):
    """序列化的实体类型定义"""
    name: str
    description: str = ""
    parent: Optional[str] = None
    attributes: List[OntologyMatchAttribute] = Field(default_factory=list)


class OntologyMatchEdgeType(BaseModel):
    """序列化的边类型定义"""
    name: str
    description: str = ""
    source_types: List[str] = Field(default_factory=list)
    target_types: List[str] = Field(default_factory=list)


class OntologyMatchRequest(BaseModel):
    """跨系统本体匹配请求体"""
    other_ontology: Dict[str, Any] = Field(..., description="待匹配本体的序列化定义")
    max_types: int = Field(100, ge=1, le=1000, description="结构匹配 O(N²) 车挡器")


class OntologyExtractRequest(BaseModel):
    """关系抽取请求体"""
    text: str = Field(..., min_length=1, description="待抽取文本")


def _ontology_from_dict(data: Dict[str, Any]) -> "OntologyService":
    """将序列化的本体定义重建为临时 OntologyService（仅用于匹配，不入库）。"""
    from core.ontology_v2 import AttrType, AttributeDef, EdgeTypeDef, EntityTypeDef, OntologyService
    svc = OntologyService()
    for t in data.get("entity_types", []):
        attrs = [AttributeDef(name=a["name"],
                              type=AttrType(a["type"]) if a.get("type") in AttrType._value2member_map_
                              else AttrType.STRING)
                 for a in t.get("attributes", [])]
        svc.register_entity_type(EntityTypeDef(
            name=t["name"], description=t.get("description", ""),
            parent=t.get("parent"), attributes=attrs))
    for e in data.get("edge_types", []):
        svc.register_edge_type(EdgeTypeDef(
            name=e["name"], description=e.get("description", ""),
            source_types=e.get("source_types", []),
            target_types=e.get("target_types", [])))
    return svc


@router.get("/ontology/export", summary="导出本体为 OWL/Turtle")
async def ontology_export(
    format: str = Query("turtle", description="导出格式（当前仅支持 turtle/ttl）"),
    deps: Services = Depends(get_services),
) -> Response:
    """将当前 Ontology v2 导出为标准 OWL/Turtle（GET 无副作用）。"""
    from core.ontology_owl import OntologyOwlExporter
    fmt = format.lower()
    if fmt not in ("turtle", "ttl"):
        raise HTTPException(status_code=400, detail=f"Unsupported format '{format}', only turtle")
    turtle = OntologyOwlExporter().export_turtle(deps.ontology_v2)
    return Response(content=turtle, media_type="text/turtle; charset=utf-8")


@router.post("/ontology/match", summary="跨系统本体匹配")
async def ontology_match(
    req: OntologyMatchRequest,
    deps: Services = Depends(get_services),
) -> dict:
    """将请求体中的序列化本体与当前本体对齐，返回匹配报告。"""
    from core.ontology_matcher import OntologyMatcher
    try:
        other = _ontology_from_dict(req.other_ontology)
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid other_ontology: {e}")
    matcher = OntologyMatcher(max_types=req.max_types)
    return matcher.match_report(deps.ontology_v2, other)


# 全局单例关系抽取器（注入 LLM 客户端，跨请求复用动态关系缓存）
_relation_extractor: Optional[Any] = None


def _get_relation_extractor() -> Any:
    """懒加载全局关系抽取器：注入全局单例 LLM 客户端，复用动态关系缓存。

    LLMClient 初始化失败时退化为纯正则（llm_client=None）。
    """
    global _relation_extractor
    if _relation_extractor is None:
        from core.llm_client import LLMClient
        from core.relation_extractor import RelationExtractor
        llm_client = None
        try:
            llm_client = LLMClient()  # 自动从 config/settings.py 读取 llm 段
        except Exception as e:
            logger.warning("RelationExtractor LLMClient init skipped (regex-only): %s", e)
        _relation_extractor = RelationExtractor(llm_client=llm_client)
    return _relation_extractor


@router.post("/ontology/relations/extract", summary="关系抽取（正则 + 动态关系缓存）")
async def ontology_relations_extract(
    req: OntologyExtractRequest,
    deps: Services = Depends(get_services),
) -> dict:
    """同步混合抽取：正则结果 + 上次 LLM 发现的动态关系缓存匹配（无 LLM 调用）。"""
    extractor = _get_relation_extractor()
    triples = extractor.extract_hybrid(req.text)
    return {
        "count": len(triples),
        "triples": [
            {
                "subject": t.subject,
                "relation": t.relation,
                "object": t.obj,
                "confidence": t.confidence,
                "attributes": t.attributes,
            }
            for t in triples
        ],
    }
