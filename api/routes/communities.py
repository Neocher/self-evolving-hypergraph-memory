"""
社区与冲突路由 (communities, conflicts)
"""

from api.routes._deps import (
    router, Services, get_services, _now, logger,
    set_trace_id, record_request,
    Depends, Query, HTTPException,
    CommunityInfo, CommunityListResponse,
    WriteReconciler, ConflictLogger, Strategy,
)


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
    # 收集所有 episode ID 批量查询
    all_episode_ids = set()
    for r in rows:
        e_a = r.get("c.episode_a", "")
        e_b = r.get("c.episode_b", "")
        if e_a:
            all_episode_ids.add(e_a)
        if e_b:
            all_episode_ids.add(e_b)

    episode_map = {}
    if all_episode_ids and deps.kuzu_store is not None:
        try:
            ep_rows = deps.kuzu_store.query_cypher(
                "MATCH (e:EpisodeNode) WHERE e.id IN $ids RETURN e.id, e.version",
                {"ids": list(all_episode_ids)}
            )
            for er in ep_rows:
                if isinstance(er, (list, tuple)):
                    episode_map[str(er[0])] = int(er[1]) if len(er) > 1 else 1
                elif isinstance(er, dict):
                    episode_map[str(er.get("e.id", ""))] = int(er.get("e.version", 1))
        except Exception:
            logger.exception("Failed to query episode versions for %d conflicts", len(all_episode_ids))

    for r in rows:
        conflict_entry = {
            "id": r.get("c.id", ""),
            "episode_a": r.get("c.episode_a", ""),
            "episode_b": r.get("c.episode_b", ""),
            "rule_id": r.get("c.rule_id", ""),
            "detected_at": r.get("c.detected_at", 0.0),
            "resolved": r.get("c.resolved", False),
        }
        conflict_entry["episode_a_version"] = episode_map.get(conflict_entry["episode_a"], 1)
        conflict_entry["episode_b_version"] = episode_map.get(conflict_entry["episode_b"], 1)
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
