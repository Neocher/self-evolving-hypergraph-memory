"""
梦境界面路由 (trigger, reset, notify, candidates)
"""

from api.routes._deps import (
    router, Services, get_services, _now, logger,
    set_trace_id, record_request,
    Depends, HTTPException,
    DreamTriggerResponse,
)


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
    将梦境候选中的 PRUNE 和 MERGE 操作应用到生产 GraphLite 数据库。
    操作不可逆，请先 review。
    """
    from api.models import DreamApplyResponse

    store = getattr(deps, "dream_candidate_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Dream candidate store not available")

    if deps.graphlite_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")

    success = store.apply_candidate(dream_id, deps.graphlite_store)
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
