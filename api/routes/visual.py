"""
视觉记忆路由 (visual CRUD, heatmap)
"""

import os

from api.routes._deps import (
    router, Services, get_services, _now, logger,
    set_trace_id, record_request,
    Depends, HTTPException, Query,
    uuid, base64, np,
    qsubmit,
)

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
    if deps.graphlite_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")

    visual_id = str(uuid.uuid4())
    created_at = _now()

    os.makedirs(VISUALS_DIR, exist_ok=True)
    image_path = os.path.join(VISUALS_DIR, f"{visual_id}.png")
    try:
        image_data = base64.b64decode(image_b64)
        with open(image_path, "wb") as f:
            f.write(image_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    emb = deps.encoder.embed(caption)
    if emb is None:
        raise HTTPException(status_code=500, detail="Embedding failed")
    emb_array = emb.reshape(-1).astype(np.float32)

    # 【v5.24】写串行化：VisualNode INSERT 经写队列提交，不阻塞事件循环
    await qsubmit(deps, deps.graphlite_store.create_visual_node, {
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
    if deps.graphlite_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")

    rows = deps.graphlite_store.get_visual_nodes(limit)
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
    if deps.graphlite_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")

    node = deps.graphlite_store.get_visual_node(visual_id)
    if not node:
        raise HTTPException(status_code=404, detail="Visual node not found")

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
    if deps.graphlite_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")

    node = deps.graphlite_store.get_visual_node(visual_id)
    if not node:
        raise HTTPException(status_code=404, detail="Visual node not found")

    caption = node.get("caption", "")
    words = caption.strip().split()
    total = max(len(words), 1)
    heat_regions = []
    for i, w in enumerate(words):
        pos_ratio = i / total
        weight = 1.0 - 0.5 * abs(pos_ratio - 0.5) * 2
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
