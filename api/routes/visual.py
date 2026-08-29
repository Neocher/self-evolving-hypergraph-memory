"""
视觉记忆路由 (visual CRUD, heatmap)
"""

import asyncio
import os

from api.routes._deps import (
    router, Services, get_services, _now, logger,
    set_trace_id, record_request,
    Depends, HTTPException, Query,
    uuid, base64, np,
    qsubmit_visual_index,
)

VISUALS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "visuals")

# 【P1-2】CLIP 冷启动首次加载预算（秒）：模型首次下载/加载常 >10s，
# 放宽到 30s（对齐 write.py _MEDIA_WARMUP_TIMEOUT），失败/超时 → 500 降级
_VISUAL_EMBED_TIMEOUT = 30.0


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

    【P1-2】embedding 用 CLIP 文本编码（512d）@ 512→384 随机投影落库——
    与 /memories/multimodal 图像路径、检索 query 侧同处 CLIP 投影空间，
    修复原 bge 文本编码器 512d 直落被 prewarm 跳过、节点永不可检索的缺陷。
    """
    start = _now()
    set_trace_id()

    image_b64 = req.get("image_base64", "")
    caption = req.get("caption", "")
    source = req.get("source", "user")
    if not image_b64 or not caption:
        raise HTTPException(status_code=400, detail="image_base64 and caption required")
    if deps.graph_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")

    # CLIP 嵌入器（共享写路径实例，懒加载创建）
    clip = getattr(deps, "_clip_embedder", None)
    if clip is None:
        try:
            from multimodal.embedders import ClipEmbedder
            clip = ClipEmbedder()
            deps._clip_embedder = clip
        except Exception:
            clip = None
    if clip is None or not getattr(clip, "available", True):
        raise HTTPException(status_code=503, detail="CLIP embedder not available")

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

    # 【P1-2】caption → CLIP 文本 512d（与检索 query 同模型空间）；
    # 专用媒体线程池执行（不占默认池）+ 冷启动预算，不阻塞事件循环
    from api.routes.write import _MEDIA_EXECUTOR
    try:
        emb = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                _MEDIA_EXECUTOR, clip.embed_text, caption),
            timeout=_VISUAL_EMBED_TIMEOUT,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Embedding failed")
    if emb is None:
        raise HTTPException(status_code=500, detail="Embedding failed")
    emb_512 = np.asarray(emb, dtype=np.float32).reshape(-1)
    if emb_512.shape[0] != 512:
        raise HTTPException(status_code=500, detail="Unexpected CLIP embedding dimension")

    # 512→384 随机投影（seed 42 列归一，与 multimodal 写路径 / 检索侧同空间）
    proj = getattr(deps, "_clip_projection", None)
    if proj is None:
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((512, 384), dtype=np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        deps._clip_projection = proj
    emb_384 = (emb_512 @ proj).astype(np.float32)

    # 【v5.24】写串行化：VisualNode INSERT 经写队列提交，不阻塞事件循环
    # 【P2-2】qsubmit_visual_index：成功/超时路径都补索引（超时=已入队迟到完成，
    # DB 仍会落库，旧代码此处不执行 add_visual_node → 节点入库但不可检索）
    visual_node = {
        "id": visual_id,
        "image_path": image_path,
        "caption": caption,
        "embedding": emb_384.tolist(),
        "source": source,
        "created_at": created_at,
    }
    await qsubmit_visual_index(deps, deps.graph_store.create_visual_node, visual_node)

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
    if deps.graph_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")

    rows = deps.graph_store.get_visual_nodes(limit)
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
    if deps.graph_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")

    node = deps.graph_store.get_visual_node(visual_id)
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
    if deps.graph_store is None:
        raise HTTPException(status_code=503, detail="GraphLite store not available")

    node = deps.graph_store.get_visual_node(visual_id)
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
