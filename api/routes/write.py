"""
记忆写入路由 (sensory, episodes, promote)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from core.defense import MemoryDefenseVerdict

from api.routes._deps import (
    router, Services, get_services, _now, logger,
    _embed_queue, _embed_queue_lock, flush_faiss_buffer,
    set_trace_id, record_request,
    Depends, Request, JSONResponse, HTTPException,
    uuid, np, base64, json, time,
    SensoryResponse, SensoryRecord,
    MultimodalRecord, MultimodalResponse,
    EpisodeCreate, EpisodeResponse,
    PromoteRequest, PromoteResponse,
)

# 【P1-3】外部调用超时（秒）：防御预检 / CLIP 图像嵌入 / Whisper 转录
_EXTERNAL_CALL_TIMEOUT = 10.0

# 【M3-a】冷启动 warmup 超时（秒）：CLIP/Whisper 模型首次加载常 >10s，
# 首个媒体写入若用 10s 预算会超时。首次嵌入放宽到 30s，完成后切回常规超时。
_MEDIA_WARMUP_TIMEOUT = 30.0

# 【M3】媒体嵌入专用线程池（私有实例，max_workers=2，不共享默认 ThreadPoolExecutor）。
# wait_for 超时无法取消已运行的线程：若与 search 共用默认池，反复超时会把池占满，
# 导致检索 to_thread(retrieve) 被排到卡死 worker 之后 → 新 DoS 向量。
# 独占小池将卡死线程隔离在媒体嵌入路径内，不影响读路径。
_MEDIA_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shm-media-embed")


@router.post("/memories/sensory", summary="写入感觉缓冲区 (Layer1)")
async def write_sensory(
    record: SensoryRecord,
    deps: Services = Depends(get_services),
) -> SensoryResponse:
    """
    将原始文本写入 Layer1 环形缓冲区。
    缓冲区满时自动挤出最旧记录并通知梦境调度器。
    """
    start = _now()
    set_trace_id()

    record_id = str(uuid.uuid4())
    buf = getattr(deps.graphlite_store, "_sensory_buffer", None)
    buffer_usage = 0

    if buf is not None:
        buf.append({"id": record_id, "content": record.content,
                     "source": record.source, "created_at": start,
                     "namespace": record.namespace,
                     "visibility": record.visibility})
        buffer_usage = len(buf)
        if hasattr(buf, "is_full") and buf.is_full():
            evicted = buf.evict_oldest()
            if deps.dream_scheduler:
                await deps.dream_scheduler.on_node_created()
    else:
        # 无环形缓冲区：直接写入 GraphLite EpisodeNode 作为兜底
        deps.graphlite_store.create_episode({
            "id": record_id,
            "content": record.content,
            "source": record.source,
            "visibility": record.visibility,
            "created_at": start,
            "tau_initial": 1.0,
        })
        # 命名空间链接
        if record.namespace:
            deps.graphlite_store.ensure_session(record.namespace)
            deps.graphlite_store.link_to_session(record.namespace, record_id)
        if deps.dream_scheduler:
            await deps.dream_scheduler.on_node_created()

    record_request("POST", "/memories/sensory", "200", _now() - start)
    return SensoryResponse(record_id=record_id, buffer_usage=buffer_usage)


@router.post("/memories/multimodal", summary="多模态记忆写入 (图像/音频/视频+文本)")
async def write_multimodal(
    req: MultimodalRecord,
    request: Request,
    deps: Services = Depends(get_services),
) -> MultimodalResponse:
    """写入多模态记忆。

    接收 Base64 编码的媒体文件（图像/音频/视频）+ 可选文本描述。
    媒体文件存储到 data/media/，嵌入向量写入 VisualNode。
    音频自动转录（Whisper）并与文本一起索引。
    """
    start = _now()
    set_trace_id()

    episode_id = str(uuid.uuid4())
    created_at = _now()
    media_paths: list[str] = []
    unembedded_paths: list[str] = []  # 【M3-a】已落盘但嵌入失败/超时的媒体（保留文件）
    transcription: Optional[str] = None
    visual_node_id: Optional[str] = None
    text = req.text

    # 初始化多模态组件（懒加载）
    clip = getattr(deps, "_clip_embedder", None)
    if clip is None:
        try:
            from multimodal.embedders import ClipEmbedder
            clip = ClipEmbedder()
            deps._clip_embedder = clip
        except ImportError:
            clip = None
        except Exception:
            clip = None
    whisper = getattr(deps, "_whisper_embedder", None)
    if whisper is None:
        from multimodal.embedders import WhisperEmbedder
        try:
            whisper = WhisperEmbedder()
            deps._whisper_embedder = whisper
        except Exception:
            whisper = None
    store = getattr(deps, "_media_store", None)
    if store is None:
        from multimodal.store import MediaStore
        try:
            store = MediaStore()
            deps._media_store = store
        except Exception:
            logger.exception("MediaStore init failed")
            store = None

    # 【M3-a】冷启动 warmup：首个媒体嵌入尝试放宽超时（模型加载常 >10s），
    # 避免首写被误超时 → 修复前超时会删除已落盘的用户媒体文件（数据丢失）。
    media_timeout = (_MEDIA_WARMUP_TIMEOUT
                     if not getattr(deps, "_media_warmup_done", False)
                     else _EXTERNAL_CALL_TIMEOUT)

    # ── 处理图像 ──
    image_embeddings: list[np.ndarray] = []
    # 【B-复审】warmup 预算消耗跟踪：仅在实际尝试过媒体嵌入/转录后置位
    # _media_warmup_done。纯文本 multimodal（无 images/audio/video）不消耗
    # 预算，首个真实媒体写入仍保留 30s 模型加载超时（原无条件置位导致
    # 纯文本首请求后首个媒体写入退化到 10s 常规超时）。
    media_warmup_consumed = False
    for b64_str in req.images:
        try:
            if len(b64_str) > 10 * 1024 * 1024:  # 10MB max per media item
                logger.warning("Image too large (%d bytes), skipping", len(b64_str))
                continue
            img_bytes = base64.b64decode(b64_str)
        except Exception:
            continue
        saved_path: Optional[str] = None
        if store is not None:
            saved_path = store.save_image(img_bytes)
            media_paths.append(saved_path)
        if clip is not None:
            media_warmup_consumed = True
            try:
                # 【M3】专用线程池（_MEDIA_EXECUTOR）执行嵌入，不占默认池
                # 【M3-a】使用 warmup/常规超时（首次放宽到 30s）
                loop = asyncio.get_running_loop()
                emb = await asyncio.wait_for(
                    loop.run_in_executor(_MEDIA_EXECUTOR, clip.embed_image, img_bytes),
                    timeout=media_timeout,
                )
            except Exception:
                # 【P1-3】【M3】【M3-a】超时/失败降级：跳过该图像嵌入，但保留
                # 已落盘文件（仅标记"未嵌入"）——瞬时故障不导致用户媒体数据丢失
                logger.warning("CLIP embed_image failed or timed out, skipping image")
                emb = None
                if saved_path is not None and saved_path in media_paths:
                    unembedded_paths.append(saved_path)
            if emb is not None:
                image_embeddings.append(emb)

    # ── 处理音频 ──
    audio_texts: list[str] = []
    for b64_str in req.audio:
        try:
            if len(b64_str) > 10 * 1024 * 1024:
                logger.warning("Audio too large (%d bytes), skipping", len(b64_str))
                continue
            aud_bytes = base64.b64decode(b64_str)
        except Exception:
            continue
        saved_path: Optional[str] = None
        if store is not None:
            saved_path = store.save_audio(aud_bytes)
            media_paths.append(saved_path)
        if whisper is not None:
            media_warmup_consumed = True
            try:
                # 【M3】专用线程池执行转录，不占默认池
                # 【M3-a】使用 warmup/常规超时（首次放宽到 30s）
                loop = asyncio.get_running_loop()
                seg = await asyncio.wait_for(
                    loop.run_in_executor(_MEDIA_EXECUTOR, whisper.transcribe, aud_bytes),
                    timeout=media_timeout,
                )
            except Exception:
                # 【P1-3】【M3】【M3-a】超时/失败降级：跳过该音频转录，但保留
                # 已落盘文件（仅标记"未嵌入"），不删除用户文件
                logger.warning("Whisper transcribe failed or timed out, skipping audio")
                seg = None
                if saved_path is not None and saved_path in media_paths:
                    unembedded_paths.append(saved_path)
            if seg:
                audio_texts.append(seg)

    # ── 处理视频 ──
    for b64_str in req.video:
        try:
            if len(b64_str) > 10 * 1024 * 1024:
                logger.warning("Video too large (%d bytes), skipping", len(b64_str))
                continue
            vid_bytes = base64.b64decode(b64_str)
        except Exception:
            continue
        if store is not None:
            path = store.save_video(vid_bytes)
            media_paths.append(path)

    # 【M3-a】嵌入尝试完成（成功/失败）→ 冷启动 warmup 结束，后续用常规超时
    # 【B-复审】仅实际尝试过嵌入/转录才置位：纯文本请求不消耗 warmup 预算
    if media_warmup_consumed:
        deps._media_warmup_done = True

    # ── 合并文本 ──
    text_parts: list[str] = []
    if text:
        text_parts.append(text)
    if audio_texts:
        transcription = " ".join(audio_texts)
        text_parts.append(f"[audio transcription]: {transcription}")
    merged_text = "\n".join(text_parts) if text_parts else ""

    # ── 写入 VisualNode（有图像时）──
    visual_emb = None
    if image_embeddings and clip is not None:
        # 多图像取平均作为视觉嵌入
        visual_emb = np.mean(image_embeddings, axis=0).astype(np.float32)

        # 投影 512 → 384 以匹配 VisualNode schema（随机投影桥）
        try:
            proj = getattr(deps, "_clip_projection", None)
            if proj is None:
                rng = np.random.default_rng(42)
                proj = rng.standard_normal((512, 384), dtype=np.float32)
                proj /= np.linalg.norm(proj, axis=0, keepdims=True)  # 单位列向量
                deps._clip_projection = proj
            emb_384 = visual_emb @ proj  # (512,) @ (512, 384) → (384,)

            visual_node_id = str(uuid.uuid4())
            if deps.graphlite_store is not None:
                deps.graphlite_store.create_visual_node({
                    "id": visual_node_id,
                    "image_path": media_paths[0] if media_paths else "",
                    "caption": merged_text[:1024],
                    "embedding": emb_384.tolist(),
                    "source": req.source,
                    "created_at": created_at,
                })
        except Exception:
            logger.exception("VisualNode creation failed (non-fatal)")

    # ── 写入 EpisodeNode（文本索引）──
    if merged_text and deps.graphlite_store is not None:
        deps.graphlite_store.create_episode({
            "id": episode_id,
            "content": merged_text,
            "source": req.source,
            "visibility": req.visibility,
            "created_at": created_at,
            "tau_initial": 1.0,
        })
        if req.namespace:
            deps.graphlite_store.ensure_session(req.namespace)
            deps.graphlite_store.link_to_session(req.namespace, episode_id)

        # 通知梦境调度器
        if deps.dream_scheduler:
            await deps.dream_scheduler.on_activity()
            await deps.dream_scheduler.on_node_created()

        # 异步入队文本 embedding
        with _embed_queue_lock:
            _embed_queue.append((episode_id, merged_text, created_at))

    record_request("POST", "/memories/multimodal", "200", _now() - start)
    return MultimodalResponse(
        episode_id=episode_id,
        visual_node_id=visual_node_id,
        text=merged_text[:200] if merged_text else None,
        media_paths=media_paths,
        transcription=transcription,
        created_at=created_at,
        unembedded_paths=unembedded_paths,
    )


@router.post("/memories/episodes", summary="直接创建情节节点 (Layer2)")
async def create_episode(
    req: EpisodeCreate,
    request: Request,
    deps: Services = Depends(get_services),
) -> EpisodeResponse:
    """直接创建 Layer2 情节节点，可选强制提升。"""
    start = _now()
    set_trace_id()

    episode_id = str(uuid.uuid4())
    created_at = _now()
    tau_initial = 1.0

    # τ 值计算
    if deps.tau_engine:
        tau_initial = deps.tau_engine.compute_strength(created_at)
        if tau_initial < deps.tau_engine.config.decay_threshold and not req.force_promote:
            raise HTTPException(status_code=400, detail="τ below threshold; use force_promote=true")

    # SSM门控过滤：低价值内容跳过持久化
    # 【FIX 2026-08-06】force_promote=true 应绕过 gate — "强制提升"语义即无条件持久化。
    # 原实现 gate 不认 force_promote, warmup 后随机初始化权重输出 ≈0.45 < 0.5
    # 阈值导致所有正常写入被过滤(记忆系统静默丢记忆, benchmark 复现)。
    if (deps.ssm_gate is not None and deps.tau_engine is not None
            and not req.force_promote):
        features = np.array([
            float(len(req.content)),            # 内容长度
            float(created_at - time.time()),    # 时间衰减信号
        ], dtype=np.float32)
        if features.shape[0] != deps.ssm_gate.config.input_dim:
            features = np.pad(features, (0, max(0, deps.ssm_gate.config.input_dim - features.shape[0])),
                             mode="constant")[:deps.ssm_gate.config.input_dim]
        # 懒初始化 hidden_state
        try:
            deps.ssm_gate.hidden_state
        except AttributeError:
            deps.ssm_gate.hidden_state = deps.ssm_gate.reset_state()
        hidden_prev = deps.ssm_gate.hidden_state
        hidden, gate_value = deps.ssm_gate.step(hidden_prev, features)
        deps.ssm_gate.hidden_state = hidden
        if not deps.ssm_gate.should_keep(gate_value):
            logger.debug("SSM gate filtered episode", content_len=len(req.content), gate=float(gate_value))
            # 【FIX 2026-08-09】过滤 → 负信号学习，防止"写入越多门越松"退化（P0-1）
            # 【P0-1】用 hidden(决策时 SSM 状态)而非 hidden_prev(step 前状态)：
            # step() 内 MLP.forward(hidden) 做决策，learn() 必须重放同一状态。
            try:
                deps.ssm_gate.learn(gate_value, 0.0, hidden)
            except Exception:
                logger.warning("SSM gate learn failed (non-fatal)", exc_info=True)
            return EpisodeResponse(episode_id=episode_id, status="filtered", tau_initial=0.0,
                                   created_at=created_at,  # 【FIX】缺 created_at → 500
                                   content=req.content, source=req.source)

    # [Defense] 记忆投毒预检（在 GraphLite 写入前执行）
    defense_verdict = None
    defense_reason = ""
    if deps.defense_engine and deps.defense_engine.config.enabled:
        # 【FIX】pre_check 是 async 函数，缺 await 导致 TypeError: cannot unpack coroutine
        try:
            verdict, reason = await asyncio.wait_for(
                deps.defense_engine.pre_check(
                    content=req.content, source=req.source, created_at=created_at,
                ),
                timeout=_EXTERNAL_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            # 【M2】超时降级改为 QUARANTINE（fail-closed 而非 fail-open）：
            # fail-open 是投毒窗口——高并发写入下 pre_check 内 asyncio.Lock 串行化
            # R1/R3/R4/R5，锁等待超时若放行（ALLOW）可被攻击者绕过 R1 限流/
            # R4 防重复/R5 信任衰减；降级为 QUARANTINE 使超时写入只隔离不放行。
            logger.warning("Defense pre_check timed out after %.1fs, quarantining write",
                           _EXTERNAL_CALL_TIMEOUT)
            verdict, reason = MemoryDefenseVerdict.QUARANTINE, "defense_timeout"
        defense_verdict = verdict
        defense_reason = reason
        if verdict.value == "block":
            logger.warning("Defense BLOCKED write", source=req.source, reason=reason)
            record_request("POST", "/memories/episodes", "403", _now() - start)
            return JSONResponse(
                status_code=403,
                content={"error": "blocked_by_defense", "reason": reason},
            )
        elif verdict.value == "quarantine":
            logger.warning("Defense QUARANTINE write", source=req.source, reason=reason)

    # [Ontology] 写时验证（v1 — 冲突检测）
    ontology_note = None
    val_result = None
    if deps.ontology_validator is not None:
        val_result = deps.ontology_validator.write_validate(req.content, episode_id)
        if not val_result.passed:
            logger.warning("Ontology write_validate rejected", content=req.content[:50],
                          reason=f"confidence={val_result.confidence:.2f}, conflicts={val_result.conflict_count}")
        if val_result.conflict_count > 0:
            ontology_note = f"[本体警告] 与 {val_result.conflict_count} 条已有事实存在矛盾"
            for c in val_result.contradictions:
                try:
                    conflict_id = c.get("conflict_id", "")
                    conflict_node_id = f"conflict_{episode_id}_{conflict_id}"
                    # GraphLite 不支持 MERGE：MATCH 存在性检查 + INSERT（幂等）
                    if not deps.graphlite_store.execute_cypher(
                        "MATCH (c:ConflictNode {id: $id}) RETURN c",
                        {"id": conflict_node_id},
                    ):
                        deps.graphlite_store.execute_cypher(
                            "INSERT (:ConflictNode {id: $id, episode_a: $a, episode_b: $b, "
                            "rule_id: $rule, detected_at: $t, resolved: false})",
                            {"id": conflict_node_id,
                             "a": episode_id, "b": conflict_id,
                             "rule": "write_validate", "t": _now()})
                except Exception:
                    logger.warning("Conflict node creation failed for %s / %s", episode_id, conflict_id)
            # P2: 通知梦境调度器有冲突产生
            try:
                if deps.dream_scheduler:
                    await deps.dream_scheduler.on_conflict_detected()
            except Exception:
                logger.warning("Failed to notify dream scheduler of conflict")
    episode_data = {
        "id": episode_id,
        "content": req.content,
        "source": req.source,
        "visibility": req.visibility,
        "created_at": created_at,
        "tau_initial": tau_initial,
    }
    # 本体字段: 验证器提取到了就存独立字段, 供矛盾检测等值匹配 (b64 根治)
    if val_result is not None:
        if val_result.ontology_type:
            episode_data["ontology_type"] = val_result.ontology_type
        if val_result.entity_name:
            episode_data["entity_name"] = val_result.entity_name
        if val_result.entity_value:
            episode_data["entity_value"] = val_result.entity_value
    deps.graphlite_store.create_episode(episode_data)

    # 【FIX 2026-08-09】写入成功 → 正信号学习（P0-1 learn 闭环）
    # 【P0-1】用 hidden(决策时 SSM 状态)而非 hidden_prev:
    # step() 内 MLP.forward(hidden) 做决策，learn() 重放同一状态.
    if (deps.ssm_gate is not None and deps.tau_engine is not None
            and not req.force_promote):
        try:
            deps.ssm_gate.learn(gate_value, 1.0, hidden)
        except Exception:
            logger.warning("SSM gate learn failed (non-fatal)", exc_info=True)

    # [Defense] 隔离标记：QUARANTINE 判定的节点写入后标记隔离
    if defense_verdict is not None and defense_verdict.value == "quarantine":
        if deps.quarantine_store is not None:
            deps.quarantine_store.quarantine(episode_id, defense_reason, req.source)
            logger.info("Node %s quarantined after write: %s", episode_id[:12], defense_reason[:80])

    # 命名空间链接
    if req.namespace:
        deps.graphlite_store.ensure_session(req.namespace)
        deps.graphlite_store.link_to_session(req.namespace, episode_id)

    # [Ontology v2] 写时类型验证
    if deps.ontology_v2 is not None:
        try:
            v2_result = deps.ontology_v2.validate_write(req.content)
            if not v2_result.passed and v2_result.errors:
                for err in v2_result.errors:
                    logger.warning("Ontology v2 validation: %s → %s", err.field, err.message)
        except Exception:
            logger.exception("Ontology v2 write validation error (non-fatal)")

    # [Step 1] 关系抽取：批量 GraphLite 操作（减少 3N→2 次往返）
    triples = None
    if deps.graphlite_store is not None and len(req.content) > 50:
        try:
            from core.relation_extractor import RelationExtractor
            rext = RelationExtractor()
            triples = rext.extract(req.content)
            if triples:
                # 批量创建实体节点（GraphLite 不支持 MERGE/多语句：逐条 MATCH + INSERT）
                seen_entities = set()
                for t in triples:
                    for entity_name in (t.subject, t.obj):
                        if not entity_name:
                            continue  # 空串守卫：避免写入哨兵节点
                        if entity_name not in seen_entities:
                            seen_entities.add(entity_name)
                            if not deps.graphlite_store.execute_cypher(
                                "MATCH (n:OntologyEntity {name: $name}) RETURN n",
                                {"name": entity_name},
                            ):
                                deps.graphlite_store.execute_cypher(
                                    "INSERT (n:OntologyEntity {name: $name, type: 'discovered'})",
                                    {"name": entity_name},
                                )
                # 批量创建关系边（GraphLite 不支持 MERGE：MATCH 边存在性 + INSERT）
                for t in triples:
                    if not deps.graphlite_store.execute_cypher(
                        "MATCH (a:OntologyEntity {name: $subj})"
                        "-[:RELATES_TO]->"
                        "(b:OntologyEntity {name: $obj}) RETURN a",
                        {"subj": t.subject, "obj": t.obj},
                    ):
                        deps.graphlite_store.execute_cypher(
                            "MATCH (a:OntologyEntity {name: $subj}), "
                            "(b:OntologyEntity {name: $obj}) "
                            "INSERT (a)-[:RELATES_TO {relation: $rel}]->(b)",
                            {"subj": t.subject, "obj": t.obj, "rel": t.relation},
                        )
            if triples:
                logger.info("Relation extraction: %d typed edges", len(triples))
        except Exception:
            logger.exception("Relation extraction error (non-fatal)")

    # [Step 2] 置信度累积
    if deps.evidence_tracker is not None:
        try:
            evidence_count = deps.evidence_tracker.record(
                req.content, source=req.source,
                metadata={"episode_id": episode_id},
            )
            if evidence_count > 1:
                logger.info("Evidence tracker: count=%d for %s", evidence_count, req.content[:40])
        except Exception:
            logger.exception("Evidence tracker error (non-fatal)")

    # [Step 3] 实体消歧 — 仅对有一定信息量的内容执行
    if deps.graphlite_store is not None and len(req.content) > 80:
        try:
            from core.entity_resolver import EntityResolver
            resolver = EntityResolver(graphlite_store=deps.graphlite_store)
            result = resolver.process(req.content)
            if result.get("alias_count", 0) > 0:
                logger.info("Entity resolver: %d alias edges, %d entities",
                            result["alias_count"], len(result.get("entities", [])))
        except Exception:
            logger.exception("Entity resolver error (non-fatal)")

    # [Phase3] 写入时提取实体共现 → 建 RELATES_TO 边（保留旧逻辑作为fallback）
    if deps.ontology_validator is not None and triples is None:
        try:
            rel_count = deps.ontology_validator.extract_and_relate(req.content)
            if rel_count > 0:
                logger.info("Write-time entity relations: %d edges (fallback)", rel_count)
        except Exception:
            logger.exception("Entity relations extraction failed (non-fatal)")

    # 【P6】异步 embedding：入队后立即返回（不阻塞写入响应）
    with _embed_queue_lock:
        _embed_queue.append((episode_id, req.content, created_at))

    # 不再在写入路径中同步消费队列——由 poll loop 每5秒 flush 一次
    # 避免了写入延迟因 FAISS 编码而膨胀 200-400ms

    if deps.dream_scheduler:
        await deps.dream_scheduler.on_activity()
        await deps.dream_scheduler.on_node_created()

    # 【P0】自动超边创建：检测同源节点形成时态/情节超边
    try:
        await _auto_create_hyperedges(episode_id, req.source, req.content, deps)
    except Exception:
        logger.exception("Auto hyperedge creation failed for episode %s", episode_id)

    # 【P0-①】会话观测节点：通过 X-Session-Id header 关联记忆到同一会话
    try:
        session_id = request.headers.get("X-Session-Id") or request.headers.get("x-session-id")
        if session_id and deps.graphlite_store is not None:
            session_node_id = deps.graphlite_store.get_or_create_session(
                session_id, metadata=json.dumps({"source": req.source})
            )
            if session_node_id:
                deps.graphlite_store.link_session_member(session_node_id, episode_id)
    except Exception:
        logger.exception("Session memory link failed for episode %s", episode_id)

    record_request("POST", "/memories/episodes", "200", _now() - start)
    return EpisodeResponse(
        episode_id=episode_id,
        content=req.content[:200],
        tau_initial=tau_initial,
        created_at=created_at,
        source=req.source,
    )


@router.post("/memories/episodes/batch", summary="批量创建情节节点 (Layer2)")
async def create_episodes_batch(
    reqs: list[EpisodeCreate],
    deps: Services = Depends(get_services),
) -> dict:
    """批量创建情节节点。逐条调用核心写入逻辑, 返回每条结果。

    用于 benchmark/数据导入等批量场景(比逐条 HTTP 调用少 RTT 开销)。
    注意: 仍受服务端编码速度限制(CPU embedding 1.4s/条)。
    """
    start = _now()
    set_trace_id()
    results: list[dict] = []
    created_entries: list[tuple[str, str]] = []
    for req in reqs:
        episode_id = str(uuid.uuid4())
        created_at = _now()
        tau_initial = 1.0
        try:
            if deps.graphlite_store is not None:
                deps.graphlite_store.create_episode({
                    "id": episode_id, "content": req.content, "source": req.source,
                    "visibility": req.visibility, "created_at": created_at,
                    "tau_initial": tau_initial,
                    "metadata": req.metadata,
                })
            # namespace 关联(同单条逻辑)
            if req.namespace and deps.graphlite_store is not None:
                deps.graphlite_store.ensure_session(req.namespace)
                deps.graphlite_store.link_to_session(req.namespace, episode_id)
            # 【P1-1】超边创建改为批量合并 (循环外每 source 2 次查询),
            # 不再逐条触发 (原: 每项 2 次 MATCH 全表扫描)。
            created_entries.append((episode_id, req.source))
            # 【P1-1·6.1】补入 embedding 队列 (原批量路径缺失 → 批量导入数据
            # 不进 FAISS、检索全空)。µs 级入队, 编码由 poll loop 异步消费;
            # 隔离节点由 _process_embed_queue 的 quarantined_set 跳过。
            with _embed_queue_lock:
                _embed_queue.append((episode_id, req.content, created_at))
            # 梦境通知 (批量写入计入写压力/累积计数, 与单条路径一致)
            if deps.dream_scheduler is not None:
                await deps.dream_scheduler.on_activity()
                await deps.dream_scheduler.on_node_created()
            results.append({"episode_id": episode_id, "status": "created"})
        except Exception as e:
            results.append({"episode_id": episode_id, "status": "error", "error": str(e)})

    # 【P1-1】批量超边创建: 每 source 只查 1 次 recent 窗口
    try:
        await _auto_create_hyperedges_batch(created_entries, deps)
    except Exception:
        logger.exception("Batch hyperedge creation failed (non-fatal)")

    record_request("POST", "/memories/episodes/batch", "200", _now() - start)
    return {"status": "ok", "count": len(results), "results": results}


@router.get("/memories/episodes/{episode_id}", summary="查询情节节点")
async def get_episode(
    episode_id: str,
    deps: Services = Depends(get_services),
) -> EpisodeResponse:
    """按 ID 查询单个情节节点。"""
    start = _now()
    set_trace_id()

    node = deps.graphlite_store.get_episode(episode_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

    record_request("GET", "/memories/episodes/{episode_id}", "200", _now() - start)
    return EpisodeResponse(
        episode_id=node["id"],
        content=node.get("content", ""),
        tau_initial=node.get("tau_initial", 1.0),
        created_at=node.get("created_at", 0.0),
        source=node.get("source", "unknown"),
    )


@router.post("/memories/promote", summary="Layer1 → Layer2 提升")
async def promote_to_episode(
    req: PromoteRequest,
    deps: Services = Depends(get_services),
) -> PromoteResponse:
    """将 Layer1 感觉记录提升为 Layer2 情节节点。"""
    start = _now()
    set_trace_id()

    episode_id = str(uuid.uuid4())
    created_at = _now()
    tau = 1.0

    if deps.tau_engine:
        tau = deps.tau_engine.compute_strength(created_at)

    # 尝试从 GraphLite 查找原始记录内容
    content = ""
    existing = deps.graphlite_store.get_episode(req.sensory_record_id)
    if existing:
        content = existing.get("content", "")
    else:
        content = "promoted_record"

    deps.graphlite_store.create_episode({
        "id": episode_id,
        "content": content,
        "source": "promoted",
        "created_at": created_at,
        "tau_initial": tau,
    })

    # 【FIX 2026-08-07】promote 后必须入 embedding 队列，否则提升的 episode
    # 永远不会进 FAISS 向量索引（检索链路 L1/L2 全空，数据不可检索）。
    # 对齐 episodes/batch 路由的异步队列写法（poll loop 每 5s flush）。
    if content and content != "promoted_record":
        with _embed_queue_lock:
            _embed_queue.append((episode_id, content, created_at))

    count = 0
    if deps.hebbian_updater and deps.dream_scheduler:
        count = 1

    if deps.dream_scheduler:
        await deps.dream_scheduler.on_activity()

    record_request("POST", "/memories/promote", "200", _now() - start)
    return PromoteResponse(
        episode_id=episode_id,
        tau_initial=tau,
        hebbian_connections_updated=count,
    )


# ─── 【P6】异步 embedding 队列消费 ─────────────────────────


def _process_embed_queue(deps: Services) -> int:
    """消费 embedding 队列：异步编码并加入 FAISS 缓冲。"""
    global _embed_queue
    if not deps.encoder:
        return 0
    with _embed_queue_lock:
        batch = _embed_queue[:]
        _embed_queue.clear()
    if not batch:
        return 0
    count = 0
    # 【Defense】预先获取隔离节点 ID 集合用于快速排除
    quarantined_set: set[str] = set()
    if deps.quarantine_store is not None:
        quarantined_set = deps.quarantine_store.get_quarantined_ids()

    for episode_id, content, created_at in batch:
        # 隔离节点不加入 FAISS
        if episode_id in quarantined_set:
            logger.debug("Embed queue: skip quarantined node %s", episode_id[:12])
            continue
        try:
            emb = deps.encoder.embed(content)
            if emb is not None and deps.faiss_index is not None:
                faiss_id = int(uuid.uuid5(uuid.NAMESPACE_OID, str(episode_id)).int & ((1 << 63) - 1))
                emb_array = emb.reshape(1, -1).astype(np.float32)
                with deps._faiss_buffer_lock:
                    deps._faiss_buffer.append((faiss_id, emb_array.flatten(), episode_id))
                try:
                    if deps.hebbian_updater and deps.graphlite_store:
                        deps.hebbian_updater.update(
                            {episode_id: 1.0}, deps.graphlite_store.get_all_connections()
                        )
                except Exception:
                    logger.exception("Hebbian update failed for %s", episode_id)
                count += 1
        except Exception:
            logger.exception("FAISS add_with_ids failed for %s", episode_id)
    # 批量 flush FAISS
    flush_faiss_buffer(deps)
    if count:
        logger.debug("Embed queue: processed %d items", count)
    return count


# ─── 【P0】写入时自动创建超边 ────────────────────────────


async def _auto_create_hyperedges(episode_id: str, source: str, content: str, deps: Services) -> int:
    """
    写入新情节节点后自动检测并创建超边：
    - 时态超边：同一 source 在 300s 内写入的节点
    - 情节超边：同一 source 在 3600s 内连续写入的节点
    """
    if deps.hyperedge_manager is None or deps.graphlite_store is None:
        return 0

    try:
        created = 0
        now = _now()

        # 时态超边：最近 300s 内的同源节点
        recent_rows = deps.graphlite_store.query_cypher(
            "MATCH (e:EpisodeNode) WHERE e.id <> $id AND e.source = $src "
            "AND e.created_at >= $cutoff "
            "RETURN e.id ORDER BY e.created_at DESC LIMIT 5",
            {"id": episode_id, "src": source, "cutoff": now - 300},
        )
        recent_ids = []
        for row in recent_rows:
            if isinstance(row, (list, tuple)):
                recent_ids.append(str(row[0]))
            elif isinstance(row, dict):
                recent_ids.append(str(row.get("id", "")))
        if len(recent_ids) >= 2:
            # 用所有最近的 + 新节点一起创建时态超边
            member_ids = [episode_id] + recent_ids[:4]
            deps.hyperedge_manager.create_temporal_hyperedge(
                member_ids=member_ids,
                start_time=now - 300,
                end_time=now,
            )
            created += 1
            logger.info("Auto-created TEMPORAL hyperedge: %d members (source=%s)", len(member_ids), source)
        elif len(recent_ids) == 1:
            member_ids = [episode_id, recent_ids[0]]
            deps.hyperedge_manager.create_temporal_hyperedge(
                member_ids=member_ids, start_time=now - 300, end_time=now,
            )
            created += 1
            logger.info("Auto-created TEMPORAL hyperedge (pair): source=%s", source)

        # 情节超边：同一 source 在 3600s 内的节点池
        window_rows = deps.graphlite_store.query_cypher(
            "MATCH (e:EpisodeNode) WHERE e.id <> $id AND e.source = $src "
            "AND e.created_at >= $cutoff_window "
            "RETURN e.id ORDER BY e.created_at DESC LIMIT 20",
            {"id": episode_id, "src": source, "cutoff_window": now - 3600},
        )
        window_ids = []
        for row in window_rows:
            if isinstance(row, (list, tuple)):
                window_ids.append(str(row[0]))
            elif isinstance(row, dict):
                window_ids.append(str(row.get("id", "")))
        # 如果同一 source 在 1h 内有 5+ 个节点 → 创建情节超边
        if len(window_ids) >= 4:
            member_ids = [episode_id] + window_ids[:7]
            deps.hyperedge_manager.create_episode_hyperedge(
                member_ids=member_ids,
                topic=f"batch_{source}_{int(now)}",
            )
            created += 1
            logger.info("Auto-created EPISODE hyperedge: %d members (source=%s)", len(member_ids), source)

        return created
    except Exception as e:
        logger.warning("Auto-hyperedge creation failed (non-fatal): %s", e)
        return 0


async def _auto_create_hyperedges_batch(entries: list[tuple[str, str]], deps: Services) -> int:
    """P1-1: 批量超边创建 — 每 source 只查 1 次 recent 窗口, 聚合创建超边。

    原批量路径逐条调 _auto_create_hyperedges (每项 2 次 MATCH, n=30 共 60 次查询)。
    改为每 source 2 次 MATCH (时态窗口 + 情节窗口), 创建 ≤1 条时态 + ≤1 条情节超边。
    entries: [(episode_id, source), ...]
    """
    if deps.hyperedge_manager is None or deps.graphlite_store is None or not entries:
        return 0
    from collections import defaultdict
    by_source: dict[str, list[str]] = defaultdict(list)
    for eid, src in entries:
        by_source[src].append(str(eid))
    created = 0
    now = _now()
    try:
        for src, new_ids in by_source.items():
            # 时态超边: 一次性查 recent 300s 窗口 (每 source 1 次)
            recent_rows = deps.graphlite_store.query_cypher(
                "MATCH (e:EpisodeNode) WHERE e.source = $src "
                "AND e.created_at >= $cutoff "
                "RETURN e.id ORDER BY e.created_at DESC LIMIT 5",
                {"src": src, "cutoff": now - 300},
            )
            recent_ids: list[str] = []
            for row in recent_rows:
                if isinstance(row, (list, tuple)):
                    recent_ids.append(str(row[0]))
                elif isinstance(row, dict):
                    recent_ids.append(str(row.get("id", "")))
            # 合并本批新节点去重 (批内节点已落库, 查询可能已含它们)
            merged = list(dict.fromkeys(recent_ids + new_ids))[:5]
            if len(merged) >= 2:
                deps.hyperedge_manager.create_temporal_hyperedge(
                    member_ids=merged, start_time=now - 300, end_time=now,
                )
                created += 1
                logger.info("Batch TEMPORAL hyperedge: %d members (source=%s)", len(merged), src)
            # 情节超边: 3600s 窗口 (每 source 1 次)
            window_rows = deps.graphlite_store.query_cypher(
                "MATCH (e:EpisodeNode) WHERE e.source = $src "
                "AND e.created_at >= $cutoff "
                "RETURN e.id ORDER BY e.created_at DESC LIMIT 20",
                {"src": src, "cutoff": now - 3600},
            )
            window_ids: list[str] = []
            for row in window_rows:
                if isinstance(row, (list, tuple)):
                    window_ids.append(str(row[0]))
                elif isinstance(row, dict):
                    window_ids.append(str(row.get("id", "")))
            merged_w = list(dict.fromkeys(window_ids + new_ids))[:8]
            if len(merged_w) >= 4:
                deps.hyperedge_manager.create_episode_hyperedge(
                    member_ids=merged_w, topic=f"batch_{src}_{int(now)}",
                )
                created += 1
                logger.info("Batch EPISODE hyperedge: %d members (source=%s)", len(merged_w), src)
        return created
    except Exception as e:
        logger.warning("Batch auto-hyperedge creation failed (non-fatal): %s", e)
        return 0
