"""
GatewayAPI — 统一的 SHM 核心接口
===============================
所有协议适配器 (MCP / A2A / ACP / CLI) 通过它访问 SHM。

每个方法从 api/_routes.py 提取同名端点的纯业务逻辑，
去掉 @router、Depends、HTTP 参数解析和 observability 埋点。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from api._routes import Services
from api.models import (
    HealthStatus,
    HyperedgeListResponse,
    HyperedgeResponse,
    MultimodalResponse,
    RetrieveResponse,
    EpisodicResult,
    SearchVectorResponse,
    SearchVectorResult,
    SensoryResponse,
    EpisodeResponse,
    DreamTriggerResponse,
    AuditTrace,
    AuditOperation,
    CommunityListResponse,
    CommunityInfo,
    HyperedgeType as APIHyperedgeType,
)
from graph.hyperedge import HyperedgeType as CoreHyperedgeType
from observability.health import HealthChecker


# ── 凭据扫描 ──

_CREDENTIAL_PATTERNS = [
    r'(?i)(api[_-]?key|apikey|secret[_-]?key|access[_-]?key)\s*[:=]{1}\s*[\'"]?[A-Za-z0-9_\-]{16,}',
    r'(?i)sk-[A-Za-z0-9]{20,}',
    r'(?i)Bearer\s+[A-Za-z0-9_\-\.]{20,}',
    r'(?i)(-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----)',
    r'(?i)(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}',
    r'(?i)(xox[bpsar]-[A-Za-z0-9\-]{10,})',
    r'(?i)(AKIA[0-9A-Z]{16})',
]


def _scan_credentials(content: str) -> list[str]:
    """扫描文本中是否包含疑似凭据/密钥内容。返回匹配的规则描述列表。"""
    import re
    findings: list[str] = []
    for pattern in _CREDENTIAL_PATTERNS:
        if re.search(pattern, content):
            findings.append(f"matched pattern: {pattern[:50]}...")
    return findings


class GatewayAPI:
    """统一的 SHM 核心接口 — 所有协议适配器都通过它访问 SHM。

    用法:
        svc: Services = ...      # 由 app.py 初始化
        api = GatewayAPI(svc)
        result = await api.retrieve("some query")
    """

    def __init__(self, services: Services) -> None:
        self._svc = services
        self._logger = logging.getLogger("gateway-api")

    # ─── 写入 ────────────────────────────────────────────────────────────

    async def write_sensory(
        self,
        content: str,
        source: str = "api",
        namespace: Optional[str] = None,
        visibility: str = "private",
    ) -> SensoryResponse:
        """将原始文本写入 Layer1 环形缓冲区。"""
        creds = _scan_credentials(content)
        if creds:
            self._logger.warning("Credential-like content detected in write_sensory from %s: %s", source, creds)
        record_id = str(uuid.uuid4())
        created_at = time.time()
        buf = getattr(self._svc.kuzu_store, "_sensory_buffer", None)
        buffer_usage = 0

        if buf is not None:
            buf.append({
                "id": record_id,
                "content": content,
                "source": source,
                "created_at": created_at,
                "namespace": namespace,
                "visibility": visibility,
            })
            buffer_usage = len(buf)
            if hasattr(buf, "is_full") and buf.is_full():
                buf.evict_oldest()
                if self._svc.dream_scheduler:
                    await self._svc.dream_scheduler.on_node_created()
        else:
            # 无环形缓冲区：直接写入 Kuzu EpisodeNode 作为兜底
            self._svc.kuzu_store.create_episode({
                "id": record_id,
                "content": content,
                "source": source,
                "visibility": visibility,
                "created_at": created_at,
                "tau_initial": 1.0,
            })
            if namespace:
                self._svc.kuzu_store.ensure_session(namespace)
                self._svc.kuzu_store.link_to_session(namespace, record_id)
            if self._svc.dream_scheduler:
                await self._svc.dream_scheduler.on_node_created()

        return SensoryResponse(record_id=record_id, buffer_usage=buffer_usage)

    async def store_episode(
        self,
        content: str,
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
        force_promote: bool = False,
        namespace: Optional[str] = None,
        visibility: str = "private",
    ) -> EpisodeResponse:
        """直接创建 Layer2 情节节点。

        从 _routes.create_episode 提取核心逻辑：τ 计算 → SSM 门控 →
        本体验证 → 持久化 → 命名空间链接 → 关系抽取。
        """
        creds = _scan_credentials(content)
        if creds:
            self._logger.warning("Credential-like content detected in store_episode from %s: %s", source, creds)
        episode_id = str(uuid.uuid4())
        created_at = time.time()
        tau_initial = 1.0

        # τ 值计算
        if self._svc.tau_engine:
            tau_initial = self._svc.tau_engine.compute_tau(created_at)
            if tau_initial < self._svc.tau_engine.config.decay_threshold and not force_promote:
                return EpisodeResponse(
                    episode_id=episode_id, status="filtered", tau_initial=0.0,
                    content=content, source=source,
                )

        # SSM 门控过滤
        if self._svc.ssm_gate is not None and self._svc.tau_engine is not None:
            input_dim = self._svc.ssm_gate.config.input_dim
            age_hours = 0.0
            features = np.zeros(input_dim, dtype=np.float32)
            norm_len = min(float(len(content)) / 1000.0, 1.0)
            features[0] = norm_len
            if input_dim > 1:
                features[1] = age_hours
            if input_dim > 2:
                features[2] = 0.0
            try:
                _ = self._svc.ssm_gate.hidden_state
            except AttributeError:
                self._svc.ssm_gate.hidden_state = self._svc.ssm_gate.reset_state()
            hidden, gate_value = self._svc.ssm_gate.step(
                self._svc.ssm_gate.hidden_state, features
            )
            self._svc.ssm_gate.hidden_state = hidden
            if not self._svc.ssm_gate.should_keep(gate_value):
                return EpisodeResponse(
                    episode_id=episode_id, status="filtered", tau_initial=0.0,
                    content=content, source=source,
                )

        # 持久化
        self._svc.kuzu_store.create_episode({
            "id": episode_id,
            "content": content,
            "source": source,
            "visibility": visibility,
            "created_at": created_at,
            "tau_initial": tau_initial,
        })

        # 命名空间链接
        if namespace:
            self._svc.kuzu_store.ensure_session(namespace)
            self._svc.kuzu_store.link_to_session(namespace, episode_id)

        # 通知梦境调度器
        if self._svc.dream_scheduler:
            await self._svc.dream_scheduler.on_activity()
            await self._svc.dream_scheduler.on_node_created()

        return EpisodeResponse(
            episode_id=episode_id,
            content=content[:200],
            tau_initial=tau_initial,
            created_at=created_at,
            source=source,
        )

    async def store_multimodal(
        self,
        text: Optional[str] = None,
        images: Optional[list[bytes]] = None,
        audio: Optional[list[bytes]] = None,
        video: Optional[list[bytes]] = None,
        source: str = "api",
        namespace: Optional[str] = None,
        visibility: str = "private",
    ) -> MultimodalResponse:
        """写入多模态记忆（CLIP 嵌入 + 媒体文件存储 + 文本索引）。

        Args:
            text: 可选文本描述。
            images: Base64 编码的图像字节列表。
            audio: Base64 编码的音频字节列表。
            video: Base64 编码的视频字节列表。
            source: 来源标识。
            namespace: 命名空间。
            visibility: 可见性。

        Returns:
            MultimodalResponse 包含 episode_id、media_paths 等。
        """
        import base64
        import time
        import uuid
        import numpy as np

        episode_id = str(uuid.uuid4())
        created_at = time.time()
        media_paths: list[str] = []
        transcription: Optional[str] = None
        visual_node_id: Optional[str] = None

        # 懒加载多模态组件
        clip = getattr(self._svc, "_clip_embedder", None)
        if clip is None:
            try:
                from multimodal.embedders import ClipEmbedder
                clip = ClipEmbedder()
                self._svc._clip_embedder = clip
            except Exception:
                clip = None
        whisper = getattr(self._svc, "_whisper_embedder", None)
        if whisper is None:
            try:
                from multimodal.embedders import WhisperEmbedder
                whisper = WhisperEmbedder()
                self._svc._whisper_embedder = whisper
            except Exception:
                whisper = None
        store = getattr(self._svc, "_media_store", None)
        if store is None:
            try:
                from multimodal.store import MediaStore
                store = MediaStore()
                self._svc._media_store = store
            except Exception:
                store = None

        # 处理图像
        image_embs: list[np.ndarray] = []
        for img_bytes in (images or []):
            try:
                data = base64.b64decode(img_bytes) if isinstance(img_bytes, str) else img_bytes
            except Exception:
                continue
            if store is not None:
                media_paths.append(store.save_image(data))
            if clip is not None:
                emb = clip.embed_image(data)
                if emb is not None:
                    image_embs.append(emb)

        # 处理音频
        audio_texts: list[str] = []
        for aud_bytes in (audio or []):
            try:
                data = base64.b64decode(aud_bytes) if isinstance(aud_bytes, str) else aud_bytes
            except Exception:
                continue
            if store is not None:
                media_paths.append(store.save_audio(data))
            if whisper is not None:
                seg = whisper.transcribe(data)
                if seg:
                    audio_texts.append(seg)

        # 处理视频
        for vid_bytes in (video or []):
            try:
                data = base64.b64decode(vid_bytes) if isinstance(vid_bytes, str) else vid_bytes
            except Exception:
                continue
            if store is not None:
                media_paths.append(store.save_video(data))

        # 合并文本
        text_parts: list[str] = []
        if text:
            text_parts.append(text)
        if audio_texts:
            transcription = " ".join(audio_texts)
            text_parts.append(f"[audio transcription]: {transcription}")
        merged_text = "\n".join(text_parts) if text_parts else ""

        # 写入 VisualNode（有图像时）
        if image_embs and clip is not None:
            visual_emb = np.mean(image_embs, axis=0).astype(np.float32)
            try:
                proj = getattr(self._svc, "_clip_projection", None)
                if proj is None:
                    rng = np.random.default_rng(42)
                    proj = rng.standard_normal((512, 384), dtype=np.float32)
                    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
                    self._svc._clip_projection = proj
                emb_384 = visual_emb @ proj

                visual_node_id = str(uuid.uuid4())
                if self._svc.kuzu_store is not None:
                    self._svc.kuzu_store.create_visual_node({
                        "id": visual_node_id,
                        "image_path": media_paths[0] if media_paths else "",
                        "caption": merged_text[:1024],
                        "embedding": emb_384.tolist(),
                        "source": source,
                        "created_at": created_at,
                    })
            except Exception:
                self._logger.exception("VisualNode creation failed (non-fatal)")

        # 写入 EpisodeNode（文本索引）
        if merged_text and self._svc.kuzu_store is not None:
            creds = _scan_credentials(merged_text)
            if creds:
                self._logger.warning("Credential-like content detected in store_multimodal: %s", creds)
            self._svc.kuzu_store.create_episode({
                "id": episode_id,
                "content": merged_text,
                "source": source,
                "visibility": visibility,
                "created_at": created_at,
                "tau_initial": 1.0,
            })
            if namespace:
                self._svc.kuzu_store.ensure_session(namespace)
                self._svc.kuzu_store.link_to_session(namespace, episode_id)
            if self._svc.dream_scheduler:
                await self._svc.dream_scheduler.on_activity()
                await self._svc.dream_scheduler.on_node_created()

        return MultimodalResponse(
            episode_id=episode_id,
            visual_node_id=visual_node_id,
            text=merged_text[:200] if merged_text else None,
            media_paths=media_paths,
            transcription=transcription,
            created_at=created_at,
        )

    # ─── 检索 ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        strategy: Optional[str] = "auto",
        namespace: Optional[str] = None,
        include_shared: bool = True,
    ) -> RetrieveResponse:
        """粗到精三级融合检索，带 Cypher 兜底和去重。"""
        start = time.time()

        if self._svc.query_router is None:
            return RetrieveResponse(
                query=query, strategy_used=strategy or "auto",
                results=[], total_found=0, latency_ms=0, degraded=True,
            )

        try:
            results_raw = self._svc.query_router.retrieve(query)
        except Exception as exc:
            self._logger.exception("Query router failed")
            return RetrieveResponse(
                query=query, strategy_used=strategy or "auto",
                results=[], total_found=0,
                latency_ms=(time.time() - start) * 1000, degraded=True,
            )

        degraded = False

        # Cypher 兜底
        if not results_raw and self._svc.kuzu_store is not None:
            try:
                words = [w.strip().lower() for w in query.split() if len(w.strip()) > 1]
                if words:
                    params = {f"w{i}": w for i, w in enumerate(words[:5])}
                    conditions = " OR ".join(
                        f"toLower(e.content) CONTAINS $w{i}" for i in range(len(words[:5]))
                    )
                    cypher = (
                        f"MATCH (e:EpisodeNode) WHERE {conditions} "
                        f"RETURN e.id AS node_id, e.content AS content LIMIT 10"
                    )
                    fallback_rows = self._svc.kuzu_store.query_cypher(cypher, params)
                    degraded = True
                    for row in fallback_rows:
                        if isinstance(row, (list, tuple)):
                            nid, c = row[0], row[1] if len(row) > 1 else ""
                        elif isinstance(row, dict):
                            nid, c = row.get("node_id", ""), row.get("content", "")
                        else:
                            continue
                        results_raw.append({
                            "node_id": str(nid),
                            "content": str(c),
                            "score": 0.5,
                            "level": "kuzu_fallback",
                        })
            except Exception:
                self._logger.exception("Cypher fallback failed")

        # 去重 + 命名空间过滤
        if results_raw:
            seen: set = set()
            deduped = []
            ns_set: Optional[set[str]] = None
            if namespace and self._svc.kuzu_store is not None:
                try:
                    ns_rows = self._svc.kuzu_store.query_cypher(
                        "MATCH (s:SessionNode {session_id: $ns})-[:SESSION_MEMBER]->(e:EpisodeNode) "
                        "RETURN e.id",
                        {"ns": namespace},
                    )
                    ns_set = {row[0] for row in ns_rows} if ns_rows else set()
                except Exception:
                    pass
            for r in results_raw:
                key = r.get("content", "")[:100]
                if key and key not in seen:
                    if ns_set is not None and r.get("node_id", "") not in ns_set:
                        continue
                    seen.add(key)
                    deduped.append(r)
            results_raw = deduped

        # 降级标记
        if results_raw:
            first_level = (
                results_raw[0].get("level", "hypergraph")
                if isinstance(results_raw[0], dict)
                else "hypergraph"
            )
            degraded = first_level != "hypergraph"

        results = [
            EpisodicResult(
                node_id=r.get("node_id", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                retrieval_level=r.get("level", "hypergraph"),
            )
            for r in results_raw
        ]

        latency = (time.time() - start) * 1000
        return RetrieveResponse(
            query=query,
            strategy_used=strategy or "auto",
            results=results,
            total_found=len(results),
            latency_ms=round(latency, 2),
            degraded=degraded,
        )

    async def search_vector(self, query: str, limit: int = 10) -> SearchVectorResponse:
        """纯向量检索（直通 FAISS）。"""
        start = time.time()
        degraded = False

        if self._svc.encoder is None or self._svc.faiss_index is None:
            degraded = True
            return SearchVectorResponse(
                query=query, results=[], total_found=0,
                latency_ms=(time.time() - start) * 1000, degraded=True,
            )

        try:
            emb = self._svc.encoder.embed(query)
            if emb is None:
                raise ValueError("Encoder returned None")

            emb_array = emb.reshape(1, -1).astype(np.float32)
            k = min(limit, self._svc.faiss_index.ntotal) if self._svc.faiss_index.ntotal > 0 else limit
            if k == 0:
                return SearchVectorResponse(
                    query=query, results=[], total_found=0,
                    latency_ms=(time.time() - start) * 1000, degraded=False,
                )

            distances, indices = self._svc.faiss_index.search(emb_array, k)

            results: list[SearchVectorResult] = []
            faiss_id_map = getattr(self._svc, "faiss_id_map", {}) or {}
            for rank in range(len(indices[0])):
                faiss_id = int(indices[0][rank])
                if faiss_id < 0:
                    continue
                l2_dist = float(distances[0][rank])
                score = max(0.0, 1.0 - l2_dist / 2.0)
                episode_id = faiss_id_map.get(faiss_id)
                if not episode_id:
                    continue
                try:
                    node = self._svc.kuzu_store.get_episode(episode_id) if self._svc.kuzu_store else None
                    content = node.get("content", "") if node else ""
                except Exception:
                    content = ""
                results.append(SearchVectorResult(
                    node_id=episode_id,
                    content=content,
                    score=round(score, 4),
                    faiss_id=faiss_id,
                ))
        except Exception:
            self._logger.exception("Vector search failed")
            return SearchVectorResponse(
                query=query, results=[], total_found=0,
                latency_ms=(time.time() - start) * 1000, degraded=True,
            )

        latency = (time.time() - start) * 1000
        return SearchVectorResponse(
            query=query,
            results=results,
            total_found=len(results),
            latency_ms=round(latency, 2),
            degraded=degraded,
        )

    # ─── 梦境 ────────────────────────────────────────────────────────────

    async def trigger_dream(self, mode: str = "auto") -> DreamTriggerResponse:
        """显式触发梦境整合管道。"""
        if self._svc.dream_scheduler is None:
            return DreamTriggerResponse(
                accepted=False, message="Dream scheduler not available"
            )
        accepted = await self._svc.dream_scheduler.trigger_explicit()
        if accepted:
            return DreamTriggerResponse(accepted=True, message="Dream triggered successfully")
        return DreamTriggerResponse(accepted=False, message="Dream already running")

    # ─── 治理 ────────────────────────────────────────────────────────────

    async def health(self) -> HealthStatus:
        """深度健康检查，覆盖所有核心组件。"""
        start = time.time()
        checker = HealthChecker(
            graph_store=self._svc.kuzu_store,
            faiss_index=self._svc.faiss_index,
            audit_chain=self._svc.audit_chain,
            dream_scheduler=self._svc.dream_scheduler,
        )
        health = checker.check()

        stats: Dict[str, Any] = {
            "version": "",
            "version_name": "",
            "uptime_seconds": health.uptime_seconds,
            "faiss_index_size": health.faiss_index_size,
            "chain_verified": health.chain_verified,
            "node_count": health.node_count,
            "hyperedge_count": health.hyperedge_count,
            "last_dream_time": health.last_dream_time,
            "dream_run_count": health.dream_run_count,
            "circuit_breaker": health.details.get("circuit_breaker", {}),
            "memory": health.details.get("memory_usage", {}),
        }
        try:
            from shm._version import __version__, __version_name__
            stats["version"] = __version__
            stats["version_name"] = __version_name__
        except ImportError:
            pass

        return HealthStatus(
            status=health.status,
            graph_connected=health.graph_connected,
            faiss_loaded=health.faiss_loaded,
            dream_scheduler_running=health.dream_scheduler_running,
            stats=stats,
            timestamp=start,
        )

    async def get_audit(self, node_id: str) -> AuditTrace:
        """查询指定节点的完整溯源链。"""
        if self._svc.audit_chain is None:
            return AuditTrace(
                node_id=node_id, operations=[], chain_verified=False, total_blocks=0,
            )

        chain_verified = False
        try:
            chain_verified = self._svc.audit_chain.verify_chain()
        except Exception:
            self._logger.warning("Audit chain verification failed")

        ops_raw = self._svc.audit_chain.trace_node(node_id)
        chain_length = self._svc.audit_chain.chain_length

        operations = [
            AuditOperation(
                op_type=op.op_type,
                node_id=op.node_id,
                old_value=op.old_value,
                new_value=op.new_value,
                reason=op.reason,
            )
            for op in ops_raw
        ]

        return AuditTrace(
            node_id=node_id,
            operations=operations,
            chain_verified=chain_verified,
            total_blocks=chain_length,
        )

    async def cypher_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> dict:
        """执行只读 Cypher 查询（禁止写操作）。"""
        import re
        stripped = re.sub(r'"[^"]*"|\'[^\']*\'|`[^`]*`', '', query)
        blocked = re.compile(
            r"\b(?:CREATE|DELETE|SET\s+\w+|DROP|MERGE|REMOVE|DETACH|INSERT|LOAD\s+CSV)\b", re.IGNORECASE
        )
        if blocked.search(stripped):
            return {"error": "Write queries blocked: contains CREATE/DELETE/SET/DROP/MERGE/REMOVE/DETACH", "rows": [], "count": 0}
        params = params or {}
        try:
            rows = self._svc.kuzu_store.query_cypher(query, params)
            return {"rows": rows, "count": len(rows)}
        except Exception as e:
            return {"error": str(e), "rows": [], "count": 0}

    # ─── 图 ─────────────────────────────────────────────────────────────

    async def list_communities(self, limit: int = 50, offset: int = 0) -> CommunityListResponse:
        """列出所有社区 (Layer3)。"""
        try:
            rows = self._svc.kuzu_store.query_cypher(
                "MATCH (c:CommunityNode) RETURN c.* ORDER BY c.created_at DESC "
                "LIMIT $limit",
                {"offset": offset, "limit": limit},
            )
        except Exception as e:
            self._logger.exception("Failed to list communities")
            return CommunityListResponse(communities=[], total=0)

        def _to_dict(row: Any) -> dict:
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
        return CommunityListResponse(communities=communities, total=len(communities))

    async def list_hyperedges(
        self, node_id: Optional[str] = None, limit: int = 50
    ) -> HyperedgeListResponse:
        """列出超边。指定 node_id 时只返回包含该节点的超边。"""
        if node_id:
            if self._svc.hyperedge_manager is None:
                return HyperedgeListResponse(hyperedges=[], total=0)
            edges = self._svc.hyperedge_manager.get_hyperedges_by_node(node_id)
            items = [
                HyperedgeResponse(
                    id=e.id,
                    type=APIHyperedgeType(e.type.value if hasattr(e.type, "value") else str(e.type)),
                    member_ids=e.member_ids,
                    created_at=e.created_at,
                    gate_value=e.gate_value,
                    metadata=e.metadata,
                )
                for e in edges
            ]
            return HyperedgeListResponse(hyperedges=items, total=len(items))

        # 列出所有超边
        try:
            rows = self._svc.kuzu_store.query_cypher(
                "MATCH (h:HyperedgeNode) RETURN h.* ORDER BY h.created_at DESC LIMIT $limit",
                {"limit": limit},
            )
        except Exception as e:
            self._logger.exception("Failed to list hyperedges")
            return HyperedgeListResponse(hyperedges=[], total=0)

        results = []
        for row in rows:
            if isinstance(row, (list, tuple)):
                h = {"id": row[0], "type": row[1], "created_at": row[2],
                     "gate_value": row[3], "metadata": row[4]}
            elif isinstance(row, dict):
                h = {k.split(".")[-1]: v for k, v in row.items()}
            else:
                continue

            member_rows = self._svc.kuzu_store.query_cypher(
                "MATCH (h:HyperedgeNode {id: $id})-[:HYPEREDGE_MEMBER]->(e:EpisodeNode) RETURN e.id",
                {"id": h["id"]},
            )
            member_ids = []
            for mr in member_rows:
                if isinstance(mr, (list, tuple)):
                    member_ids.append(str(mr[0]))
                elif isinstance(mr, dict):
                    member_ids.append(str(mr.get("id", "")))

            import json as _j
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

        return HyperedgeListResponse(hyperedges=results, total=len(results))
