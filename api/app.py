"""
FastAPI 应用工厂
================
- 生命周期管理（startup/shutdown）
- 中间件链（trace_id, CORS, 性能监控）
- 所有服务初始化 + 依赖注入
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import load_settings, get_settings
from api.routes import router, init_services, Services
from observability.metrics import record_request
from observability.logger import get_logger, configure_logging
from api.dashboard import dashboard_router

logger = get_logger(__name__)


def _init_services() -> Services:
    """初始化所有服务组件，单个组件失败不影响整体启动。"""
    cfg = get_settings()
    svc = Services()
    errors = []

    # 1. GraphLite 图数据库 (替换RyuGraph)
    try:
        import sys as _sys
        _bindings = "/home/admin/GraphLite/bindings/python"
        _sdk = "/home/admin/GraphLite/sdk-python/src"
        for _p in [_bindings, _sdk]:
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from graph.graphlite_store import GraphLiteStore
        graphlite_cfg = type("cfg", (), {
            "database_path": str(cfg.graphlite.database_path),
            "max_threads": cfg.graphlite.max_threads,
        })()
        svc.graphlite_store = GraphLiteStore(config=graphlite_cfg)
        svc.graphlite_store.connect()
        logger.info("GraphLiteStore initialized", path=graphlite_cfg.database_path)
    except Exception as e:
        import traceback
        errors.append(f"GraphLiteStore: {e}")
        logger.warning("GraphLiteStore init failed", error=str(e), traceback=traceback.format_exc())

    # 2. FAISS 向量索引
    if svc.graphlite_store is not None:
        try:
            # ── 编码器初始化（三层降级架构） ──
            # Tier 1: Cloud API / Tier 2: Local sentence-transformers / Tier 3: TF-IDF
            import os
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"

            from embedding.encoder import create_encoder, TfidfEncoder

            try:
                logger.info("Attempting Tier 2 encoder: create_encoder()")
                svc.encoder = create_encoder(
                    model_name=cfg.embedding.model_name,
                    device=cfg.embedding.device,
                    prefer_cloud=False,  # ONNX优先，跳过云端API阻塞
                )
                svc.encoder.load()
                logger.info("TextEncoder (Tier 2) initialized", model=cfg.embedding.model_name)
            except Exception:
                logger.warning("Tier 2 encoder failed, falling back to Tier 3 (TF-IDF)", exc_info=True)
                svc.encoder = TfidfEncoder()
                svc.encoder.load()
                logger.info("TfidfEncoder (Tier 3) initialized as fallback")

        except Exception as e:
            errors.append(f"TextEncoder: {e}")
            svc.encoder = None
            logger.warning("TextEncoder init failed (fallback: embedding disabled)", error=str(e))

        try:
            from retrieval.vector_store import VectorStoreFactory
            # FAISS 维度从 encoder 动态获取（bge=512 / MiniLM-ONNX=384 / TF-IDF=384），
            # 避免 dim 不匹配报错；encoder 不可用时回退到配置默认值
            dim = getattr(svc.encoder, "dimension", None)
            if not isinstance(dim, int) or dim <= 0:
                dim = cfg.faiss.dimension
            store = VectorStoreFactory.create(
                dimension=dim,
                index_type=cfg.faiss.index_type,
                nlist=cfg.faiss.nlist,
            )
            svc.vector_store = store
            svc.faiss_index = store.index      # 保持向后兼容
            svc.faiss_dim = dim
            svc.faiss_index_type = store.index_type
            svc.faiss_nlist = store.nlist
            svc.faiss_id_map = store.id_map
            logger.info("VectorStore initialized", engine="faiss", dim=dim)
        except Exception as e:
            errors.append(f"FAISS: {e}")
            logger.warning("FAISS init failed (fallback: vector search disabled)", error=str(e))

    # 3. HyperedgeManager
    if svc.graphlite_store is not None:
        try:
            from graph.hyperedge import HyperedgeManager
            svc.hyperedge_manager = HyperedgeManager(graphlite_store=svc.graphlite_store)
            logger.info("HyperedgeManager initialized")
        except Exception as e:
            errors.append(f"HyperedgeManager: {e}")
            logger.warning("HyperedgeManager init failed", error=str(e))

    # 4. Tau 衰减引擎
    try:
        from core.tau_decay import TauDecayEngine, TauDecayConfig
        tcfg = cfg.tau
        tau_cfg = TauDecayConfig(
            tau_initial=tcfg.tau_initial,
            tau_decay_seconds=tcfg.tau_decay_seconds,
            decay_threshold=tcfg.decay_threshold,
            refresh_on_access=tcfg.refresh_on_access,
        )
        svc.tau_engine = TauDecayEngine(config=tau_cfg)
        logger.info("TauEngine initialized", decay_seconds=cfg.tau.tau_decay_seconds)
    except Exception as e:
        errors.append(f"TauEngine: {e}")
        logger.warning("TauEngine init failed", error=str(e))

    # 5. Hebbian 更新器
    try:
        from core.hebbian import SparseHebbianUpdater
        svc.hebbian_updater = SparseHebbianUpdater(config=cfg.hebbian)
        logger.info("HebbianUpdater initialized", k_sparsity=cfg.hebbian.k_sparsity)
    except Exception as e:
        errors.append(f"HebbianUpdater: {e}")
        logger.warning("HebbianUpdater init failed", error=str(e))

    # 6. SSM 门控
    try:
        from core.dual_gate import DualAdaptiveGate, DualGateConfig
        # 从 cfg.ssm 映射公共字段到 DualGateConfig
        dg = DualGateConfig(
            hidden_dim=cfg.ssm.hidden_dim,
            input_dim=cfg.ssm.input_dim,
            gate_threshold=cfg.ssm.gate_threshold,
            ssm_state_decay=cfg.ssm.state_decay,
            feat_mean_activation=cfg.ssm.feat_mean_activation,
            feat_age_hours=cfg.ssm.feat_age_hours,
            feat_access_freq=cfg.ssm.feat_access_freq,
            feat_member_count=cfg.ssm.feat_member_count,
            feat_community_density=cfg.ssm.feat_community_density,
            feat_tau_mean=getattr(cfg.ssm, 'feat_tau_mean', 5),
            feat_tau_variance=getattr(cfg.ssm, 'feat_tau_variance', 6),
            feat_connection_entropy=getattr(cfg.ssm, 'feat_connection_entropy', 7),
            seed=getattr(cfg.ssm, 'seed', 42),
        )
        svc.ssm_gate = DualAdaptiveGate(config=dg)
        logger.info("AdaptiveGate initialized")
    except Exception as e:
        errors.append(f"AdaptiveGate: {e}")
        logger.warning("AdaptiveGate init failed", error=str(e))

    # 7b. 本体验证器（[Ontology] 写时+读时验证层）
    if svc.graphlite_store is not None:
        try:
            from core.ontology_validator import OntologyValidator
            svc.ontology_validator = OntologyValidator(
                graphlite_store=svc.graphlite_store,
                encoder=svc.encoder,
                config=cfg.ontology,
            )
            logger.info("OntologyValidator initialized", enabled=cfg.ontology.enabled)
        except Exception as e:
            errors.append(f"OntologyValidator: {e}")
            logger.warning("OntologyValidator init failed (fallback: no ontology validation)", error=str(e))

    # 7c. 本体 v2（动态类型系统）
    try:
        from core.ontology_v2 import OntologyService
        svc.ontology_v2 = OntologyService()
        logger.info("Ontology v2 initialized", entity_types=len(svc.ontology_v2.entity_types),
                    edge_types=len(svc.ontology_v2.edge_types))
    except Exception as e:
        errors.append(f"OntologyV2: {e}")
        logger.warning("Ontology v2 init failed", error=str(e))

    # 7d. 置信度追踪器 (Step 2)
    try:
        from core.evidence_tracker import EvidenceTracker
        svc.evidence_tracker = EvidenceTracker()
        logger.info("Evidence tracker initialized")
    except Exception as e:
        errors.append(f"EvidenceTracker: {e}")
        logger.warning("Evidence tracker init failed", error=str(e))

    # 8. 溯源链（【FIX】移到了前面，确保dream_pipeline能接收audit_chain）
    try:
        from core.audit_chain import AuditChain
        svc.audit_chain = AuditChain()
        logger.info("AuditChain initialized")
    except Exception as e:
        errors.append(f"AuditChain: {e}")
        logger.warning("AuditChain init failed", error=str(e))

    # 7. 梦境管道 & 调度器（【FIX】移到audit_chain之后）
    try:
        from core.dream_pipeline import DreamPipeline
        # 【P0-1】LLM 客户端注入
        llm_client = None
        try:
            from core.llm_client import LLMClient
            llm_client = LLMClient()  # 自动从 config/settings.py 读取 llm 段
            logger.info("LLMClient initialized for dream synthesis (endpoint=%s, model=%s)",
                       llm_client.base_url, llm_client.model)
        except Exception as e:
            logger.warning("LLMClient init skipped (dreams will use TF-IDF fallback): %s", e)

        svc.dream_pipeline = DreamPipeline(
            tau_engine=svc.tau_engine,
            hebbian_updater=svc.hebbian_updater,
            audit_chain=svc.audit_chain,  # ← 现在有值了
            llm_client=llm_client,
            ontology_validator=svc.ontology_validator if hasattr(svc, 'ontology_validator') else None,
        )
        # 【P0-2】梦境候选存储（非破坏性模式）
        try:
            from core.dream_candidate_store import DreamCandidateStore
            svc.dream_candidate_store = DreamCandidateStore()
            logger.info("DreamCandidateStore initialized")
        except Exception as e:
            logger.warning("DreamCandidateStore init skipped: %s", e)
        from core.dream_scheduler import DreamScheduler, DreamSchedulerConfig
        dcfg = cfg.dream
        dream_cfg = DreamSchedulerConfig(
            idle_timeout_seconds=dcfg.idle_timeout_seconds,
            accum_threshold=dcfg.accum_threshold,
            min_interval_seconds=dcfg.min_interval_seconds,
            max_dream_duration_seconds=dcfg.max_dream_duration_seconds,
        )
        pipeline_fn = None
        if svc.dream_pipeline is not None and hasattr(svc.dream_pipeline, "run"):
            pipeline_fn = svc.dream_pipeline.run
        svc.dream_scheduler = DreamScheduler(
            config=dream_cfg,
            pipeline_fn=pipeline_fn,
        )
        # 【FIX】注入GraphLite引用供梦境调度器拉取数据
        svc.dream_scheduler._graphlite_store = svc.graphlite_store
        # 注入FAISS引用供梦境后增量更新索引
        svc.dream_scheduler._faiss_index = svc.faiss_index
        svc.dream_scheduler._faiss_id_map = getattr(svc, "faiss_id_map", {})
        from api.routes import incremental_faiss_update
        svc.dream_scheduler._incremental_update_fn = incremental_faiss_update
        # 【P0-2】注入候选存储引用
        if hasattr(svc, 'dream_candidate_store'):
            svc.dream_scheduler._candidate_store = svc.dream_candidate_store
        logger.info("Dream system initialized")
    except Exception as e:
        errors.append(f"DreamSystem: {e}")
        logger.warning("Dream system init failed", error=str(e))

    # 9. 查询路由
    try:
        from retrieval.query_router import QueryRouter, QueryRouterConfig as QRCfg
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        class TfidfSearchIndex:
            """包装 TF-IDF 向量化器提供 search() 接口。"""
            def __init__(self):
                # 字符级 2-4gram：兼容中文单字/短文本（默认 \b\w+\b 对 CJK 边界失效）
                self.vectorizer = TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 4), max_features=5000
                )
                self._fitted = False
            def fit(self, texts):
                if texts:
                    self.matrix = self.vectorizer.fit_transform(texts)
                    self._fitted = True
                    self.texts = texts
            def search(self, query, k=20):
                if not self._fitted or not hasattr(self, "matrix"):
                    return []
                q_vec = self.vectorizer.transform([query])
                scores = cosine_similarity(q_vec, self.matrix)[0]
                top_k = min(k, len(scores))
                if top_k == 0:
                    return []
                top_indices = np.argsort(scores)[-top_k:][::-1]
                return [(self.texts[i], float(scores[i])) for i in top_indices]

        tfidf_index = TfidfSearchIndex()
        svc.tfidf_index = tfidf_index

        qr_kwargs = {
            "graphlite_store": svc.graphlite_store,
            "faiss_index": svc.faiss_index,
            "tfidf_index": tfidf_index,
            "encoder": svc.encoder,
            # 【修复】query_router 和 _routes 共享同一个 faiss_id_map 对象
            "faiss_id_map": svc.faiss_id_map,  # 引用传递
            "episode_cache": getattr(svc, "_episode_cache", {}),  # 【Perf】共享缓存，flush_faiss_buffer 的修改对 query_router 可见
        }
        rcfg = cfg.retrieval
        qr_kwargs["config"] = QRCfg(
            tau_weight=rcfg.tau_weight,
            vector_weight=rcfg.vector_weight,
            top_k_l1=rcfg.top_k_l1,
            top_k_vector=rcfg.top_k_vector,
            top_k_keyword=rcfg.top_k_keyword,
        )
        svc.query_router = QueryRouter(**qr_kwargs)
        logger.info("QueryRouter initialized")

        # 【自演化】包装检索层，实现检索配置自演化
        try:
            from retrieval.self_evolving import SelfEvolvingRetrieval
            svc.query_router = SelfEvolvingRetrieval(svc.query_router)
            logger.info("SelfEvolvingRetrieval wrapped")
        except Exception as evo_e:
            logger.warning("SelfEvolvingRetrieval init failed", error=str(evo_e))
    except Exception as e:
        errors.append(f"QueryRouter: {e}")
        logger.warning("QueryRouter init failed", error=str(e))

    # 10. 记忆投毒防御引擎（可独立于 GraphLite 运行）
    try:
        from core.defense import MemoryDefenseEngine, DefenseConfig as CoreDefenseConfig
        _dcfg = cfg.defense
        svc.defense_engine = MemoryDefenseEngine(
            config=CoreDefenseConfig(
                enabled=_dcfg.enabled,
                silent=_dcfg.silent,
                max_writes_per_window=_dcfg.max_writes_per_window,
                write_window_seconds=_dcfg.write_window_seconds,
                drift_cosine_threshold=_dcfg.drift_cosine_threshold,
                drift_reference_window=_dcfg.drift_reference_window,
                max_entity_cooccurrence=_dcfg.max_entity_cooccurrence,
                max_repeat_exact=_dcfg.max_repeat_exact,
                repeat_dedup_window=_dcfg.repeat_dedup_window,
                trust_decay_per_block=_dcfg.trust_decay_per_block,
                trust_recovery_writes=_dcfg.trust_recovery_writes,
                initial_trust=_dcfg.initial_trust,
                block_trust_threshold=_dcfg.block_trust_threshold,
                quarantine_trust_threshold=_dcfg.quarantine_trust_threshold,
            ),
            encoder=svc.encoder,
        )
        logger.info("DefenseEngine initialized",
                     enabled=_dcfg.enabled, silent=_dcfg.silent)
    except Exception as e:
        errors.append(f"DefenseEngine: {e}")
        logger.warning("DefenseEngine init failed (fallback: no defense)", error=str(e))

    # 11. 隔离存储（依赖 GraphLite）
    if svc.graphlite_store is not None:
        try:
            from core.quarantine_store import QuarantineStore
            svc.quarantine_store = QuarantineStore(graph_store=svc.graphlite_store)
            # 启动时从 GraphLite 同步已有隔离节点
            q_count = svc.quarantine_store.refresh()
            logger.info("QuarantineStore initialized", quarantined_count=q_count)
        except Exception as e:
            errors.append(f"QuarantineStore: {e}")
            logger.warning("QuarantineStore init failed (fallback: in-memory only)", error=str(e))
    else:
        # 无 GraphLite 时使用纯内存模式
        try:
            from core.quarantine_store import QuarantineStore
            svc.quarantine_store = QuarantineStore()
            logger.info("QuarantineStore initialized (memory-only, no GraphLite)")
        except Exception as e:
            errors.append(f"QuarantineStore: {e}")

    if errors:
        logger.warning("Services initialized with errors", count=len(errors), errors=errors)
    else:
        logger.info("All services initialized successfully")

    return svc


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """应用生命周期管理"""
    # startup
    configure_logging()
    app.state.config = load_settings()
    app.state.logger = logger
    logger.info("SHM v4.0 starting up", config_path=str(get_settings().graphlite.database_path))

    # 初始化所有服务并注入
    svc = _init_services()
    init_services(svc)
    logger.info("Services injected into route handlers")

    # 启动后台重建 FAISS 索引 + TF-IDF 拟合（非阻塞）
    async def _startup_rebuild() -> None:
        try:
            from api.routes import rebuild_index
            from fastapi import Depends
            logger.info("Startup: auto-building FAISS + TF-IDF...")
            result = await rebuild_index(svc)
            idx = result.get("indexed_count", 0)
            # 注意: rebuild_index (api/routes/system.py) 内部已做 TF-IDF fit
            # (含 GraphLite 嵌套/b64 兼容解析), 此处无需重复拟合。
            logger.info("Startup: FAISS auto-build complete", indexed=idx)
        except Exception as e:
            logger.warning("Startup FAISS auto-build skipped (non-fatal): %s", e)

    async def _cleanup_dream_candidates() -> None:
        """启动时清理过期的梦境候选文件（保留最近50个）。"""
        try:
            import os, glob
            cand_dir = os.path.join(os.path.dirname(__file__), "..", "data", "dream_candidates")
            if not os.path.isdir(cand_dir):
                return
            files = sorted(glob.glob(os.path.join(cand_dir, "*.json")), key=os.path.getmtime, reverse=True)
            if len(files) > 50:
                for f in files[50:]:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                logger.info("Startup cleanup: purged %d old dream candidates (kept 50)", len(files) - 50)
        except Exception:
            pass

    startup_task = asyncio.create_task(_startup_rebuild())
    cleanup_task = asyncio.create_task(_cleanup_dream_candidates())

    # 启动梦境调度器后台轮询（每60秒检查一次触发条件）
    DREAM_POLL_INTERVAL = 60.0

    # 【P1-3】从 GraphLite SystemNode 恢复梦境调度器状态
    if svc.graphlite_store is not None and svc.dream_scheduler is not None:
        try:
            rows = svc.graphlite_store.query_cypher(
                "MATCH (s:SystemNode) WHERE s.id = 'dream_scheduler_state' "
                "RETURN s.payload"
            )
            if rows and len(rows) > 0:
                import json as _json
                row = rows[0]
                payload_str = ""
                if isinstance(row, dict):
                    payload_str = str(row.get("payload", ""))
                elif isinstance(row, (list, tuple)):
                    payload_str = str(row[0]) if len(row) > 0 else ""
                if payload_str:
                    state = _json.loads(payload_str)
                    svc.dream_scheduler.load_state(state)
                    logger.info("Dream scheduler state restored: %s", {
                        k: v for k, v in state.items() if k != "saved_at"})
        except Exception as e:
            logger.debug("Dream scheduler state restore skipped: %s", e)

    async def _dream_poll_loop() -> None:
        logger.info("Dream poll loop started", interval=DREAM_POLL_INTERVAL)
        while True:
            try:
                await asyncio.sleep(DREAM_POLL_INTERVAL)
                # ── API Key 热同步：检查 Hermes 的 .env 文件是否有更新 ──
                try:
                    if hasattr(svc, 'dream_pipeline') and svc.dream_pipeline is not None:
                        llm = getattr(svc.dream_pipeline, 'llm_client', None)
                        if llm and hasattr(llm, 'hot_reload'):
                            if llm.hot_reload():
                                logger.info("API Key hot-reloaded from ~/.hermes/.env")
                except Exception:
                    pass
                # 定期 flush FAISS 缓冲区（每 5 秒，因为写入路径不再同步 flush）
                try:
                    from api.routes import flush_faiss_buffer
                    flushed = flush_faiss_buffer(svc)
                    if flushed:
                        logger.debug("Periodic FAISS buffer flush: %d vectors", flushed)
                except Exception:
                    pass
                if svc.dream_scheduler is not None and hasattr(svc.dream_scheduler, "check_and_trigger"):
                    triggered = await svc.dream_scheduler.check_and_trigger()
                    if triggered:
                        logger.info("Dream triggered by poll loop")
                        # 【P1-3】梦境完成后保存状态
                        try:
                            if svc.graphlite_store is not None:
                                import json as _json
                                state_json = _json.dumps(svc.dream_scheduler.save_state())
                                # GraphLite 不支持 MERGE：MATCH 存在性检查 + INSERT/SET
                                if svc.graphlite_store.execute_cypher(
                                    "MATCH (s:SystemNode {id: 'dream_scheduler_state'}) RETURN s"
                                ):
                                    svc.graphlite_store.execute_cypher(
                                        "MATCH (s:SystemNode {id: 'dream_scheduler_state'}) "
                                        "SET s.payload = $payload",
                                        {"payload": state_json},
                                    )
                                else:
                                    svc.graphlite_store.execute_cypher(
                                        "INSERT (s:SystemNode {id: 'dream_scheduler_state', "
                                        "payload: $payload})",
                                        {"payload": state_json},
                                    )
                        except Exception:
                            pass
                    # 自动 apply 梦境候选
                    if hasattr(svc, "dream_candidate_store") and svc.dream_candidate_store is not None:
                        try:
                            applied, communities, deleted = svc.dream_candidate_store.auto_apply_candidates(svc.graphlite_store)
                            if applied > 0:
                                logger.info("Auto-applied %d dreams: %d communities, %d files cleaned", applied, communities, deleted)
                        except Exception:
                            logger.exception("Auto-apply error (non-fatal)")
            except asyncio.CancelledError:
                logger.info("Dream poll loop cancelled")
                break
            except Exception:
                logger.exception("Dream poll error (non-fatal)")
        logger.info("Dream poll loop ended")

    async def _hyperedge_sweep() -> None:
        """定时扫描未形成超边的节点，尝试自动创建超边。"""
        import asyncio
        HYPEREDGE_SWEEP_INTERVAL = 600.0  # 每10分钟
        await asyncio.sleep(HYPEREDGE_SWEEP_INTERVAL)  # 启动后延迟
        while True:
            try:
                await asyncio.sleep(HYPEREDGE_SWEEP_INTERVAL)
                if svc.hyperedge_manager is not None and svc.graphlite_store is not None:
                    # 扫描长时间窗口内的同源节点
                    rows = svc.graphlite_store.query_cypher(
                        "MATCH (e:EpisodeNode) WHERE e.created_at >= $cutoff "
                        "RETURN e.id, e.source, e.content ORDER BY e.created_at DESC LIMIT 100",
                        {"cutoff": time.time() - 7200},
                    )
                    if rows and len(rows) >= 2:
                        logger.debug("Hyperedge sweep: %d recent episodes found, idle check", len(rows))
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    poll_task = asyncio.create_task(_dream_poll_loop())
    hyperedge_task = asyncio.create_task(_hyperedge_sweep())

    # 【Perf】嵌入队列消费 loop — 每 5 秒 flush FAISS
    async def _embed_poll_loop() -> None:
        logger.info("Embed poll loop started (interval=5s)")
        while True:
            try:
                # 导入 consumer 函数
                from api._routes import _process_embed_queue
                _process_embed_queue(svc)
            except Exception:
                logger.exception("Embed poll error (non-fatal)")
            await asyncio.sleep(5)
    embed_task = asyncio.create_task(_embed_poll_loop())

    yield
    # shutdown
    poll_task.cancel()
    hyperedge_task.cancel()
    if svc.graphlite_store:
        svc.graphlite_store.close()
    logger.info("SHM v4.0 shutting down")


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    app = FastAPI(
        title="SHM v4.0 — 自演化超图记忆系统",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — 生产环境应配置具体域名（通过 SHM_CORS_ORIGINS 环境变量）
    origins = os.environ.get("SHM_CORS_ORIGINS", "")
    if origins:
        allow_origins = [o.strip() for o in origins.split(",")]
    else:
        allow_origins = ["http://127.0.0.1:8000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 请求体大小限制（防止 DoS）
    max_body = int(os.environ.get("SHM_MAX_BODY", str(8 * 1024 * 1024)))  # 默认 8MB
    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_body:
            return JSONResponse(status_code=413, content={"error": "request_too_large",
                "detail": f"Max body size: {max_body} bytes"})
        return await call_next(request)

    # 认证 + 速率限制（在 observe_request 之前）
    from gateway.auth import create_auth_middleware, is_dev_mode
    dev_mode = is_dev_mode()
    skip_paths = ["/health", "/metrics"]
    app.middleware("http")(create_auth_middleware(dev_mode=dev_mode, skip_paths=skip_paths))

    # 请求级中间件：trace_id + 性能监控
    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id", uuid.uuid4().hex[:16])
        request.state.trace_id = trace_id
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        record_request(request.method, request.url.path, response.status_code, duration)
        response.headers["X-Trace-Id"] = trace_id
        return response

    app.include_router(router)
    app.include_router(dashboard_router)
    return app
