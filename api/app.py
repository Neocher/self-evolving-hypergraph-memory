"""
FastAPI 应用工厂
================
- 生命周期管理（startup/shutdown）
- 中间件链（trace_id, CORS, 性能监控）
- 所有服务初始化 + 依赖注入
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config.settings import load_settings, get_settings
from api.routes import router, init_services, Services
from observability.metrics import record_request
from observability.logger import get_logger, configure_logging

logger = get_logger(__name__)


def _init_services() -> Services:
    """初始化所有服务组件，单个组件失败不影响整体启动。"""
    cfg = get_settings()
    svc = Services()
    errors = []

    # 1. Kuzu 图数据库
    try:
        from graph.kuzu_store import KuzuStore, KuzuConfig as KuzuStoreConfig
        kuzu_cfg = KuzuStoreConfig(
            database_path=str(cfg.kuzu.database_path),
            buffer_pool_size_mb=cfg.kuzu.buffer_pool_size_mb,
            max_threads=cfg.kuzu.max_threads,
        )
        svc.kuzu_store = KuzuStore(config=kuzu_cfg)
        svc.kuzu_store.connect()
        logger.info("KuzuStore initialized", path=cfg.kuzu.database_path)
    except Exception as e:
        errors.append(f"KuzuStore: {e}")
        logger.warning("KuzuStore init failed", error=str(e))

    # 2. FAISS 向量索引
    if svc.kuzu_store is not None:
        try:
            # ── 编码器初始化（TextEncoder → 失败时降级到 TF-IDF） ──
            _encoder = None
            _encoder_cls = None
            try:
                from embedding.encoder import TextEncoder
                _encoder_cls = TextEncoder
            except Exception:
                logger.warning("TextEncoder import failed (CUDA/torch not available), using TF-IDF fallback")

            import os
            _cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            _model_cache_name = f"models--sentence-transformers--{cfg.embedding.model_name.replace('/', '--')}"
            _model_cached = os.path.isdir(os.path.join(_cache_dir, _model_cache_name))

            # 尝试加载 TextEncoder（本地缓存 + 非 CUDA）
            if _model_cached and _encoder_cls is not None:
                try:
                    logger.info("Encoder model found in local cache", model=cfg.embedding.model_name)
                    svc.encoder = _encoder_cls(
                        model_name=cfg.embedding.model_name,
                        device=cfg.embedding.device,
                    )
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    svc.encoder.load()
                    logger.info("TextEncoder initialized from local cache", model=cfg.embedding.model_name)
                    _encoder_ok = True
                except Exception as load_err:
                    logger.warning("TextEncoder load failed (no CUDA), falling back to TF-IDF: %s", load_err)
                    _encoder_ok = False
            else:
                _encoder_ok = False

            # TextEncoder 不可用时降级到 TF-IDF 本地编码器
            if not _encoder_ok:
                logger.info("Using sklearn TF-IDF fallback encoder (384-dim)")
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.random_projection import SparseRandomProjection
                import numpy as _np

                class LocalFallbackEncoder:
                    def __init__(self):
                        self._model = "local_fallback"
                        self._vectorizer = TfidfVectorizer(max_features=1024, analyzer="char_wb", ngram_range=(2, 4))
                        self._projector = None
                        self._fitted = False

                    def load(self):
                        pass

                    def embed(self, text: str) -> _np.ndarray:
                        if not self._fitted:
                            self._vectorizer.fit([text])
                            self._projector = SparseRandomProjection(n_components=384, random_state=42)
                            sample = self._vectorizer.transform(["", text])
                            self._projector.fit(sample)
                            self._fitted = True
                        vec = self._vectorizer.transform([text])
                        projected = self._projector.transform(vec)
                        return projected.toarray().astype(_np.float32).flatten()

                    def embed_batch(self, texts: list) -> _np.ndarray:
                        if not self._fitted:
                            self._vectorizer.fit(texts)
                            self._projector = SparseRandomProjection(n_components=384, random_state=42)
                            sample = self._vectorizer.transform(texts)
                            self._projector.fit(sample)
                            self._fitted = True
                        vec = self._vectorizer.transform(texts)
                        projected = self._projector.transform(vec)
                        return projected.toarray().astype(_np.float32)

                    @property
                    def dimension(self) -> int:
                        return 384

                    def track_indexed_node(self, node_id: str) -> None:
                        pass
                    def remove_pruned_nodes(self, pruned_node_ids: list) -> None:
                        pass
                    def should_rebuild_index(self) -> bool:
                        return False
                    def on_dream_cycle_complete(self) -> None:
                        pass
                    @property
                    def needs_rebuild(self) -> bool:
                        return False
                    @needs_rebuild.setter
                    def needs_rebuild(self, value: bool) -> None:
                        pass
                    @property
                    def indexed_count(self) -> int:
                        return 0

                svc.encoder = LocalFallbackEncoder()
                svc.encoder.load()
                logger.info("TextEncoder initialized (local TF-IDF fallback, 384-dim)")

        except Exception as e:
            errors.append(f"TextEncoder: {e}")
            svc.encoder = None
            logger.warning("TextEncoder init failed (fallback: embedding disabled)", error=str(e))

        try:
            import faiss
            import numpy as np
            dim = cfg.faiss.dimension
            # 启动时始终用 FlatL2（不需要训练，立即可用）
            # IVFFlat 在 POST /index/rebuild 时有真实数据才切换
            base_index = faiss.IndexFlatL2(dim)
            svc.faiss_index = faiss.IndexIDMap(base_index)
            svc.faiss_dim = dim
            svc.faiss_index_type = cfg.faiss.index_type
            svc.faiss_nlist = cfg.faiss.nlist

            # 存 faiss_id → node_id 逆向映射
            svc.faiss_id_map: dict[int, str] = {}
            logger.info("FAISS index initialized", type="FlatL2", dim=dim)
        except Exception as e:
            errors.append(f"FAISS: {e}")
            logger.warning("FAISS init failed (fallback: vector search disabled)", error=str(e))

    # 3. HyperedgeManager
    if svc.kuzu_store is not None:
        try:
            from graph.hyperedge import HyperedgeManager
            svc.hyperedge_manager = HyperedgeManager(kuzu_store=svc.kuzu_store)
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
        from core.ssm_gate import SSMGate
        svc.ssm_gate = SSMGate(config=cfg.ssm)
        logger.info("SSMGate initialized")
    except Exception as e:
        errors.append(f"SSMGate: {e}")
        logger.warning("SSMGate init failed", error=str(e))

    # 7b. 本体验证器（[Ontology] 写时+读时验证层）
    if svc.kuzu_store is not None:
        try:
            from core.ontology_validator import OntologyValidator
            svc.ontology_validator = OntologyValidator(
                kuzu_store=svc.kuzu_store,
                encoder=svc.encoder,
                config=cfg.ontology,
            )
            logger.info("OntologyValidator initialized", enabled=cfg.ontology.enabled)
        except Exception as e:
            errors.append(f"OntologyValidator: {e}")
            logger.warning("OntologyValidator init failed (fallback: no ontology validation)", error=str(e))

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
            llm_client = LLMClient()
            logger.info("LLMClient initialized for dream synthesis")
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
        # 【FIX】注入Kuzu引用供梦境调度器拉取数据
        svc.dream_scheduler._kuzu_store = svc.kuzu_store
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
                self.vectorizer = TfidfVectorizer(max_features=5000)
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
            "kuzu_store": svc.kuzu_store,
            "faiss_index": svc.faiss_index,
            "tfidf_index": tfidf_index,
            "encoder": svc.encoder,
            "faiss_id_map": getattr(svc, "faiss_id_map", {}),
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
    except Exception as e:
        errors.append(f"QueryRouter: {e}")
        logger.warning("QueryRouter init failed", error=str(e))

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
    logger.info("SHM v4.0 starting up", config_path=str(get_settings().kuzu.database_path))

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
            # 拟合 TF-IDF
            if idx > 0 and hasattr(svc, "tfidf_index") and svc.tfidf_index is not None and svc.kuzu_store is not None:
                rows = svc.kuzu_store.query_cypher("MATCH (e:EpisodeNode) RETURN e.content LIMIT 10000")
                texts = []
                for row in rows:
                    if isinstance(row, (list, tuple)) and len(row) > 0:
                        texts.append(str(row[0]))
                    elif isinstance(row, dict):
                        texts.append(str(row.get("content", "")))
                if texts:
                    svc.tfidf_index.fit(texts)
                    logger.info("Startup: TF-IDF fitted with %d texts", len(texts))
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
    DREAM_POLL_INTERVAL = 300.0

    async def _dream_poll_loop() -> None:
        logger.info("Dream poll loop started", interval=DREAM_POLL_INTERVAL)
        while True:
            try:
                await asyncio.sleep(DREAM_POLL_INTERVAL)
                # 定期 flush FAISS 缓冲区
                try:
                    from api.routes import flush_faiss_buffer
                    flushed = flush_faiss_buffer(svc)
                    if flushed:
                        logger.info("Periodic FAISS buffer flush: %d vectors", flushed)
                except Exception:
                    pass
                if svc.dream_scheduler is not None and hasattr(svc.dream_scheduler, "check_and_trigger"):
                    triggered = await svc.dream_scheduler.check_and_trigger()
                    if triggered:
                        logger.info("Dream triggered by poll loop")
                    # 自动 apply 梦境候选
                    if hasattr(svc, "dream_candidate_store") and svc.dream_candidate_store is not None:
                        try:
                            applied, communities = svc.dream_candidate_store.auto_apply_candidates(svc.kuzu_store)
                            if applied > 0:
                                logger.info("Auto-applied %d dreams: %d communities created", applied, communities)
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
                if svc.hyperedge_manager is not None and svc.kuzu_store is not None:
                    # 扫描长时间窗口内的同源节点
                    rows = svc.kuzu_store.query_cypher(
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

    yield
    # shutdown
    poll_task.cancel()
    hyperedge_task.cancel()
    if svc.kuzu_store:
        svc.kuzu_store.close()
    logger.info("SHM v4.0 shutting down")


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    app = FastAPI(
        title="SHM v4.0 — 自演化超图记忆系统",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
    return app
