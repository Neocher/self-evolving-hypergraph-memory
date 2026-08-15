"""多信号检索融合引擎
====================
三路并行融合（向量 + BM25 + 实体匹配）：
  - VECTOR:    向量相似度检索（FAISS）
  - BM25:      关键词检索（sklearn TfidfVectorizer IDF + BM25 评分）
  - ENTITY:    实体名称匹配（GraphLite GQL CONTAINS）

融合权重：vector=0.5, bm25=0.3, entity=0.2
时序衰减：score = score * (1 + 1/(1 + exp(-τ/60)))
去重策略：按 content[:100] 保留最高分

向后兼容降级链：
  L1 — HYPERGRAPH: 超图检索（GraphLite + FAISS 联合）
  L2 — VECTOR:     纯向量检索（FAISS-only）
  L3 — KEYWORD:    关键词检索（TF-IDF）
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from core.user_profile import profile_hit, profile_values
from config.settings import get_settings
from graph.graphlite_store import CircuitBreakerOpen

from observability.logger import get_logger

logger = get_logger(__name__)

# 【User-Profile】内存常驻用户画像（app.py 启动时全量重建后经 set_user_profile
# 注入；单实例服务模块级共享，_deduplicate_and_sort 静态方法可直接读取）。
# 【P2-单租户语义】画像为模块级全局，检索加分/旁路 context 均不感知
# req.namespace——当前产品为单租户/单 Agent 部署，跨 namespace 画像泄露为
# 已知接受语义；多租户需将 _USER_PROFILE 改为 {namespace: profile} 键控。
_USER_PROFILE: dict = {}


def set_user_profile(profile: dict) -> None:
    """注入/更新用户画像（v5.39.0；启动全量重建后调用，内存常驻）。"""
    global _USER_PROFILE
    _USER_PROFILE = profile or {}


class RetrievalLevel(Enum):
    """三级检索级别 + 并行融合"""

    HYPERGRAPH = "hypergraph"  # L1: 超图检索（GraphLite + FAISS）
    VECTOR = "vector"  # L2: 纯向量检索（FAISS-only）
    KEYWORD = "keyword"  # L3: 关键词检索（TF-IDF）
    FUSION = "fusion"  # F: 三路并行融合（向量+BM25+实体匹配）


class RetrievalStrategy(str, Enum):
    TAU_FIRST = "tau_first"  # τ 值优先
    VECTOR_FIRST = "vector_first"  # 向量相似度优先
    HYBRID = "hybrid"  # 混合加权
    FUSION = "fusion"  # 三路并行融合加权


class FAISSUnavailable(Exception):
    """FAISS 不可用异常，触发降级到 KEYWORD"""

    pass


@dataclass
class QueryRouterConfig:
    """查询路由配置"""

    tau_weight: float = 0.4  # τ 值权重（混合模式）
    vector_weight: float = 0.6  # 向量相似度权重（混合模式）
    top_k_l1: int = 5  # L1 FAISS 检索 top-K (直接查 EpisodeNode)
    top_k_vector: int = 20  # L2 向量检索 top-K
    top_k_keyword: int = 20  # L3 关键词检索 top-K
    # 融合模式权重
    weight_fusion_vector: float = 0.35  # 向量检索权重（降权，短查询区分度不足）
    weight_fusion_bm25: float = 0.40  # BM25 关键词权重（提权，适合短关键字查询）
    weight_fusion_entity: float = 0.25  # 实体匹配权重（提权，专名精确匹配）
    # BM25 参数
    bm25_k1: float = 1.5  # BM25 k1 参数
    bm25_b: float = 0.75  # BM25 b 参数
    top_k_fusion: int = 30  # 融合检索总 top-K
    max_bm25_corpus: int = 50000  # BM25 索引最大语料数
    bm25_build_timeout: float = 30.0  # BM25 索引构建超时（秒），超时静默降级
    bm25_retry_cooldown: float = 30.0  # BM25 构建失败后重试冷却窗口（秒），避免失败后每次检索都全量重建
    # 图扩散配置（v5.26.0）
    graph_expansion_hop: int = 1  # 固定 1 跳（防图爆炸）
    graph_expansion_max: int = 20  # 扩散补充最大条数
    graph_expansion_alpha: float = 0.8  # 扩散新节点分数 = 归一化共现 × 向量尾分 × α


class QueryRouter:
    """
    查询路由 — 多信号检索融合引擎。

    融合模式（FUSION）— 三路并行融合（向量+BM25+实体匹配）：
      - 向量检索 (FAISS):   权重 0.5
      - BM25 关键词检索:     权重 0.3
      - 实体名称匹配 (GraphLite): 权重 0.2

    向后兼容降级链模式：
      - 超图检索 (HYPERGRAPH): GraphLite + FAISS 联合检索
      - 向量检索 (VECTOR): FAISS-only 降级
      - 关键词检索 (KEYWORD): TF-IDF 最终兜底

    当 GraphLite 断路器跳闸时自动降级到 VECTOR，
    当 FAISS 不可用时自动降级到 KEYWORD。
    """

    def __init__(
        self,
        graphlite_store,
        faiss_index,
        tfidf_index,
        encoder=None,
        config: Optional[QueryRouterConfig] = None,
        faiss_id_map: Optional[dict] = None,
        episode_cache: Optional[dict] = None,
    ) -> None:
        """
        Args:
            graphlite_store: GraphLiteStore 实例
            faiss_index: FAISS 向量索引
            tfidf_index: TF-IDF 关键词索引
            encoder: 文本嵌入编码器（可选）
            config: 路由配置
            faiss_id_map: FAISS int id → GraphLite UUID string 映射（用于 L1 超图检索反查）
        """
        self.graphlite_store = graphlite_store
        self.faiss_index = faiss_index
        self.tfidf_index = tfidf_index
        self.encoder = encoder
        self.faiss_id_map = faiss_id_map if faiss_id_map is not None else {}
        # 【M5】共享 Services._episode_cache（EpisodeCache: OrderedDict LRU + TTL）。
        # flush_faiss_buffer 是唯一写入方；传入 None（测试/降级）时退化为裸 dict，
        # 行为与改造前一致（恒空）。
        self._episode_cache = episode_cache if episode_cache is not None else {}
        # 【M4】CJK 通道跳过一次性标志（进程内首次 warning，不刷屏）
        self._cjk_warned = False
        self.config = config or QueryRouterConfig()
        self._time_keywords = [
            "最近",
            "刚刚",
            "刚才",
            "之前说的",
            "上一条",
            "昨天",
            "今天",
            "几分钟前",
            "上一次",
            "recent",
            "just now",
            "earlier",
            "last",
            "previous",
            "yesterday",
        ]
        # 中英文技术术语映射（all-MiniLM-L6-v2 是英文优化，中文术语→英文提升对齐）
        self._zh_en_tech_map: dict[str, str] = {
            "深度学习": "deep learning",
            "框架": "framework",
            "向量": "vector",
            "数据库": "database",
            "编码器": "encoder",
            "解码器": "decoder",
            "图数据库": "graph database",
            "神经网络": "neural network",
            "机器学习": "machine learning",
            "自然语言": "natural language",
            "训练": "training",
            "推理": "inference",
            "离线": "offline",
            "在线": "online",
            "加载": "load",
            "嵌入": "embedding",
            "相似度": "similarity",
            "搜索": "search",
            "检索": "retrieval",
            "分类": "classification",
            "回归": "regression",
            "聚类": "clustering",
            "社区": "community",
            "梦境": "dream",
            "记忆": "memory",
            "知识图谱": "knowledge graph",
            "超图": "hypergraph",
            "图查询": "graph query",
            "监听": "monitor",
            "健康": "health",
            "错误": "error",
            "恢复": "recovery",
            "备份": "backup",
            "缓存": "cache",
            "索引": "index",
            "重建": "rebuild",
            "部署": "deploy",
            "容器": "container",
            "服务器": "server",
            "爬虫": "crawler",
        }
        # BM25 检索索引（延迟初始化）
        self._bm25_vectorizer = None  # TfidfVectorizer 实例
        self._bm25_docs: list[str] = []  # 文档列表
        self._bm25_doc_ids: list[str] = []  # 文档 ID 列表
        self._bm25_doc_contents: list[str] = []  # 文档 content 列表
        self._bm25_doc_tau: list[float] = []  # 文档 tau 值列表
        self._bm25_doc_term_matrix = None  # (n_docs, n_features) sparse matrix
        self._bm25_idf: np.ndarray = None  # (n_features,) IDF 向量
        self._bm25_doc_lens: np.ndarray = None  # (n_docs,) 每个文档的 term 数
        self._bm25_avgdl: float = 0.0  # 平均文档长度
        self._bm25_ready: bool = False  # BM25 索引是否就绪
        self._bm25_built: bool = False  # 标记构建是否成功（仅成功路径置位）
        self._bm25_last_attempt: float = 0.0  # 上次构建尝试时间（失败后冷却重试）
        self._bm25_build_lock = threading.Lock()  # 【P1-2】BM25 构建锁（防并发构建竞态）
        # 【P4】BM25 构建进行中标志：构建期间 _bm25_search 返回旧索引/None，
        # 不阻塞事件循环等待构建完成；锁只保护最终 swap（短临界区）。
        self._bm25_building: bool = False

    def _normalize_query(self, query: str) -> str:
        """查询归一化：修复中文/英文混合输入，提升跨语言检索质量。

        1. 统一中文/英文标点
        2. 中文术语→英文（提升 all-MiniLM-L6-v2 对齐）
        3. 去除多余空格
        """
        q = query.strip()
        # 统一标点：中文标点→英文
        q = q.replace("，", " ").replace("。", " ").replace("；", " ")
        q = q.replace("：", " ").replace("？", " ").replace("！", " ")
        q = q.replace("「", " ").replace("」", " ").replace("『", " ").replace("』", " ")
        q = q.replace("（", " (").replace("）", ") ").replace("【", "").replace("】", "")
        # 统一空格
        q = re.sub(r"\s+", " ", q).strip()
        # 中文技术术语→英文（仅替换出现在文本中的术语）
        q_lower = q.lower()
        for zh, en in self._zh_en_tech_map.items():
            if zh in q_lower:
                q = q.replace(zh, en)
        # 最终清理多余空格
        return re.sub(r"\s+", " ", q).strip()

    # ──────────────────────────────
    # 三路并行融合引擎
    # ──────────────────────────────

    def _build_bm25_index(self) -> None:
        """构建 BM25 检索索引（同步入口，供测试/兼容直接调用）。

        【P4】重构（对比 M1-a 的整构建持锁）：
          - _bm25_building 标志防并发构建（构建中后续调用直接跳过，返回旧索引/None）
          - 锁只保护最终 swap（_swap_bm25_index 短临界区），GQL 拉取 + fit 在锁外
            以本地变量计算 → prewarm 超时遗留的 zombie 构建不会永久钉死锁
          - 失败/空语料不置位 _bm25_built（保留冷却窗口重试机会）
        """
        if (getattr(self, "_bm25_built", False)
                or getattr(self, "_bm25_ready", False)
                or getattr(self, "_bm25_building", False)):
            return
        self._bm25_building = True
        self._bm25_last_attempt = time.time()
        try:
            state = self._build_bm25_index_core()
            if state is not None:
                self._swap_bm25_index(state)
        finally:
            self._bm25_building = False

    async def _build_bm25_index_async(self) -> None:
        """异步构建 BM25：fit_transform 经 asyncio.to_thread 在线程执行。

        供 prewarm_bm25 使用（不阻塞事件循环）；构建期间查询线程见
        _bm25_building 标志即返回旧索引/None。
        """
        if (getattr(self, "_bm25_built", False)
                or getattr(self, "_bm25_ready", False)
                or getattr(self, "_bm25_building", False)):
            return
        self._bm25_building = True
        self._bm25_last_attempt = time.time()
        try:
            state = await asyncio.to_thread(self._build_bm25_index_core)
            if state is not None:
                self._swap_bm25_index(state)
        finally:
            self._bm25_building = False

    def _build_bm25_index_core(self):
        """BM25 索引构建核心（纯计算：不持锁、不改实例属性）。

        从 GraphLite Store 拉取 EpisodeNode 内容（LIMIT 下推到数据库侧），
        使用 sklearn TfidfVectorizer 计算 IDF，并预计算文档长度用于 BM25 评分。

        【P4】全程本地变量计算 → 返回 state 元组，由调用方在短临界区内一次性
        swap（_swap_bm25_index）。锁不再横跨 GQL 拉取 + fit。

        【M1】_bm25_built 仅由 _swap_bm25_index 在成功路径置位：query_cypher
        异常/空语料/TfidfVectorizer 异常时返回 None，保留重试机会。

        Returns:
            state = (vectorizer, doc_ids, doc_contents, doc_tau, doc_fact_track,
                     term_matrix, idf, doc_lens, avgdl)，失败返回 None。
        """
        if self.graphlite_store is None:
            logger.warning("BM25: graphlite_store unavailable, skipping index build")
            return None

        try:
            rows = self.graphlite_store.query_cypher(
                "MATCH (e:EpisodeNode) "
                "WHERE (e.archived IS NULL OR e.archived = false) "
                "RETURN e.id AS node_id, e.content AS content, "
                "e.tau_initial AS tau_value, e.fact_track AS fact_track "
                "ORDER BY e.created_at DESC "
                "LIMIT $limit",  # 【P1-2】LIMIT 下推数据库侧，避免全库拉取 OOM
                {"limit": self.config.max_bm25_corpus},
            )
        except Exception:
            logger.exception("BM25: failed to fetch corpus from GraphLite")
            return None

        if not rows:
            # 【B-复审】空语料非终态：不置位 _bm25_built/_bm25_ready，
            # 由 _bm25_search 的冷却窗口（bm25_retry_cooldown）自然重试。
            # 【日志降噪】空库是启动期/无数据期的正常状态：进程内首次 warning
            # 提示一次（_bm25_empty_warned 标志），后续只打 debug，避免刷屏。
            if not getattr(self, "_bm25_empty_warned", False):
                self._bm25_empty_warned = True
                logger.warning("BM25: empty corpus from GraphLite")
            else:
                logger.debug("BM25: empty corpus from GraphLite (will retry after cooldown)")
            return None

        doc_ids: list[str] = []
        doc_contents: list[str] = []
        doc_tau: list[float] = []
        doc_fact_track: list[str] = []
        corpus: list[str] = []

        for row in rows:
            if isinstance(row, dict):
                nid = row.get("node_id", "") or ""
                content = row.get("content", "") or ""
                tau = float(row.get("tau_value", 0.0) or 0.0)
                fact_track = row.get("fact_track", "active") or "active"
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                nid = str(row[0]) if row[0] is not None else ""
                content = str(row[1]) if row[1] is not None else ""
                tau = float(row[2]) if len(row) > 2 and row[2] is not None else 0.0
                fact_track = str(row[3]) if len(row) > 3 and row[3] is not None else "active"
            else:
                continue

            if not nid or not content.strip():
                continue
            doc_ids.append(nid)
            doc_contents.append(content)
            doc_tau.append(tau)
            doc_fact_track.append(fact_track)
            corpus.append(content)

        if not corpus:
            # 【B-复审】同空语料：无有效文档非终态，不置位 → 冷却窗口后重试
            # 【日志降噪】首次 warning 提示（_bm25_empty_warned），后续 debug
            if not getattr(self, "_bm25_empty_warned", False):
                self._bm25_empty_warned = True
                logger.warning("BM25: no valid documents in corpus")
            else:
                logger.debug("BM25: no valid documents in corpus (will retry after cooldown)")
            return None

        # 限制语料大小，防 OOM
        max_corpus = self.config.max_bm25_corpus
        if len(corpus) > max_corpus:
            logger.warning("BM25: corpus too large (%d), truncating to %d",
                           len(corpus), max_corpus)
            corpus = corpus[:max_corpus]
            doc_ids = doc_ids[:max_corpus]
            doc_contents = doc_contents[:max_corpus]
            doc_tau = doc_tau[:max_corpus]
            doc_fact_track = doc_fact_track[:max_corpus]

        try:
            # 中文检索：使用字符级 bigram/trigram/4-gram（与 TfidfEncoder 一致），
            # 避免 \b 词边界对连续汉字失效导致中文语义词无法匹配
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                lowercase=True,
                max_features=50000,
            )
            tf_matrix = vectorizer.fit_transform(corpus)  # (n_docs, n_features)
            idf = np.array(vectorizer.idf_)  # (n_features,)

            # 预计算文档长度（term 数）
            doc_lens = tf_matrix.sum(axis=1).A1  # (n_docs,)
            avgdl = float(doc_lens.mean()) if doc_lens.size > 0 else 1.0

            logger.info(
                "BM25 index built",
                num_docs=len(corpus),
                features=len(idf),
                avgdl=round(avgdl, 2),
            )
            return (vectorizer, doc_ids, doc_contents, doc_tau, doc_fact_track,
                    tf_matrix, idf, doc_lens, avgdl)
        except Exception:
            logger.exception("BM25: TfidfVectorizer failed")
            return None

    def _swap_bm25_index(self, state) -> None:
        """短临界区：持 _bm25_build_lock 一次性 swap BM25 索引状态。

        state 由 _build_bm25_index_core 本地计算返回，此处只做赋值，
        锁不横跨 GQL 拉取 + fit（prewarm 超时 zombie 不会永久钉死锁）。
        """
        (vectorizer, doc_ids, doc_contents, doc_tau, doc_fact_track,
         term_matrix, idf, doc_lens, avgdl) = state

        def _assign() -> None:
            self._bm25_vectorizer = vectorizer
            self._bm25_doc_ids = doc_ids
            self._bm25_doc_contents = doc_contents
            self._bm25_doc_tau = doc_tau
            self._bm25_doc_fact_track = doc_fact_track
            self._bm25_doc_term_matrix = term_matrix
            self._bm25_idf = idf
            self._bm25_doc_lens = doc_lens
            self._bm25_avgdl = avgdl
            # 【M1】成功路径末尾置位：失败/空语料（core 返回 None）不置位，保留重试机会
            self._bm25_ready = True
            self._bm25_built = True

        lock = getattr(self, "_bm25_build_lock", None)
        if lock is not None:
            with lock:
                _assign()
        else:
            _assign()

    async def prewarm_bm25(self) -> None:
        """异步预热 BM25 索引（启动时调用，不阻塞事件循环）。

        - 构建经 _build_bm25_index_async：fit_transform 放线程池（asyncio.to_thread）
        - 受 bm25_build_timeout 超时保护；超时/失败静默降级（_bm25_ready=False），
          首个查询走延迟构建或直接返回空，不影响启动
        - 超时取消后 _bm25_building 在 finally 复位，zombie 构建不持锁（_swap 短临界区）
        """
        try:
            await asyncio.wait_for(
                self._build_bm25_index_async(),
                timeout=self.config.bm25_build_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("BM25: prewarm timed out after %.1fs, degrading silently",
                           self.config.bm25_build_timeout)
        except Exception:
            logger.exception("BM25: prewarm failed, degrading silently")

    def _bm25_search(self, query: str, k: int = 20) -> list[dict]:
        """BM25 关键词检索。

        使用 sklearn TfidfVectorizer 的 IDF 权重实现 BM25 评分。
        BM25(q, d) = Σ IDF(qᵢ) * (TF(qᵢ, d) * (k1 + 1)) / (TF(qᵢ, d) + k1 * (1 - b + b * |d| / avgdl))

        Args:
            query: 查询文本
            k: 返回 top-K 结果

        Returns:
            [{"node_id", "content", "score", "tau_value", "level": "bm25"}, ...]
        """
        # 【M1】懒构建：失败不置位 _bm25_built → 保留重试机会；
        # 以 bm25_retry_cooldown 为冷却窗口，避免失败后每次检索都触发全量重建
        # 【P4】构建进行中（_bm25_building）→ 不阻塞等待，返回旧索引/None；
        # 旧索引可用（_bm25_ready）时直接使用，否则返回空。
        if not self._bm25_built:
            if getattr(self, "_bm25_building", False):
                if not self._bm25_ready:
                    logger.debug("BM25: index building in progress, returning empty")
                    return []
            else:
                last_attempt = getattr(self, "_bm25_last_attempt", 0.0)
                if (last_attempt == 0.0
                        or time.time() - last_attempt >= self.config.bm25_retry_cooldown):
                    self._build_bm25_index()
        if not self._bm25_ready or self._bm25_vectorizer is None:
            # 【日志降噪】索引未就绪（空库/构建中）是正常状态，debug 即可
            logger.debug("BM25: index not ready")
            return []

        cfg = self.config
        k1, b = cfg.bm25_k1, cfg.bm25_b

        query_vec = self._bm25_vectorizer.transform([query])  # (1, n_features)
        query_indices = query_vec.indices  # non-zero feature ids
        if query_indices.size == 0:
            return []

        n_docs = self._bm25_doc_term_matrix.shape[0]
        scores = np.zeros(n_docs, dtype=np.float64)

        for qf_idx in query_indices:
            idf = self._bm25_idf[qf_idx]
            col = self._bm25_doc_term_matrix[:, qf_idx]
            # 【CSR 行索引修复】列切片 .indices 是列号（恒 0），须用 nonzero()[0] 取行号；
            # 修复前多文档共享 term 时 BM25 分全部累加到 docs[0]，其余文档恒 0 被跳过
            rows = col.nonzero()[0]
            if rows.size == 0:
                continue
            vals = col.data
            numerator = vals * (k1 + 1.0)
            denominator = vals + k1 * (1.0 - b + b * self._bm25_doc_lens[rows] / self._bm25_avgdl)
            scores[rows] += idf * numerator / denominator

        # 找出 top-k
        top_k = min(k, n_docs)
        if top_k == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        # 【Core-Boost】旧索引缓存兼容：无 _bm25_doc_fact_track 时缺省 active
        fact_tracks = getattr(self, "_bm25_doc_fact_track", [])
        results: list[dict] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            results.append(
                {
                    "node_id": self._bm25_doc_ids[idx],
                    "content": self._bm25_doc_contents[idx],
                    "score": round(score, 6),
                    "tau_value": self._bm25_doc_tau[idx],
                    "fact_track": fact_tracks[idx] if idx < len(fact_tracks) else "active",
                    "level": "bm25",
                }
            )

        return results

    def _entity_match(self, query: str, k: int = 20) -> list[dict]:
        """实体匹配检索。

        从查询中提取候选实体名（unigram + bigram），
        逐一匹配 GraphLite 中的 EpisodeNode 内容。

        【P0-3】中文子串匹配不可用：GraphLite Rust lexer 不支持 UTF-8，
        _gql_value 对非 ASCII 文本做 b64 块编码——无子串保持性，
        CONTAINS 对中文不保证命中。中文检索主通道依赖向量（FAISS）/ BM25。

        Args:
            query: 查询文本
            k: 每个实体的匹配上限，总结果聚合去重

        Returns:
            [{"node_id", "content", "score", "tau_value", "level": "entity_match"}, ...]
        """
        if self.graphlite_store is None:
            logger.warning("Entity match: graphlite_store unavailable")
            return []

        tokens = [t.lower().strip() for t in query.split() if len(t.strip()) > 1]
        if not tokens:
            return []

        # 生成候选实体：bigram → unigram（bigram 优先，更精准）
        candidates: list[str] = []
        seen_phrases: set[str] = set()
        for i in range(len(tokens) - 1):
            phrase = f"{tokens[i]} {tokens[i + 1]}"
            if phrase not in seen_phrases:
                candidates.append(phrase)
                seen_phrases.add(phrase)
        for t in tokens:
            if t not in seen_phrases:
                candidates.append(t)
                seen_phrases.add(t)

        if not candidates:
            return []

        # 合并所有候选词为单个 OR CONTAINS 查询，避免 N+1 模式
        try:
            params: dict[str, str] = {}
            conditions: list[str] = []
            for i, term in enumerate(candidates):
                pkey = f"t{i}"
                params[pkey] = term.lower()
                conditions.append(f"e.content CONTAINS ${pkey}")
            where_clause = " OR ".join(conditions)
            cypher = (
                f"MATCH (e:EpisodeNode) WHERE ({where_clause}) "
                f"AND (e.archived IS NULL OR e.archived = false) "
                f"RETURN e.id AS node_id, e.content AS content, "
                f"e.tau_initial AS tau_value, e.fact_track AS fact_track "
                f"LIMIT $limit"
            )
            params["limit"] = k * 2
            rows = self.graphlite_store.query_cypher(cypher, params)
        except Exception:
            logger.exception("Entity match OR query failed")
            return []

        seen_ids: set[str] = set()
        results: list[dict] = []
        candidates_lower = set(c.lower() for c in candidates)

        for row in rows:
            if isinstance(row, dict):
                nid = row.get("node_id", "") or ""
                content = row.get("content", "") or ""
                tau = float(row.get("tau_value", 0.0) or 0.0)
                fact_track = row.get("fact_track", "active") or "active"
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                nid = str(row[0]) if row[0] is not None else ""
                content = str(row[1]) if row[1] is not None else ""
                tau = float(row[2]) if len(row) > 2 and row[2] is not None else 0.0
                fact_track = str(row[3]) if len(row) > 3 and row[3] is not None else "active"
            else:
                continue

            if not nid or nid in seen_ids:
                continue
            seen_ids.add(nid)

            # 评分：取所有匹配候选中的最高分
            content_lower = content.lower()
            best_score = 0.0
            for term_lower in candidates_lower:
                if term_lower in content_lower:
                    match_ratio = len(term_lower) / max(len(content), 1)
                    score = min(1.0, 0.3 + 0.7 * match_ratio)
                    if score > best_score:
                        best_score = score

            if best_score == 0.0:
                # fallback: 部分词匹配
                term_tokens = content_lower.split()
                match_count = sum(1 for t in candidates_lower if t in content_lower)
                best_score = 0.1 + 0.2 * (match_count / max(len(candidates_lower), 1))

            results.append({
                "node_id": nid,
                "content": content,
                "score": round(best_score, 4),
                "tau_value": tau,
                "fact_track": fact_track,
                "level": "entity_match",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    @staticmethod
    def _apply_time_decay(results: list[dict]) -> list[dict]:
        """时序衰减加权。

        对每个结果的 τ 值做时间衰减加权：
        score = score * (1 + 1 / (1 + exp(-τ / 60)))

        τ 值越高（越重要/新鲜），衰减因子的 boost 越大（1x ~ 2x）。
        """
        for r in results:
            tau = float(r.get("tau_value", 0.0))
            boost = 1.0 + 1.0 / (1.0 + np.exp(-tau / 60.0))
            r["score"] = round(r["score"] * boost, 6)
        return results

    def _fuse_results(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        entity_results: list[dict],
    ) -> list[dict]:
        """三路结果加权融合。

        1. 各路内部归一化得分到 [0, 1]
        2. 加权混合：vector * w_v + bm25 * w_b + entity * w_e
        3. 时序衰减：基于 τ 值的时间衰减
        4. 去重：按 content[:100] 保留最高分

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25 检索结果
            entity_results: 实体匹配结果

        Returns:
            融合后的结果列表（按 score 降序）
        """
        cfg = self.config
        w_v = cfg.weight_fusion_vector
        w_b = cfg.weight_fusion_bm25
        w_e = cfg.weight_fusion_entity

        def _normalize_scores(items: list[dict]) -> None:
            """就地 min-max 归一化得分到 [0, 1]"""
            if not items:
                return
            scores = [r["score"] for r in items]
            lo, hi = min(scores), max(scores)
            if hi - lo < 1e-9:
                for r in items:
                    r["score"] = 1.0
            else:
                for r in items:
                    r["score"] = (r["score"] - lo) / (hi - lo)

        _normalize_scores(vector_results)
        _normalize_scores(bm25_results)
        _normalize_scores(entity_results)

        # 加权融合
        fused: dict[str, dict] = {}
        for results, weight, source in [
            (vector_results, w_v, "vector"),
            (bm25_results, w_b, "bm25"),
            (entity_results, w_e, "entity"),
        ]:
            for r in results:
                key = r.get("node_id", "")
                if not key:
                    continue
                weighted_score = r["score"] * weight
                if key not in fused:
                    r["score"] = weighted_score
                    r["_source"] = source
                    r["level"] = f"fusion_{source}"
                    fused[key] = r
                else:
                    fused[key]["score"] += weighted_score
                    # 合并来源标记
                    existing_src = fused[key].get("_source", "")
                    if source not in existing_src:
                        fused[key]["_source"] = f"{existing_src}+{source}"
                        fused[key]["level"] = "fusion_multi"

        all_results = list(fused.values())

        # 时序衰减
        self._apply_time_decay(all_results)

        # 去重 + 排序收敛到 retrieve() _finish 统一出口（此处不再重复，
        # 避免与 _finish 的 _deduplicate_and_sort 双重 boost）
        return all_results

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        level: RetrievalLevel = RetrievalLevel.HYPERGRAPH,
        include_archived: bool = False,
    ) -> list[dict]:
        """多信号检索融合入口。

        支持两种模式：
          - 降级链模式（HYPERGRAPH/VECTOR/KEYWORD）— 向后兼容
          - 融合模式（FUSION）— 三路并行融合（向量+BM25+实体匹配）

        Args:
            query: 查询文本
            query_embedding: 预计算的查询向量（None 则通过 encoder 编码）
            level: 检索级别（默认从 L1 开始，传入 FUSION 使用并行融合）
            include_archived: 是否包含已归档节点（默认 False，排除 archived=true）

        Returns:
            检索结果列表 [...]
        """
        # 查询归一化：中文标点统一 + 中文技术术语→英文
        raw_query = query  # 保留原始查询（BM25 通道需要未归一化的中文原文）
        query = self._normalize_query(query)

        def _finish(results: list[dict]) -> list[dict]:
            # 【Core-Boost】统一出口应用 core 轨 ×1.1 boost + 去重排序——
            # 覆盖 L1/L2/L3/L4/FUSION/graph_expansion 全部路径（原仅 FUSION/L1
            # 经 _deduplicate_and_sort，降级链 L2/L3/L4 直接返回原始列表无 boost）。
            # 【P2】非 include_archived 时先过滤归档再去重：避免同一 content[:100]
            # 的 archived 高分项在去重时挤掉 active 低分项，再过滤后结果为空。
            # 【v5.41 社区扩召回】补充非替代：统一去重/排序/boost 前 append 社区成员
            # （闭包可捕获 query/query_embedding/raw_query；候选经
            # _deduplicate_and_sort 单点去重 + core/画像 boost + 钳制，不双重放大）
            results = self._community_expansion(results, query, query_embedding, raw_query)
            if not include_archived:
                results = self._filter_archived(results)
            return self._deduplicate_and_sort(results)

        strategy = self.detect_strategy(query)
        logger.info("Retrieval started", query=query[:80], level=level.value, strategy=strategy)

        # F — 三路并行融合（向量 + BM25 + 实体匹配）
        if level == RetrievalLevel.FUSION:
            return _finish(self._fusion_retrieve(query, query_embedding, raw_query))

        # 从指定级别开始，逐级尝试（空结果自动级联）
        results: list[dict] = []
        if level == RetrievalLevel.HYPERGRAPH:
            try:
                results = self._hypergraph_retrieve(query, query_embedding)
            except CircuitBreakerOpen:
                logger.warning("L1 circuit breaker open, cascading to L2")
                r = self.retrieve(query, query_embedding, RetrievalLevel.VECTOR, include_archived=include_archived)
                self._tag_degraded(r, level="l1_circuit_breaker")
                return r
            except FAISSUnavailable:
                logger.warning("L1 FAISS unavailable, cascading to L2")
                r = self.retrieve(query, query_embedding, RetrievalLevel.VECTOR, include_archived=include_archived)
                self._tag_degraded(r, level="l1_faiss_unavailable")
                return r
            if results:
                return _finish(results)
            logger.info("L1 empty, cascading to L2")
            return self.retrieve(query, query_embedding, RetrievalLevel.VECTOR, include_archived=include_archived)

        if level == RetrievalLevel.VECTOR:
            try:
                results = self._vector_retrieve(query, query_embedding)
            except FAISSUnavailable:
                logger.warning("L2 FAISS unavailable, cascading to L3")
                r = self.retrieve(query, query_embedding, RetrievalLevel.KEYWORD, include_archived=include_archived)
                self._tag_degraded(r, level="l2_faiss_unavailable")
                return r
            if results:
                return _finish(results)
            logger.info("L2 empty, cascading to L3")
            r = self.retrieve(query, query_embedding, RetrievalLevel.KEYWORD, include_archived=include_archived)
            self._tag_degraded(r, level="l2_empty")
            return r

        # L3 keyword + L4 GraphLite fallback
        try:
            results = self._keyword_retrieve(query)
        except Exception as e:
            return _finish(self._graphlite_text_fallback(query, str(e)))
        if results:
            return _finish(results)
        logger.info("L3 empty, trying L4 GraphLite fallback")
        return _finish(self._graphlite_text_fallback(query, "L3 empty"))

    def _fusion_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        raw_query: Optional[str] = None,
    ) -> list[dict]:
        """三路并行融合检索。

        同时运行向量、BM25、实体匹配三条通道，输出加权融合结果。

        【P6】vector/bm25/entity 三路经 ThreadPoolExecutor(max_workers=3) 并行
        执行（docstring 的"并行"变真）；逐通道收结果保持现有 try/except 降级，
        结果合并逻辑（_fuse_results）不变。【M4】CJK 跳过实体通道逻辑不破坏：
        CJK 查询不提交 entity 任务（省一次全表扫描），一次性 warning 标志保留。

        Args:
            query: 归一化后的查询文本
            query_embedding: 预计算的查询向量（None 则通过 encoder 编码）
            raw_query: 未归一化的原始查询（语料为原始中文，BM25 通道必须用它）

        Returns:
            融合检索结果列表
        """
        cfg = self.config

        # 【M4】CJK 查询跳过实体通道：GraphLite lexer 不支持 UTF-8，CONTAINS 对
        # 中文无子串保持性（b64 块编码），通道恒空——纯省一次全表扫描。
        # 在提交任务前判断：CJK 时根本不提交 entity 任务。
        skip_entity = any("一" <= ch <= "鿿" for ch in query)
        if skip_entity and not self._cjk_warned:
            self._cjk_warned = True
            logger.warning("Fusion: CJK query detected, skipping entity channel (CONTAINS not UTF-8 safe)")

        vector_results: list[dict] = []
        bm25_results: list[dict] = []
        entity_results: list[dict] = []

        def _run_vector():
            return self._vector_retrieve(query, query_embedding)

        def _run_bm25():
            # BM25 通道用未归一化的原始查询：语料为原始中文，归一化后无交集
            bm25_query = raw_query if raw_query is not None else query
            return self._bm25_search(bm25_query, cfg.top_k_vector)

        def _run_entity():
            return self._entity_match(query, cfg.top_k_keyword)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures: dict = {}
            futures["vector"] = pool.submit(_run_vector)
            futures["bm25"] = pool.submit(_run_bm25)
            if not skip_entity:
                futures["entity"] = pool.submit(_run_entity)

            for channel in ("vector", "bm25", "entity"):
                fut = futures.get(channel)
                if fut is None:
                    continue
                try:
                    result = fut.result()
                except FAISSUnavailable:
                    logger.warning("Fusion: %s channel unavailable, skipping", channel)
                    continue
                except Exception:
                    logger.exception("Fusion: %s channel failed", channel)
                    continue
                if channel == "vector":
                    vector_results = result
                elif channel == "bm25":
                    bm25_results = result
                else:
                    entity_results = result

        logger.info(
            "Fusion retrieval results",
            vector=len(vector_results),
            bm25=len(bm25_results),
            entity=len(entity_results),
        )

        return self._fuse_results(vector_results, bm25_results, entity_results)

    def _hypergraph_retrieve(
        self, query: str, query_embedding: Optional[np.ndarray] = None
    ) -> list[dict]:
        """
        L1 超图检索 + L2 向量检索融合降级（GraphLite + FAISS 联合）。

        1. 编码查询为向量
        2. FAISS 搜索最相关的 episode
        3. 通过 GraphLite 回查获取节点详情
        4. τ 值加权排序
        """
        if query_embedding is None:
            query_embedding = self._encode_query(query)
        if query_embedding is None:
            raise FAISSUnavailable("No encoder available for query embedding")

        try:
            distances, indices = self.faiss_index.search(query_embedding, self.config.top_k_l1)
            episode_scores = list(zip(indices[0], distances[0]))
        except Exception as e:
            raise FAISSUnavailable(f"FAISS search failed: {e}") from e

        valid_pairs = [
            (ep_id, score) for ep_id, score in episode_scores if ep_id >= 0
        ]
        uuid_map = {
            ep_id: self.faiss_id_map.get(ep_id, str(ep_id))
            for ep_id, _ in valid_pairs
        }

        # GraphLite 批量回查获取节点详情
        node_uuids = list(uuid_map.values())
        episodes_dict: dict[str, dict] = {}
        missing = [u for u in node_uuids if u not in self._episode_cache]
        if missing and self.graphlite_store is not None and hasattr(self.graphlite_store, 'get_episodes_batch'):
            try:
                episodes_dict = {
                    ep["id"]: ep
                    for ep in self.graphlite_store.get_episodes_batch(missing)
                }
            except CircuitBreakerOpen:
                raise  # 熔断跳闸：向上传播，由 retrieve() L613 级联到 L2
            except Exception:
                episodes_dict = {}

        results: list[dict] = []
        for faiss_id, score in valid_pairs:
            node_uuid = uuid_map[faiss_id]
            # episode_cache 优先
            if node_uuid in self._episode_cache:
                episode = dict(self._episode_cache[node_uuid])
                episode["id"] = node_uuid
            elif node_uuid in episodes_dict:
                episode = episodes_dict[node_uuid]
            else:
                continue
            if episode:
                episode["score"] = round(1.0 / (1.0 + float(score)), 4)
                episode["level"] = "l1_faiss"
                episode["_source"] = "faiss"
                episode["fact_track"] = episode.get("fact_track", "active")
                episode["node_id"] = episode.pop("id", "")
                results.append(episode)

        # Graph expansion (v5.26.0): 从向量种子沿超边扩散
        if results and self.graphlite_store is not None:
            try:
                seeds = [r["node_id"] for r in results[:5] if r.get("node_id")]
                existing_ids = {r["node_id"] for r in results}
                tail_score = results[-1]["score"] if results else 0.0
                expansion = self._graph_expansion(seeds, existing_ids, tail_score)
                if expansion:
                    results = results + expansion
            except Exception:
                pass  # 扩散失败静默回退，纯向量结果不受影响

        # 去重 + 排序收敛到 retrieve() _finish 统一出口（含 core boost）
        return results

    def _graph_expansion(
        self, seeds: list[str], existing_ids: set[str], tail_score: float
    ) -> list[dict]:
        """从向量种子沿超边扩散获取邻居节点。整个方法 try/except 包裹，异常返回 []。"""
        if not seeds:
            return []
        try:
            all_neighbors = self.graphlite_store.get_hypergraph_neighbors(
                seeds, self.config.graph_expansion_max
            )
        except CircuitBreakerOpen:
            return []
        except Exception as e:
            logger.error("Graph expansion failed: %s", e)
            return []

        results: list[dict] = []
        alpha = self.config.graph_expansion_alpha

        # 邻居 dict 无 archived 字段（get_hypergraph_neighbors 只回 id/content/co_occurrence）
        # 且 HYPEREDGE_MEMBER 边在归档后仍保留——回查补 archived 交 _filter_archived 过滤。
        neighbor_ids: list[str] = []
        for sid in seeds:
            for nb in all_neighbors.get(sid, []):
                nid = nb.get("id", "")
                if nid:
                    neighbor_ids.append(nid)
        archived_ids = self._lookup_archived_ids(neighbor_ids)
        # 【Core-Boost】邻居回查补 fact_track（get_hypergraph_neighbors 只回
        # id/content/co_occurrence），core 邻居在 _finish 统一出口同样获 ×1.1 boost
        fact_tracks = self._lookup_fact_track(neighbor_ids)

        for sid in seeds:
            for nb in all_neighbors.get(sid, []):
                nid = nb.get("id", "")
                content = nb.get("content", "")
                if not nid or nid in existing_ids:
                    continue
                if not content:
                    continue
                cooc = nb.get("co_occurrence", 0)
                score = round(1.0 / (1.0 + max(cooc, 0)) * tail_score * alpha, 6)
                results.append({
                    "node_id": nid,
                    "content": content,
                    "score": score,
                    "archived": nid in archived_ids,
                    "fact_track": fact_tracks.get(nid, "active"),
                    "level": "graph_expansion",
                    "_source": "graph",
                })

        # 跨种子截断：汇总所有种子的邻居后，按 score 降序取 top graph_expansion_max 条
        if len(results) > self.config.graph_expansion_max:
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:self.config.graph_expansion_max]

        return results

    def _community_expansion(
        self,
        results: list[dict],
        query: str,
        query_embedding: Optional[np.ndarray],
        raw_query: str,
    ) -> list[dict]:
        """【v5.41 社区扩召回】补充非替代：种子 → 社区（BM25-on-summary）→ 成员 append。

        链路（对齐超边扩召回）：
          1. seeds = 前 5 个检索结果的 node_id
          2. get_communities_by_seeds(seeds) → 所属社区；relevance = BM25(query, summary)
          3. relevance < threshold(0.5) 丢弃；相关社区 → get_community_members → 成员（排除种子）
          4. 扩展分 = relevance × min(种子分) × boost(0.6)（相对尾分缩放，严格低于种子）

        插入点在 retrieve() _finish 去重/排序前——候选经 _deduplicate_and_sort 统一
        去重、core/画像 boost、score 钳制（不双重放大）。GraphLite 失败/开关关闭 →
        静默返回原 results（主检索零回归，永不抛异常）。
        """
        try:
            cfg = get_settings().retrieval.community_expansion
            if not getattr(cfg, "enabled", True) or not results:
                return results
            store = getattr(self, "graphlite_store", None)
            if store is None or not hasattr(store, "get_communities_by_seeds"):
                return results
            seeds = [r.get("node_id") for r in results[:5] if r.get("node_id")]
            if not seeds:
                return results
            communities = store.get_communities_by_seeds(seeds)
            if not isinstance(communities, list) or not communities:
                return results
            min_seed_score = min(
                (float(r.get("score") or 0.0) for r in results[:5] if r.get("score")),
                default=0.0,
            )
            if min_seed_score <= 0.0:
                return results
            boost = float(getattr(cfg, "boost", 0.6))
            threshold = float(getattr(cfg, "threshold", 0.5))
            max_members = int(getattr(cfg, "max_members", 10))
            relevance = self._community_relevance(
                raw_query or query,
                [c.get("summary", "") or "" for c in communities],
            )
            existing_ids = {r.get("node_id") for r in results if r.get("node_id")}
            extra: list[dict] = []
            for comm, rel_score in zip(communities, relevance):
                if rel_score < threshold:
                    continue
                members = store.get_community_members(
                    comm.get("community_id", ""), limit=max_members
                )
                if not isinstance(members, list):
                    continue
                for m in members:
                    mid = m.get("member_id", "") or ""
                    content = m.get("content", "") or ""
                    if not mid or mid in existing_ids or not content:
                        continue
                    score = round(rel_score * min_seed_score * boost, 6)
                    if score <= 0.0:
                        continue
                    extra.append({
                        "node_id": mid,
                        "content": content,
                        "score": score,
                        "tau_value": float(m.get("tau_value") or 0.0),
                        "archived": m.get("archived", False),
                        "fact_track": m.get("fact_track") or "active",
                        "level": "community_expansion",
                        "_source": "community",
                    })
            if extra:
                logger.info(
                    "Community expansion appended",
                    candidates=len(extra),
                    boost=boost,
                )
                results = results + extra
            return results
        except Exception:
            logger.debug(
                "Community expansion degraded, returning original results", exc_info=True
            )
            return results

    def _community_relevance(self, query: str, summaries: list[str]) -> list[float]:
        """BM25-on-summary 相关度（[0,1]）：社区 summary 语料上的 BM25 + 单调归一。

        CC 修正 #1：keywords 未落库（GraphLite 只写 id/name/summary/leiden_score/
        created_at 5 字段）→ 相关度用 summary 词法（≤800 字散文含 Keywords 行）。
        与主 BM25 同特征空间（char_wb 2-4gram + IDF，k1=1.5/b=0.75）；归一化
        rel = bm25/(1+bm25)：无词法重叠 → 0（阈值闸口生效），强匹配 → 趋近 1。
        """
        if not summaries or not query:
            return [0.0] * len(summaries)
        try:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), lowercase=True,
                max_features=50000,
            )
            tf = vectorizer.fit_transform(summaries)
            idf = np.array(vectorizer.idf_)
            doc_lens = tf.sum(axis=1).A1
            avgdl = float(doc_lens.mean()) if doc_lens.size > 0 else 1.0
            q_vec = vectorizer.transform([query])
            k1, b = 1.5, 0.75
            scores = np.zeros(len(summaries), dtype=np.float64)
            for qf_idx in q_vec.indices:
                idf_w = idf[qf_idx]
                col = tf[:, qf_idx]
                # 【CSR 行索引修复】列切片 .indices 是列号（恒 0），须用 nonzero()[0] 取行号；
                # 修复前多社区时所有 BM25 分累加到 summaries[0]，其余社区恒 0
                vals = col.data
                rows = col.nonzero()[0]
                if rows.size == 0:
                    continue
                numerator = vals * (k1 + 1.0)
                denominator = vals + k1 * (1.0 - b + b * doc_lens[rows] / avgdl)
                scores[rows] += idf_w * numerator / denominator
            return (scores / (scores + 1.0)).tolist()
        except Exception:
            return [0.0] * len(summaries)

    def _vector_retrieve(
        self, query: str, query_embedding: Optional[np.ndarray] = None
    ) -> list[dict]:
        """
        L2 纯向量检索（FAISS + GraphLite 回查）。

        在节点向量空间中检索，通过 GraphLite 回查补充节点内容、tau 值等字段。
        参考 /search/vector 路由的实现方式。

        Raises:
            FAISSUnavailable: FAISS 索引不可用
        """
        if query_embedding is None:
            query_embedding = self._encode_query(query)
        if query_embedding is None:
            raise FAISSUnavailable("No encoder available for query embedding")

        try:
            distances, indices = self.faiss_index.search(query_embedding, self.config.top_k_vector)
            node_scores = list(zip(indices[0], distances[0]))
        except Exception as e:
            raise FAISSUnavailable(f"FAISS search failed: {e}") from e

        valid_pairs = [(faiss_id, score) for faiss_id, score in node_scores if faiss_id >= 0]

        # 通过 faiss_id_map 将 FAISS int ID → GraphLite UUID string
        uuid_map: dict[int, str] = {}
        for faiss_id, _ in valid_pairs:
            uuid = self.faiss_id_map.get(faiss_id)
            if uuid:
                uuid_map[faiss_id] = uuid

        # 批量回查 GraphLite 获取节点详情（content、tau_value 等）
        episodes_dict: dict[str, dict] = {}
        if uuid_map and self.graphlite_store is not None and hasattr(self.graphlite_store, 'get_episodes_batch'):
            try:
                episodes_dict = {
                    ep["id"]: ep
                    for ep in self.graphlite_store.get_episodes_batch(list(uuid_map.values()))
                }
            except CircuitBreakerOpen:
                episodes_dict = {}  # 熔断降级：静默跳过回查（内容为空，不刷异常日志）
            except Exception:
                logger.exception(
                    "_vector_retrieve: GraphLite batch lookup failed, results will have empty content"
                )

        # 构建结果集：优先使用 GraphLite 回查的数据，fallback 到空 content
        results: list[dict] = []
        for faiss_id, score in valid_pairs:
            node_uuid = uuid_map.get(faiss_id, str(faiss_id))
            episode = episodes_dict.get(node_uuid, {})
            results.append({
                "node_id": node_uuid,
                "content": episode.get("content", ""),
                "score": round(1.0 / (1.0 + float(score)), 4),
                "tau_value": episode.get("tau_initial", 0.0),
                "archived": episode.get("archived", False),
                "fact_track": episode.get("fact_track", "active"),
                "level": RetrievalLevel.VECTOR.value,
            })

        return results

    def _keyword_retrieve(self, query: str) -> list[dict]:
        """
        L3 关键词检索（TF-IDF 最终兜底）。

        不依赖向量编码或图数据库，纯文本关键词匹配。
        """
        if self.tfidf_index is None:
            raise RuntimeError("TF-IDF index not available, all retrieval levels failed")

        try:
            keyword_results = self.tfidf_index.search(query, self.config.top_k_keyword)
        except Exception as e:
            raise RuntimeError(f"TF-IDF keyword search failed: {e}") from e

        items: list[tuple[str, str, float]] = []
        node_ids: list[str] = []
        for item in keyword_results:
            if isinstance(item, tuple) and len(item) >= 2:
                doc_id, score = item[0], item[1]
                content = item[2] if len(item) > 2 else ""
            else:
                doc_id, score = str(item), 0.0
                content = ""
            nid = str(doc_id)
            items.append((nid, str(content), float(score)))
            node_ids.append(nid)
        # TF-IDF 快照语料在 app.py/system.py fit，无法在此构建期过滤——回查补 archived + fact_track
        archived_ids = self._lookup_archived_ids(node_ids)
        fact_tracks = self._lookup_fact_track(node_ids)
        return [
            {
                "node_id": doc_id,
                "content": content,
                "score": score,
                "tau_value": 0.0,
                "archived": doc_id in archived_ids,
                "fact_track": fact_tracks.get(doc_id, "active"),
                "level": RetrievalLevel.KEYWORD.value,
            }
            for doc_id, content, score in items
        ]

    def _graphlite_text_fallback(self, query: str, error_context: str = "") -> list[dict]:
        """
        L4 GraphLite GQL 全文兜底检索。

        当 FAISS 和 TF-IDF 均不可用时，直接查询 GraphLite 数据库，
        使用 Cypher CONTAINS 做文本匹配。

        【P0-3】中文子串匹配不可用：b64 块编码无子串保持性，
        CONTAINS 对中文不保证命中。中文检索主通道依赖向量/BM25，L4 仅英文有效。

        Returns:
            检索结果列表 [{"node_id", "content", "score", "level": "graphlite_fallback"}, ...]
             失败时返回空列表（不抛异常）。
        """
        if self.graphlite_store is None:
            logger.warning("L4 fallback: graphlite_store unavailable")
            return []

        # 【M4】CJK 查询直接返回 []：GraphLite CONTAINS 对中文无子串保持性
        # （b64 块编码），L4 兜底通道对中文恒空——纯省一次全表扫描。
        if any("一" <= ch <= "鿿" for ch in query):
            if not self._cjk_warned:
                self._cjk_warned = True
                logger.warning("L4 fallback: CJK query skipped (CONTAINS not UTF-8 safe)")
            return []

        logger.info("L4 fallback: querying GraphLite directly", query=query[:80])
        try:
            # 提取关键词（取前5个有意义的词）
            words = [w.strip().lower() for w in query.split() if len(w.strip()) > 1]
            search_terms = words[:5]

            if not search_terms:
                return []

            # 构建多个 CONTAINS 条件（大小写不敏感）
            params = {f"w{i}": w.lower() for i, w in enumerate(search_terms)}
            conditions = " OR ".join(
                f"e.content CONTAINS $w{i}" for i in range(len(search_terms))
            )
            cypher = (
                f"MATCH (e:EpisodeNode) WHERE ({conditions}) "
                f"AND (e.archived IS NULL OR e.archived = false) "
                f"RETURN e.id AS node_id, e.content AS content, e.tau_initial AS tau_value, "
                f"e.fact_track AS fact_track "
                "LIMIT $limit"
            )
            params["limit"] = self.config.top_k_keyword
            rows = self.graphlite_store.query_cypher(cypher, params)
            results = []
            for row in rows:
                if isinstance(row, dict):
                    results.append(
                        {
                            "node_id": row.get("node_id", ""),
                            "content": row.get("content", ""),
                            "score": 0.5,
                            "tau_value": row.get("tau_value", 0.0),
                            "fact_track": row.get("fact_track", "active"),
                            "level": "graphlite_fallback",
                        }
                    )
                elif isinstance(row, (list, tuple)) and len(row) >= 2:
                    results.append(
                        {
                            "node_id": str(row[0]),
                            "content": str(row[1]),
                            "score": 0.5,
                            "tau_value": float(row[2]) if len(row) > 2 else 0.0,
                            "fact_track": str(row[3]) if len(row) > 3 else "active",
                            "level": "graphlite_fallback",
                        }
                    )
            logger.info("L4 fallback results", count=len(results))
            return results
        except Exception:
            logger.exception("L4 GraphLite text fallback failed")
            return []

    def detect_strategy(self, query_text: str) -> RetrievalStrategy:
        """
        检测查询的时间敏感性，选择检索策略。

        Args:
            query_text: 用户查询文本

        Returns:
            推荐的检索策略
        """
        has_time_keyword = any(kw in query_text.lower() for kw in self._time_keywords)
        if has_time_keyword:
            return RetrievalStrategy.TAU_FIRST
        if len(query_text) > 50:
            return RetrievalStrategy.VECTOR_FIRST
        # 【PERF 2026-08-07】纯英文/ASCII 查询直接走 VECTOR_FIRST:
        # HYBRID 的实体匹配通道(CONTAINS 全表扫描)对英文 bigram 收益低且大库下
        # 拖慢整条检索(1946 节点实测卡死, benchmark 验证 vector 通道 recall 0.9)。
        # 中文查询保持 HYBRID(实体匹配对中文有区分度)。
        if not any('\u4e00' <= ch <= '\u9fff' for ch in query_text):
            return RetrievalStrategy.VECTOR_FIRST
        return RetrievalStrategy.HYBRID

    def hybrid_score(self, tau_score: float, vector_score: float) -> float:
        """
        计算混合得分。

        final_score = tau_weight * tau_score + vector_weight * vector_score
        """
        return self.config.tau_weight * tau_score + self.config.vector_weight * vector_score

    def fuse_results(
        self,
        results: list[dict],
        time_field: str = "created_at",
        score_field: str = "score",
    ) -> list[dict]:
        """
        融合检索结果并应用时间衰减因子。

        时间衰减公式: score = score * (1 + 1/(1 + exp(-age_hours/24)))
        效果: 24小时内新数据权重接近翻倍, 7天前衰减到0.5左右

        Args:
            results: 检索结果列表，每项需包含 score_field 和可选的 time_field
            time_field: 创建时间字段名（unix timestamp）
            score_field: 得分字段名

        Returns:
            时间衰减加权后的结果列表（按新 score 降序）
        """
        now = time.time()
        for r in results:
            try:
                created_at = r.get(time_field)
                if created_at is None:
                    continue
                age_hours = (now - float(created_at)) / 3600.0
                # 时间衰减因子: score * (1 + sigmoid(-age_hours/24))
                decay = 1.0 + 1.0 / (1.0 + math.exp(-age_hours / 24.0))
                r[score_field] = round(r.get(score_field, 0.0) * decay, 4)
            except (ValueError, TypeError, KeyError):
                continue  # 降级：异常时返回原分

        # 按新 score 降序排列
        results.sort(key=lambda x: x.get(score_field, 0.0), reverse=True)
        return results

    def _encode_query(self, query: str) -> Optional[np.ndarray]:
        """编码查询文本为 2D 向量 (1, dim)，FAISS 要求 2D 输入"""
        if self.encoder is None:
            return None
        try:
            emb = self.encoder.embed(query)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            return emb
        except Exception:
            logger.exception("Query encoding failed")
            return None

    @staticmethod
    def _tag_degraded(results: list[dict], level: str) -> None:
        for r in results:
            if "_degradation_level" not in r:
                r["_degradation_level"] = level

    @staticmethod
    def _deduplicate_and_sort(results: list[dict]) -> list[dict]:
        """按 content[:100] 去重并按 score 降序排列（core 轨 ×1.1 / 画像命中 ×1.2）。

        【P1】分数契约：boost 后钳制 score ∈ [0, 1]——EpisodicResult.score 约束
        le=1.0（api/models.py），越界会让 /memories/retrieve 构造响应抛
        ValidationError → 500（如 1.0×1.2=1.2）。钳制不破坏排序语义（单调）。
        """
        # 【Core-Boost】core 事实温和加分 + 【Profile-Boost】画像值命中加分
        profile_vals = profile_values(_USER_PROFILE)
        for r in results:
            if r.get("fact_track") == "core":
                r["score"] *= 1.1
            if profile_hit(r.get("content", ""), profile_vals):
                r["score"] *= 1.2
            # 【P1】钳制上界 1.0：core ×1.1 / 画像 ×1.2 可能突破 1.0，
            # 契约越界 → EpisodicResult ValidationError → 生产检索 500
            r["score"] = min(1.0, r["score"])
        seen: set[str] = set()
        unique: list[dict] = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            content = r.get("content", "")
            key = (content or r.get("node_id") or "")[:100]
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def search_profile(self, query: str) -> dict:
        """旁路：返回与 query 相关的画像上下文块（prepend 到 prompt，不参与主排序）。

        消费点：api/routes/search.py /memories/retrieve 注入响应
        RetrieveResponse.profile_context（SelfEvolvingRetrieval 包装时经 _qr 解包调用）。
        """
        if not _USER_PROFILE:
            return {"matched": False, "context": ""}
        hits = []
        for group, entries in _USER_PROFILE.items():
            if not isinstance(entries, dict):
                continue
            for value, info in entries.items():
                if value and str(value) in query:
                    weight = info.get("weight", 0.0) if isinstance(info, dict) else 0.0
                    hits.append({"group": group, "value": value, "weight": weight})
        if not hits:
            return {"matched": False, "context": ""}
        lines = "\n".join(f"- {h['group']}: {h['value']} (weight {h['weight']:.1f})" for h in hits)
        return {"matched": True, "context": f"【用户画像】\n{lines}", "hits": hits}

    @staticmethod
    def _filter_archived(results: list[dict]) -> list[dict]:
        """排除已归档节点（archived=true）。旧节点无 archived 字段视为未归档，不排除。"""
        return [r for r in results
                if r.get("archived") not in (True, "true", 1)]

    def _lookup_archived_ids(self, node_ids: list[str]) -> set[str]:
        """批量回查节点归档状态，返回已归档 id 集合（回查失败返回空集，不误过滤）。"""
        if not node_ids or self.graphlite_store is None:
            return set()
        try:
            if not hasattr(self.graphlite_store, "get_episodes_batch"):
                return set()
            eps = self.graphlite_store.get_episodes_batch(list(dict.fromkeys(node_ids)))
            if not isinstance(eps, (list, tuple)):
                return set()
            return {
                str(ep.get("id", ""))
                for ep in eps
                if isinstance(ep, dict) and ep.get("archived") in (True, "true", 1)
            }
        except Exception:
            return set()

    def _lookup_fact_track(self, node_ids: list[str]) -> dict[str, str]:
        """批量回查节点 fact_track，返回 {node_id: fact_track}（回查失败返回空，缺省 active）。"""
        if not node_ids or self.graphlite_store is None:
            return {}
        try:
            if not hasattr(self.graphlite_store, "get_episodes_batch"):
                return {}
            eps = self.graphlite_store.get_episodes_batch(list(dict.fromkeys(node_ids)))
            if not isinstance(eps, (list, tuple)):
                return {}
            return {
                str(ep.get("id", "")): ep.get("fact_track", "active")
                for ep in eps
                if isinstance(ep, dict)
            }
        except Exception:
            return {}
