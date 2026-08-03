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
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from graph.graphlite_store import CircuitBreakerOpen

from observability.logger import get_logger

logger = get_logger(__name__)


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
        self._episode_cache = episode_cache or {}  # 【Perf】共享 Services._episode_cache
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
        """构建 BM25 检索索引（防并发构建竞态 + 空语料冷却重试）。

        【M1-a】改进（对比 P1-2 的阻塞锁）：
          - 先无锁快速检查 _bm25_built/_bm25_ready：已构建/已就绪则直接返回
          - try-acquire（非阻塞）：拿不到锁说明已有构建进行中（可能被 prewarm
            超时遗留的 zombie 线程持有）→ 直接返回，不阻塞查询线程
          - 持锁后二次检查：等锁期间另一线程可能已完成构建 → 杜绝并发双重构建
        """
        if getattr(self, "_bm25_built", False) or getattr(self, "_bm25_ready", False):
            return
        lock = getattr(self, "_bm25_build_lock", None)
        if lock is None:
            self._bm25_last_attempt = time.time()
            self._build_bm25_index_core()
            return
        # 非阻塞 try-acquire：zombie 持锁时跳过本次构建，不卡住查询线程
        if not lock.acquire(blocking=False):
            logger.debug("BM25: build already in progress (or lock held), skipping")
            return
        try:
            # 持锁后二次检查：等待期间另一线程可能已完成构建
            if getattr(self, "_bm25_built", False) or getattr(self, "_bm25_ready", False):
                return
            self._bm25_last_attempt = time.time()
            self._build_bm25_index_core()
        finally:
            lock.release()

    def _build_bm25_index_core(self) -> None:
        """BM25 索引构建核心逻辑（调用方需持有 _bm25_build_lock）。

        从 GraphLite Store 拉取 EpisodeNode 内容（LIMIT 下推到数据库侧），
        使用 sklearn TfidfVectorizer 计算 IDF，并预计算文档长度用于 BM25 评分。

        【M1】_bm25_built 仅在成功路径置位：query_cypher 异常/空语料/
        TfidfVectorizer 异常时保留重试机会，避免"失败一次即永久降级"。
        """
        if self.graphlite_store is None:
            logger.warning("BM25: graphlite_store unavailable, skipping index build")
            return

        try:
            rows = self.graphlite_store.query_cypher(
                "MATCH (e:EpisodeNode) "
                "RETURN e.id AS node_id, e.content AS content, "
                "e.tau_initial AS tau_value "
                "ORDER BY e.created_at DESC "
                "LIMIT $limit",  # 【P1-2】LIMIT 下推数据库侧，避免全库拉取 OOM
                {"limit": self.config.max_bm25_corpus},
            )
        except Exception:
            logger.exception("BM25: failed to fetch corpus from GraphLite")
            return

        if not rows:
            logger.warning("BM25: empty corpus from GraphLite")
            # 【B-复审】空语料非终态：不置位 _bm25_built/_bm25_ready，
            # 由 _bm25_search 的冷却窗口（bm25_build_timeout）自然重试。
            # 修复前（M1-a）空库 prewarm 置终态 → 语料累积后进程内永不重建。
            return

        self._bm25_doc_ids.clear()
        self._bm25_doc_contents.clear()
        self._bm25_doc_tau.clear()
        corpus: list[str] = []

        for row in rows:
            if isinstance(row, dict):
                nid = row.get("node_id", "") or ""
                content = row.get("content", "") or ""
                tau = float(row.get("tau_value", 0.0) or 0.0)
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                nid = str(row[0]) if row[0] is not None else ""
                content = str(row[1]) if row[1] is not None else ""
                tau = float(row[2]) if len(row) > 2 and row[2] is not None else 0.0
            else:
                continue

            if not nid or not content.strip():
                continue
            self._bm25_doc_ids.append(nid)
            self._bm25_doc_contents.append(content)
            self._bm25_doc_tau.append(tau)
            corpus.append(content)

        if not corpus:
            logger.warning("BM25: no valid documents in corpus")
            # 【B-复审】同空语料：无有效文档非终态，不置位 → 冷却窗口后重试
            return

        # 限制语料大小，防 OOM
        max_corpus = self.config.max_bm25_corpus
        if len(corpus) > max_corpus:
            logger.warning("BM25: corpus too large (%d), truncating to %d",
                           len(corpus), max_corpus)
            corpus = corpus[:max_corpus]
            self._bm25_doc_ids = self._bm25_doc_ids[:max_corpus]
            self._bm25_doc_contents = self._bm25_doc_contents[:max_corpus]
            self._bm25_doc_tau = self._bm25_doc_tau[:max_corpus]

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

            self._bm25_vectorizer = vectorizer
            self._bm25_doc_term_matrix = tf_matrix
            self._bm25_idf = idf
            self._bm25_doc_lens = doc_lens
            self._bm25_avgdl = avgdl
            # 【M1】成功路径末尾置位：失败（上方各 return/except）不置位，保留重试机会
            self._bm25_ready = True
            self._bm25_built = True
            logger.info(
                "BM25 index built",
                num_docs=len(corpus),
                features=len(idf),
                avgdl=round(avgdl, 2),
            )
        except Exception:
            logger.exception("BM25: TfidfVectorizer failed")

    async def prewarm_bm25(self) -> None:
        """异步预热 BM25 索引（启动时调用，不阻塞事件循环）。

        - 构建放到线程池执行（asyncio.to_thread），避免阻塞事件循环
        - 受 bm25_build_timeout 超时保护；超时/失败静默降级（_bm25_ready=False），
          首个查询走延迟构建或直接返回空，不影响启动
        """
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._build_bm25_index),
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
        # 以 bm25_build_timeout 为冷却窗口，避免失败后每次检索都触发全量重建
        if not self._bm25_built:
            last_attempt = getattr(self, "_bm25_last_attempt", 0.0)
            if (last_attempt == 0.0
                    or time.time() - last_attempt >= self.config.bm25_build_timeout):
                self._build_bm25_index()
        if not self._bm25_ready or self._bm25_vectorizer is None:
            logger.warning("BM25: index not ready")
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
            rows = col.indices
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
                    "level": "bm25",
                }
            )

        return results

    def _entity_match(self, query: str, k: int = 20) -> list[dict]:
        """实体匹配检索。

        从查询中提取候选实体名（unigram + bigram），
        逐一匹配 GraphLite 中的 EpisodeNode 内容。

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
                conditions.append(f"toLower(e.content) CONTAINS ${pkey}")
            where_clause = " OR ".join(conditions)
            cypher = (
                f"MATCH (e:EpisodeNode) WHERE {where_clause} "
                f"RETURN e.id AS node_id, e.content AS content, "
                f"e.tau_initial AS tau_value "
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
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                nid = str(row[0]) if row[0] is not None else ""
                content = str(row[1]) if row[1] is not None else ""
                tau = float(row[2]) if len(row) > 2 and row[2] is not None else 0.0
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

        # 去重 + 排序（按 content[:100]）
        return self._deduplicate_and_sort(all_results)

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        level: RetrievalLevel = RetrievalLevel.HYPERGRAPH,
    ) -> list[dict]:
        """多信号检索融合入口。

        支持两种模式：
          - 降级链模式（HYPERGRAPH/VECTOR/KEYWORD）— 向后兼容
          - 融合模式（FUSION）— 三路并行融合（向量+BM25+实体匹配）

        Args:
            query: 查询文本
            query_embedding: 预计算的查询向量（None 则通过 encoder 编码）
            level: 检索级别（默认从 L1 开始，传入 FUSION 使用并行融合）

        Returns:
            检索结果列表 [...]
        """
        # 查询归一化：中文标点统一 + 中文技术术语→英文
        raw_query = query  # 保留原始查询（BM25 通道需要未归一化的中文原文）
        query = self._normalize_query(query)

        strategy = self.detect_strategy(query)
        logger.info("Retrieval started", query=query[:80], level=level.value, strategy=strategy)

        # F — 三路并行融合（向量 + BM25 + 实体匹配）
        if level == RetrievalLevel.FUSION:
            return self._fusion_retrieve(query, query_embedding, raw_query)

        # 从指定级别开始，逐级尝试（空结果自动级联）
        results: list[dict] = []
        if level == RetrievalLevel.HYPERGRAPH:
            try:
                results = self._hypergraph_retrieve(query, query_embedding)
            except CircuitBreakerOpen:
                logger.warning("L1 circuit breaker open, cascading to L2")
                r = self.retrieve(query, query_embedding, RetrievalLevel.VECTOR)
                self._tag_degraded(r, level="l1_circuit_breaker")
                return r
            except FAISSUnavailable:
                logger.warning("L1 FAISS unavailable, cascading to L2")
                r = self.retrieve(query, query_embedding, RetrievalLevel.VECTOR)
                self._tag_degraded(r, level="l1_faiss_unavailable")
                return r
            if results:
                return results
            logger.info("L1 empty, cascading to L2")
            return self.retrieve(query, query_embedding, RetrievalLevel.VECTOR)

        if level == RetrievalLevel.VECTOR:
            try:
                results = self._vector_retrieve(query, query_embedding)
            except FAISSUnavailable:
                logger.warning("L2 FAISS unavailable, cascading to L3")
                r = self.retrieve(query, query_embedding, RetrievalLevel.KEYWORD)
                self._tag_degraded(r, level="l2_faiss_unavailable")
                return r
            if results:
                return results
            logger.info("L2 empty, cascading to L3")
            r = self.retrieve(query, query_embedding, RetrievalLevel.KEYWORD)
            self._tag_degraded(r, level="l2_empty")
            return r

        # L3 keyword + L4 GraphLite fallback
        try:
            results = self._keyword_retrieve(query)
        except Exception as e:
            return self._graphlite_text_fallback(query, str(e))
        if results:
            return results
        logger.info("L3 empty, trying L4 GraphLite fallback")
        return self._graphlite_text_fallback(query, "L3 empty")

    def _fusion_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        raw_query: Optional[str] = None,
    ) -> list[dict]:
        """三路并行融合检索。

        同时运行向量、BM25、实体匹配三条通道，输出加权融合结果。

        Args:
            query: 归一化后的查询文本
            query_embedding: 预计算的查询向量（None 则通过 encoder 编码）
            raw_query: 未归一化的原始查询（语料为原始中文，BM25 通道必须用它）

        Returns:
            融合检索结果列表
        """
        cfg = self.config

        # 1. 向量通道
        vector_results: list[dict] = []
        try:
            vector_results = self._vector_retrieve(query, query_embedding)
        except FAISSUnavailable:
            logger.warning("Fusion: vector channel unavailable, skipping")
        except Exception:
            logger.exception("Fusion: vector channel failed")

        # 2. BM25 通道（用未归一化的原始查询：语料为原始中文，归一化后无交集）
        bm25_results: list[dict] = []
        try:
            bm25_query = raw_query if raw_query is not None else query
            bm25_results = self._bm25_search(bm25_query, cfg.top_k_vector)
        except Exception:
            logger.exception("Fusion: BM25 channel failed")

        # 3. 实体匹配通道
        entity_results: list[dict] = []
        try:
            entity_results = self._entity_match(query, cfg.top_k_keyword)
        except Exception:
            logger.exception("Fusion: entity match channel failed")

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
        if node_uuids and self.graphlite_store is not None and hasattr(self.graphlite_store, 'get_episodes_batch'):
            try:
                episodes_dict = {
                    ep["id"]: ep
                    for ep in self.graphlite_store.get_episodes_batch(node_uuids)
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
                episode["node_id"] = episode.pop("id", "")
                results.append(episode)

        return self._deduplicate_and_sort(results)

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

        results: list[dict] = []
        for item in keyword_results:
            if isinstance(item, tuple) and len(item) >= 2:
                doc_id, score = item[0], item[1]
                content = item[2] if len(item) > 2 else ""
            else:
                doc_id, score = str(item), 0.0
                content = ""
            results.append(
                {
                    "node_id": str(doc_id),
                    "content": str(content),
                    "score": float(score),
                    "tau_value": 0.0,
                    "level": RetrievalLevel.KEYWORD.value,
                }
            )
        return results

    def _graphlite_text_fallback(self, query: str, error_context: str = "") -> list[dict]:
        """
        L4 GraphLite GQL 全文兜底检索。

        当 FAISS 和 TF-IDF 均不可用时，直接查询 GraphLite 数据库，
        使用 Cypher CONTAINS 做文本匹配。

        Returns:
            检索结果列表 [{"node_id", "content", "score", "level": "graphlite_fallback"}, ...]
            失败时返回空列表（不抛异常）。
        """
        if self.graphlite_store is None:
            logger.warning("L4 fallback: graphlite_store unavailable")
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
                f"toLower(e.content) CONTAINS $w{i}" for i in range(len(search_terms))
            )
            cypher = (
                f"MATCH (e:EpisodeNode) WHERE {conditions} "
                f"RETURN e.id AS node_id, e.content AS content, e.tau_initial AS tau_value "
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
        """按 content[:100] 去重并按 score 降序排列"""
        seen: set[str] = set()
        unique: list[dict] = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            content = r.get("content", "")
            key = content[:100]
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
