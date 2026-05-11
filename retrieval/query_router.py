"""
三级降级链查询路由
===============
[Harness Fix] 三级检索降级链：
  L1 — HYPERGRAPH: 超图检索（Kuzu + FAISS 联合）
  L2 — VECTOR:     纯向量检索（FAISS-only）
  L3 — KEYWORD:    关键词检索（TF-IDF）

当上游降级时自动回退到下一级别。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import numpy as np

from graph.kuzu_store import CircuitBreakerOpen

logger = logging.getLogger(__name__)


class RetrievalLevel(Enum):
    """三级检索级别"""
    HYPERGRAPH = "hypergraph"  # L1: 超图检索（Kuzu + FAISS）
    VECTOR = "vector"          # L2: 纯向量检索（FAISS-only）
    KEYWORD = "keyword"        # L3: 关键词检索（TF-IDF）


class RetrievalStrategy(str, Enum):
    TAU_FIRST = "tau_first"         # τ 值优先
    VECTOR_FIRST = "vector_first"   # 向量相似度优先
    HYBRID = "hybrid"               # 混合加权


class FAISSUnavailable(Exception):
    """FAISS 不可用异常，触发降级到 KEYWORD"""
    pass


@dataclass
class QueryRouterConfig:
    """查询路由配置"""
    tau_weight: float = 0.4         # τ 值权重（混合模式）
    vector_weight: float = 0.6      # 向量相似度权重（混合模式）
    top_k_l1: int = 5               # L1 FAISS 检索 top-K (直接查 EpisodeNode)
    top_k_vector: int = 20          # L2 向量检索 top-K
    top_k_keyword: int = 20         # L3 关键词检索 top-K


class QueryRouter:
    """
    查询路由。

    支持三级降级链：
    - 超图检索 (HYPERGRAPH): Kuzu + FAISS 联合检索
    - 向量检索 (VECTOR): FAISS-only 降级
    - 关键词检索 (KEYWORD): TF-IDF 最终兜底

    当 Kuzu 断路器跳闸时自动降级到 VECTOR，
    当 FAISS 不可用时自动降级到 KEYWORD。
    """

    def __init__(
        self,
        kuzu_store,
        faiss_index,
        tfidf_index,
        encoder=None,
        config: Optional[QueryRouterConfig] = None,
    ) -> None:
        """
        Args:
            kuzu_store: KuzuStore 实例
            faiss_index: FAISS 向量索引（需有 search(embedding, k) -> list[(id, score)] 方法）
            tfidf_index: TF-IDF 关键词索引（需有 search(query, k) -> list[(id, score)] 方法）
            encoder: 文本嵌入编码器（可选，需有 embed(text) -> np.ndarray 方法）
            config: 路由配置
        """
        self.kuzu_store = kuzu_store
        self.faiss_index = faiss_index
        self.tfidf_index = tfidf_index
        self.encoder = encoder
        self.config = config or QueryRouterConfig()
        self._time_keywords = [
            "最近", "刚刚", "刚才", "之前说的", "上一条",
            "昨天", "今天", "几分钟前", "上一次",
            "recent", "just now", "earlier", "last",
            "previous", "yesterday",
        ]

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        level: RetrievalLevel = RetrievalLevel.HYPERGRAPH,
    ) -> list[dict]:
        """
        带三级降级链 + Kuzu Cypher 最终兜底的检索入口。

        L1 — HYPERGRAPH: Kuzu + FAISS 联合检索
        L2 — VECTOR:     FAISS-only
        L3 — KEYWORD:    TF-IDF 关键词检索
        L4 — KUZU_FALLBACK: 直接 Cypher LIKE 查询（所有上游降级后的最终兜底）

        Args:
            query: 查询文本
            query_embedding: 预计算的查询向量（None 则通过 encoder 编码）
            level: 起始检索级别（默认从 L1 开始）

        Returns:
            检索结果列表 [...]

        Raises:
            RuntimeError: 四级全部降级失败
        """
        strategy = self.detect_strategy(query)
        logger.info(
            "Retrieval started", query=query[:80], level=level.value, strategy=strategy
        )
        try:
            if level == RetrievalLevel.HYPERGRAPH:
                return self._hypergraph_retrieve(query, query_embedding)
            elif level == RetrievalLevel.VECTOR:
                return self._vector_retrieve(query, query_embedding)
            else:
                return self._keyword_retrieve(query)
        except CircuitBreakerOpen:
            logger.warning("L1 hypergraph failed (circuit breaker open), falling back to L2 vector")
            return self.retrieve(query, query_embedding, RetrievalLevel.VECTOR)
        except FAISSUnavailable:
            logger.warning("L2 vector failed (FAISS unavailable), falling back to L3 keyword")
            return self.retrieve(query, query_embedding, RetrievalLevel.KEYWORD)
        except Exception as e:
            logger.exception("L1-L3 all failed, trying L4 Kuzu fallback")
            return self._kuzu_text_fallback(query, str(e))

    def _hypergraph_retrieve(
        self, query: str, query_embedding: Optional[np.ndarray] = None
    ) -> list[dict]:
        """
        L1 超图检索（Kuzu + FAISS 联合）。

        1. 编码查询为向量
        2. FAISS 搜索最相关的超边
        3. 通过 Kuzu 展开超边成员节点
        4. τ 值加权排序

        Raises:
            CircuitBreakerOpen: Kuzu 断路器跳闸
            FAISSUnavailable: FAISS 索引不可用
        """
        if query_embedding is None:
            query_embedding = self._encode_query(query)
        if query_embedding is None:
            raise FAISSUnavailable("No encoder available for query embedding")

        try:
            distances, indices = self.faiss_index.search(
                query_embedding, self.config.top_k_l1
            )
            episode_scores = list(zip(indices[0], distances[0]))
        except Exception as e:
            raise FAISSUnavailable(f"FAISS search failed: {e}") from e

        results: list[dict] = []
        for ep_id, score in episode_scores:
            if ep_id < 0:
                continue
            try:
                episode = self.kuzu_store.get_episode(str(ep_id))
            except CircuitBreakerOpen:
                raise
            except Exception:
                continue
            if episode:
                episode["score"] = float(score)
                episode["level"] = "l1_faiss"
                episode["_source"] = "faiss"
                results.append(episode)

        return self._deduplicate_and_sort(results)

    def _vector_retrieve(
        self, query: str, query_embedding: Optional[np.ndarray] = None
    ) -> list[dict]:
        """
        L2 纯向量检索（FAISS-only）。

        直接在节点向量空间中检索，不依赖 Kuzu 超边展开。

        Raises:
            FAISSUnavailable: FAISS 索引不可用
        """
        if query_embedding is None:
            query_embedding = self._encode_query(query)
        if query_embedding is None:
            raise FAISSUnavailable("No encoder available for query embedding")

        try:
            distances, indices = self.faiss_index.search(
                query_embedding, self.config.top_k_vector
            )
            node_scores = list(zip(indices[0], distances[0]))
        except Exception as e:
            raise FAISSUnavailable(f"FAISS search failed: {e}") from e

        return [
            {
                "node_id": str(node_id) if node_id >= 0 else "",
                "content": "",
                "score": float(score),
                "tau_value": 0.0,
                "level": RetrievalLevel.VECTOR.value,
            }
            for node_id, score in node_scores
            if node_id >= 0
        ]

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
            results.append({
                "node_id": str(doc_id),
                "content": str(content),
                "score": float(score),
                "tau_value": 0.0,
                "level": RetrievalLevel.KEYWORD.value,
            })
        return results

    def _kuzu_text_fallback(
        self, query: str, error_context: str = ""
    ) -> list[dict]:
        """
        L4 Kuzu Cypher 全文兜底检索。

        当 FAISS 和 TF-IDF 均不可用时，直接查询 Kuzu 数据库，
        使用 Cypher CONTAINS 做文本匹配。

        Returns:
            检索结果列表 [{"node_id", "content", "score", "level": "kuzu_fallback"}, ...]
            失败时返回空列表（不抛异常）。
        """
        if self.kuzu_store is None:
            logger.warning("L4 fallback: kuzu_store unavailable")
            return []

        logger.info("L4 fallback: querying Kuzu directly", query=query[:80])
        try:
            # 提取关键词（取前5个有意义的词）
            words = [w.strip().lower() for w in query.split() if len(w.strip()) > 1]
            search_terms = words[:5]

            if not search_terms:
                return []

            # 构建多个 CONTAINS 条件
            params = {f"w{i}": w for i, w in enumerate(search_terms)}
            conditions = " OR ".join(
                f"e.content CONTAINS $w{i}" for i in range(len(search_terms))
            )
            cypher = (
                f"MATCH (e:EpisodeNode) WHERE {conditions} "
                f"RETURN e.id AS node_id, e.content AS content, e.tau_initial AS tau_value "
                f"LIMIT {self.config.top_k_keyword}"
            )
            rows = self.kuzu_store.query_cypher(cypher, params)
            results = []
            for row in rows:
                rd = dict(row) if hasattr(row, "items") else row
                results.append({
                    "node_id": rd.get("node_id", ""),
                    "content": rd.get("content", ""),
                    "score": 0.5,
                    "tau_value": rd.get("tau_value", 0.0),
                    "level": "kuzu_fallback",
                })
            logger.info("L4 fallback results", count=len(results))
            return results
        except Exception:
            logger.exception("L4 Kuzu text fallback failed")
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
        return (
            self.config.tau_weight * tau_score
            + self.config.vector_weight * vector_score
        )

    def _encode_query(self, query: str) -> Optional[np.ndarray]:
        """编码查询文本为向量，无 encoder 时返回 None"""
        if self.encoder is None:
            return None
        try:
            return self.encoder.embed(query)
        except Exception:
            logger.exception("Query encoding failed")
            return None

    @staticmethod
    def _deduplicate_and_sort(results: list[dict]) -> list[dict]:
        """按 node_id 去重并按 score 降序排列"""
        seen: set[str] = set()
        unique: list[dict] = []
        # 已按 score 排序，保留首次出现的（最高分）记录
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            nid = r["node_id"]
            if nid and nid not in seen:
                seen.add(nid)
                unique.append(r)
        return unique
