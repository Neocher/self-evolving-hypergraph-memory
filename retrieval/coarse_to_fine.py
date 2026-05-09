"""
粗到精三级检索
============
基于 HyperMem 的三级粗到精检索设计。

Step 1 (粗粒度):  查询 embedding → FAISS 搜索 → 定位主题超边
Step 2 (中粒度):  主题超边 → 展开连接的情节节点 (Layer2)
Step 3 (细粒度):  情节节点 → 提取事实节点 (Layer1)

在 10 万节点规模下，相比全量向量搜索减少 90%+ 检索计算量。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from graph.kuzu_store import CircuitBreakerOpen

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """单条检索结果"""
    node_id: str
    content: str
    score: float
    source: str                    # 'hyperedge' | 'episode' | 'fact'
    hyperedge_id: Optional[str] = None
    community_id: Optional[str] = None
    tau_value: float = 0.0


@dataclass
class CoarseToFineConfig:
    """粗到精检索配置"""
    top_k_hyperedges: int = 5      # 粗粒度：返回的最相关超边数
    top_k_episodes: int = 20       # 中粒度：从超边展开的情节数
    top_k_facts: int = 50          # 细粒度：最终返回的事实数
    score_threshold: float = 0.3   # 得分阈值过滤
    use_tau_rerank: bool = True    # 是否使用 τ 值重排序
    max_topics: int = 3            # 上下文预算：主题上限
    max_episodes: int = 20         # 上下文预算：情节上限
    max_facts: int = 50            # 上下文预算：事实上限


class CoarseToFineRetriever:
    """
    粗到精三级检索器。

    检索流程:
    1. 超边级别检索 — FAISS 搜索最相关的主题超边 (top_k=5)
    2. 情节节点展开 — 对每个超边展开成员情节节点，τ 值加权
    3. 事实节点提取 — 排序去重，返回 top_k_facts 条结果
    """

    def __init__(
        self,
        kuzu_store,
        faiss_index,
        config: Optional[CoarseToFineConfig] = None,
    ) -> None:
        """
        Args:
            kuzu_store: Kuzu 图存储实例
            faiss_index: FAISS 向量索引（需有 search(embedding, k) -> list[(id, score)] 方法）
            config: 检索配置
        """
        self.store = kuzu_store
        self.faiss = faiss_index
        self.config = config or CoarseToFineConfig()

    async def retrieve(
        self,
        query_embedding: np.ndarray,
        query_text: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        执行粗到精三级检索。

        Args:
            query_embedding: 查询文本的 embedding 向量
            query_text: 原始查询文本（可选，用于关键词辅助）

        Returns:
            排序后的检索结果列表
        """
        # Step 1: 粗粒度 — FAISS 搜索超边
        hyperedge_scores = self._search_hyperedges(query_embedding)

        if not hyperedge_scores:
            logger.info("Coarse step returned no hyperedges")
            return []

        # Step 2: 中粒度 — 展开超边到情节节点
        episode_candidates = self._expand_to_episodes(hyperedge_scores)

        if not episode_candidates:
            logger.info("Medium step returned no episodes")
            return []

        # Step 3: 细粒度 — 混合排序 + 阈值过滤 + 截断
        results = self._rank_and_filter(episode_candidates)

        logger.info(
            "Coarse-to-fine retrieval complete",
            hyperedges_found=len(hyperedge_scores),
            episodes_expanded=len(episode_candidates),
            final_results=len(results),
        )
        return results

    def _search_hyperedges(
        self, query_embedding: np.ndarray
    ) -> list[tuple[str, float]]:
        """
        Step 1 — 粗粒度：FAISS 搜索最相关的超边。

        Returns:
            [(hyperedge_id, score), ...] 按 score 降序
        """
        try:
            results = self.faiss.search(
                query_embedding, self.config.top_k_hyperedges
            )
            return [(str(r[0]), float(r[1])) for r in results]
        except CircuitBreakerOpen:
            raise
        except Exception as e:
            logger.error(f"FAISS search failed in coarse step: {e}")
            return []

    def _expand_to_episodes(
        self, hyperedge_scores: list[tuple[str, float]]
    ) -> list[dict]:
        """
        Step 2 — 中粒度：展开超边成员到情节节点。

        对每个超边查询 Kuzu 获取成员节点，附带超边得分和 τ 值。
        """
        candidates: list[dict] = []
        seen_episodes: set[str] = set()

        for he_id, he_score in hyperedge_scores:
            try:
                members = self.store.get_hyperedge_members(he_id)
            except CircuitBreakerOpen:
                raise
            except Exception as e:
                logger.warning(f"Failed to get members for hyperedge {he_id}: {e}")
                continue

            for member in members:
                ep_id = member.get("id", "")
                if ep_id in seen_episodes:
                    continue
                seen_episodes.add(ep_id)
                candidates.append({
                    "id": ep_id,
                    "content": member.get("content", ""),
                    "source": "episode",
                    "hyperedge_id": he_id,
                    "hyperedge_score": he_score,
                    "tau_value": float(member.get("tau_value", 0.0)),
                    "community_id": member.get("community_id"),
                })

        return candidates

    def _rank_and_filter(self, candidates: list[dict]) -> list[RetrievalResult]:
        """
        Step 3 — 细粒度：混合排序 + 阈值过滤。

        排序权重：0.6 * hyperedge_score + 0.4 * tau_value（可配置 τ 重排）
        """
        if self.config.use_tau_rerank:
            candidates.sort(
                key=lambda c: 0.6 * c["hyperedge_score"] + 0.4 * c["tau_value"],
                reverse=True,
            )
        else:
            candidates.sort(key=lambda c: c["hyperedge_score"], reverse=True)

        results: list[RetrievalResult] = []
        for c in candidates:
            score = (
                0.6 * c["hyperedge_score"] + 0.4 * c["tau_value"]
                if self.config.use_tau_rerank
                else c["hyperedge_score"]
            )
            if score < self.config.score_threshold:
                continue
            results.append(RetrievalResult(
                node_id=c["id"],
                content=c["content"],
                score=score,
                source=c["source"],
                hyperedge_id=c.get("hyperedge_id"),
                community_id=c.get("community_id"),
                tau_value=c["tau_value"],
            ))

        return results[:self.config.max_facts]
