"""
概念记忆引擎（Conceptual Memory）
===============================
对应五层记忆架构中的最高层「概念记忆」。

功能：
1. 分析多个 Communities，检测跨社区的主题/框架
2. 将多个社区抽象为统一的概念节点
3. 概念随着证据增加而自动升级（从 hypothesis → theory → framework）

数据流：
CommunityNode → 概念检测 → ConceptualNode → CONCEPTUAL_FRAMEWORK → CommunityNode
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


ABSTRACTION_LEVELS = ["hypothesis", "theory", "framework", "paradigm"]


@dataclass
class ConceptualMemoryConfig:
    min_communities_for_concept: int = 2  # 至少几个社区才能抽象为概念
    min_confidence: float = 0.3
    llm_summarize: bool = True


class ConceptualMemoryEngine:
    """
    概念记忆引擎。

    分析社区报告，检测跨社区的抽象主题，
    将多社区证据整合为概念节点。
    """

    def __init__(self, config: Optional[ConceptualMemoryConfig] = None,
                 graphlite_store=None, llm_client=None):
        self.config = config or ConceptualMemoryConfig()
        self._graphlite_store = graphlite_store
        self._llm_client = llm_client

    def set_graphlite_store(self, store) -> None:
        self._graphlite_store = store

    def analyze_communities(self, communities: list[dict]) -> list[dict]:
        """分析多个社区，发现跨社区概念。
        
        Args:
            communities: 社区列表，每个含 id, name, summary, keywords, nodes
            
        Returns:
            发现的概念列表
        """
        if len(communities) < self.config.min_communities_for_concept:
            logger.info("概念记忆: 社区数 %d < %d，跳过",
                       len(communities), self.config.min_communities_for_concept)
            return []

        # 提取关键词聚合
        keyword_map: dict[str, list[dict]] = {}
        for c in communities:
            for kw in c.get("keywords", []):
                kw_lower = kw.lower().strip()
                if kw_lower not in keyword_map:
                    keyword_map[kw_lower] = []
                keyword_map[kw_lower].append(c)

        # 找到跨社区的关键词（出现在 >= 2 社区）
        concepts: list[dict] = []
        for keyword, source_comms in keyword_map.items():
            if len(source_comms) >= self.config.min_communities_for_concept:
                concept = self._build_concept(keyword, source_comms)
                if concept:
                    concepts.append(concept)

        # 持久化到 GraphLite
        for concept in concepts:
            if self._graphlite_store is not None:
                try:
                    node_id = self._graphlite_store.create_conceptual_node(concept)
                    concept["id"] = node_id
                    # 链接到源社区
                    for comm in concept.get("_source_communities", []):
                        self._graphlite_store.link_conceptual_framework(
                            node_id, comm.get("id", ""), weight=concept["confidence"]
                        )
                except Exception as e:
                    logger.warning("概念持久化失败: %s", e)
            # 清理内部字段
            concept.pop("_source_communities", None)

        logger.info("概念记忆: 从 %d 个社区发现 %d 个跨社区概念",
                   len(communities), len(concepts))
        return concepts

    def _build_concept(self, keyword: str,
                       source_communities: list[dict]) -> Optional[dict]:
        """构建一个概念节点。"""
        if not source_communities:
            return None

        # 根据出现频率决定抽象层级
        freq = len(source_communities)
        total = max(len(source_communities), 1)
        confidence = min(1.0, 0.3 + 0.2 * (freq / total))

        if freq >= 4 and confidence > 0.7:
            level = "theory"
        elif freq >= 3:
            level = "theory"
        else:
            level = "hypothesis"

        # 从社区生成概念描述
        summaries = [c.get("summary", "") for c in source_communities if c.get("summary")]
        description = f"跨社区概念: '{keyword}' 出现在 {freq} 个社区中"

        comm_ids = [c.get("id", "") for c in source_communities if c.get("id")]
        
        return {
            "id": str(uuid.uuid4()),
            "concept_name": keyword,
            "description": description,
            "abstraction_level": level,
            "confidence": round(confidence, 3),
            "created_at": time.time(),
            "source_communities": json.dumps(comm_ids, ensure_ascii=False),
            "_source_communities": source_communities,
        }

    def get_concepts(self, level: str = None) -> list[dict]:
        """查询概念记忆。"""
        if self._graphlite_store is None:
            return []
        try:
            if level:
                return self._graphlite_store.get_concepts_by_level(level)
            # 全部层级
            results = []
            for lv in ABSTRACTION_LEVELS:
                results.extend(self._graphlite_store.get_concepts_by_level(lv))
            return results
        except Exception:
            return []
