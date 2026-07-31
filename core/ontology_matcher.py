"""
Ontology Matching — 跨系统本体匹配
==================================
对齐两个独立 OntologyService 的类型，返回匹配候选列表。

匹配策略（design_ontology_gaps.md v2）:
  · exact      → 名称精确匹配，score=1.0（二元，非阈值）
  · lexical    → difflib.SequenceMatcher 相似度 >= 0.75（含别名归一化）
  · structural → 邻居类集合 Jaccard >= 0.6（仅当类型数 <= max_types）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Set

from core.ontology_v2 import OntologyService

logger = logging.getLogger(__name__)

LEXICAL_THRESHOLD = 0.75
STRUCTURAL_THRESHOLD = 0.6


@dataclass
class MatchResult:
    """单个类型匹配结果"""
    source: str      # 源本体中的类型名
    target: str      # 目标本体中的类型名
    score: float     # 0~1
    method: str      # exact | lexical | structural


class OntologyMatcher:
    """本体匹配器 — 基于名称 + 结构的两阶段对齐。"""

    def __init__(self, max_types: int = 100):
        """max_types: 结构匹配的 O(N²) 车挡器（超过则跳过结构匹配）"""
        self.max_types = max_types
        self._threshold = LEXICAL_THRESHOLD

    # ─── 主入口 ─────────────────────────────────────────────

    def match(self, src: OntologyService, dst: OntologyService) -> List[MatchResult]:
        """匹配两个本体，返回排序后的 MatchResult 列表。

        优先级: exact > lexical > structural（同一类型对只保留最高优先级结果）。
        """
        # 自匹配（src is dst）：验收标准 100% exact — 只生成 exact，跳过 lexical/structural
        if src is dst:
            results = [
                MatchResult(source=t.name, target=t.name, score=1.0, method="exact")
                for t in src.list_entity_types()
            ]
            results.sort(key=lambda m: (m.source, m.target))
            return results

        src_names = [t.name for t in src.list_entity_types()]
        dst_names = [t.name for t in dst.list_entity_types()]
        results: List[MatchResult] = []

        # 1. 名称精确匹配（二元 score=1.0）
        exact_pairs = set()
        for s in src_names:
            for d in dst_names:
                if self._exact_eq(s, d):
                    exact_pairs.add((s, d))
                    results.append(MatchResult(source=s, target=d, score=1.0, method="exact"))

        # 2. 词法相似度（未精确匹配的类型对）
        for s in src_names:
            for d in dst_names:
                if (s, d) in exact_pairs:
                    continue
                ratio = SequenceMatcher(None, self._normalize(s), self._normalize(d)).ratio()
                if ratio >= self._threshold:
                    results.append(MatchResult(source=s, target=d, score=round(ratio, 4),
                                               method="lexical"))
                    exact_pairs.add((s, d))  # 已匹配的类型对不再参与结构匹配

        # 3. 结构相似度（邻居类集合 Jaccard，受 max_types 限制）
        if len(src_names) <= self.max_types and len(dst_names) <= self.max_types:
            src_neighbors = {t.name: self._neighbors(src, t.name) for t in src.list_entity_types()}
            dst_neighbors = {t.name: self._neighbors(dst, t.name) for t in dst.list_entity_types()}
            for s in src_names:
                for d in dst_names:
                    if (s, d) in exact_pairs:  # 已被 exact/lexical 覆盖
                        continue
                    jaccard = self._jaccard(src_neighbors.get(s, set()), dst_neighbors.get(d, set()))
                    if jaccard >= STRUCTURAL_THRESHOLD:
                        results.append(MatchResult(source=s, target=d,
                                                   score=round(jaccard, 4), method="structural"))
        else:
            logger.warning(
                "OntologyMatcher: structural matching skipped (%d vs %d types > max_types=%d)",
                len(src_names), len(dst_names), self.max_types)

        # 排序：score 降序，同分按方法优先级
        method_rank = {"exact": 0, "lexical": 1, "structural": 2}
        results.sort(key=lambda m: (-m.score, method_rank[m.method], m.source, m.target))
        return results

    def match_report(self, src: OntologyService, dst: OntologyService) -> Dict[str, object]:
        """返回匹配汇总报告。"""
        results = self.match(src, dst)
        by_method: Dict[str, int] = {}
        for m in results:
            by_method[m.method] = by_method.get(m.method, 0) + 1
        return {
            "total_matches": len(results),
            "by_method": by_method,
            "avg_score": round(sum(m.score for m in results) / len(results), 4) if results else 0.0,
            "thresholds": {
                "lexical": LEXICAL_THRESHOLD,
                "structural": STRUCTURAL_THRESHOLD,
            },
            "max_types": self.max_types,
            "structural_computed": (len(src.list_entity_types()) <= self.max_types
                                    and len(dst.list_entity_types()) <= self.max_types),
            "matches": [
                {"source": m.source, "target": m.target, "score": m.score, "method": m.method}
                for m in results
            ],
        }

    # ─── 内部工具 ───────────────────────────────────────────

    @staticmethod
    def _normalize(name: str) -> str:
        """别名归一化：小写 + 非字母数字折叠为空格 + 压缩空白。"""
        norm = re.sub(r"[^A-Za-z0-9]", " ", name.lower())
        return re.sub(r"\s+", " ", norm).strip()

    @classmethod
    def _exact_eq(cls, a: str, b: str) -> bool:
        """精确相等（含大小写不敏感的规范化相等）。"""
        return a == b or cls._normalize(a) == cls._normalize(b)

    def _neighbors(self, ontology: OntologyService, type_name: str) -> Set[str]:
        """类型 T 的邻居集合：参与边 + 父类型 + 子类型（归一化小写）。"""
        neighbors: Set[str] = set()
        tdef = ontology.get_entity_type(type_name)
        if tdef is None:
            return neighbors
        if tdef.parent:
            neighbors.add(self._normalize(tdef.parent))
        for edef in ontology.list_edge_types():
            if type_name in edef.source_types or type_name in edef.target_types:
                neighbors.add(self._normalize(edef.name))
        for other in ontology.list_entity_types():
            if other.parent == type_name:
                neighbors.add(self._normalize(other.name))
        return neighbors

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        """两个邻居集合的 Jaccard 相似度。"""
        if not a and not b:
            return 0.0
        union = a | b
        return len(a & b) / len(union)
