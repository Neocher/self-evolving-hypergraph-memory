"""
OntologyValidator — 轻量Kuzu本体验证层
=======================================
为 SHM v4 提供写时验证 + 读时验证，消除实体级事实矛盾导致的幻觉。

写时验证: 新事实写入前，检查Kuzu中是否存在矛盾的已有事实
读时验证: 检索结果返回前，做一致性交叉检验 + 置信度打分

零新依赖（仅用已有的 Kuzu + FAISS + sckit-learn）。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── 本休类型定义 ────────────────────────────────────────────

# 默认的本体类型体系：每个类型对应一组冲突检测规则
ONTOLOGY_TYPES: dict[str, dict[str, Any]] = {
    "person_birth": {
        "description": "人的出生日期/地点",
        "conflict_keys": ["person", "birth", "出生于", "生于", "出生"],
        "contradiction_pattern": "same_entity_diff_value",
    },
    "person_death": {
        "description": "人的死亡日期/地点",
        "conflict_keys": ["person", "death", "死于", "逝世", "去世"],
        "contradiction_pattern": "same_entity_diff_value",
    },
    "organization_founded": {
        "description": "组织/公司成立时间",
        "conflict_keys": ["org", "founded", "成立于", "成立", "创办", "创立"],
        "contradiction_pattern": "same_entity_diff_value",
    },
    "scientific_claim": {
        "description": "科学声明/事实陈述",
        "conflict_keys": ["claim", "finding", "导致", "实验", "证明"],
        "contradiction_pattern": "contradictory_claim",
    },
    "event_date": {
        "description": "事件发生时间",
        "conflict_keys": ["event", "date", "举行", "召开", "于.*年"],
        "contradiction_pattern": "same_entity_diff_value",
    },
    "location_fact": {
        "description": "地理位置事实",
        "conflict_keys": ["location", "place", "位于", "地处"],
        "contradiction_pattern": "same_entity_diff_value",
    },
    "relationship": {
        "description": "人与人/组织间的关系",
        "conflict_keys": ["relation", "between", "关系", "婚姻", "夫妻"],
        "contradiction_pattern": "same_entity_diff_value",
    },
    "generic_fact": {
        "description": "通用事实（无法归类时使用）",
        "conflict_keys": [],
        "contradiction_pattern": "embedding_contradiction",
    },
}

# 矛盾规则模板（Cypher 模式）
CONTRADICTION_RULES: dict[str, str] = {
    "same_entity_diff_value": """
        MATCH (existing:EpisodeNode)
        WHERE existing.id <> $new_id
          AND existing.ontology_type = $ontology_type
          AND existing.content CONTAINS $entity_name
          AND NOT existing.content CONTAINS $new_value
        RETURN existing.id AS conflict_id,
               existing.content AS conflict_content,
               existing.trust_score AS conflict_trust,
               existing.tau_value AS conflict_tau
        LIMIT 5
    """,
    "contradictory_claim": """
        MATCH (existing:EpisodeNode)
        WHERE existing.id <> $new_id
          AND existing.ontology_type = $ontology_type
          AND existing.content CONTAINS $entity_name
        RETURN existing.id AS conflict_id,
               existing.content AS conflict_content,
               existing.trust_score AS conflict_trust,
               existing.tau_value AS conflict_tau
        LIMIT 5
    """,
}


@dataclass
class OntologyConfig:
    """本体验证器配置"""
    enabled: bool = True
    write_validation: bool = True
    read_validation: bool = True
    confidence_threshold: float = 0.3
    contradiction_threshold: float = 0.7
    max_contradictions_per_fact: int = 5
    reject_on_contradiction: bool = False
    conflict_penalty_factor: float = 0.5


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    ontology_type: str = "generic_fact"
    entity_name: str = ""
    confidence: float = 1.0
    contradictions: List[Dict[str, Any]] = None
    conflict_count: int = 0

    def __post_init__(self):
        if self.contradictions is None:
            self.contradictions = []


@dataclass
class ReadValidationResult:
    """读取时验证结果（附加到每个检索结果上）"""
    episode_id: str
    original_score: float
    ontology_confidence: float
    adjusted_score: float
    conflict_count: int
    has_conflicts: bool
    conflict_note: str = ""


class OntologyValidator:
    """
    轻量本体验证器
    ===============
    在 SHM 的写路径和读路径中注入，不阻塞现有流程。

    写路径:
        write_validate(content) → (passed, ontology_type, entity_name, confidence)
    读路径:
        read_validate(results) → [(episode_id, adjusted_score, conflict_note), ...]
    """

    def __init__(
        self,
        kuzu_store=None,
        encoder=None,
        config: Optional[OntologyConfig] = None,
    ):
        self.kuzu = kuzu_store
        self.encoder = encoder
        self.config = config or OntologyConfig()

    # ─── 实体提取 ─────────────────────────────────────────────

    def _extract_entities(self, text: str) -> List[str]:
        """
        从文本中提取候选实体名。
        使用简单的模式匹配（人名、组织名、地名等首字母大写的词/中文专名）。
        """
        entities = []
        # 英文：提取连续大写词（人名/地名）
        en_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text, re.ASCII)
        entities.extend(en_entities)
        # 中文：提取有意义的实体（人物/组织/地点）
        cn_entities = []
        # 先尝试提取"XX公司/XX大学/XX科技"等组织名
        org_matches = re.findall(r'[\u4e00-\u9fff]{2,6}(?:公司|集团|有限|科技|大学|学院|银行|证券)', text)
        cn_entities.extend(org_matches)
        # 提取"XX出生于/毕业于/任职于"前的人名
        person_matches = re.findall(r'([\u4e00-\u9fff]{2,3}?)(?:出生于|毕业于|任职于|生于)', text)
        cn_entities.extend(person_matches)
        # 回退：提取2-4字中文实体（在结构词边界处分隔）
        if not cn_entities:
            cn_entities = []
            for run in re.findall(r'[\u4e00-\u9fff]+', text):
                # 用多字结构词做分词（避免单字分裂实体名，如被"因"分裂"爱因斯坦"）
                parts = re.split(
                    r'(?:因为|所以|但是|虽然|而且|或者|并且|然而|因此|'
                    r'如果|那么|由于|为了|在于|位于|就是|不是|而是|只是|还有|'
                    r'以及|或者|还是|直到|关于|对于|根据|按照|通过|经过)',
                    run
                )
                for p in parts:
                    if 2 <= len(p) <= 4:
                        cn_entities.append(p)
        stop_words = {'我们', '他们', '这个', '那个', '什么', '如何', '可以', '进行', '一个'}
        entities.extend(e for e in cn_entities if e not in stop_words)
        return list(set(entities))

    def _classify_ontology_type(self, text: str, entities: List[str]) -> str:
        """根据文本内容推断本体类型。"""
        text_lower = text.lower()
        for otype, info in ONTOLOGY_TYPES.items():
            keys = info["conflict_keys"]
            if any(k in text_lower for k in keys):
                return otype
        return "generic_fact"

    def _extract_values(self, text: str, ontology_type: str) -> Dict[str, str]:
        """提取用于矛盾检测的关键值。"""
        values = {}
        # 提取年份（re.ASCII保证\b不被Unicode中文干扰）
        years = re.findall(r'\b(?:19|20)\d{2}\b', text, re.ASCII)
        if years:
            values["year"] = years[0]
        # 提取日期（re.ASCII保证\b不被Unicode中文干扰）
        dates = re.findall(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', text, re.ASCII)
        if dates:
            values["date"] = dates[0]
        return values

    # ─── 写时验证 ─────────────────────────────────────────────

    def write_validate(
        self,
        content: str,
        episode_id: str = "",
        embedding: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """
        写入前验证：检查新事实是否与已有事实矛盾。

        Args:
            content: 新事实文本
            episode_id: 新事实的ID（如果已分配）
            embedding: 新事实的embedding向量（可选）

        Returns:
            ValidationResult
        """
        if not self.config.enabled or not self.config.write_validation:
            return ValidationResult(passed=True)

        result = ValidationResult(passed=True)

        # 1. 提取实体
        entities = self._extract_entities(content)
        if not entities:
            return result

        # 2. 推断本体类型
        ontology_type = self._classify_ontology_type(content, entities)
        result.ontology_type = ontology_type

        # 3. 提取关键值
        values = self._extract_values(content, ontology_type)

        # 4. 对每个实体检查矛盾
        entity_name = entities[0]
        result.entity_name = entity_name

        if self.kuzu is not None and entity_name:
            try:
                rule = CONTRADICTION_RULES.get(
                    ONTOLOGY_TYPES.get(ontology_type, {}).get(
                        "contradiction_pattern", "contradictory_claim"
                    )
                )
                if not rule:
                    return result

                # 执行Kuzu查询
                params = {
                    "new_id": episode_id,
                    "ontology_type": ontology_type,
                    "entity_name": entity_name,
                    "new_value": values.get("year", values.get("date", "")),
                }
                conflicts = self.kuzu.execute_cypher(rule, params)

                if conflicts:
                    result.conflict_count = len(conflicts)
                    result.contradictions = [
                        {
                            "conflict_id": c.get("conflict_id", ""),
                            "conflict_content": c.get("conflict_content", ""),
                            "conflict_trust": c.get("conflict_trust", 0.5),
                            "conflict_tau": c.get("conflict_tau", 0.5),
                        }
                        for c in conflicts[:self.config.max_contradictions_per_fact]
                    ]

                    # 计算置信度：基于已有事实的数量和信任度
                    avg_trust = np.mean([
                        c.get("conflict_trust", 0.5)
                        for c in result.contradictions
                    ])
                    penalty = self.config.conflict_penalty_factor * result.conflict_count
                    result.confidence = max(0.1, 1.0 - penalty)

                    # 判断是否拦截
                    if result.confidence < self.config.confidence_threshold:
                        result.passed = not self.config.reject_on_contradiction
                    else:
                        result.passed = True

            except Exception as e:
                logger.warning("Ontology write_validate failed: %s", e)
                return ValidationResult(passed=True)

        return result

    # ─── 读时验证 ─────────────────────────────────────────────

    def read_validate(
        self,
        results: List[Dict[str, Any]],
        query: str = "",
    ) -> List[ReadValidationResult]:
        """
        读取后验证：对检索结果做一致性检验 + 置信度修正。

        Args:
            results: 检索结果列表（含 id, score, tau_value, trust_score, content 等字段）
            query: 原始查询文本

        Returns:
            带有调整后分数和冲突注释的结果列表
        """
        if not self.config.enabled or not self.config.read_validation:
            return [
                ReadValidationResult(
                    episode_id=r.get("id", ""),
                    original_score=r.get("score", 0.0),
                    ontology_confidence=1.0,
                    adjusted_score=r.get("score", 0.0),
                    conflict_count=0,
                    has_conflicts=False,
                )
                for r in results
            ]

        validated = []
        for r in results:
            ep_id = r.get("id", "")
            tau = r.get("tau_value", r.get("tau", 0.5))
            trust = r.get("trust_score", 0.5)
            score = r.get("score", 0.0)
            content = r.get("content", r.get("txt_content", ""))

            # 1. 提取实体
            entities = self._extract_entities(content)

            # 2. 计算本体置信度
            ontology_conf = 1.0
            conflict_count = 0

            if entities and self.kuzu is not None:
                try:
                    # 检查是否有矛盾标记
                    otype = self._classify_ontology_type(content, entities)
                    rule = CONTRADICTION_RULES.get(
                        ONTOLOGY_TYPES.get(otype, {}).get(
                            "contradiction_pattern", "contradictory_claim"
                        )
                    )
                    if rule:
                        params = {
                            "new_id": ep_id,
                            "ontology_type": otype,
                            "entity_name": entities[0],
                            "new_value": "",
                        }
                        conflicts = self.kuzu.execute_cypher(rule, params)
                        if conflicts:
                            conflict_count = len(conflicts)
                            ontology_conf = max(
                                0.1,
                                1.0 - self.config.conflict_penalty_factor * conflict_count
                            )
                except Exception as e:
                    logger.warning("Ontology read_validate failed for %s: %s", ep_id, e)

            # 3. 综合分数 = 原始分数 × τ值 × 本体置信度
            tau_factor = min(1.0, tau / 0.5) if tau > 0 else 0.5
            adjusted = score * tau_factor * ontology_conf

            note = ""
            if conflict_count > 0:
                note = f"[本体矛盾] 关于「{entities[0]}」有 {conflict_count} 条冲突记录，此条置信度已下调"

            validated.append(ReadValidationResult(
                episode_id=ep_id,
                original_score=score,
                ontology_confidence=round(ontology_conf, 3),
                adjusted_score=round(adjusted, 4),
                conflict_count=conflict_count,
                has_conflicts=conflict_count > 0,
                conflict_note=note,
            ))

        # 按调整后分数降序排列
        validated.sort(key=lambda x: x.adjusted_score, reverse=True)
        return validated
