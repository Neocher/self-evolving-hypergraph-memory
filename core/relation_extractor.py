"""
Relation Extractor — 关系抽取细化
==================================
将粗粒度的「共现边」细化为语义精确的「谓词边」。

之前: (EntityA)-[:RELATES_TO {relation: 'co_occur'}]->(EntityB)
之后: (EntityA)-[:RELATES_TO {relation: 'FOUNDED', attr_year: '2002'}]->(EntityB)
      (EntityA)-[:RELATES_TO {relation: 'CEO_OF'}]->(EntityB)
      (EntityA)-[:RELATES_TO {relation: 'ACQUIRED', attr_amount: '7.5B'}]->(EntityB)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─── 关系模式定义 ────────────────────────────────────────────

# 每条模式: (关系类型, 正则表达式, 属性提取函数)
# 正则捕获组: 1=subject, 2=object, (可选3,4,5=属性值)
# 只匹配英文全名（双大写词开头）或有上下文的中文

RelationPattern = Tuple[str, str, Optional[str]]

RELATION_PATTERNS: List[RelationPattern] = [
    # ── 英文模式 ──

    # X founded Y
    ("FOUNDED",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+founded\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b',
     None),

    # X is (the) CEO/president/chairman/head/leader of Y
    ("LEADS",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+is\s+(?:the\s+)?'
     r'(?:CEO|president|chairman|chairwoman|head|leader|founder|director|manager)'
     r'(?:\s+and\s+(?:CEO|president|chairman|CEO|CTO|CFO))?\s+of\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b',
     None),

    # X acquired / bought / purchased Y
    ("ACQUIRED",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:acquired|bought|purchased)\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b',
     "for\s+[$]?([0-9,.]+[BMK]?)"),

    # X released / launched / published Y
    ("RELEASED",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:released|launched|published|announced|introduced)\s+'
     r'([A-Z][a-zA-Z][a-zA-Z0-9]+(?:\s[A-Za-z0-9]+)*)\b',
     None),

    # X is located in / based in Y
    ("LOCATED_IN",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+is\s+'
     r'(?:located|based|headquartered)(?:\s+in)\s+'
     r'([A-Za-z][A-Za-z\s,]+?)(?:\.|,|\s+and|\s*$|\.)',
     None),

    # X works at / is employed by / joined Y
    ("WORKS_AT",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:works\s+at|is\s+employed\s+by|joined)\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b',
     None),

    # X created / authored / built / developed Y
    ("CREATED",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:created|authored|built|developed|designed|wrote)\s+'
     r'([A-Za-z0-9][A-Za-z0-9\s]{1,30}?)(?:\.|,|;|$)',
     None),

    # X invested in Y
    ("INVESTED_IN",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:invested\s+in|funded)\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b',
     "invested\s+[$]?([0-9,.]+[BMK]?)"),

    # X partnered with Y / X collaborated with Y
    ("PARTNERED_WITH",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:partnered\s+with|collaborated\s+with|teamed\s+up\s+with)\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b',
     None),

    # ── 中文模式 ──

    # X 创立了 Y / X 创建了 Y / X 成立了 Y
    ("FOUNDED",
     r'([\u4e00-\u9fff]{2,8}(?:公司|集团|科技|有限|先生|女士|博士)?)'
     r'(?:创立了|创建了|成立了|创办了|建立了)'
     r'([\u4e00-\u9fff]{2,10}(?:公司|集团|大学|银行|科技|有限|研究院)?)',
     None),

    # X 担任 Y 的 CEO / 总裁 / 董事长
    ("LEADS",
     r'([\u4e00-\u9fff]{2,6}(?:先生|女士|博士|教授|老师)?)'
     r'(?:担任|出任|就任)'
     r'([\u4e00-\u9fff]{2,10}(?:公司|集团|科技|有限|大学|研究院|银行)?)'
     r'的(?:CEO|总裁|董事长|总经理|主席|董事|校长|院长|主任)',
     None),

    # X 收购了 Y
    ("ACQUIRED",
     r'([\u4e00-\u9fff]{2,10}(?:公司|集团|科技|有限)?)'
     r'(?:收购了|并购了|收购)'
     r'([\u4e00-\u9fff]{2,10}(?:公司|集团|科技|有限)?)',
     "花费\s*([0-9,.]+[亿万元美元]*)"),

    # X 发布了 Y / X 推出了 Y（中文）
    ("RELEASED",
     r'([\u4e00-\u9fff]{2,10}(?:公司|集团|科技|有限)?)'
     r'(?:发布了|推出了|公布了|宣布了)'
     r'([\w\u4e00-\u9fff]{2,30}?)'
     r'(?:[,，。.、]|$)',
     None),

    # X 位于 Y / X 地处 Y
    ("LOCATED_IN",
     r'([\u4e00-\u9fff]{2,10}(?:公司|集团|大学|科技|有限)?)'
     r'(?:位于|地处|坐落于)'
     r'([\u4e00-\u9fff]{2,8}(?:省|市|区|县|路|街道|大厦|广场)?)',
     None),
]


# ─── 关系抽取引擎 ───────────────────────────────────────────

@dataclass
class RelationTriple:
    """抽取出的关系三元组"""
    subject: str
    relation: str          # FOUNDED / LEADS / ACQUIRED / ...
    obj: str
    confidence: float      # 0~1
    attributes: Dict[str, Any] = field(default_factory=dict)  # 额外属性


@dataclass
class DynamicRelationInfo:
    """LLM 发现的关系缓存条目（轻量，不承载 EdgeTypeDef 约束）。

    新关系只注册在此缓存中，**不写入 OntologyService**（防污染，
    人工确认后才可固化到本体）。
    """
    name: str
    description: str = ""
    discovery_count: int = 0
    last_seen: float = 0.0


class RelationExtractor:
    """关系抽取器 — 将文本中的语义关系提取为结构化三元组"""

    # LLM 返回置信度缺失/非法时的回退值
    LLM_CONFIDENCE_FALLBACK = 0.75

    def __init__(self, llm_client: Any = None):
        """llm_client 可选注入（None = 仅正则，不调用 LLM）"""
        self._llm_client = llm_client
        self._dynamic_relations: Dict[str, DynamicRelationInfo] = {}
        # 动态关系涉及的实体对（以 (relation, subj, obj) 为键，extract_hybrid 复用缓存的依据）
        self._dynamic_relation_entities: Set[Tuple[str, str, str]] = set()
        # 编译所有模式
        self._patterns: List[Tuple] = []
        for rel_type, pattern_str, attr_pattern in RELATION_PATTERNS:
            try:
                compiled = re.compile(pattern_str)
                attr_compiled = re.compile(attr_pattern) if attr_pattern else None
                self._patterns.append((rel_type, compiled, attr_compiled))
            except re.error as e:
                logger.warning("Bad regex for %s: %s", rel_type, e)

        logger.info("RelationExtractor initialized: %d patterns", len(self._patterns))

    def extract(self, text: str) -> List[RelationTriple]:
        """从文本中提取关系三元组（纯同步，仅正则，向后兼容）"""
        triples, _ = self._extract_with_spans(text)
        return triples

    def _extract_with_spans(self, text: str) -> Tuple[List[RelationTriple], List[Tuple[int, int]]]:
        """正则抽取 + 返回所有命中片段的 (start, end) 跨度。

        跨度供 extract_async 计算未命中片段（送给 LLM 的文本）。
        """
        triples: List[RelationTriple] = []
        spans: List[Tuple[int, int]] = []
        # 避免重复（同关系+同实体对）
        seen: set[tuple] = set()

        for rel_type, pattern, attr_pattern in self._patterns:
            for m in pattern.finditer(text):
                spans.append((m.start(), m.end()))
                # subject 和 obj 分别在第1、第2捕获组
                subject = m.group(1).strip()
                obj = m.group(2).strip()

                # 过滤太短的结果
                if len(subject) < 2 or len(obj) < 2:
                    continue
                if subject.lower() == obj.lower():
                    continue  # 自己关联自己

                # 提取额外属性
                attrs: Dict[str, Any] = {}
                if attr_pattern:
                    am = attr_pattern.search(text)
                    if am:
                        val = am.group(1).strip() if am.lastindex and am.lastindex >= 1 else ""
                        if val:
                            attrs["value"] = val

                triple_key = (rel_type, subject.lower(), obj.lower())
                if triple_key in seen:
                    continue
                seen.add(triple_key)

                triple = RelationTriple(
                    subject=subject,
                    relation=rel_type,
                    obj=obj,
                    confidence=0.85,  # 固定高置信度（正则匹配）
                    attributes=attrs,
                )
                triples.append(triple)

        # 去重并按关系类型排序
        triples.sort(key=lambda t: t.relation)
        if triples:
            logger.info("Extracted %d relation triples from text", len(triples))
        return triples, spans

    def extract_and_report(self, text: str) -> Dict[str, Any]:
        """提取并返回可读报告"""
        triples = self.extract(text)
        if not triples:
            return {"count": 0, "triples": []}

        return {
            "count": len(triples),
            "triples": [
                {
                    "subject": t.subject,
                    "relation": t.relation,
                    "object": t.obj,
                    "confidence": t.confidence,
                    "attributes": t.attributes,
                }
                for t in triples
            ],
        }

    # ─── LLM 增强（v5.19）────────────────────────────────────

    async def extract_async(self, text: str, ontology: Any = None) -> List[RelationTriple]:
        """异步混合抽取：先正则快速路径，未命中片段送 LLM。

        Args:
            text: 输入文本
            ontology: 可选 OntologyService，仅用于 prompt 提示（不写入）

        Returns:
            正则 + LLM 合并后的三元组列表（去重，按 relation 排序）
        """
        regex_triples, spans = self._extract_with_spans(text)
        if not self._llm_client:
            return regex_triples

        uncovered = self._uncovered_sentences(text, spans)
        if not uncovered:
            return regex_triples

        llm_triples = await self._llm_extract(uncovered, ontology)
        merged = self._merge_triples(regex_triples, llm_triples)
        if merged:
            logger.info("Extract async: %d regex + %d llm = %d triples",
                        len(regex_triples), len(llm_triples), len(merged))
        return merged

    def extract_hybrid(self, text: str) -> List[RelationTriple]:
        """同步混合抽取：正则结果 + 上次 LLM 缓存的动态关系匹配（无 LLM 调用）。

        仅基于 _dynamic_relations 缓存中 subject/object 均在文本中出现的
        关系；不发起任何网络调用，可在同步上下文安全使用。
        """
        regex_triples = self.extract(text)
        text_lower = text.lower()
        cached: List[RelationTriple] = []
        for rel, subj, obj in self._dynamic_relation_entities:
            if self._in_text(text_lower, subj) and self._in_text(text_lower, obj):
                cached.append(RelationTriple(
                    subject=subj, relation=rel, obj=obj,
                    confidence=0.85,  # 缓存命中复用（非实时 LLM 判定）
                    attributes={},
                ))
        # _merge_triples 与正则结果去重（同键保留高置信度）并按 relation 排序
        return self._merge_triples(regex_triples, cached)

    @staticmethod
    def _in_text(text_lower: str, entity: str) -> bool:
        """实体是否出现在文本中：英文按词边界匹配，中文直接包含。

        避免子串误匹配（如 "AI" 误中 "brain"）；中文无空格分词，
        \\b 不适用，直接用包含判断。
        """
        ent = entity.lower()
        if not ent:
            return False
        if re.search(r"[\u4e00-\u9fff]", ent):
            return ent in text_lower
        return re.search(r"\b" + re.escape(ent) + r"\b", text_lower) is not None

    async def _llm_extract(self, fragments: List[str], ontology: Any = None) -> List[RelationTriple]:
        """将未命中片段发送给 LLM，解析 JSON 三元组（非法 confidence 回退 0.75）。"""
        hints = self._ontology_hints(ontology)
        prompt = self._llm_prompt("\n".join(fragments), hints)
        try:
            raw = await self._llm_client.chat(
                messages=[
                    {"role": "system",
                     "content": "You extract semantic relations from text. "
                                "Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
        except Exception:
            logger.warning("RelationExtractor: LLM call failed, regex-only result",
                           exc_info=True)
            return []
        if not raw:
            return []
        return self._parse_llm_json(raw)

    @staticmethod
    def _llm_prompt(fragments: str, hints: str) -> str:
        """构建 LLM 抽取 prompt（要求输出 confidence 字段）。"""
        return (
            "Extract all semantic relations from the following text fragment(s).\n\n"
            f"{fragments}\n\n"
            "Respond with a JSON array only, each item:\n"
            '{"subject": "entity", "relation": "RELATION_NAME", '
            '"object": "entity", "confidence": 0.0-1.0}\n'
            "confidence must be a number in [0, 1]. "
            "Use uppercase snake_case relation names.\n"
            f"{hints}"
        )

    @staticmethod
    def _ontology_hints(ontology: Any) -> str:
        """从可选本体提取类型提示（仅提示用，不修改本体）。"""
        if ontology is None:
            return ""
        try:
            types = [t.name for t in ontology.list_entity_types()]
            edges = [e.name for e in ontology.list_edge_types()]
        except Exception:
            return ""
        if not types and not edges:
            return ""
        return ("Known entity types: " + ", ".join(types[:50])
                + " | Known edge types: " + ", ".join(edges[:50]))

    def _parse_llm_json(self, raw: str) -> List[RelationTriple]:
        """解析 LLM 返回的 JSON（容忍 markdown 代码围栏）。

        confidence 缺失/非法/越界 → 回退 0.75。
        新关系注册到 _dynamic_relations（不写入 OntologyService）。
        """
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("RelationExtractor: LLM returned non-JSON output")
            return []
        if not isinstance(data, list):
            data = [data]

        triples: List[RelationTriple] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject", "")).strip()
            obj = str(item.get("object", "")).strip()
            rel = str(item.get("relation", "")).strip()
            if not subject or not obj or not rel:
                continue
            confidence = self._validate_confidence(item.get("confidence"))
            triples.append(RelationTriple(
                subject=subject, relation=rel.upper(), obj=obj,
                confidence=confidence, attributes={},
            ))
            self._register_dynamic(rel, description="LLM-discovered relation")
            self._dynamic_relation_entities.add((rel.upper(), subject, obj))
        return triples

    @staticmethod
    def _validate_confidence(value: Any) -> float:
        """校验 LLM 置信度 ∈ [0,1]；缺失/非法/越界回退 0.75。"""
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return RelationExtractor.LLM_CONFIDENCE_FALLBACK
        if not 0.0 <= conf <= 1.0:
            return RelationExtractor.LLM_CONFIDENCE_FALLBACK
        return round(conf, 4)

    def _register_dynamic(self, name: str, description: str = "") -> None:
        """注册/更新动态关系缓存（仅内存，不写入 OntologyService）。"""
        key = name.upper()
        info = self._dynamic_relations.get(key)
        if info is None:
            info = DynamicRelationInfo(name=key, description=description)
            self._dynamic_relations[key] = info
        info.discovery_count += 1
        info.last_seen = time.time()

    @staticmethod
    def _uncovered_sentences(text: str, spans: List[Tuple[int, int]]) -> List[str]:
        """找出未被任何正则命中覆盖的句子片段（送 LLM 的候选）。"""
        if not spans:
            return [s for s in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if s.strip()]
        merged = RelationExtractor._merge_spans(spans)
        sentences = [s for s in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if s.strip()]
        uncovered: List[str] = []
        pos = 0
        for sent in sentences:
            start = text.find(sent, pos)
            pos = start + len(sent) if start >= 0 else pos
            covered = any(a <= start < b or a < start + len(sent) <= b
                          for a, b in merged)
            if not covered:
                uncovered.append(sent)
        return uncovered

    @staticmethod
    def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """合并重叠的 (start, end) 跨度列表。"""
        if not spans:
            return []
        ordered = sorted(spans)
        merged = [list(ordered[0])]
        for s, e in ordered[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(a, b) for a, b in merged]

    @staticmethod
    def _merge_triples(*groups: List[RelationTriple]) -> List[RelationTriple]:
        """合并多组三元组并按 relation 排序（同键保留高置信度）。"""
        by_key: Dict[Tuple[str, str, str], RelationTriple] = {}
        for group in groups:
            for t in group:
                key = (t.relation.upper(), t.subject.lower(), t.obj.lower())
                prev = by_key.get(key)
                if prev is None or t.confidence > prev.confidence:
                    by_key[key] = t
        triples = sorted(by_key.values(), key=lambda t: t.relation)
        return triples
