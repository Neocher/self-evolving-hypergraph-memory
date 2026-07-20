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

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


class RelationExtractor:
    """关系抽取器 — 将文本中的语义关系提取为结构化三元组"""

    def __init__(self):
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
        """从文本中提取关系三元组"""
        triples: List[RelationTriple] = []
        # 避免重复（同关系+同实体对）
        seen: set[tuple] = set()

        for rel_type, pattern, attr_pattern in self._patterns:
            for m in pattern.finditer(text):
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
        return triples

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
