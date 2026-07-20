"""
Entity Resolver — 实体消歧 + 指代消解
=======================================
在写入时自动处理：

  实体消歧: "Elon Musk" ≠ "Apple" (公司)", "apple" (水果)
            → 根据上下文判断，合并同实体别名

  指代消解: "Elon Musk founded SpaceX. He later announced Starship."
            → "He" = "Elon Musk"

用法:
    resolver = EntityResolver(kuzu_store)
    resolved = resolver.resolve("Elon Musk founded SpaceX. He later announced Starship.")
    # → "Elon Musk founded SpaceX. Elon later announced Starship."
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─── 指代消解模式 ────────────────────────────────────────────

# 英文代词 → 可能的实体类型映射
EN_PRONOUN_MAP = {
    "he": "Person",
    "him": "Person",
    "his": "Person",
    "she": "Person",
    "her": "Person",
    "it": "Organization,Product,Technology,Location,Concept",
    "its": "Organization,Product,Technology,Location,Concept",
    "they": "Organization,Product,Technology",
    "them": "Organization,Product,Technology",
    "their": "Organization,Product,Technology",
    "this": "Concept,Product",
    "that": "Concept,Product",
}

# 中文代词 → 可能的实体类型映射
CN_PRONOUN_MAP = {
    "他": "Person",
    "她": "Person",
    "它": "Organization,Product,Technology,Concept",
    "他们": "Organization,Person",
    "她们": "Person",
    "它们": "Organization,Product,Technology,Concept",
    "这家公司": "Organization",
    "该公司": "Organization",
    "这个产品": "Product",
    "该产品": "Product",
    "这项技术": "Technology",
    "该技术": "Technology",
    "这个人": "Person",
    "此人": "Person",
    "这类": "Concept",
}

# 英文指代表达
EN_REFERENCE_PATTERNS = [
    (re.compile(r'\bthe\s+(?:former|latter)\b', re.IGNORECASE), "dual"),
    (re.compile(r'\bthe\s+(?:company|firm|organization)\b', re.IGNORECASE), "Organization"),
    (re.compile(r'\bthe\s+(?:CEO|founder|president|executive|leader)\b', re.IGNORECASE), "Person"),
    (re.compile(r'\bthe\s+(?:product|software|app|platform|tool)\b', re.IGNORECASE), "Product,Technology"),
    (re.compile(r'\bthe\s+(?:technology|framework|system)\b', re.IGNORECASE), "Technology"),
]

# 中文指代表达
CN_REFERENCE_PATTERNS = [
    (re.compile(r'前者'), "dual"),
    (re.compile(r'后者'), "dual"),
    (re.compile(r'该公司|该企业'), "Organization"),
    (re.compile(r'该人|此人'), "Person"),
    (re.compile(r'该产品|该商品'), "Product"),
]


# ─── 实体消歧模式 ────────────────────────────────────────────

# 同实体不同名的归一化规则
ALIAS_NORMALIZE = [
    # 英文公司后缀归一化
    (re.compile(r'\b(Inc|Incorporated|Corp|Corporation|Ltd|Limited|LLC|Co)\b\.?$', re.IGNORECASE), ""),
    # 中文公司后缀
    (re.compile(r'(?:有限公司|有限责任公司|股份有限公司)$'), ""),
    # 去掉引号
    (re.compile(r'["""'']'), ""),
]

# 上下文消歧（歧义词 → 可能的上文关键词）
DISAMBIGUATION_CONTEXT: Dict[str, List[str]] = {
    "apple": [
        "公司", "发布", "iPhone", "iPad", "Mac", "库克", "手机",
        "company", "released", "iPhone", "CEO", "Tim Cook",
    ],
    "亚马逊": [
        "公司", "电商", "AWS", "云计算", "创始人", "Jeff Bezos",
    ],
    "crm": [
        "销售", "客户", "系统", "软件", "平台",
    ],
}


# ─── 实体解析引擎 ───────────────────────────────────────────

class EntityResolver:
    """实体消歧 + 指代消解引擎"""

    def __init__(self, kuzu_store=None):
        self.kuzu_store = kuzu_store
        # 最近的实体记录（用于指代消解）
        self._recent_entities: List[dict] = []

    # ═══════════════════════════════════════════════════════════
    # 指代消解
    # ═══════════════════════════════════════════════════════════

    def resolve_content(self, content: str) -> str:
        """对内容做指代消解，返回消解后的文本

        策略:
          1. 提取内容中的实体
          2. 扫描代词，找到最近的匹配实体替换
          3. 对歧义实体做上下文消歧
        """
        # 1. 提取当前内容中的候选实体
        current_entities = self._extract_entities(content)
        if not current_entities:
            # 没有实体→无法消解
            return content

        # 2. 扫描并替换英文代词
        resolved = self._resolve_pronouns(content, current_entities)

        # 3. 扫描并替换中文代词
        resolved = self._resolve_cn_pronouns(resolved, current_entities)

        # 4. 更新最近实体记录
        self._recent_entities = current_entities[-10:]  # keep last 10

        return resolved

    def _extract_entities(self, content: str) -> List[dict]:
        """从文本中提取所有候选实体"""
        entities = []
        # 英文人名（双大写词）
        for m in re.finditer(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', content):
            entities.append({"name": m.group(1), "type": "Person", "position": m.start()})
        # 英文缩写名
        for m in re.finditer(r'\b([A-Z]{2,})\b', content):
            name = m.group(1)
            if len(name) >= 3:
                entities.append({"name": name, "type": "Organization", "position": m.start()})
        # 中文人名（带称谓）
        for m in re.finditer(r'([\u4e00-\u9fff]{2,4}(?:先生|女士|博士|教授|老师))', content):
            entities.append({"name": m.group(1), "type": "Person", "position": m.start()})
        # 中文组织名
        for m in re.finditer(r'([\u4e00-\u9fff]{2,8}(?:公司|集团|大学|科技|有限))', content):
            entities.append({"name": m.group(1), "type": "Organization", "position": m.start()})
        # 按位置排序
        entities.sort(key=lambda e: e["position"])
        return entities

    def _resolve_pronouns(self, content: str, entities: List[dict]) -> str:
        """替换英文代词"""
        result = content
        for pronoun, target_type_str in EN_PRONOUN_MAP.items():
            target_types = target_type_str.split(",")
            pattern = re.compile(r'\b' + pronoun + r'\b', re.IGNORECASE)
            for m in pattern.finditer(content):
                # 找到代词前最近的匹配类型实体
                replacement = self._find_nearest_entity(
                    m.start(), entities, target_types, content
                )
                if replacement:
                    start, end = m.start(), m.end()
                    result = result[:start] + replacement + result[end:]
        return result

    def _resolve_cn_pronouns(self, content: str, entities: List[dict]) -> str:
        """替换中文代词"""
        result = content
        for pronoun, target_type_str in CN_PRONOUN_MAP.items():
            target_types = target_type_str.split(",")
            pattern = re.compile(re.escape(pronoun))
            for m in pattern.finditer(content):
                replacement = self._find_nearest_entity(
                    m.start(), entities, target_types, content
                )
                if replacement:
                    start, end = m.start(), m.end()
                    result = result[:start] + replacement + result[end:]
        return result

    @staticmethod
    def _find_nearest_entity(
        pos: int, entities: List[dict],
        target_types: List[str], content: str
    ) -> Optional[str]:
        """找到代词前最近的匹配类型实体"""
        best = None
        for e in entities:
            if e["position"] < pos and e["type"] in target_types:
                if best is None or e["position"] > best["position"]:
                    best = e
        if best:
            return best["name"]
        return None

    # ═══════════════════════════════════════════════════════════
    # 实体消歧
    # ═══════════════════════════════════════════════════════════

    def disambiguate(self, text: str) -> Dict[str, str]:
        """对文本中的歧义实体做上下文消歧

        Returns: {原始词 → 消歧后类型}
        如: {"apple" → "Organization"} 或 {"apple" → "Concept"}
        """
        result = {}
        text_lower = text.lower()

        for ambiguous_word, keywords in DISAMBIGUATION_CONTEXT.items():
            if ambiguous_word in text_lower:
                # 检查上下文是否包含任何关键词
                matched_keywords = [kw for kw in keywords if kw in text_lower]
                if matched_keywords:
                    # 有歧义但上下文已明确
                    if any(kw in ["公司", "发布", "iPhone", "CEO"] for kw in matched_keywords):
                        result[ambiguous_word] = "Organization"
                    elif any(kw in ["水果", "吃", "一个", "apple", "day"] for kw in matched_keywords):
                        result[ambiguous_word] = "Concept"
                    else:
                        result[ambiguous_word] = "Organization"  # 默认企业实体
                else:
                    # 上下文不足→标注为需人工确认
                    result[ambiguous_word] = "ambiguous"

        return result

    # ═══════════════════════════════════════════════════════════
    # 别名合并（关联到 Kuzu）
    # ═══════════════════════════════════════════════════════════

    def link_aliases(self, entities: List[dict]) -> int:
        """在 Kuzu 图中创建 ALIAS_OF 边

        当检测到同一实体的不同表达时（如 "Elon" == "Elon Musk"），
        在图中创建 alias 链接。
        """
        if self.kuzu_store is None or len(entities) < 2:
            return 0

        count = 0
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]
                if e1["type"] != e2["type"]:
                    continue
                # 检查是否是同一实体的不同表达
                if self._is_same_entity(e1["name"], e2["name"]):
                    try:
                        self.kuzu_store.execute_cypher(
                            "MATCH (a:OntologyEntity {name: $n1}) "
                            "MATCH (b:OntologyEntity {name: $n2}) "
                            "MERGE (a)-[:ALIAS_OF]->(b)",
                            {"n1": e1["name"], "n2": e2["name"]},
                        )
                        count += 1
                    except Exception:
                        pass

        if count:
            logger.info("EntityResolver: created %d ALIAS_OF edges", count)
        return count

    @staticmethod
    def _is_same_entity(name1: str, name2: str) -> bool:
        """判断两个名称是否指向同一实体"""
        n1, n2 = name1.lower().strip(), name2.lower().strip()
        if n1 == n2:
            return False  # 完全一样不需要alias
        # 子串包含关系
        if n1 in n2 or n2 in n1:
            return True
        # 姓氏匹配（英文）
        parts1 = n1.split()
        parts2 = n2.split()
        if len(parts1) >= 2 and len(parts2) >= 2:
            if parts1[0] == parts2[0] and parts1[-1] == parts2[-1]:
                return True
        return False

    # ═══════════════════════════════════════════════════════════
    # 一站式调用
    # ═══════════════════════════════════════════════════════════

    def process(self, content: str) -> Dict[str, Any]:
        """一站式处理：指代消解 + 实体消歧 + 别名链接

        Returns:
            resolved_text: 消解后的文本
            entities: 提取的实体
            disambiguation: 消歧结果
            alias_count: 别名链接数
        """
        # 1. 指代消解
        resolved = self.resolve_content(content)

        # 2. 提取实体
        entities = self._extract_entities(resolved)

        # 3. 实体消歧
        disambiguation = self.disambiguate(resolved)

        # 4. 别名链接
        alias_count = self.link_aliases(entities)

        return {
            "original": content,
            "resolved_text": resolved if resolved != content else None,
            "entities": [
                {"name": e["name"], "type": e["type"]}
                for e in entities
            ],
            "disambiguation": disambiguation if disambiguation else None,
            "alias_count": alias_count,
        }
