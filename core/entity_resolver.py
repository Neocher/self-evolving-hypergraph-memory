"""
Entity Resolver — 实体消歧 + 指代消解
=======================================
在写入时自动处理：

  实体消歧: "Elon Musk" ≠ "Apple" (公司)", "apple" (水果)
            → 根据上下文判断，合并同实体别名

  指代消解: "Elon Musk founded SpaceX. He later announced Starship."
            → "He" = "Elon Musk"

用法:
    resolver = EntityResolver(graphlite_store)
    resolved = resolver.resolve("Elon Musk founded SpaceX. He later announced Starship.")
    # → "Elon Musk founded SpaceX. Elon later announced Starship."
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from datetime import datetime
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

# P1-2 读写两侧共用的实体名归一化：小写 + 去尾词（Inc/Corp/Ltd/Company 等）
# 与 ALIAS_NORMALIZE 同源但更强：多后缀词（"Apple Inc"）、大小写、空白统一，
# 供读侧 _extract_query_entities 候选与写侧 entity_id 共用同一匹配基准。
_ENTITY_NORMALIZE_SUFFIX_RE = re.compile(
    r'\b(?:inc|incorporated|corp|corporation|ltd|limited|llc|co|company|group|'
    r'holdings|technologies|technology|systems|software|gmbh|ag)\b\.?$',
    re.IGNORECASE,
)


def normalize_entity_name(name: str) -> str:
    """实体名归一化（读写两侧共用）：小写化 + 去尾词后缀 + 压缩空白。

    写侧 entity_id（"Apple Inc"）与读侧查询候选（"Apple" / "apple"）经同一
    函数归一化（→ "apple"）后做包含/前缀匹配，消除精确匹配不一致漏检。
    """
    n = re.sub(r'\s+', ' ', str(name).strip().lower())
    n = _ENTITY_NORMALIZE_SUFFIX_RE.sub('', n)
    return n.strip().strip('.,，。')

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

    def __init__(self, graphlite_store=None):
        self.graphlite_store = graphlite_store
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
    # 别名合并（关联到 GraphLite）
    # ═══════════════════════════════════════════════════════════

    def link_aliases(self, entities: List[dict]) -> int:
        """在 GraphLite 图中创建 ALIAS_OF 边

        当检测到同一实体的不同表达时（如 "Elon" == "Elon Musk"），
        在图中创建 alias 链接。
        """
        if self.graphlite_store is None or len(entities) < 2:
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
                        # GraphLite 不支持 MERGE：MATCH 边存在性检查 + INSERT（幂等）
                        if not self.graphlite_store.execute_cypher(
                            "MATCH (a:OntologyEntity {name: $n1})"
                            "-[:ALIAS_OF]->"
                            "(b:OntologyEntity {name: $n2}) RETURN a",
                            {"n1": e1["name"], "n2": e2["name"]},
                        ):
                            self.graphlite_store.execute_cypher(
                                "MATCH (a:OntologyEntity {name: $n1}), "
                                "(b:OntologyEntity {name: $n2}) "
                                "INSERT (a)-[:ALIAS_OF]->(b)",
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
    # P0-1 实体-属性-时间三维建模（属性时间版本链编排）
    # ═══════════════════════════════════════════════════════════

    def update_properties_from_triples(self, triples: List[Any]) -> int:
        """从关系三元组 attributes 构建实体属性时间版本链（决策 2）。

        属性值来源：RelationTriple.attributes（如 ACQUIRED 的 value='3B'、
        attr_year='2014'，正则确定性零 LLM）。attr_name 由关系派生命名
        `{relation.lower()}_{key}`。P1-1：attr_year 作为时间锚注入 valid_from
        （不单独成属性版本）；其余键按属性值建版本。
        无 graphlite_store / 无属性 → 0。返回新建版本数。
        """
        if self.graphlite_store is None or not triples:
            return 0
        created = 0
        for t in triples:
            attrs = dict(getattr(t, "attributes", None) or {})
            if not attrs:
                continue
            # P1-1: attr_year → valid_from（该年 1 月 1 日时间戳）；不建独立属性版本
            valid_from = None
            year_raw = attrs.pop("attr_year", None)
            if year_raw:
                try:
                    valid_from = datetime(int(str(year_raw)), 1, 1).timestamp()
                except (TypeError, ValueError):
                    logger.warning("EntityResolver: invalid attr_year %r, ignore", year_raw)
            for attr_key, attr_val in attrs.items():
                if attr_val is None or str(attr_val).strip() == "":
                    continue
                attr_name = f"{t.relation.lower()}_{attr_key}"
                created += self._update_property_version(
                    t.subject, attr_name, str(attr_val), valid_from=valid_from,
                )
        if created:
            logger.info("EntityResolver: created %d property version(s)", created)
        return created

    def _update_property_version(
        self, entity_id: str, attr_name: str, value: str,
        valid_from: Optional[float] = None,
    ) -> int:
        """单属性版本编排（GraphLite 无 MERGE：查存在→插 两段式）。

        1. 查 (entity_id, attr_name) 最新版本
        2. 值相同 → no-op（幂等，返回 0）
        3. 值不同 → store 创建新版本：旧版本打 expired_at + SUPERSEDES 边
           （store 内拆独立 execute）。时间语义（P0-1-R2 N2 乱序修复）：
           - now > last_ts：常规更新，supersede 当前最新
           - now == last_ts：同微秒 → 旧版 + 1ms bump（排序稳定）
           - now < last_ts：历史版本插入，不抬时间戳——supersedes_id 取
             "当前最新但 valid_from < now"的版本，无则无 supersedes 但
             保留 valid_from（乱序写入时间链语义不反向）
        4. 写后惰性裁剪 N=8（决策 5）
        Returns: 1=创建新版本，0=no-op
        """
        store = self.graphlite_store
        latest = store.get_latest_property_version(entity_id, attr_name)
        now = valid_from if valid_from is not None else time.time()
        supersedes_id: Optional[str] = None
        superseded_by: Optional[str] = None
        if latest is not None:
            # 【N2-P2 同值历史写入】no-op 判定纳入时序方向：
            # - 同值 + 顺序写入（now >= last_ts，值未变）→ 幂等 no-op
            # - 同值 + 乱序历史写入（now < last_ts，补历史时点确认）→ 建历史版本，
            #   否则 at_time 查询该历史时点会因 valid_from > at_ts 全部跳过
            if latest.get("value") == value and now >= float(
                latest.get("valid_from") or 0.0
            ):
                return 0
            last_ts = float(latest.get("valid_from") or 0.0)
            if now == last_ts:
                # 同微秒防重：bump 后按常规更新挂链（等价 supersede 最新版）
                now = last_ts + 0.001
                supersedes_id = latest["id"]
            elif now > last_ts:
                supersedes_id = latest["id"]
            else:
                # 历史版本插入（Codex R3 P1）：同时定位前驱 P（valid_from < now
                # 的最新版）与后继 S（valid_from > now 的最早版），传 supersedes_id=P、
                # superseded_by=S → store 双挂链（P→new→S + 双向 expired_at），
                # 任意乱序下血统链完整。
                pred, succ = self._chain_neighbors_for(entity_id, attr_name, now)
                supersedes_id = pred
                superseded_by = succ
        store.create_property_version(
            entity_id, attr_name, value,
            valid_from=now,
            supersedes_id=supersedes_id,
            superseded_by=superseded_by,
        )
        try:
            store.prune_property_versions(entity_id, attr_name)
        except Exception:
            logger.warning("property version prune failed for %s.%s (non-fatal)",
                           entity_id, attr_name)
        return 1

    def _chain_neighbors_for(
        self, entity_id: str, attr_name: str, now: float,
    ) -> tuple[Optional[str], Optional[str]]:
        """历史版本插入的链邻居：前驱 P（valid_from < now 的最新版）与后继 S
        （valid_from > now 的最早版）。

        返回 (pred_id, succ_id)；无前驱/无后继对应 None。调用方传
        supersedes_id=pred、superseded_by=succ → store 双挂链 P→new→S +
        双向 expired_at，任意乱序下血统链完整（Codex R3 P1 修复）。
        """
        pred: Optional[str] = None
        succ: Optional[str] = None
        for v in self.graphlite_store.get_property_versions(entity_id, attr_name):
            try:
                vf = float(v.get("valid_from") or 0.0)
            except (TypeError, ValueError):
                continue
            if vf < now:
                pred = v.get("id")          # 取最后一个（循环后 = valid_from 最大者）
            elif vf > now and succ is None:
                succ = v.get("id")          # 取第一个（valid_from 最小者，ASC 序）
        return pred, succ

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
