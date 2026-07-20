"""
Entity Discovery — 自动本体发现引擎
====================================
扫描已有记忆数据，自动发现实体、类型、属性模式。

核心流程:
  scan() → 提取候选实体 → 聚类消歧 → 评分排序 → 生成类型定义提案

用法:
    POST /ontology/discover           # 扫描并返回候选
    POST /ontology/discover/apply      # 将候选批量注册到 Ontology v2
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.ontology_v2 import (
    OntologyService, EntityTypeDef, AttributeDef, EdgeTypeDef,
    EdgeAttributeDef, AttrType,
)

logger = logging.getLogger(__name__)


# ─── 候选实体模型 ────────────────────────────────────────────

@dataclass
class EntityProposal:
    """一个候选实体（聚类后的发现结果）"""
    canonical_name: str                    # 规范化名称（最常用形式）
    aliases: List[str] = field(default_factory=list)  # 别名列表
    occurrences: int = 0                   # 出现次数
    source_count: int = 0                  # 来源文本数
    confidence: float = 0.0                # 置信度 (0~1)
    inferred_type: str = "Concept"         # 推断的实体类型
    sample_texts: List[str] = field(default_factory=list)  # 样本文本


@dataclass
class TypeProposal:
    """一个候选实体类型定义"""
    name: str
    description: str = ""
    parent: Optional[str] = None
    attributes: List[AttributeDef] = field(default_factory=list)
    entity_count: int = 0                  # 属于该类型的实体数
    sample_entities: List[str] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    """一次自动发现的完整结果"""
    total_nodes_scanned: int = 0
    candidate_entities: List[EntityProposal] = field(default_factory=list)
    proposed_types: List[TypeProposal] = field(default_factory=list)
    scan_time_ms: float = 0.0


# ─── NER 模式定义 ────────────────────────────────────────────

# 英文实体模式
EN_PERSON = re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b')  # Elon Musk
EN_ORG = re.compile(r'\b[A-Z][a-zA-Z]+(?:Inc|Corp|Ltd|LLC|Co)\b', re.IGNORECASE)  # OpenAI Inc
EN_TECH = re.compile(r'\b(?:GPT-\d+|BERT|CLIP|ViT|ResNet|DALL.E|T5|LLaMA)\b')
EN_PROGRAMMING = re.compile(r'\b(?:Python|JavaScript|TypeScript|Rust|Go|Java|Swift|Kotlin)\b', re.IGNORECASE)
EN_PRODUCT = re.compile(r'\b[A-Z][a-z]+[\d.]+(?:[-\s][A-Za-z\d]+)*\b')  # iPhone 16
EN_EMAIL = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b')
EN_URL = re.compile(r'https?://[^\s,，。]+')
EN_HASHTAG = re.compile(r'#\w+')

# 中文实体模式
CN_ORG = re.compile(r'[\u4e00-\u9fff]{2,8}(?:公司|集团|大学|银行|科技|有限|研究院|中心)')
CN_PERSON_PATTERN = re.compile(r'[\u4e00-\u9fff]{2,4}(?:先生|女士|博士|教授|同学|经理|老师|同志)')
CN_LOCATION = re.compile(r'[\u4e00-\u9fff]{2,4}(?:省|市|区|县|路|街道|大厦|广场|桥)')
CN_ENTITY = re.compile(r'[\u4e00-\u9fff]{2,6}(?:科技|技术|系统|平台|引擎|框架|模型|算法)')
CN_REPORTED = re.compile(r'[\u4e00-\u9fff]{2,4}表示|[\u4e00-\u9fff]{2,4}指出|[\u4e00-\u9fff]{2,4}认为')

# 类型推断关键词映射
TYPE_HINTS: Dict[str, List[str]] = {
    "Person": [
        "founded", "CEO", "president", "founder", "author",
        "born", "出生", "担任", "毕业于", "创始人", "董事长",
    ],
    "Organization": [
        "Inc", "Corp", "Ltd", "company", "organization",
        "公司", "集团", "大学", "银行", "基金", "研究院",
    ],
    "Product": [
        "released", "launched", "version", "产品",
        "iPhone", "iPad", "Mac", "model",
    ],
    "Technology": [
        "framework", "library", "language", "tool",
        "database", "platform", "engine",
        "技术", "框架", "引擎", "语言", "工具",
    ],
    "Location": [
        "位于", "地处", "省", "市", "区", "县",
        "Mountain View", "California", "Beijing",
    ],
    "Event": [
        "conference", "summit", "meeting", "launch",
        "会议", "大会", "峰会", "发布会",
    ],
}

# 高频停用词（排除非实体的高频词）
STOP_WORDS: Set[str] = {
    "Hello", "World", "Hello World", "Test", "Testing",
    "Note", "Notes", "Summary", "Detail",
    "User", "Admin", "Guest", "System",
    "The", "This", "That", "There", "Here",
    "Page", "File", "Data", "Info", "Information",
    "Type", "Name", "Value", "Key", "Code",
    "Step", "Steps", "Item", "Items",
    "Below", "Above", "Following", "Previous",
    "Example", "Sample", "Please", "Click",
    "Question", "Answer", "Response", "Request",
    "Chapter", "Section", "Part", "Introduction",
    # 中文停用
    "我们", "他们", "它们", "你们", "这个", "那个",
    "什么", "怎么", "如何", "哪些", "这些", "那些",
    "可以", "需要", "应该", "能够", "可能",
    "进行", "通过", "关于", "按照", "根据",
    # 上下文但不构成命名实体
    "过去", "将来", "现在", "今天", "明天", "昨天",
    "首先", "其次", "最后", "然后", "之后",
}

# 单字符大写缩写（保留）
SINGLE_CHAR_UPPER = re.compile(r'\b[A-Z]\b')


# ─── 实体发现引擎 ───────────────────────────────────────────

class EntityDiscoveryEngine:
    """实体自动发现引擎"""

    def __init__(self, ontology: Optional[OntologyService] = None):
        self.ontology = ontology

    # ═══════════════════════════════════════════════════════════
    # 主扫描方法
    # ═══════════════════════════════════════════════════════════

    def scan(self, contents: List[str],
             min_occurrences: int = 2,
             max_candidates: int = 50) -> DiscoveryResult:
        """扫描文本列表，发现候选实体和类型"""
        start = time.time()
        result = DiscoveryResult(total_nodes_scanned=len(contents))

        # 1. 提取候选实体
        raw_entities = self._extract_all(contents)
        logger.info("Extracted %d raw entity mentions", len(raw_entities))

        # 2. 聚类消歧（合并相同实体的不同写法）
        clusters = self._cluster_entities(raw_entities)
        logger.info("Clustered into %d candidate entities", len(clusters))

        # 3. 评分排序
        proposals = []
        for canonical, aliases, count in clusters:
            if count < min_occurrences:
                continue
            # 排除停用词
            if canonical in STOP_WORDS:
                continue
            # 排除单字符
            if len(canonical) <= 1:
                continue
            # 推断类型
            inferred = self._infer_type(canonical, aliases)
            props = EntityProposal(
                canonical_name=canonical,
                aliases=sorted(aliases - {canonical}),
                occurrences=count,
                source_count=count,
                confidence=min(1.0, count / 10),
                inferred_type=inferred,
                sample_texts=[],
            )
            proposals.append(props)

        # 按出现次数排序
        proposals.sort(key=lambda p: -p.occurrences)
        result.candidate_entities = proposals[:max_candidates]

        # 4. 从候选实体中推导类型定义
        result.proposed_types = self._derive_type_proposals(proposals)

        result.scan_time_ms = round((time.time() - start) * 1000, 1)
        logger.info("Discovery scan complete: %d candidates, %d types in %.1fms",
                     len(result.candidate_entities), len(result.proposed_types),
                     result.scan_time_ms)
        return result

    # ═══════════════════════════════════════════════════════════
    # 候选实体提取
    # ═══════════════════════════════════════════════════════════

    def _extract_all(self, contents: List[str]) -> List[Tuple[str, str]]:
        """从文本列表中提取所有候选实体，返回 [(实体名, 推断类型), ...]"""
        entities: List[Tuple[str, str]] = []

        for text in contents:
            if not text or len(text) < 4:
                continue

            # ---- 英文实体 ----
            # 人名（双大写词）
            for m in EN_PERSON.finditer(text):
                name = m.group().strip()
                entities.append((name, "Person"))

            # 公司名（Inc/Corp/Ltd）
            for m in EN_ORG.finditer(text):
                name = m.group().strip()
                entities.append((name, "Organization"))

            # 技术品牌（GPT-4, BERT 等）
            for m in EN_TECH.finditer(text):
                name = m.group().strip()
                entities.append((name, "Technology"))

            # 编程语言
            for m in EN_PROGRAMMING.finditer(text):
                name = m.group().strip()
                entities.append((name, "Technology"))

            # 大写字首字母缩写（多字母）
            acronyms = re.findall(r'\b(?:[A-Z][a-z]?[A-Z][a-z]?|[A-Z]{3,})\b', text)
            for ac in acronyms:
                if len(ac) >= 2 and ac not in {"The", "This", "That", "With", "From"}:
                    entities.append((ac, "Organization"))

            # ---- 中文实体 ----
            for m in CN_ORG.finditer(text):
                name = m.group().strip()
                entities.append((name, "Organization"))

            for m in CN_PERSON_PATTERN.finditer(text):
                # 提取称谓前的名字
                name = m.group().strip()
                base = re.sub(r'(?:先生|女士|博士|教授|同学|经理|老师|同志)$', '', name)
                if base and len(base) >= 2:
                    entities.append((base, "Person"))

            for m in CN_LOCATION.finditer(text):
                name = m.group().strip()
                entities.append((name, "Location"))

            for m in CN_REPORTED.finditer(text):
                name = re.sub(r'(?:表示|指出|认为)$', '', m.group())
                if name and len(name) >= 2:
                    entities.append((name, "Person"))

            # ---- 通用标签 ----
            for m in EN_HASHTAG.finditer(text):
                tag = m.group().strip("#")
                if tag and len(tag) > 2:
                    entities.append((tag, "Concept"))

        return entities

    # ═══════════════════════════════════════════════════════════
    # 实体聚类消歧
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_name(name: str) -> str:
        """规范化实体名（用于聚类匹配）"""
        n = name.lower().strip()
        # 去除所有格
        n = re.sub(r"'s$|'$", "", n)
        # 去除标点
        n = re.sub(r'[^\w\u4e00-\u9fff]', '', n)
        return n

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """Levenshtein 编辑距离"""
        a_norm = EntityDiscoveryEngine._normalize_name(a)
        b_norm = EntityDiscoveryEngine._normalize_name(b)
        if a_norm == b_norm:
            return 0
        # 快速短截：如果长度差 > 2，跳过
        if abs(len(a_norm) - len(b_norm)) > 3:
            return 999
        # 简单编辑距离
        m, n = len(a_norm), len(b_norm)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                cost = 0 if a_norm[i - 1] == b_norm[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = temp
        return dp[n]

    def _cluster_entities(
        self, entities: List[Tuple[str, str]]
    ) -> List[Tuple[str, Set[str], int]]:
        """对候选实体做聚类消歧，返回 [(规范名, 别名集合, 总次数), ...]"""
        # 第一步：按规范化名分组
        groups: Dict[str, Counter] = defaultdict(Counter)
        for name, etype in entities:
            norm = self._normalize_name(name)
            groups[norm][name] += 1

        # 第二步：合并相似实体（编辑距离 <= 2 视为同一实体）
        norm_names = list(groups.keys())
        clusters: List[Set[str]] = []
        assigned: Set[str] = set()

        for i, n1 in enumerate(norm_names):
            if n1 in assigned:
                continue
            cluster = {n1}
            assigned.add(n1)
            for j in range(i + 1, len(norm_names)):
                n2 = norm_names[j]
                if n2 in assigned:
                    continue
                if self._edit_distance(n1, n2) <= 2:
                    cluster.add(n2)
                    assigned.add(n2)
            clusters.append(cluster)

        # 第三步：合并交集（如有）
        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if clusters[i] & clusters[j]:
                        clusters[i] |= clusters[j]
                        clusters[j] = set()
                        merged = True
            clusters = [c for c in clusters if c]

        # 第四步：选择规范名（最高频的原始实体名）
        result = []
        for cluster in clusters:
            all_names = set()
            total_count = 0
            name_freq: Counter = Counter()
            for norm_name in cluster:
                for raw_name, cnt in groups[norm_name].items():
                    all_names.add(raw_name)
                    name_freq[raw_name] += cnt
                    total_count += cnt
            canonical = name_freq.most_common(1)[0][0] if name_freq else list(cluster)[0]
            result.append((canonical, all_names, total_count))

        result.sort(key=lambda x: -x[2])
        return result

    # ═══════════════════════════════════════════════════════════
    # 类型推断
    # ═══════════════════════════════════════════════════════════

    def _infer_type(self, name: str, aliases: Set[str]) -> str:
        """根据名称推断实体类型"""
        name_lower = name.lower()
        all_names = {name_lower} | {a.lower() for a in aliases}

        for etype, keywords in TYPE_HINTS.items():
            for n in all_names:
                if any(kw.lower() in n for kw in keywords):
                    return etype

        # 启发式规则
        # 中文×公司 → Organization
        if re.search(r'[\u4e00-\u9fff]', name):
            if re.search(r'(?:公司|集团|大学|银行|科技|有限)', name):
                return "Organization"
            if re.search(r'(?:先生|女士|博士|教授|老师)', name):
                return "Person"
            if re.search(r'(?:省|市|区|县|路|街道)', name):
                return "Location"
            return "Concept"

        # 英文
        if EN_PROGRAMMING.search(name) or EN_TECH.search(name):
            return "Technology"
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', name):
            return "Person"
        if re.match(r'^[A-Z][a-zA-Z\d]+(?:Inc|Corp|Ltd|Co)$', name, re.IGNORECASE):
            return "Organization"
        if re.match(r'^[A-Z]{2,}$', name):  # 全大写缩写
            return "Organization"

        return "Concept"

    # ═══════════════════════════════════════════════════════════
    # 类型定义推导
    # ═══════════════════════════════════════════════════════════

    def _derive_type_proposals(
        self, proposals: List[EntityProposal]
    ) -> List[TypeProposal]:
        """从候选实体中推导类型定义"""
        # 按推断类型分组
        type_groups: Dict[str, List[EntityProposal]] = defaultdict(list)
        for p in proposals:
            type_groups[p.inferred_type].append(p)

        type_proposals = []

        # 已有的基线类型
        existing_types = set()
        if self.ontology:
            existing_types = set(self.ontology.entity_types.keys())

        # 对每个出现 >= 2 的实体类型的组，生成类型定义
        for etype, ents in type_groups.items():
            if len(ents) < 2:
                continue

            # 如果类型已存在，但仍然提出新实体
            if etype in existing_types:
                type_proposals.append(TypeProposal(
                    name=etype + "Discovered",
                    description=f"自动发现的{etype}类型扩展",
                    parent=etype if etype in existing_types else None,
                    entity_count=len(ents),
                    sample_entities=[e.canonical_name for e in ents[:10]],
                ))
            else:
                type_proposals.append(TypeProposal(
                    name=etype,
                    description=f"自动发现的{etype}实体类型",
                    parent=None,
                    entity_count=len(ents),
                    sample_entities=[e.canonical_name for e in ents[:10]],
                ))

        return type_proposals

    # ═══════════════════════════════════════════════════════════
    # 应用到 Ontology v2
    # ═══════════════════════════════════════════════════════════

    def apply_to_ontology(
        self, result: DiscoveryResult,
        auto_register: bool = True,
    ) -> Dict[str, Any]:
        """将发现结果应用到 Ontology v2"""
        if self.ontology is None:
            return {"status": "error", "message": "No OntologyService available"}

        stats = {"types_registered": 0, "skipped_existing": 0}

        for tp in result.proposed_types:
            # 如果类型已存在，跳过
            if tp.name in self.ontology.entity_types:
                stats["skipped_existing"] += 1
                continue
            if auto_register:
                try:
                    self.ontology.register_entity_type(EntityTypeDef(
                        name=tp.name,
                        description=tp.description,
                        parent=tp.parent,
                        attributes=tp.attributes,
                    ))
                    stats["types_registered"] += 1
                    logger.info("Auto-registered entity type: %s", tp.name)
                except Exception as e:
                    logger.warning("Failed to register type '%s': %s", tp.name, e)

        return {
            "status": "ok",
            "types_registered": stats["types_registered"],
            "skipped_existing": stats["skipped_existing"],
            "total_candidates": len(result.candidate_entities),
            "total_types_proposed": len(result.proposed_types),
            "scan_time_ms": result.scan_time_ms,
        }
