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
        "位于", "地处", "省",
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

# 常见的非实体全大写缩写（不应被归为 Organization）
COMMON_ACRONYMS: Set[str] = {
    "AI", "RMB", "USD", "EUR", "GBP", "JPY", "CNY", "HKD",
    "API", "CPU", "GPU", "TPU", "RAM", "ROM", "SSD", "HDD",
    "URL", "HTML", "CSS", "JSON", "XML", "YAML", "SQL",
    "PDF", "PNG", "JPEG", "GIF", "SVG", "MP3", "MP4",
    "HTTP", "HTTPS", "FTP", "SSH", "DNS", "DHCP", "TCP", "UDP",
    "OSA", "CAGR", "ECG", "CPAP", "GT", "GTIN", "EAN", "UPC",
    "GDP", "NLP", "ML", "DL", "RL", "CV", "ASR", "TTS",
    "CEO", "CTO", "CFO", "COO", "CIO", "VP", "SVP", "EVP",
    "HR", "PR", "QA", "QC", "R&D",
    "ID", "UUID", "GUID",
    "P0", "P1", "P2", "P3", "P4",
    "v1", "v2", "v3", "v4", "v5",
}

# 常见产品关键词（用于区分 "Product" vs "Person" 中的双词大写匹配）
PRODUCT_KEYWORDS: Set[str] = {
    "Band", "Watch", "Phone", "Pro", "Max", "Mini", "Air", "Ultra",
    "Series", "Edition", "Device", "Model", "Version", "一代", "二代",
    "版", "型", "号", "系列",
}

# 双大写词但明显不是人名的模式（形容词+名词/语言/地区）
NOT_PERSON_TWO_WORD: Set[str] = {
    # 方向词
    "North", "South", "East", "West", "Northern", "Southern", "Eastern", "Western",
    # 颜色
    "Red", "Blue", "Green", "Yellow", "White", "Black", "Dark", "Light",
    # 大小/范围
    "Major", "Minor", "Large", "Small", "Mini", "Micro", "Macro", "Ultra",
    "Global", "Local", "Regional", "National", "International",
    # 通用形容词
    "Smart", "Home", "Open", "Close", "Fast", "Slow", "High", "Low",
    "New", "Old", "Big", "Little", "Great", "Best", "Top", "Key",
    # 中文拼音（常为产品名）
    "Huawei", "Xiaomi", "Honor", "Redmi", "Realme", "Vivo", "Oppo",
    # 数字+词
    "V2", "V3", "V4", "V5", "X2", "X3", "Pro", "Max", "Air",
    # 地区修饰
    "Chinese", "Japanese", "Korean", "European", "American", "British",
    "Asian", "African", "Indian", "Russian",
}

# 中文财经概念关键词（不应被识别为位置）
CN_FINANCE_CONCEPT: Set[str] = {
    "刚需", "市场", "需求", "供应", "供给", "消费", "经济",
    "行业", "产业", "领域", "赛道", "风口", "红利",
    "价值", "价格", "规模", "增速", "增长", "下降",
    "占比", "份额", "渗透", "装机", "存量", "增量",
}

# ─── 上下文消歧关键词映射 ──────────────────────────────────────
# 每种类型对应的上下文关键词（出现在实体名前后 N 个字内）
# 匹配越多 → 该类型置信度越高

CONTEXT_HINTS: Dict[str, List[str]] = {
    "Person": [
        # 英文
        "founded", "is the", "CEO of", "CTO of", "CFO of", "founder of",
        "president of", "chairman of", "director of", "manager of",
        "born in", "said", "says", "announced", "tweeted", "posted",
        "works at", "joined", "left", "resigned from",
        # 中文
        "担任", "先生", "女士", "博士", "教授", "老师",
        "表示", "认为", "指出", "说", "宣布",
        "创始人", "CEO", "总裁", "董事长", "总经理",
        "毕业于", "出生于",
    ],
    "Organization": [
        # 英文
        "Inc", "Corp", "Ltd", "LLC", "Company", "company",
        "headquartered", "based in", "acquired", "released",
        "announced", "launched", "founded",
        "subsidiary of", "division of", "partnered with",
        # 中文
        "公司", "集团", "有限", "科技", "银行",
        "总部", "位于", "成立于", "发布", "推出",
        "收购了", "投资", "合作",
    ],
    "Product": [
        # 英文
        "released", "launched", "announced", "introduced",
        "model", "version", "series", "edition",
        "priced at", "costs", "buy", "purchase", "download",
        # 中文
        "发布", "推出", "售价", "价格", "购买",
        "产品", "新款", "升级", "版本", "系列",
    ],
    "Technology": [
        # 英文
        "framework", "library", "tool", "platform", "engine",
        "language", "database", "protocol", "standard",
        "open-source", "built with", "written in",
        # 中文
        "框架", "引擎", "平台", "技术", "工具",
        "语言", "库", "协议", "标准",
    ],
    "Location": [
        # 英文
        "located in", "based in", "headquartered in",
        "city of", "in the", "near", "visit",
        # 中文
        "位于", "地处", "坐落", "省", "市", "区", "县",
    ],
    "Event": [
        # 英文
        "conference", "summit", "meeting", "convention",
        "held in", "took place", "organized",
        # 中文
        "会议", "大会", "峰会", "论坛", "展览",
    ],
}


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

        # 3. 评分排序（带上下文投票）
        proposals = []
        for canonical, aliases, count, contexts in clusters:
            if count < min_occurrences:
                continue
            # 排除停用词
            if canonical in STOP_WORDS:
                continue
            # 排除单字符
            if len(canonical) <= 1:
                continue
            # 上下文投票推断类型
            inferred = self._infer_type(canonical, aliases, contexts)
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

    def _extract_all(self, contents: List[str]) -> List[Tuple[str, str, str]]:
        """从文本列表中提取所有候选实体，返回 [(实体名, 推断类型, 上下文), ...]"""
        entities: List[Tuple[str, str, str]] = []

        for text in contents:
            if not text or len(text) < 4:
                continue

            text_lower = text.lower()

            # ---- 英文实体 ----
            # 人名（双大写词）
            for m in EN_PERSON.finditer(text):
                name = m.group().strip()
                ctx = self._extract_context(text, m.start(), m.end())
                entities.append((name, "Person", ctx))

            # 公司名（Inc/Corp/Ltd）
            for m in EN_ORG.finditer(text):
                name = m.group().strip()
                ctx = self._extract_context(text, m.start(), m.end())
                entities.append((name, "Organization", ctx))

            # 技术品牌（GPT-4, BERT 等）
            for m in EN_TECH.finditer(text):
                name = m.group().strip()
                ctx = self._extract_context(text, m.start(), m.end())
                entities.append((name, "Technology", ctx))

            # 编程语言
            for m in EN_PROGRAMMING.finditer(text):
                name = m.group().strip()
                ctx = self._extract_context(text, m.start(), m.end())
                entities.append((name, "Technology", ctx))

            # 大写字首字母缩写（多字母）
            acronyms = re.findall(r'\b(?:[A-Z][a-z]?[A-Z][a-z]?|[A-Z]{3,})\b', text)
            for ac in acronyms:
                if len(ac) >= 2 and ac not in {"The", "This", "That", "With", "From"}:
                    pos = text_lower.find(ac.lower())
                    ctx = self._extract_context(text, pos, pos + len(ac)) if pos >= 0 else ""
                    entities.append((ac, "Concept", ctx))

            # ---- 中文实体 ----
            for m in CN_ORG.finditer(text):
                name = m.group().strip()
                ctx = self._extract_context(text, m.start(), m.end())
                entities.append((name, "Organization", ctx))

            for m in CN_PERSON_PATTERN.finditer(text):
                name = m.group().strip()
                base = re.sub(r'(?:先生|女士|博士|教授|同学|经理|老师|同志)$', '', name)
                if base and len(base) >= 2:
                    ctx = self._extract_context(text, m.start(), m.end())
                    entities.append((base, "Person", ctx))

            for m in CN_LOCATION.finditer(text):
                name = m.group().strip()
                ctx = self._extract_context(text, m.start(), m.end())
                entities.append((name, "Location", ctx))

            for m in CN_REPORTED.finditer(text):
                name = re.sub(r'(?:表示|指出|认为)$', '', m.group())
                if name and len(name) >= 2:
                    ctx = self._extract_context(text, m.start(), m.end())
                    entities.append((name, "Person", ctx))

            # ---- 通用标签 ----
            for m in EN_HASHTAG.finditer(text):
                tag = m.group().strip("#")
                if tag and len(tag) > 2:
                    ctx = self._extract_context(text, m.start(), m.end())
                    entities.append((tag, "Concept", ctx))

        return entities

    @staticmethod
    def _extract_context(text: str, start: int, end: int, window: int = 40) -> str:
        """提取实体周围的上下文片段（前后各 window 个字符）"""
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end].strip()

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
        self, entities: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, Set[str], int, List[str]]]:
        """对候选实体做聚类消歧，返回 [(规范名, 别名集合, 总次数, 上下文列表), ...]"""
        # 第一步：按规范化名分组
        groups: Dict[str, Counter] = defaultdict(Counter)
        contexts: Dict[str, List[str]] = defaultdict(list)  # norm → context list
        for name, etype, ctx in entities:
            norm = self._normalize_name(name)
            groups[norm][name] += 1
            if ctx:
                contexts[norm].append(ctx)

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

        # 第四步：选择规范名（最高频的原始实体名） + 收集上下文
        result = []
        for cluster in clusters:
            all_names = set()
            total_count = 0
            name_freq: Counter = Counter()
            cluster_contexts: List[str] = []
            for norm_name in cluster:
                for raw_name, cnt in groups[norm_name].items():
                    all_names.add(raw_name)
                    name_freq[raw_name] += cnt
                    total_count += cnt
                cluster_contexts.extend(contexts.get(norm_name, []))
            canonical = name_freq.most_common(1)[0][0] if name_freq else list(cluster)[0]
            result.append((canonical, all_names, total_count, cluster_contexts[:20]))

        result.sort(key=lambda x: -x[2])
        return result

    # ═══════════════════════════════════════════════════════════
    # 类型推断
    # ═══════════════════════════════════════════════════════════

    def _infer_type(self, name: str, aliases: Set[str],
                    contexts: List[str] = None) -> str:
        """根据名称 + 上下文投票推断实体类型

        策略：
          1. 名称启发式规则（针对明显特征）
          2. 上下文关键词投票（每个上下文片段对 CONTEXT_HINTS 投票）
          3. 按投票数加权，最高票类型胜出
          4. 如无上下文 → 回退到旧规则
        """
        name_lower = name.lower()
        all_names = {name_lower} | {a.lower() for a in aliases}
        ctx_list = contexts or []

        # ── 第一关：名称启发式规则（高优先级） ──

        # 全大写缩写 → Concept（除非明确匹配 Organization 关键词）
        if re.match(r'^[A-Z][A-Z\d]+$', name) and name in COMMON_ACRONYMS:
            return "Concept"

        for etype, keywords in TYPE_HINTS.items():
            for n in all_names:
                if any(kw.lower() in n for kw in keywords):
                    return etype

        # 中文启发式
        if re.search(r'[\u4e00-\u9fff]', name):
            if re.search(r'(?:公司|集团|大学|银行|科技|有限)', name):
                return "Organization"
            if re.search(r'(?:先生|女士|博士|教授|老师)', name):
                return "Person"
            if re.search(r'(?:省|区|县|路|街道)', name):
                return "Location"
            if re.search(r'(?:市|大厦|广场)', name):
                if any(kw in name for kw in CN_FINANCE_CONCEPT):
                    return "Concept"
                if re.search(r'[\u4e00-\u9fff]{3,6}市', name):
                    return "Concept"
                return "Location"
            return "Concept"

        # 英文启发式
        if EN_PROGRAMMING.search(name) or EN_TECH.search(name):
            return "Technology"
        if re.match(r'^[A-Z][a-zA-Z\d]+(?:Inc|Corp|Ltd|Co)$', name, re.IGNORECASE):
            return "Organization"
        if re.match(r'^[A-Z]{2,}$', name):
            return "Concept"

        # ── 第二关：上下文关键词投票（核心改进） ──
        if ctx_list:
            votes: Dict[str, int] = defaultdict(int)
            ctx_combined = " ".join(ctx_list).lower()

            for etype, keywords in CONTEXT_HINTS.items():
                for kw in keywords:
                    if kw.lower() in ctx_combined:
                        votes[etype] += 1

            # 如果有投票结果，选最高票
            if votes:
                best_type = max(votes, key=votes.get)
                best_score = votes[best_type]
                # 至少需要 2 票才可信
                if best_score >= 2:
                    return best_type
                # 1 票时，只有不是 Person 才采用
                if best_type != "Person":
                    return best_type

        # ── 第三关：双大写词降级处理（最后把关） ──
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', name):
            parts = name.split()
            if any(p in PRODUCT_KEYWORDS for p in parts):
                return "Product"
            if any(p in NOT_PERSON_TWO_WORD for p in parts):
                return "Concept"
            return "Person"

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
