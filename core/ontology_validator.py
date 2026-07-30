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

# 确保日志输出到 stderr（即使 structlog 不可用）
_log = logging.getLogger(__name__)
if not _log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    _log.addHandler(_handler)
    _log.propagate = False
logger = _log


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
    },    "domain_info": {
        "description": "域名注册/IP映射/Whois信息",
        "conflict_keys": ["domain", "dns", "ip", "解析", "注册"],
        "contradiction_pattern": "same_entity_diff_value",
    },
    "ip_address": {
        "description": "IP地址地理位置/归属",
        "conflict_keys": ["ip", "地理位置", "归属", "asn"],
        "contradiction_pattern": "same_entity_diff_value",
    },
    "url_link": {
        "description": "URL结构/内容类型/状态",
        "conflict_keys": ["url", "http", "https", "status", "响应"],
        "contradiction_pattern": "same_entity_diff_value",
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
    semantic_threshold: float = 0.85  # P0-② 语义归一化阈值


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
        self._ontology_synced = False  # lazy sync on first use
        self._candidate_entities: dict[str, int] = {}  # P5
        # P0-②: 语义对齐缓存
        self._entity_embeddings: dict[str, np.ndarray] = {}  # entity_name → vector
        self._entity_list: list[str] = []  # for batch encoding
        self._entity_emb_matrix: Optional[np.ndarray] = None  # (N, dim) matrix

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

    # ─── 实体提取增强 ─────────────────────────────────────────

    # 已知实体 → 本体类型 映射表（搜索时类型一致性校验用）
    ENTITY_TYPE_MAP: dict[str, str] = {
        # --- 深度学习框架 ---
        "pytorch": "deep_learning_framework",
        "tensorflow": "deep_learning_framework",
        "jax": "deep_learning_framework",
        "mxnet": "deep_learning_framework",
        "paddlepaddle": "deep_learning_framework",
        "onnx": "deep_learning_framework",
        "keras": "deep_learning_framework",
        "theano": "deep_learning_framework",
        "caffe": "deep_learning_framework",
        # --- 机器学习模型/架构 ---
        "transformer": "ml_model",
        "bert": "ml_model",
        "gpt": "ml_model",
        "gpt-4": "ml_model",
        "gpt-4o": "ml_model",
        "gpt-3.5": "ml_model",
        "llama": "ml_model",
        "llama3": "ml_model",
        "mistral": "ml_model",
        "qwen": "ml_model",
        "deepseek": "ml_model",
        "deepseek-v3": "ml_model",
        "claude": "ml_model",
        "gemini": "ml_model",
        "clip": "ml_model",
        "vit": "ml_model",
        "resnet": "ml_model",
        "textencoder": "ml_model",
        "sentencetransformer": "ml_model",
        "all-minilm-l6-v2": "ml_model",
        "word2vec": "ml_model",
        "whisper": "ml_model",
        "stable-diffusion": "ml_model",
        "dalle": "ml_model",
        # --- 硬件 ---
        "cpu": "hardware",
        "gpu": "hardware",
        "tpu": "hardware",
        "nvidia": "hardware",
        "amd": "hardware",
        "intel": "hardware",
        "cuda": "hardware",
        "rocm": "hardware",
        "mps": "hardware",
        "asic": "hardware",
        "fpga": "hardware",
        # --- 数据库/向量搜索 ---
        "faiss": "vector_database",
        "milvus": "vector_database",
        "pinecone": "vector_database",
        "weaviate": "vector_database",
        "chromadb": "vector_database",
        "qdrant": "vector_database",
        "kuzu": "graph_database",
        "neo4j": "graph_database",
        "arangodb": "graph_database",
        "redis": "database",
        "postgresql": "database",
        "mysql": "database",
        "mongodb": "database",
        "clickhouse": "database",
        "elasticsearch": "database",
        # --- 知名公司 ---
        "tesla": "company",
        "spacex": "company",
        "openai": "company",
        "google": "company",
        "apple": "company",
        "microsoft": "company",
        "meta": "company",
        "amazon": "company",
        "aws": "cloud_platform",
        "azure": "cloud_platform",
        "gcp": "cloud_platform",
        "alibaba": "company",
        "tencent": "company",
        "baidu": "company",
        "bytedance": "company",
        "huawei": "company",
        "xiaomi": "company",
        "ibm": "company",
        "oracle": "company",
        "salesforce": "company",
        "netflix": "company",
        "uber": "company",
        "airbnb": "company",
        "spotify": "company",
        "shopify": "company",
        "twitter": "company",
        "linkedin": "company",
        "github": "company",
        "gitlab": "company",
        "redhat": "company",
        # --- 知名人物 ---
        "elon musk": "person",
        "sam altman": "person",
        "tim cook": "person",
        "satya nadella": "person",
        "sundar pichai": "person",
        "mark zuckerberg": "person",
        "jeff bezos": "person",
        "bill gates": "person",
        "steve jobs": "person",
        "larry page": "person",
        "sergey brin": "person",
        "jack ma": "person",
        # --- 中文互联网平台 ---
        "qq": "internet_platform",
        "weixin": "internet_platform",
        "wechat": "internet_platform",
        "taobao": "internet_platform",
        "douyin": "internet_platform",
        "tiktok": "internet_platform",
        "bilibili": "internet_platform",
        "sina": "internet_platform",
        "sohu": "internet_platform",
        "netease": "internet_platform",
        "zhihu": "internet_platform",
        "xiaohongshu": "internet_platform",
        "meituan": "internet_platform",
        "didi": "internet_platform",
        "jd": "internet_platform",
        "pinduoduo": "internet_platform",
        "kuaishou": "internet_platform",
        # --- 基础设施 ---
        "docker": "infrastructure",
        "kubernetes": "infrastructure",
        "k8s": "infrastructure",
        "terraform": "infrastructure",
        "ansible": "infrastructure",
        "jenkins": "infrastructure",
        "fastapi": "web_framework",
        "flask": "web_framework",
        "django": "web_framework",
        "spring": "web_framework",
        "uvicorn": "web_server",
        "nginx": "web_server",
        "apache": "web_server",
        # --- 系统/AI ---
        "shm": "memory_system",
        "hermes": "ai_agent",
        "cursor": "ide",
        # --- 中文技术术语 ---
        "深度学习": "chinese_tech",
        "向量数据库": "chinese_tech",
        "知识图谱": "chinese_tech",
        "搜索引擎": "chinese_tech",
        "推荐系统": "chinese_tech",
        "自然语言": "chinese_tech",
        "机器学习": "chinese_tech",
        "图数据库": "chinese_tech",
        "神经网络": "chinese_tech",
        "编码器": "chinese_tech",
        "解码器": "chinese_tech",
        # --- 编程语言 ---
        "python": "programming_language",
        "rust": "programming_language",
        "go": "programming_language",
        "javascript": "programming_language",
        "typescript": "programming_language",
        "java": "programming_language",
        "swift": "programming_language",
        "kotlin": "programming_language",
        # --- 操作系统 ---
        "linux": "os",
        "ubuntu": "os",
        "centos": "os",
        "debian": "os",
        "alpine": "os",
        "macos": "os",
        "windows": "os",
        "freebsd": "os",
        # --- 数据格式/处理 ---
        "json": "data_format",
        "yaml": "data_format",
        "toml": "data_format",
        "csv": "data_format",
        "parquet": "data_format",
        "xml": "data_format",
        "numpy": "data_processing",
        "pandas": "data_processing",
        "polars": "data_processing",
        "spark": "data_processing",
        # --- 网络协议/服务 ---
        "http": "network_protocol",
        "https": "network_protocol",
        "smtp": "network_protocol",
        "websocket": "network_protocol",
        "grpc": "network_protocol",
        "dns": "network_service",
        "cdn": "network_service",
        "ddos": "network_service",
        "vpn": "network_service",
    }
    # 实体类型 → 类别（用于泛化匹配）
    # 实体类型 -> 类别（用于泛化匹配）
    ENTITY_TYPE_CATEGORIES: dict[str, str] = {
        "deep_learning_framework": "ml_infra",
        "ml_model": "ml_infra",
        "hardware": "infrastructure",
        "vector_database": "data_infra",
        "graph_database": "data_infra",
        "database": "data_infra",
        "company": "organization",
        "person": "people",
        "cloud_platform": "organization",
        "internet_platform": "web_service",
        "infrastructure": "infrastructure",
        "web_framework": "software",
        "web_server": "software",
        "memory_system": "system",
        "ai_agent": "ai_software",
        "ai_assistant": "ai_software",
        "ai_platform": "ai_software",
        "chinese_tech": "technology",
        "data_format": "data",
        "data_processing": "data_infra",
        "os": "platform",
        "programming_language": "language",
        "ide": "software",
        "network_protocol": "infrastructure",
        "network_service": "infrastructure",
    }
    def _extract_types(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取实体及其类型。

        返回: [{"entity": "PyTorch", "type": "deep_learning_framework",
                  "category": "ml_infra", "matched": True}, ...]
        """
        text_lower = text.lower()
        found: List[Dict[str, str]] = []

        # 1. 查已知实体词典（按名称降序排列，最长实体优先匹配避免短名覆盖长名）
        sorted_entities = sorted(self.ENTITY_TYPE_MAP.items(), key=lambda x: len(x[0]), reverse=True)
        for entity, etype in sorted_entities:
            if entity in text_lower:
                try:
                    has_cjk = any(ord(c) > 0x2E80 for c in entity)
                    if has_cjk or re.search(r'\b' + re.escape(entity) + r'\b', text_lower, re.ASCII):
                        category = self.ENTITY_TYPE_CATEGORIES.get(etype, "unknown")
                        found.append({
                            "entity": entity,
                            "type": etype,
                            "category": category,
                            "matched": True,
                        })
                except Exception:
                    category = self.ENTITY_TYPE_CATEGORIES.get(etype, "unknown")
                    found.append({
                        "entity": entity,
                        "type": etype,
                        "category": category,
                        "matched": True,
                    })

        # 2. 正则模式匹配（未在词典中但可推断类型）
        patterns = {
            "ml_model": [
                r'\b(?:model|encoder|decoder|transformer|embedding|tokenizer)\b',
            ],
            "hardware": [
                r'\b(?:cpu|gpu|tpu|ram|vram|memory|processor|chip)\b',
            ],
            "infrastructure": [
                r'\b(?:server|cloud|cluster|container|deploy|pipeline)\b',
            ],
            "programming_language": [
                r'\bpython\b', r'\bjavascript\b', r'\btypescript\b',
                r'\brust\b', r'\bgo\b', r'\bjava\b', r'\bc\+\+\b',
            ],
            "data_format": [
                r'\b(?:json|yaml|toml|csv|xml|parquet)\b',
            ],
        }
        for etype, pats in patterns.items():
            for pat in pats:
                m = re.search(pat, text_lower)
                if m:
                    raw = m.group(0)
                    # 避免重复
                    if not any(f["entity"] == raw for f in found):
                        category = self.ENTITY_TYPE_CATEGORIES.get(etype, "unknown")
                        found.append({
                            "entity": raw,
                            "type": etype,
                            "category": category,
                            "matched": False,  # 推断匹配（非精确词典）
                        })

        # 去重
        seen = set()
        unique = []
        for f in found:
            key = f"{f['entity']}|{f['type']}"
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    def _compute_type_overlap(
        self,
        query_types: List[Dict[str, str]],
        result_types: List[Dict[str, str]],
    ) -> float:
        """计算查询与检索结果的实体类型重叠度。

        score = w1 * exact_type_overlap + w2 * category_overlap

        精确类型重叠: 查询实体A[type=X] × 结果实体B[type=X]
        类别重叠:      查询实体A[cat=Y] × 结果实体B[cat=Y]
        """
        if not query_types or not result_types:
            return 0.5  # 无法判断时返回中性值

        exact_hits = 0
        cat_hits = 0

        q_types = set(f["type"] for f in query_types)
        r_types = set(f["type"] for f in result_types)
        q_cats = set(f["category"] for f in query_types)
        r_cats = set(f["category"] for f in result_types)

        # 精确类型重叠
        exact_overlap = q_types & r_types
        exact_hits = len(exact_overlap)

        # 类别重叠
        cat_overlap = q_cats & r_cats
        cat_hits = len(cat_overlap)

        # 归一化分数
        total_unique = len(q_types | r_types)
        if total_unique == 0:
            return 0.5
        exact_ratio = exact_hits / total_unique
        cat_ratio = cat_hits / max(len(q_cats | r_cats), 1)

        # 加权: 精确匹配权重高
        return round(0.6 * exact_ratio + 0.4 * cat_ratio, 3)

    # ─── Phase 2: Kuzu 本体图同步 + 拓扑验证 ─────────────────────

    def _ensure_ontology_schema(self) -> None:
        """确保 Kuzu 中存在本体论节点/边表（幂等）。"""
        if self.kuzu is None:
            return
        try:
            self.kuzu.execute_cypher(
                "CREATE NODE TABLE IF NOT EXISTS OntologyType ("
                "name STRING, category STRING, PRIMARY KEY (name))",
                {},
            )
            self.kuzu.execute_cypher(
                "CREATE NODE TABLE IF NOT EXISTS OntologyEntity ("
                "name STRING, type STRING, category STRING, PRIMARY KEY (name))",
                {},
            )
            self.kuzu.execute_cypher(
                "CREATE REL TABLE IF NOT EXISTS IS_A "
                "(FROM OntologyEntity TO OntologyType)",
                {},
            )
            self.kuzu.execute_cypher(
                "CREATE REL TABLE IF NOT EXISTS RELATES_TO "
                "(FROM OntologyEntity TO OntologyEntity, relation STRING)",
                {},
            )
        except Exception as e:
            logger.warning("Failed to create ontology schema: %s", e)

    def sync_entity_types_to_kuzu(self) -> int:
        """同步 ENTITY_TYPE_MAP 到 Kuzu，创建节类型/实体节点 + IS_A 边。

        Returns: 同步的实体数
        """
        if self.kuzu is None:
            return 0
        self._ensure_ontology_schema()
        count = 0
        for entity, etype in self.ENTITY_TYPE_MAP.items():
            category = self.ENTITY_TYPE_CATEGORIES.get(etype, "unknown")
            try:
                # 创建类型节点（只对主键 name MERGE）
                self.kuzu.execute_cypher(
                    "MERGE (t:OntologyType {name: $type}) "
                    "SET t.category = $cat",
                    {"type": etype, "cat": category},
                )
                # 创建实体节点 + IS_A 边（只对主键 name MERGE，防止重复键冲突）
                self.kuzu.execute_cypher(
                    "MERGE (e:OntologyEntity {name: $name}) "
                    "SET e.type = $etype, e.category = $cat "
                    "WITH e "
                    "MATCH (t:OntologyType {name: $type}) "
                    "MERGE (e)-[:IS_A]->(t)",
                    {"name": entity, "etype": etype, "cat": category, "type": etype},
                )
                count += 1
            except Exception as e:
                logger.warning("Failed to sync entity %s→%s: %s", entity, etype, e)
        logger.info("Ontology synced to Kuzu: %d entities", count)
        return count

    def _extract_entity_cooccurrence(self, content: str) -> List[str]:
        """从文本中找出哪些 ENTITY_TYPE_MAP 实体共同出现。

        用于推断实体间关系（同一段话中出现的实体可能有关联）。
        返回: 共现实体名列表（小写去重）
        """
        text_lower = content.lower()
        found_names: List[str] = []
        for entity in self.ENTITY_TYPE_MAP:
            if entity in text_lower:
                try:
                    has_cjk = any(ord(c) > 0x2E80 for c in entity)
                    if has_cjk or re.search(r'\b' + re.escape(entity) + r'\b', text_lower, re.ASCII):
                        found_names.append(entity)
                except Exception:
                    if entity in text_lower:
                        found_names.append(entity)
        return list(set(found_names))

    def _populate_relationships(self) -> int:
        """扫描 EpisodeNode 内容，提取实体共现关系 → 创建 RELATES_TO 边。

        Returns: 新增的关系数
        """
        if self.kuzu is None:
            return 0
        self._ensure_ontology_schema()
        try:
            rows = self.kuzu.execute_cypher(
                "MATCH (e:EpisodeNode) RETURN e.content LIMIT 5000",
                {},
            )
        except Exception as e:
            logger.warning("Failed to query EpisodeNode contents: %s", e)
            return 0
        rel_count = 0
        for row in rows:
            content = row.get("content", "") if isinstance(row, dict) else str(row[0]) if isinstance(row, (list, tuple)) and len(row) > 0 else ""
            if not content:
                continue
            entities = self._extract_entity_cooccurrence(content)
            if len(entities) < 2:
                continue
            # 同一段话中每对实体创建 RELATES_TO 边
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    try:
                        self.kuzu.execute_cypher(
                            "MATCH (a:OntologyEntity {name: $a_name}) "
                            "MATCH (b:OntologyEntity {name: $b_name}) "
                            "MERGE (a)-[:RELATES_TO {relation: 'co_occur'}]->(b) "
                            "MERGE (b)-[:RELATES_TO {relation: 'co_occur'}]->(a)",
                            {"a_name": entities[i], "b_name": entities[j]},
                        )
                        rel_count += 1
                    except Exception as e:
                        logger.warning("Failed to create RELATES_TO edge: %s", e)
        return rel_count

    def _learn_candidate_entities(self, content: str, threshold: int = 3) -> None:
        """P5: discover unknown entity candidates."""
        import re
        candidates = re.findall(r'\b[A-Z][a-zA-Z0-9._-]{2,49}\b', content)
        for c in candidates:
            c_lower = c.lower()
            if c_lower not in self.ENTITY_TYPE_MAP and c_lower not in self._candidate_entities:
                self._candidate_entities[c_lower] = 1
            elif c_lower not in self.ENTITY_TYPE_MAP:
                self._candidate_entities[c_lower] = self._candidate_entities.get(c_lower, 0) + 1
                freq = self._candidate_entities[c_lower]
                if freq == threshold:
                    logger.info("P5: New entity candidate '%s' appeared %d times", c, freq)

    def extract_and_relate(self, content: str) -> int:
        """写入时提取实体共现关系 → 创建 RELATES_TO 边。

        每次新记忆写入时调用，确保拓扑图即时更新。
        Returns: 创建的关系数
        """
        if self.kuzu is None:
            return 0
        # lazy sync on first write
        if not self._ontology_synced:
            self.sync_entity_types_to_kuzu()
            self._ontology_synced = True
        self._learn_candidate_entities(content)
        entities = self._extract_entity_cooccurrence(content)
        if len(entities) < 2:
            return 0
        count = 0
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                try:
                    self.kuzu.execute_cypher(
                        "MATCH (a:OntologyEntity {name: $a_name}) "
                        "MATCH (b:OntologyEntity {name: $b_name}) "
                        "MERGE (a)-[:RELATES_TO {relation: 'co_occur'}]->(b) "
                        "MERGE (b)-[:RELATES_TO {relation: 'co_occur'}]->(a)",
                        {"a_name": entities[i], "b_name": entities[j]},
                    )
                    count += 1
                except Exception as e:
                    logger.warning("Topology path query failed: %s", e)
        return count

    # ─── P1: 本体层次距离 ─────────────────────────────────────

    # 类别层次树：category → parent_category（用于计算本体距离）
    ONTOLOGY_TREE: dict[str, str] = {
        # 根 → AI/技术
        "ml_infra": "ai_technology",
        "ai_software": "ai_technology",
        "ai_assistant": "ai_technology",
        "ai_platform": "ai_technology",
        # 根 → 软件/开发
        "software": "development",
        "programming_language": "development",
        "language": "development",
        "ide": "development",
        "platform": "development",
        "data_format": "development",
        # 根 → 数据/基础设施
        "data_infra": "data_engineering",
        "data": "data_engineering",
        "data_processing": "data_engineering",
        # 根 → 基础设施
        "infrastructure": "infrastructure",
        "system": "infrastructure",
        "memory_system": "infrastructure",
        # 根 → 组织/人
        "organization": "human_world",
        "people": "human_world",
        # 根 → 网络/服务
        "web_service": "internet",
        "web_platform": "internet",
        # 根 → 技术（通用）
        "technology": "general_tech",

        # entity_type → category 已经由 ENTITY_TYPE_CATEGORIES 定义
    }

    def _get_entity_type(self, entity_name: str) -> str:
        """获取实体的类型。"""
        return self.ENTITY_TYPE_MAP.get(entity_name.lower(), "")

    def _get_category(self, entity_type: str) -> str:
        """获取类型所属的类别。"""
        return self.ENTITY_TYPE_CATEGORIES.get(entity_type, "")

    def _get_parent_category(self, category: str) -> str:
        """获取类别的父类别（用于距离计算）。"""
        return self.ONTOLOGY_TREE.get(category, "")

    def _compute_ontological_distance(self, entity_a: str, entity_b: str) -> float:
        """计算两个实体之间的本体层次距离。

        层级: entity_name → entity_type → category → parent_category

        Returns:
            1.0 同实体
            0.9 同类型（同为deep_learning_framework）
            0.7 同类别（同为ml_infra）
            0.5 同父类别（同为ai_technology下）
            0.3 不同分支
        """
        if entity_a.lower() == entity_b.lower():
            return 1.0

        ta = self._get_entity_type(entity_a)
        tb = self._get_entity_type(entity_b)
        if not ta or not tb:
            return 1.0  # 未知实体，保底

        if ta == tb:
            return 0.9

        ca = self._get_category(ta)
        cb = self._get_category(tb)
        if not ca or not cb:
            return 1.0
        if ca == cb:
            return 0.7

        pa = self._get_parent_category(ca)
        pb = self._get_parent_category(cb)
        if not pa or not pb:
            return 0.5
        if pa == pb:
            return 0.5

        return 0.3

    def _compute_topology_score(
        self,
        query_entities: List[Dict[str, str]],
        result_content: str,
    ) -> float:
        """拓扑验证：查询实体与结果实体的关系路径一致性。

        1. 从结果内容中提取实体
        2. 对查询与结果中共同出现的实体对，检查 Kuzu 中是否有 RELATES_TO 路径
        3. 有路径 → 1.0（高度一致）；无路径 → 0.6（中性）
        Returns: 拓扑置信度 (0.0 ~ 1.0)
        """
        if not query_entities or self.kuzu is None:
            return 1.0

        result_entity_names = self._extract_entity_cooccurrence(result_content)
        if not result_entity_names:
            return 1.0

        query_entity_names = [q["entity"] for q in query_entities]
        # 找查询与结果的共同实体
        shared = set(query_entity_names) & set(result_entity_names)
        # 找查询与结果的不同实体对（需要检查关系）
        query_only = [e for e in query_entity_names if e not in shared]
        result_only = [e for e in result_entity_names if e not in shared]

        if not query_only or not result_only:
            return 1.0

        # 检查是否有 RELATES_TO 路径连接查询和结果的实体
        path_found = 0
        total_checked = 0
        for qe in query_only:
            for re_name in result_only:
                total_checked += 1
                try:
                    result = self.kuzu.execute_cypher(
                        "MATCH (a:OntologyEntity {name: $a_name}) "
                        "MATCH (b:OntologyEntity {name: $b_name}) "
                        "MATCH (a)-[:RELATES_TO*1..3]-(b) "
                        "RETURN count(*) AS cnt LIMIT 1",
                        {"a_name": qe, "b_name": re_name},
                    )
                    if result and len(result) > 0:
                        row = result[0]
                        cnt = row.get("cnt", 0) if isinstance(row, dict) else int(row[0]) if isinstance(row, (list, tuple)) else 0
                        if cnt > 0:
                            path_found += 1
                except Exception as e:
                    logger.debug("Topology path query failed: %s", e)

        # P3: also check paths between shared entities
        shared_list = list(shared)
        for i in range(len(shared_list)):
            for j in range(i + 1, len(shared_list)):
                total_checked += 1
                try:
                    result = self.kuzu.execute_cypher(
                        "MATCH (a:OntologyEntity {name: $a_name}) "
                        "MATCH (b:OntologyEntity {name: $b_name}) "
                        "OPTIONAL MATCH (a)-[:RELATES_TO*1..3]-(b) "
                        "RETURN count(*) AS cnt",
                        {"a_name": shared_list[i], "b_name": shared_list[j]},
                    )
                    if result and len(result) > 0:
                        row = result[0]
                        cnt = row.get("cnt", 0) if isinstance(row, dict) else int(row[0]) if isinstance(row, (list, tuple)) else 0
                        if cnt > 0:
                            path_found += 1
                except Exception:
                    pass

        if total_checked == 0:
            return 1.0
        ratio = path_found / total_checked
        # 有路径 → 1.0 (确认关系); 无路径 → 0.6 (非关系但也不是矛盾)
        return round(0.6 + 0.4 * ratio, 3)

    # ─── P0-②: 语义对齐 ─────────────────────────────────────

    def _build_entity_embeddings(self) -> None:
        """预计算 ENTITY_TYPE_MAP 所有实体的 embedding（复用 encoder）。"""
        if self.encoder is None:
            return
        names = list(self.ENTITY_TYPE_MAP.keys())
        if not names:
            return
        try:
            if hasattr(self.encoder, 'embed_batch'):
                embs = self.encoder.embed_batch(names)
                if embs is not None and len(embs) == len(names):
                    for i, name in enumerate(names):
                        self._entity_embeddings[name] = embs[i]
                    self._entity_list = names
                    self._entity_emb_matrix = np.array(embs)
                    logger.info("Semantic alignment: %d entity embeddings built", len(names))
            else:
                for name in names:
                    emb = self.encoder.embed(name)
                    if emb is not None:
                        self._entity_embeddings[name] = emb
                self._entity_list = names
        except Exception as e:
            logger.warning("Entity embedding build failed (semantic alignment degraded): %s", e)

    def _semantic_normalize(self, raw_entity: str) -> str:
        """用语义相似度将提取的实体名映射到标准实体名。

        例: "deeplearning" → "deep_learning_framework" 的词表中找最近邻
             "Deep Learning" → "deep_learning_framework"
        Returns: 标准实体名（无匹配则返回原串）
        """
        if self.encoder is None or not self._entity_embeddings:
            return raw_entity

        try:
            raw_vec = self.encoder.embed(raw_entity)
            if raw_vec is None or self._entity_emb_matrix is None:
                return raw_entity
            raw_vec = raw_vec.reshape(1, -1)
            # 余弦相似度
            norms = np.linalg.norm(self._entity_emb_matrix, axis=1)
            raw_norm = np.linalg.norm(raw_vec)
            if raw_norm == 0 or np.any(norms == 0):
                return raw_entity
            dots = self._entity_emb_matrix @ raw_vec.T
            sims = (dots.flatten()) / (norms * raw_norm + 1e-10)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= self.config.semantic_threshold:
                matched = self._entity_list[best_idx]
                if matched.lower() != raw_entity.lower():
                    logger.debug("Semantic normalize: '%s' → '%s' (cos=%.3f)",
                                 raw_entity, matched, best_score)
                return matched
            return raw_entity
        except Exception as e:
            logger.debug("Semantic normalize failed for '%s': %s", raw_entity, e)
            return raw_entity

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

        # 0. 语义对齐：构建/使用 entity embeddings
        if not self._entity_embeddings and self.encoder is not None:
            try:
                self._build_entity_embeddings()
            except Exception:
                pass

        # 1. 提取实体
        entities = self._extract_entities(content)
        if not entities:
            return result

        # 1b. 语义归一化：将提取实体映射到标准实体名
        normalized_entities = []
        for ent in entities:
            norm = self._semantic_normalize(ent)
            normalized_entities.append(norm)
        # 用归一化后的实体替换
        entities = list(set(normalized_entities))

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

        [增强] 新增实体类型一致性评分：
        1. 从查询文本提取实体类型
        2. 对每条候选结果对比类型重叠度
        3. 综合分数 = original_score × type_overlap × τ_factor × conflict_penalty
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

        # lazy: 首次调用时同步实体类型到 Kuzu 并建立关系图
        if not self._ontology_synced:
            self.sync_entity_types_to_kuzu()
            rels = self._populate_relationships()
            logger.info("Ontology graph synced: %d relationships from content", rels)
            self._ontology_synced = True

        # 0. 提取查询的实体类型
        query_types = self._extract_types(query) if query else []

        # 0b. 语义归一化查询实体
        if query_types and self.encoder is not None:
            if not self._entity_embeddings:
                try:
                    self._build_entity_embeddings()
                except Exception:
                    pass
            for qt in query_types:
                qt["entity"] = self._semantic_normalize(qt.get("entity", ""))

        validated = []
        for r in results:
            ep_id = r.get("id", "")
            tau = r.get("tau_value", r.get("tau", 0.5)) or 0.5
            trust = r.get("trust_score", 0.5)
            score = r.get("score", 0.0)
            content = r.get("content", r.get("txt_content", ""))

            # 1. 提取实体（传统方法）
            entities = self._extract_entities(content)

            # 2. 提取结果中的类型（新方法）
            result_types = self._extract_types(content)

            # 3. 计算类型一致性重叠度
            type_overlap = self._compute_type_overlap(query_types, result_types)

            # 4. 检查 Kuzu 矛盾（原有逻辑，容错处理）
            ontology_conf = 1.0
            conflict_count = 0

            if entities and self.kuzu is not None and len(entities) > 0:
                try:
                    otype = self._classify_ontology_type(content, entities)
                    rule = CONTRADICTION_RULES.get(
                        ONTOLOGY_TYPES.get(otype, {}).get(
                            "contradiction_pattern", "contradictory_claim"
                        )
                    )
                    if rule and ep_id:
                        params = {
                            "new_id": ep_id,
                            "ontology_type": otype,
                            "entity_name": entities[0],
                            "new_value": "",
                        }
                        conflicts = self.kuzu.execute_cypher(rule, params)
                        if conflicts and isinstance(conflicts, list):
                            conflict_count = len(conflicts)
                            ontology_conf = max(
                                0.1,
                                1.0 - self.config.conflict_penalty_factor * conflict_count
                            )
                except Exception as e:
                    # Kuzu 矛盾查询不可用时不阻塞类型一致性评分
                    logger.debug("Contradiction query skipped (non-fatal): %s", e)

            # 5. 拓扑验证：检查查询与结果实体间的关系路径
            topology_score = self._compute_topology_score(query_types, content)

            # 6. 综合分数: 惩罚不匹配 + 奖励超匹配，突破原始 FAISS 天花板
            tau_factor = min(1.0, tau / 0.5) if tau > 0 else 0.5
            confidence_bonus = round(type_overlap * topology_score * tau_factor * ontology_conf, 3)

            # 【P4】查询无实体类型→无法做本体判断→返回原始分
            if not query_types:
                adjusted = score * ontology_conf
            # 高置信度阈值：实体类型全匹配 + 拓扑有路径 → 直接给满分
            elif confidence_bonus >= 0.95 and self.kuzu is not None:
                adjusted = min(0.9999, score + 0.3)
            else:
                # 乘法惩罚(不匹配时) + 保底(0.2)防止完全归零
                adjusted = score * (0.2 + 0.8 * confidence_bonus)

            note = ""
            if conflict_count > 0:
                note = f"[本体矛盾] 关于「{entities[0]}」有 {conflict_count} 条冲突记录，此条置信度已下调"
            if query_types and result_types and type_overlap < 0.3:
                qtypes_str = ", ".join(sorted(set(f["type"] for f in query_types)))
                rtypes_str = ", ".join(sorted(set(f["type"] for f in result_types)))
                note = f"[类型不匹配] 查询实体类型({qtypes_str}) ≠ 结果类型({rtypes_str})，分数已下调"

            validated.append(ReadValidationResult(
                episode_id=ep_id,
                original_score=score,
                ontology_confidence=round(type_overlap * ontology_conf, 3),
                adjusted_score=round(adjusted, 4),
                conflict_count=conflict_count,
                has_conflicts=conflict_count > 0 or type_overlap < 0.3,
                conflict_note=note,
            ))

        # 按调整后分数降序排列
        validated.sort(key=lambda x: x.adjusted_score, reverse=True)
        return validated
