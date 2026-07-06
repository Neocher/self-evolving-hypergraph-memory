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
        self._ontology_synced = False  # lazy sync on first use

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
        # 深度学习框架
        "pytorch": "deep_learning_framework",
        "tensorflow": "deep_learning_framework",
        "jax": "deep_learning_framework",
        "mxnet": "deep_learning_framework",
        "paddlepaddle": "deep_learning_framework",
        "onnx": "deep_learning_framework",
        # 机器学习模型/架构
        "transformer": "ml_model",
        "bert": "ml_model",
        "gpt": "ml_model",
        "llama": "ml_model",
        "clip": "ml_model",
        "vit": "ml_model",
        "resnet": "ml_model",
        "textencoder": "ml_model",
        "sentencetransformer": "ml_model",
        "all-minilm-l6-v2": "ml_model",
        "word2vec": "ml_model",
        # 硬件
        "cpu": "hardware",
        "gpu": "hardware",
        "tpu": "hardware",
        "nvidia": "hardware",
        "amd": "hardware",
        "intel": "hardware",
        "cuda": "hardware",
        "rocm": "hardware",
        "mps": "hardware",
        # 计算/部署技术
        "faiss": "vector_database",
        "kuzu": "graph_database",
        "neo4j": "graph_database",
        "redis": "database",
        "docker": "infrastructure",
        "kubernetes": "infrastructure",
        "fastapi": "web_framework",
        "uvicorn": "web_server",
        # 系统
        "shm": "memory_system",
        "hermes": "ai_agent",
        "cursor": "ide",
        "claude": "ai_assistant",
        "deepseek": "ai_assistant",
        "openai": "ai_platform",
        # 中文互联网平台
        "baidu": "internet_platform",
        "qq": "internet_platform",
        "weixin": "internet_platform",
        "wechat": "internet_platform",
        "taobao": "internet_platform",
        "alibaba": "internet_platform",
        "tencent": "internet_platform",
        "huawei": "internet_platform",
        "xiaomi": "internet_platform",
        "douyin": "internet_platform",
        "tiktok": "internet_platform",
        "bilibili": "internet_platform",
        "sina": "internet_platform",
        "sohu": "internet_platform",
        "netease": "internet_platform",
        "zhihu": "internet_platform",
        "meituan": "internet_platform",
        "didi": "internet_platform",
        "jd": "internet_platform",
        "bytedance": "internet_platform",
        "pinduoduo": "internet_platform",
        "kuaishou": "internet_platform",
        # 网络服务/协议
        "dns": "network_service",
        "cdn": "network_service",
        "smtp": "network_service",
        "pop3": "network_service",
        "imap": "network_service",
        "spf": "network_service",
        "mx": "network_service",
        "ns1": "network_service",
        "ns2": "network_service",
        "ns3": "network_service",
        "ns4": "network_service",
        "cname": "network_service",
        "aaaa": "network_service",
        "srv": "network_service",
        "txt": "network_service",
        "http": "network_service",
        "https": "network_service",
        "websocket": "network_service",
        # 文件/数据格式
        "json": "data_format",
        "yaml": "data_format",
        "toml": "data_format",
        "csv": "data_format",
        "parquet": "data_format",
        "numpy": "data_processing",
        "pandas": "data_processing",
        # 操作系统/环境
        "linux": "os",
        "ubuntu": "os",
        "centos": "os",
        "python": "programming_language",
        "rust": "programming_language",
        "go": "programming_language",
        "javascript": "programming_language",
        "typescript": "programming_language",
    }

    # 实体类型 → 类别（用于泛化匹配）
    ENTITY_TYPE_CATEGORIES: dict[str, str] = {
        "deep_learning_framework": "ml_infra",
        "ml_model": "ml_infra",
        "hardware": "infrastructure",
        "vector_database": "data_infra",
        "graph_database": "data_infra",
        "database": "data_infra",
        "infrastructure": "infrastructure",
        "web_framework": "software",
        "web_server": "software",
        "memory_system": "system",
        "ai_agent": "ai_software",
        "ai_assistant": "ai_software",
        "data_format": "data",
        "data_processing": "data_infra",
        "os": "platform",
        "programming_language": "language",
        "ide": "software",
        "ai_platform": "ai_software",
        "internet_platform": "web_service",
        "network_service": "infrastructure",
    }

    def _extract_types(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取实体及其类型。

        返回: [{"entity": "PyTorch", "type": "deep_learning_framework",
                  "category": "ml_infra", "matched": True}, ...]
        """
        text_lower = text.lower()
        found: List[Dict[str, str]] = []

        # 1. 查已知实体词典
        for entity, etype in self.ENTITY_TYPE_MAP.items():
            if entity in text_lower:
                # re.ASCII 确保 \\b 只匹配 ASCII 单词边界（CJK 字符不是 \\w）
                try:
                    if re.search(r'\b' + re.escape(entity) + r'\b', text_lower, re.ASCII):
                        category = self.ENTITY_TYPE_CATEGORIES.get(etype, "unknown")
                        found.append({
                            "entity": entity,
                            "type": etype,
                            "category": category,
                            "matched": True,
                        })
                except Exception:
                    # 回退：直接子串匹配
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
                # 创建类型节点
                self.kuzu.execute_cypher(
                    "MERGE (t:OntologyType {name: $type, category: $cat})",
                    {"type": etype, "cat": category},
                )
                # 创建实体节点 + IS_A 边
                self.kuzu.execute_cypher(
                    "MERGE (e:OntologyEntity {name: $name, type: $etype, category: $cat}) "
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
                    if re.search(r'\b' + re.escape(entity) + r'\b', text_lower, re.ASCII):
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

        if total_checked == 0:
            return 1.0
        ratio = path_found / total_checked
        # 有路径 → 1.0 (确认关系); 无路径 → 0.6 (非关系但也不是矛盾)
        return round(0.6 + 0.4 * ratio, 3)

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
                adjusted = score
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
