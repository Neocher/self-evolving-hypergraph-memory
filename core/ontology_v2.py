"""
Ontology v2 — 动态实体/边类型系统
=================================
对标 Zep Ontology + Neo4j Property Graph。

核心能力:
  · 动态类型注册（API，无需重启）
  · 属性类型系统（string/int/float/date/text_embedding 等）
  · 边约束（源/目标实体类型白名单）
  · 类型继承（Person ← Employee）
  · 写时/读时验证
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ─── 属性类型枚举 ────────────────────────────────────────────

class AttrType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    STRING_LIST = "string[]"
    TEXT_EMBEDDING = "text_embedding"   # 自动 embedding 的文本字段
    ENTITY_REF = "entity_ref"            # 引用另一个实体的 ID


# ─── Pydantic-like 数据模型定义（dataclass 版，轻量无 pydantic 依赖） ──

@dataclass
class AttributeDef:
    """属性定义"""
    name: str
    type: AttrType = AttrType.STRING
    required: bool = False
    indexed: bool = False          # 是否在 GraphLite 中建索引
    description: str = ""
    default: Any = None
    min_value: Optional[float] = None   # 数值类型最小值
    max_value: Optional[float] = None   # 数值类型最大值
    enum_values: Optional[List[str]] = None  # 枚举值列表
    # 【P0-1】temporal 属性走 PropertyVerNode 时间版本链（默认 False 向后兼容，零迁移）
    temporal: bool = False


@dataclass
class EdgeAttributeDef:
    """边属性定义（简化版，只有标量类型）"""
    name: str
    type: AttrType = AttrType.STRING
    required: bool = False
    description: str = ""


@dataclass
class EntityTypeDef:
    """实体类型定义"""
    name: str
    description: str = ""
    parent: Optional[str] = None          # 父类型（继承）
    attributes: List[AttributeDef] = field(default_factory=list)

    def get_all_attributes(self, ontology: "OntologyService") -> List[AttributeDef]:
        """获取所有属性（含继承的父类型属性）"""
        attrs = list(self.attributes)
        if self.parent and self.parent in ontology.entity_types:
            parent = ontology.entity_types[self.parent]
            # 父类型属性在前，子类型在后
            parent_attrs = parent.get_all_attributes(ontology)
            parent_names = {a.name for a in parent_attrs}
            # 子类型同名属性覆盖父类型
            merged = {a.name: a for a in parent_attrs}
            for a in attrs:
                merged[a.name] = a
            return list(merged.values())
        return attrs


@dataclass
class EdgeTypeDef:
    """边类型定义"""
    name: str
    description: str = ""
    source_types: List[str] = field(default_factory=list)   # 空=允许所有
    target_types: List[str] = field(default_factory=list)   # 空=允许所有
    attributes: List[EdgeAttributeDef] = field(default_factory=list)
    symmetry: bool = False          # 是否对称边（双向）


# ─── 验证结果 ────────────────────────────────────────────────

@dataclass
class ValidationError:
    field: str
    message: str
    code: str = "validation_error"


@dataclass
class WriteValidationResult:
    passed: bool
    entity_type: Optional[str] = None
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ─── 核心服务 ────────────────────────────────────────────────

class OntologyService:
    """本体服务 — 管理实体类型、边类型、验证"""

    # GraphLite 字段名保留集
    RESERVED_NAMES = {
        "id", "content", "embedding", "created_at",
        "tau_initial", "tau_value", "trust_score", "ontology_type", "source",
    }

    def __init__(self):
        self.entity_types: Dict[str, EntityTypeDef] = {}
        self.edge_types: Dict[str, EdgeTypeDef] = {}
        self._graphlite_store = None
        # 启动时注入基础类型
        self._init_baseline()

    def _init_baseline(self) -> None:
        """注入与旧版 ontology_validator.py 兼容的基础类型"""
        base = EntityTypeDef(
            name="_BaseNode",
            description="所有实体类型的基类",
            attributes=[
                AttributeDef(name="content", type=AttrType.STRING, required=True, description="内容文本"),
                AttributeDef(name="source", type=AttrType.STRING, required=False, description="来源"),
            ],
        )
        self.entity_types["_BaseNode"] = base

        # 常见实体类型（来自旧版 ENTITY_TYPE_MAP 的汇总）
        common_types = {
            "Person": {"description": "人物个体", "parent": None},
            "Organization": {"description": "组织/公司/机构", "parent": None},
            "MlModel": {"description": "机器学习模型", "parent": None},
            "Framework": {"description": "框架/库", "parent": None},
            "Hardware": {"description": "硬件设备", "parent": None},
            "Location": {"description": "地理位置", "parent": None},
            "Event": {"description": "事件", "parent": None},
            "Document": {"description": "文档/文章", "parent": None},
            "Concept": {"description": "抽象概念/术语", "parent": None},
            "Agent": {"description": "AI Agent（模拟角色）", "parent": "Person"},
        }
        for name, info in common_types.items():
            if name not in self.entity_types:
                self.entity_types[name] = EntityTypeDef(
                    name=name,
                    description=info["description"],
                    parent=info["parent"],
                )

        # 常见边类型
        common_edges = {
            "FOLLOWS": {"desc": "关注关系", "src": ["Person", "Agent"], "tgt": ["Person", "Organization", "Agent"]},
            "REPLIES_TO": {"desc": "回复关系", "src": ["Person", "Agent"], "tgt": ["Person", "Agent"]},
            "MENTIONS": {"desc": "提及关系", "src": ["Person", "Agent"], "tgt": ["Person", "Organization", "Concept"]},
            "BELONGS_TO": {"desc": "所属关系", "src": ["Person", "Organization"], "tgt": ["Organization", "Location"]},
            "LOCATED_IN": {"desc": "位于", "src": ["Person", "Organization", "Event"], "tgt": ["Location"]},
            "CREATED": {"desc": "创建/创作", "src": ["Person", "Agent"], "tgt": ["Document", "Concept"]},
            "RELATES_TO": {"desc": "关联（通用）", "src": [], "tgt": []},
        }
        for name, info in common_edges.items():
            if name not in self.edge_types:
                self.edge_types[name] = EdgeTypeDef(
                    name=name,
                    description=info["desc"],
                    source_types=info["src"],
                    target_types=info["tgt"],
                )

        logger.info("Ontology baseline loaded: %d entity types, %d edge types",
                     len(self.entity_types), len(self.edge_types))

    # ─── 实体类型 CRUD ───────────────────────────────────────

    def register_entity_type(self, type_def: EntityTypeDef) -> EntityTypeDef:
        """注册实体类型（同名覆盖）"""
        # 验证保留名
        for attr in type_def.attributes:
            if attr.name in self.RESERVED_NAMES:
                raise ValueError(f"Attribute name '{attr.name}' is reserved")
        # 验证父类型存在
        if type_def.parent and type_def.parent not in self.entity_types:
            raise ValueError(f"Parent type '{type_def.parent}' not found. Register it first.")
        self.entity_types[type_def.name] = type_def
        logger.info("Entity type registered: %s (parent=%s, %d attributes)",
                     type_def.name, type_def.parent, len(type_def.attributes))
        return type_def

    def get_entity_type(self, name: str) -> Optional[EntityTypeDef]:
        return self.entity_types.get(name)

    def list_entity_types(self) -> List[EntityTypeDef]:
        return list(self.entity_types.values())

    def delete_entity_type(self, name: str) -> bool:
        if name == "_BaseNode" or name in ("Person",):
            raise ValueError(f"Cannot delete baseline type '{name}'")
        # 检查是否有其他类型继承自它
        for t in self.entity_types.values():
            if t.parent == name:
                raise ValueError(f"Cannot delete '{name}': '{t.name}' inherits from it")
        return self.entity_types.pop(name, None) is not None

    # ─── 边类型 CRUD ─────────────────────────────────────────

    def register_edge_type(self, edge_def: EdgeTypeDef) -> EdgeTypeDef:
        """注册边类型（同名覆盖）"""
        # 验证源/目标类型存在（如果指定了白名单）
        for st in edge_def.source_types:
            if st not in self.entity_types:
                raise ValueError(f"Source type '{st}' not registered")
        for tt in edge_def.target_types:
            if tt not in self.entity_types:
                raise ValueError(f"Target type '{tt}' not registered")
        self.edge_types[edge_def.name] = edge_def
        logger.info("Edge type registered: %s (src=%s, tgt=%s)",
                     edge_def.name, edge_def.source_types, edge_def.target_types)
        return edge_def

    def get_edge_type(self, name: str) -> Optional[EdgeTypeDef]:
        return self.edge_types.get(name)

    def list_edge_types(self) -> List[EdgeTypeDef]:
        return list(self.edge_types.values())

    def delete_edge_type(self, name: str) -> bool:
        return self.edge_types.pop(name, None) is not None

    # ─── 写时验证 ─────────────────────────────────────────────

    def validate_write(self, content: str, ontology_type: Optional[str] = None,
                       attributes: Optional[Dict[str, Any]] = None) -> WriteValidationResult:
        """写时验证：内容是否符合注册的实体类型定义"""
        errors: List[ValidationError] = []
        warnings: List[str] = []

        # 1. 推断或使用指定的实体类型
        etype_name = ontology_type or self._infer_type(content)
        result = WriteValidationResult(passed=True, entity_type=etype_name)

        if etype_name not in self.entity_types:
            # 未知类型 → 宽松通过（仅记录）
            warnings.append(f"Unknown entity type '{etype_name}', skipped validation")
            result.warnings = warnings
            return result

        etype = self.entity_types[etype_name]
        all_attrs = etype.get_all_attributes(self)
        # 拷贝副本，避免原地修改调用方传入的 dict
        attributes = dict(attributes or {})

        # 2. 验证必填属性
        for attr in all_attrs:
            if attr.required and attr.name not in attributes:
                if attr.type == AttrType.TEXT_EMBEDDING:
                    # text_embedding 用 content 自动填充
                    attributes[attr.name] = content
                else:
                    errors.append(ValidationError(
                        field=attr.name,
                        message=f"Required attribute '{attr.name}' is missing",
                        code="missing_required",
                    ))

        # 3. 验证属性类型
        for attr_name, attr_value in attributes.items():
            attr_def = next((a for a in all_attrs if a.name == attr_name), None)
            if attr_def is None:
                warnings.append(f"Unknown attribute '{attr_name}', skipped type check")
                continue

            type_ok, err_msg = self._check_type(attr_value, attr_def)
            if not type_ok:
                errors.append(ValidationError(
                    field=attr_name,
                    message=f"Type mismatch: {err_msg}",
                    code="type_mismatch",
                ))

            # 数值范围检查
            if attr_def.type in (AttrType.INTEGER, AttrType.FLOAT) and attr_value is not None:
                try:
                    val = float(attr_value)
                    if attr_def.min_value is not None and val < attr_def.min_value:
                        errors.append(ValidationError(attr_name, f"Value {val} < min {attr_def.min_value}", "value_out_of_range"))
                    if attr_def.max_value is not None and val > attr_def.max_value:
                        errors.append(ValidationError(attr_name, f"Value {val} > max {attr_def.max_value}", "value_out_of_range"))
                except (ValueError, TypeError):
                    pass

            # 枚举值检查
            if attr_def.enum_values and attr_value not in attr_def.enum_values:
                errors.append(ValidationError(attr_name, f"Value '{attr_value}' not in {attr_def.enum_values}", "invalid_enum"))

        result.passed = len(errors) == 0
        result.errors = errors
        result.warnings = warnings
        return result

    def validate_edge(self, edge_type: str, source_type: str, target_type: str) -> WriteValidationResult:
        """边类型验证：源/目标实体类型是否符合白名单"""
        result = WriteValidationResult(passed=True)
        if edge_type not in self.edge_types:
            result.warnings.append(f"Unknown edge type '{edge_type}', skipped validation")
            return result

        edef = self.edge_types[edge_type]

        # 如果白名单为空，允许所有
        if edef.source_types and source_type not in edef.source_types:
            result.passed = False
            result.errors.append(ValidationError(
                "source_type",
                f"Source type '{source_type}' not allowed for edge '{edge_type}'. Allowed: {edef.source_types}",
                "invalid_source_type",
            ))

        if edef.target_types and target_type not in edef.target_types:
            result.passed = False
            result.errors.append(ValidationError(
                "target_type",
                f"Target type '{target_type}' not allowed for edge '{edge_type}'. Allowed: {edef.target_types}",
                "invalid_target_type",
            ))

        return result

    # ─── 读时类型感知评分 ─────────────────────────────────────

    def compute_type_match_score(self, query_types: List[str], result_types: List[str]) -> float:
        """计算查询类型与结果类型的匹配度（0~1）"""
        if not query_types or not result_types:
            return 0.5

        q_set = set(t.lower() for t in query_types)
        r_set = set(t.lower() for t in result_types)

        # 精确匹配
        exact = q_set & r_set
        # 类型层次匹配（父类型）
        hierarchical = set()
        for rt in r_set:
            for typ in self.entity_types.values():
                if typ.name.lower() == rt and typ.parent:
                    hierarchical.add(typ.parent.lower())

        overlap = exact | (q_set & hierarchical)
        if not overlap:
            return 0.3  # 完全不匹配 → 低分

        return min(1.0, 0.5 + 0.1 * len(overlap))

    # ─── 内部工具 ─────────────────────────────────────────────

    def _infer_type(self, content: str) -> str:
        """根据内容推断实体类型（简单启发式）"""
        content_lower = content.lower()
        # 启发式匹配
        type_hints = [
            ("Person", ["person", "human", "individual", "名", "人", "者", "委员"]),
            ("Organization", ["company", "organization", "corp", "inc", "ltd", "集团", "公司", "有限", "组织"]),
            ("Location", ["located", "位于", "地处", "city", "country", "region", "省", "市", "区"]),
            ("Event", ["event", "conference", "summit", "meeting", "会议", "大会", "峰会"]),
            ("Document", ["report", "article", "paper", "document", "报告", "文章", "论文"]),
            ("Agent", ["agent", "模拟角色", "#role", "#agent"]),
        ]
        for etype, keywords in type_hints:
            if any(k in content_lower for k in keywords):
                return etype
        return "Concept"

    @staticmethod
    def _check_type(value: Any, attr_def: AttributeDef) -> tuple[bool, str]:
        """检查值是否符合属性类型定义"""
        if value is None and not attr_def.required:
            return True, ""

        try:
            if attr_def.type == AttrType.STRING:
                if not isinstance(value, str):
                    return False, f"Expected string, got {type(value).__name__}"
            elif attr_def.type == AttrType.INTEGER:
                if isinstance(value, str):
                    int(value)  # 尝试转型
                elif not isinstance(value, int):
                    return False, f"Expected integer, got {type(value).__name__}"
            elif attr_def.type == AttrType.FLOAT:
                float(value)
            elif attr_def.type == AttrType.BOOLEAN:
                if not isinstance(value, bool):
                    return False, f"Expected boolean, got {type(value).__name__}"
            elif attr_def.type == AttrType.DATE:
                # 简单格式检查 YYYY-MM-DD
                import re
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(value)):
                    return False, f"Expected date (YYYY-MM-DD), got '{value}'"
            elif attr_def.type == AttrType.STRING_LIST:
                if not isinstance(value, list):
                    return False, f"Expected list, got {type(value).__name__}"
            elif attr_def.type == AttrType.TEXT_EMBEDDING:
                if not isinstance(value, str):
                    return False, f"Expected string for embedding, got {type(value).__name__}"
            # ENTITY_REF: 运行时检查（需查询GraphLite），这里跳过
        except (ValueError, TypeError):
            return False, f"Cannot convert '{value}' to {attr_def.type.value}"
        return True, ""
