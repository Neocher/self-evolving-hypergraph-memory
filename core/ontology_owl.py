"""
Ontology OWL 互操作层 — 将 Ontology v2 导出为标准 OWL/Turtle
=============================================================
对标 W3C OWL 2 序列化。只导出，不推理、不导入（导入留待 v5.20）。

映射规则（design_ontology_gaps.md v2）:
  · EntityTypeDef  → owl:Class（类型继承 → rdfs:subClassOf）
  · AttributeDef   → owl:DatatypeProperty（IRI 作用域命名 {TypeName}_{AttrName}）
  · EdgeTypeDef    → owl:ObjectProperty（source/target 白名单 → rdfs:domain/rdfs:range）
  · EdgeAttributeDef → owl:AnnotationProperty（附加到 ObjectProperty 上）
  · AttrType       → 完整 9 类型映射（含 datetime/text_embedding/entity_ref）
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

from core.ontology_v2 import AttrType, EdgeTypeDef, EntityTypeDef, OntologyService

try:
    from shm._version import __version__ as _SHM_VERSION
except ImportError:  # 测试/独立环境下无版本包时回退
    _SHM_VERSION = "5.19.0"

logger = logging.getLogger(__name__)

# 命名空间
_OWL = "http://www.w3.org/2002/07/owl#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_XSD = "http://www.w3.org/2001/XMLSchema#"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_SHM = "http://shm.local/ontology#"

# AttrType → OWL 数据类型 完整映射（含全部 9 种枚举值）
ATTR_TYPE_TO_XSD: Dict[AttrType, str] = {
    AttrType.STRING: f"{_XSD}string",
    AttrType.INTEGER: f"{_XSD}int",
    AttrType.FLOAT: f"{_XSD}double",
    AttrType.BOOLEAN: f"{_XSD}boolean",
    AttrType.DATE: f"{_XSD}date",
    AttrType.DATETIME: f"{_XSD}dateTime",
    AttrType.STRING_LIST: f"{_RDF}Seq",
    AttrType.TEXT_EMBEDDING: f"{_XSD}string",
    AttrType.ENTITY_REF: f"{_XSD}anyURI",
}

# 需要附加 SHM 专用注释的特殊类型
_SPECIAL_ATTR_COMMENTS = {
    AttrType.TEXT_EMBEDDING: "SHM-specific: text_embedding",
    AttrType.ENTITY_REF: "SHM-specific: entity_ref",
}


class OntologyOwlExporter:
    """将 OntologyService 导出为 OWL/Turtle 文本。"""

    @staticmethod
    def _local_name(name: str) -> str:
        """将类型/属性名转换为合法的 Turtle QName 局部名。"""
        local = re.sub(r"[^A-Za-z0-9_]", "_", name)
        if not local or local[0].isdigit():
            local = "_" + local
        return local

    @staticmethod
    def _escape(text: str) -> str:
        """转义 Turtle 字符串字面量中的特殊字符。"""
        return (text.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n"))

    def _find_duplicate_attr_names(self, ontology: OntologyService) -> Set[str]:
        """检测多个实体类型共用的属性名（同名属性 IRI 作用域警告依据）。"""
        seen: Dict[str, str] = {}
        dupes: Set[str] = set()
        for tdef in ontology.list_entity_types():
            for attr in tdef.attributes:
                if attr.name in seen and seen[attr.name] != tdef.name:
                    dupes.add(attr.name)
                else:
                    seen[attr.name] = tdef.name
        return dupes

    def export_turtle(self, ontology: OntologyService) -> str:
        """导出本体为 Turtle 文本。"""
        lines: List[str] = []
        dupes = self._find_duplicate_attr_names(ontology)

        lines.append("# SHM Ontology export (Turtle/OWL)")
        lines.append(f"# Version: {_SHM_VERSION} | entity types: {len(ontology.entity_types)} | "
                     f"edge types: {len(ontology.edge_types)}")
        lines.append("")
        lines.append("@prefix owl:  <http://www.w3.org/2002/07/owl#> .")
        lines.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
        lines.append("@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .")
        lines.append("@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
        lines.append(f"@prefix shm:  <{_SHM}> .")
        lines.append("")

        # 本体头
        lines.append("shm: a owl:Ontology ;")
        lines.append(f'    owl:versionInfo "{self._escape(_SHM_VERSION)}" ;')
        if dupes:
            dup_list = ", ".join(sorted(dupes))
            lines.append(f'    rdfs:comment "Duplicate attribute names across types '
                         f'(scoped per-type IRIs emitted): {self._escape(dup_list)}" .')
        else:
            lines.append('    rdfs:comment "SHM Ontology" .')
        lines.append("")

        # 类定义（实体类型 + 继承）
        valid_types = {t.name for t in ontology.list_entity_types()}
        lines.append("# ─── Classes ───")
        for tdef in ontology.list_entity_types():
            lines.append(self._class_block(tdef, valid_types))
        lines.append("")

        # 属性定义（DatatypeProperty，IRI 作用域命名）
        lines.append("# ─── DatatypeProperties (scoped: {TypeName}_{AttrName}) ───")
        for tdef in ontology.list_entity_types():
            for attr in tdef.attributes:
                lines.append(self._datatype_block(tdef, attr))
        lines.append("")

        # 边定义（ObjectProperty）
        lines.append("# ─── ObjectProperties ───")
        for edef in ontology.list_edge_types():
            lines.append(self._object_property_block(edef))
            for eattr in edef.attributes:
                lines.append(self._annotation_block(edef, eattr))
        lines.append("")

        return "\n".join(lines)

    # ─── 各块生成 ────────────────────────────────────────────

    def _class_block(self, tdef: EntityTypeDef, valid_types: Set[str]) -> str:
        """EntityTypeDef → owl:Class（含 rdfs:subClassOf 继承）"""
        lines = [f"shm:{self._local_name(tdef.name)} a owl:Class ;"]
        lines.append(f'    rdfs:label "{self._escape(tdef.name)}" ;')
        if tdef.description:
            lines.append(f'    rdfs:comment "{self._escape(tdef.description)}" ;')
        if tdef.parent and tdef.parent in valid_types:
            lines.append(f"    rdfs:subClassOf shm:{self._local_name(tdef.parent)} ;")
        lines[-1] = lines[-1].rstrip(" ;") + " ."
        return "\n".join(lines)

    def _datatype_block(self, tdef: EntityTypeDef, attr) -> str:
        """AttributeDef → owl:DatatypeProperty，IRI 用 {TypeName}_{AttrName} 作用域限定"""
        local = f"{self._local_name(tdef.name)}_{self._local_name(attr.name)}"
        rng = ATTR_TYPE_TO_XSD.get(attr.type, f"{_XSD}string")
        lines = [f"shm:{local} a owl:DatatypeProperty ;"]
        lines.append(f'    rdfs:label "{self._escape(attr.name)}" ;')
        lines.append(f"    rdfs:domain shm:{self._local_name(tdef.name)} ;")
        lines.append(f"    rdfs:range <{rng}> ;")
        if attr.description:
            lines.append(f'    rdfs:comment "{self._escape(attr.description)}" ;')
        special = _SPECIAL_ATTR_COMMENTS.get(attr.type)
        if special:
            lines.append(f'    rdfs:comment "{special}" ;')
        lines[-1] = lines[-1].rstrip(" ;") + " ."
        return "\n".join(lines)

    def _object_property_block(self, edef: EdgeTypeDef) -> str:
        """EdgeTypeDef → owl:ObjectProperty（白名单 → domain/range）"""
        lines = [f"shm:{self._local_name(edef.name)} a owl:ObjectProperty ;"]
        lines.append(f'    rdfs:label "{self._escape(edef.name)}" ;')
        if edef.description:
            lines.append(f'    rdfs:comment "{self._escape(edef.description)}" ;')
        if edef.source_types:
            dom = ", ".join(f"shm:{self._local_name(s)}" for s in edef.source_types)
            lines.append(f"    rdfs:domain {dom} ;")
        if edef.target_types:
            rng = ", ".join(f"shm:{self._local_name(t)}" for t in edef.target_types)
            lines.append(f"    rdfs:range {rng} ;")
        lines[-1] = lines[-1].rstrip(" ;") + " ."
        return "\n".join(lines)

    def _annotation_block(self, edef: EdgeTypeDef, eattr) -> str:
        """EdgeAttributeDef → owl:AnnotationProperty（rdfs:seeAlso 挂到所属边）"""
        local = f"{self._local_name(edef.name)}_{self._local_name(eattr.name)}"
        lines = [f"shm:{local} a owl:AnnotationProperty ;"]
        lines.append(f'    rdfs:label "{self._escape(eattr.name)}" ;')
        lines.append(f"    rdfs:seeAlso shm:{self._local_name(edef.name)} ;")
        if eattr.description:
            lines.append(f'    rdfs:comment "{self._escape(eattr.description)}" ;')
        lines[-1] = lines[-1].rstrip(" ;") + " ."
        return "\n".join(lines)

    # ─── 持久化 ─────────────────────────────────────────────

    def save(self, ontology: OntologyService, path: str) -> None:
        """导出 Turtle 并写入文件。"""
        turtle = self.export_turtle(ontology)
        with open(path, "w", encoding="utf-8") as f:
            f.write(turtle)
        logger.info("Ontology exported to %s (%d bytes)", path, len(turtle.encode("utf-8")))
