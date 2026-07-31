"""
Ontology OWL 导出单元测试
==========================
覆盖 design_ontology_gaps.md v2 模块1：
  · Turtle 输出可被 rdflib 解析（语法验证）
  · 全部 9 种 AttrType 映射覆盖
  · 属性 IRI 作用域命名 {TypeName}_{AttrName}
  · 类型继承 → rdfs:subClassOf
  · 边 → owl:ObjectProperty（domain/range 白名单）
  · 边属性 → owl:AnnotationProperty
  · save() 落盘后可再解析
"""
from __future__ import annotations

import pytest
import rdflib

from core.ontology_owl import OntologyOwlExporter, ATTR_TYPE_TO_XSD
from core.ontology_v2 import (
    AttrType,
    AttributeDef,
    EdgeAttributeDef,
    EdgeTypeDef,
    EntityTypeDef,
    OntologyService,
)

OWL = "http://www.w3.org/2002/07/owl#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
SHM = "http://shm.local/ontology#"
XSD = "http://www.w3.org/2001/XMLSchema#"


def _build_ontology() -> OntologyService:
    """构造覆盖全部 9 种 AttrType 的测试本体。"""
    svc = OntologyService()
    svc.register_entity_type(EntityTypeDef(
        name="Person",
        description="人物个体",
        parent="_BaseNode",
        attributes=[
            AttributeDef(name="full_name", type=AttrType.STRING),
            AttributeDef(name="age", type=AttrType.INTEGER),
            AttributeDef(name="height", type=AttrType.FLOAT),
            AttributeDef(name="is_active", type=AttrType.BOOLEAN),
            AttributeDef(name="birth_date", type=AttrType.DATE),
            AttributeDef(name="last_seen", type=AttrType.DATETIME),
            AttributeDef(name="tags", type=AttrType.STRING_LIST),
            AttributeDef(name="summary_embedding", type=AttrType.TEXT_EMBEDDING),
            AttributeDef(name="employer_ref", type=AttrType.ENTITY_REF),
        ],
    ))
    svc.register_entity_type(EntityTypeDef(
        name="Company",
        description="公司",
        parent="_BaseNode",
        attributes=[
            # 与 Person.full_name 同名 → 触发 IRI 作用域 + 头部警告
            AttributeDef(name="full_name", type=AttrType.STRING),
        ],
    ))
    svc.register_edge_type(EdgeTypeDef(
        name="EMPLOYED_AT",
        description="雇佣关系",
        source_types=["Person"],
        target_types=["Company"],
        attributes=[EdgeAttributeDef(name="since", type=AttrType.DATE,
                                      description="入职年份")],
    ))
    return svc


@pytest.fixture
def ontology() -> OntologyService:
    return _build_ontology()


@pytest.fixture
def graph(ontology: OntologyService) -> rdflib.Graph:
    """导出 Turtle 并用 rdflib 解析成图。"""
    turtle = OntologyOwlExporter().export_turtle(ontology)
    g = rdflib.Graph()
    g.parse(data=turtle, format="turtle")
    return g


def _shm(name: str) -> rdflib.URIRef:
    return rdflib.URIRef(SHM + name)


# ─── Turtle 语法验证 ────────────────────────────────────


class TestTurtleParsable:
    def test_export_parses_as_turtle(self, ontology: OntologyService):
        """导出文本应能被 rdflib 以 turtle 格式解析（无异常）。"""
        turtle = OntologyOwlExporter().export_turtle(ontology)
        g = rdflib.Graph()
        g.parse(data=turtle, format="turtle")
        assert len(g) > 0

    def test_export_contains_ontology_header(self, ontology: OntologyService):
        """应包含 owl:versionInfo 与命名空间前缀。"""
        turtle = OntologyOwlExporter().export_turtle(ontology)
        assert "@prefix owl:" in turtle
        assert "@prefix rdfs:" in turtle
        assert "@prefix xsd:" in turtle
        assert "@prefix rdf:" in turtle
        assert "owl:versionInfo" in turtle

    def test_duplicate_attr_names_emit_warning(self):
        """同名属性跨类型时头部应有 rdfs:comment 警告。"""
        svc = _build_ontology()
        turtle = OntologyOwlExporter().export_turtle(svc)
        assert "Duplicate attribute names across types" in turtle


# ─── 类与继承 ───────────────────────────────────────────


class TestClasses:
    def test_entity_type_is_owl_class(self, graph: rdflib.Graph):
        assert (_shm("Person"), rdflib.RDF.type, rdflib.OWL.Class) in graph

    def test_inheritance_emits_subclassof(self, graph: rdflib.Graph):
        """Person 继承 _BaseNode → rdfs:subClassOf。"""
        assert (_shm("Person"), rdflib.RDFS.subClassOf, _shm("_BaseNode")) in graph


# ─── AttrType 完整映射（9 种）────────────────────────────


class TestAttrTypeMapping:
    def test_all_nine_attr_types_present(self, graph: rdflib.Graph):
        """9 种 AttrType 全部导出为 DatatypeProperty。"""
        attr_names = ["full_name", "age", "height", "is_active",
                      "birth_date", "last_seen", "tags",
                      "summary_embedding", "employer_ref"]
        for name in attr_names:
            prop = _shm(f"Person_{name}")
            assert (prop, rdflib.RDF.type, rdflib.OWL.DatatypeProperty) in graph, \
                f"DatatypeProperty {name} missing"

    def test_attr_range_mapping(self, graph: rdflib.Graph):
        """AttrType → xsd 类型映射逐一验证。"""
        expected_ranges = {
            "Person_full_name": XSD + "string",
            "Person_age": XSD + "int",
            "Person_height": XSD + "double",
            "Person_is_active": XSD + "boolean",
            "Person_birth_date": XSD + "date",
            "Person_last_seen": XSD + "dateTime",
            "Person_tags": "http://www.w3.org/1999/02/22-rdf-syntax-ns#Seq",
            "Person_summary_embedding": XSD + "string",
            "Person_employer_ref": XSD + "anyURI",
        }
        for prop, rng in expected_ranges.items():
            assert (_shm(prop), rdflib.RDFS.range, rdflib.URIRef(rng)) in graph, \
                f"{prop} range != {rng}"

    def test_text_embedding_has_shm_comment(self, graph: rdflib.Graph):
        """text_embedding 应带 SHM-specific 注释。"""
        prop = _shm("Person_summary_embedding")
        comments = set(graph.objects(prop, rdflib.RDFS.comment))
        assert "SHM-specific: text_embedding" in {str(c) for c in comments}

    def test_entity_ref_has_shm_comment(self, graph: rdflib.Graph):
        """entity_ref 应带 SHM-specific 注释。"""
        prop = _shm("Person_employer_ref")
        comments = set(graph.objects(prop, rdflib.RDFS.comment))
        assert "SHM-specific: entity_ref" in {str(c) for c in comments}

    def test_mapping_table_covers_all_enum_values(self):
        """ATTR_TYPE_TO_XSD 应覆盖 AttrType 全部枚举。"""
        assert set(ATTR_TYPE_TO_XSD.keys()) == set(AttrType)


# ─── IRI 作用域命名 ─────────────────────────────────────


class TestIriScoping:
    def test_same_attr_name_scoped_per_type(self, graph: rdflib.Graph):
        """同名属性 full_name 在 Person/Company 各自独立作用域。"""
        assert (_shm("Person_full_name"), rdflib.RDF.type, rdflib.OWL.DatatypeProperty) in graph
        assert (_shm("Company_full_name"), rdflib.RDF.type, rdflib.OWL.DatatypeProperty) in graph
        # 作用域后的属性应带有 domain 限定
        assert (_shm("Person_full_name"), rdflib.RDFS.domain, _shm("Person")) in graph
        assert (_shm("Company_full_name"), rdflib.RDFS.domain, _shm("Company")) in graph


# ─── 边定义 ─────────────────────────────────────────────


class TestEdges:
    def test_edge_is_object_property(self, graph: rdflib.Graph):
        assert (_shm("EMPLOYED_AT"), rdflib.RDF.type, rdflib.OWL.ObjectProperty) in graph

    def test_edge_domain_range_whitelist(self, graph: rdflib.Graph):
        """source/target 白名单 → rdfs:domain/rdfs:range。"""
        assert (_shm("EMPLOYED_AT"), rdflib.RDFS.domain, _shm("Person")) in graph
        assert (_shm("EMPLOYED_AT"), rdflib.RDFS.range, _shm("Company")) in graph

    def test_edge_attribute_is_annotation_property(self, graph: rdflib.Graph):
        """EdgeAttributeDef → owl:AnnotationProperty（修复 #10）。"""
        prop = _shm("EMPLOYED_AT_since")
        assert (prop, rdflib.RDF.type, rdflib.OWL.AnnotationProperty) in graph
        # 附加到所属 ObjectProperty
        assert (prop, rdflib.RDFS.seeAlso, _shm("EMPLOYED_AT")) in graph


# ─── save() ─────────────────────────────────────────────


class TestSave:
    def test_save_writes_parseable_turtle(self, ontology: OntologyService, tmp_path):
        path = tmp_path / "ontology.ttl"
        OntologyOwlExporter().save(ontology, str(path))
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "owl:Ontology" in content
        g = rdflib.Graph()
        g.parse(data=content, format="turtle")
        assert (_shm("Person"), rdflib.RDF.type, rdflib.OWL.Class) in g
