"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.19.1"
__version_info__ = (5, 19, 1)
__version_name__ = "fix: cdlib社区检测兼容 + Louvain next_comm 边界修复 (design_cdlib_fix v2)"
__release_date__ = "2026-07-31"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心变更 — 本体系统三大差距补齐 (2026-07-31):
  • 新增 core/ontology_owl.py: OntologyService → OWL/RDF Turtle 导出
    (owl:Class / DatatypeProperty 作用域命名 / ObjectProperty / AnnotationProperty,
     AttrType 9 类型完整映射)
  • 新增 core/ontology_matcher.py: 跨系统本体匹配 (exact/lexical/structural,
     自匹配 100% exact, max_types=100 车挡器)
  • 增强 core/relation_extractor.py: LLM 混合抽取 (extract_async + extract_hybrid,
     DynamicRelationInfo 缓存, confidence 回退 0.75, 防污染不写入 OntologyService)
  • API: GET /ontology/export | POST /ontology/match | POST /ontology/relations/extract
  • 三体协奏管道交付 (CC设计审查 11 问题 → OpenCode 编码 → Codex 审核修复)

📦 依赖变化:
  • 新增: rdflib (仅测试用 Turtle 语法验证)

📌 前置 (v5.18.x):
  • 图引擎: RyuGraph → GraphLite (GQL 动态属性, schema 运行时演进)
  • 多协议网关: MCP :8002 | A2A :8001 | ACP :8770 | HTTP :8000 | CLI 终端

⚙️ 部署: deploy.sh 一键部署 | Docker 就绪"""
