# SHM 本体系统三大差距补齐设计 v2（v5.19.0）

## 背景

Ontology v2 (core/ontology_v2.py) 已有：动态类型注册、属性类型系统、边约束、类型继承、写时/读时验证。
Entity Discovery (core/entity_discovery.py) 已实现自动本体发现。
三大缺口：
1. **OWL 互操作层** — 无法导出为标准 OWL/RDF
2. **跨系统本体匹配 (Ontology Matching)** — 无法对齐两个独立本体
3. **LLM 驱动关系抽取深度** — Relation Extractor (230行) 仅正则+固定置信度0.85

## 目标版本

v5.19.0 — 三个新模块 + 现有模块增强，全部通过 pytest

---

## 模块1: core/ontology_owl.py — OWL 互操作层（新文件 ~220行）

```python
class OntologyOwlExporter:
    def export_turtle(self, ontology: OntologyService) -> str
    def save(self, ontology: OntologyService, path: str) -> None
```

**映射规则（含 CC 审查修复）：**

| SHM 概念 | OWL 表达 | 说明 |
|:---------|:---------|:-----|
| EntityTypeDef | `owl:Class` | 类型继承 → `rdfs:subClassOf` |
| AttributeDef | `owl:DatatypeProperty` | **IRI 作用域限定**：`{TypeName}_{AttrName}`，避免同名属性 domain 交集语义错误 |
| EdgeTypeDef | `owl:ObjectProperty` | source/target 白名单 → rdfs:domain/rdfs:range |
| EdgeAttributeDef | `owl:AnnotationProperty` | 附加到 ObjectProperty 上 |
| AttrType 映射 | 见下方完整表 | 含所有枚举值 |

**AttrType 完整映射表（修复 #3）：**
```
string         → xsd:string
integer        → xsd:int
float          → xsd:double
boolean        → xsd:boolean
date           → xsd:date
datetime       → xsd:dateTime
string[]       → rdf:Seq
text_embedding → xsd:string + rdfs:comment "SHM-specific: text_embedding"
entity_ref     → xsd:anyURI + rdfs:comment "SHM-specific: entity_ref"
```

**同名属性处理（修复 #2）：** 属性 IRI 使用 `{TypeName}_{AttrName}` 作用域限定；当同一 `{AttrName}` 存在于多个类型时，对每个类型独立输出带作用域的 DatatypeProperty，并在 Turtle 头加 `rdfs:comment` 警告。

**导出内容：** 命名空间前缀（owl/rdfs/xsd/rdf）+ owl:versionInfo + 类定义 + 属性定义 + 边定义。

---

## 模块2: core/ontology_matcher.py — 本体匹配（新文件 ~200行）

```python
@dataclass
class MatchResult:
    source: str; target: str; score: float; method: str
    # method ∈ {exact, lexical, structural}
    # exact → score=1.0（二元，非阈值）
    # lexical → score∈[0,1]（difflib.SequenceMatcher + 别名归一化）
    # structural → score∈[0,1]（邻居类集合 Jaccard）

class OntologyMatcher:
    def __init__(self, max_types: int = 100):  # O(N²) 车挡器（修复 #7）
    def match(self, src: OntologyService, dst: OntologyService) -> List[MatchResult]:
        # 1. 名称精确匹配：method=exact, score=1.0（修复 #4：exact 是二元条件，非范围）
        # 2. 词法相似度：score >= 0.75 时收录
        # 3. 结构相似度：仅当 N <= max_types 时计算；超过则跳过并 warning（修复 #7）
    def match_report(self, src, dst) -> Dict
```

**自匹配验收**（修复 #4）：`all(m.method == "exact" and m.score == 1.0 for m in results)`

---

## 模块3: core/relation_extractor.py — LLM 增强（现有文件 +~150行）

**修复 #1（同步/异步拆分）：** 保持 `extract()` 同步向后兼容，新增 `async extract_async()`：

```python
class DynamicRelationInfo:  # 修复 #8：轻量缓存类型，不承载 EdgeTypeDef 约束
    name: str
    description: str = ""
    discovery_count: int = 0
    last_seen: float = 0.0

class RelationExtractor:
    def __init__(self, llm_client=None):  # llm_client 可选注入，None=仅正则
        self._dynamic_relations: Dict[str, DynamicRelationInfo] = {}

    def extract(self, text: str) -> List[RelationTriple]:
        """纯同步，仅正则，向后兼容（签名不变）"""

    async def extract_async(self, text: str, ontology: OntologyService = None) -> List[RelationTriple]:
        """异步混合：先正则快速路径，未命中片段送 LLM"""

    def extract_hybrid(self, text: str) -> List[RelationTriple]:
        """同步包装器：正则结果 + 上次 LLM 缓存的动态关系匹配（无 LLM 调用）"""
```

**LLM prompt 规范（修复 #5）：** 要求 LLM 输出：
```json
[{"subject": "...", "relation": "...", "object": "...", "confidence": 0.0-1.0}]
```
- 解析成功且 confidence 在 [0,1] → 使用 LLM 置信度
- 缺失/非法 confidence → 回退 0.75，method="llm"
- 新关系注册到 `_dynamic_relations`（name/description/discovery_count/last_seen），**不写入 OntologyService**（防污染）

---

## API 暴露（api/routes/ontology.py）

```python
GET  /ontology/export?format=turtle   # 修复 #6：GET（无副作用），Content-Type: text/turtle
POST /ontology/match                  # body: {other_ontology: {...}} → 匹配报告
POST /ontology/relations/extract      # body: {text} → 混合抽取结果（同步包装器）
```

---

## 验收标准（含测试策略）

1. `pytest tests/` 全部通过
2. 新增 `test_ontology_owl.py`：
   - 导出 Turtle 能被 `rdflib` 解析（语法验证）
   - 含全部 9 种 AttrType 的映射覆盖
3. 新增 `test_ontology_matcher.py`：
   - 自匹配：`all(m.method=="exact" and m.score==1.0)`
   - 词法：改名副本匹配 lexical
4. 新增 `test_relation_llm.py`（修复 #9）：
   - **单元测试**：mock LLM 响应（`mock.chat.return_value = json`），验证解析逻辑
   - **集成测试**：`@pytest.mark.integration`，无 DEEPSEEK_API_KEY 时 skip
5. Health API 版本号 v5.19.0

---

## 不做的事（范围控制，含 CC 补充）

- 不做 OWL 推理引擎（只导出，不推理）
- **不做 OWL 导入/RDF 解析（仅导出；OWL→Ontology v2 映射留待 v5.20）**（修复 #11）
- 不做完整 Ontology Matching 评测基准（先做基础相似度）
- 不自动把 LLM 关系写入 Ontology v2（防污染，人工确认）
- **边属性导出为 AnnotationProperty，不做复杂 OWL 属性链**（修复 #10）

---

## CC 审查修复清单（v1 → v2）

| # | 严重度 | 修复 |
|---|:------:|:-----|
| 1 | 🔴 | extract() 同步向后兼容 + async extract_async() 混合 |
| 2 | 🔴 | 属性 IRI 作用域命名 {TypeName}_{AttrName} |
| 3 | 🔴 | AttrType 完整映射（datetime/text_embedding/entity_ref） |
| 4 | 🟠 | exact = 二元 score=1.0，非阈值 |
| 5 | 🟠 | LLM prompt 要求输出 confidence，非法则回退 0.75 |
| 6 | 🟠 | GET /ontology/export |
| 7 | 🟠 | max_types=100 车挡器 |
| 8 | 🟠 | DynamicRelationInfo 轻量缓存类型 |
| 9 | 🟡 | 单元测试 mock LLM + integration 标记 |
| 10 | 🟡 | 边属性 → AnnotationProperty |
| 11 | 🟡 | 明确不做 OWL 导入（留 v5.20） |
