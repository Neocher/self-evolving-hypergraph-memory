# P2 设计任务书 — Schema 演化深化（CC 审查）

## 背景

SHM 沿路线图推进：P0-1 v5.47（实体-属性-时间）→ P0-2 v5.48（Agentic）→ P1 v5.49（MESA 梦境→检索闭环，58.0%）→ **P2 Schema 演化（93%+，对标 MindMemOS 94.03）**。

P2 核心：**Schema 自演化深化——从类型级到属性级闭环**。对标 MindMemOS `_apply_property_operations`（属性操作）与 SchemaSearchExpander（schema 驱动检索）。

## 现状（已实测确认，勿重跑）

### v5.38 Ontology-Evolution（core/ontology_evolution.py，359 行）
- `evolve_once`（:272）：SYNTHESIZE 后聚合社区 topics/report → **1 次 LLM 决策** → new_type / merge_existing / skip
- `_apply_new_type` / `_apply_merge`：类型级 create/update（conflict_keys 去重，max 1 类型/轮，守卫：≥2 非泛 conflict_keys，跨类型 key 冲突 skip）
- `classify_with_extended`（:328）：合并后 first-match-wins 分类（遍历 {**extended, **ONTOLOGY_TYPES}）
- `OntologyEvolution`（:337）：extended_path + llm_client 包装
- 落盘：`data/ontology_extended.json`（gitignore，原子写 temp+rename，失败 → skip 不声称成功）
- 已知坑（v5.38 实录）：dict 展开顺序原生优先 `{**extended, **ONTOLOGY_TYPES}`；merge 原生类型 skip 防污染；max 1/轮

### 现有 schema 相关（P0-1 实体-属性-时间）
- PropertyVerNode {{id, entity_id, attr_name, value, valid_from, expired_at}} + SUPERSEDES 血统（属性级版本链）
- entity_resolver 编排：RelationTriple.attributes → attr_name 派生 {{relation}}_{{key}}
- 检索侧 `_property_temporal_retrieve`：属性时间通道

### P2 缺口（对标 MindMemOS 94.03）
1. **属性级 schema 演化缺失**：LLM 只决策"新类型/合并类型"，不决策"属性值冲突 → 属性合并/分裂/废弃"（MindMemOS `_apply_property_operations`）
2. **无反馈闭环**：演化决策不感知检索命中率（P1 MESA 已有 mesa_hit_count 模式可复用）
3. **schema 消费浅**：classify_with_extended 只做分类，不驱动检索通道（MindMemOS SchemaSearchExpander 差距）

## 设计目标（P0 最小闭环 + 零回归）

1. **属性级演化**：LLM 决策扩展到属性操作——发现属性值冲突（同实体同属性不同值）/属性冗余（两属性语义等价）/属性分裂（一属性拆多义）→ 更新 extended schema（属性级 conflict_keys/别名合并）
2. **反馈闭环**：演化后的 schema 检索命中率（property_temporal 通道）→ 驱动后续演化决策（复用 P1 mesa_avg_score 模式）
3. **检索消费深化**：schema 属性别名/合并 → 检索时 query 扩展（属性词 → 别名集），提升属性通道召回
4. **零回归**：默认关闭/LLM 失败降级，不干扰主通道

## 审查要求（只做 read_file/grep 静态分析，不实测）

1. 属性级演化的 LLM 决策结构：如何扩展现有 prompt（summaries + current schema → 类型 + 属性操作）？max 1 操作/轮守卫怎么设计？
2. 属性冲突检测：现有 PropertyVerNode 数据能支撑"同实体同属性不同值"检测吗？还是需要新查询？
3. 属性合并/别名的 schema 表示：extended JSON 加 `attr_aliases` 字段？检索侧怎么消费（query 扩展时机）？
4. 反馈闭环：复用 P1 mesa_avg_score 模式还是新机制？信号源（property_temporal 命中率）怎么统计？
5. 检索消费深化：query 扩展在 QueryRouter 哪个环节做？会污染现有融合通道吗（零回归）？
6. Karpathy 检查：哪些是过度设计（无消费方）？哪些是真需求？

## 输出格式

- 设计决策表（每项：方案 + 理由 + 涉及文件）
- 推荐实施范围（P0/P1/P2 分级）
- 一句话方案摘要
- 关键假设（调用链证据，不臆测）
