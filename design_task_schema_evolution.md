# CC 设计任务书：SHM Schema 自进化 P0-② —— 实体属性与关系的自我进化

## 背景

SHM（自演化超图记忆系统）已完成两阶段 schema 基础：
- **v6.2.0 实体落库闭环（P0-①）**：`graph/overgraph_store.py` 新增 EntityNode 体系——`create_entity`（sha1 确定性 key 幂等）/`get_entity`/`get_entity_by_id`/`link_entity_to_episode`（EntityNode→EpisodeNode MENTIONS 边）/`get_entity_episodes`；`core/dream_pipeline.py` 的 `_persist_entities` 在梦境 PERSIST 阶段消费 `community.entity_links` 落库。
- **阶段4-1 Schema 模式蒸馏**：`core/schema_distiller.py` + `retrieval/query_router.py` 的 `_schema_recall`（SchemaNode :Conceptual 标签）已存在。

## 目标（本任务设计范围）

**Schema 自进化 P0-②：实体【属性】与【关系】的自我进化**——从 Episode 内容中自动提取实体属性和实体间语义关系，落库 EntityNode，带证据追踪与置信度累积，随数据演化（新增/强化/修正），并被检索通道利用。

1. **属性自进化**：实体属性（如 Person 的 title/company、Organization 的 industry/founded）从 Episode 内容提取→写入 EntityNode props→按证据累积置信度→演化（新证据强化；冲突证据修正/多值）
2. **关系自进化**：实体间语义谓词关系（FOUNDED/LEADS/WORK_AT/LOCATED_IN/ACQUIRED 等）从内容抽取→EntityNode-EntityNode 谓词边→置信度/权重→演化
3. **闭环**：梦境管道集成（扩展 `_persist_entities` 或新增 persist 步骤）+ 检索通道（属性/关系辅助 `get_entity_episodes` 类召回）

## 现状盘点（先 read_file/grep 核实再设计）

- `graph/overgraph_store.py`：EntityNode 5 方法 + `create_schema_node`/`query_schema_nodes`/`create_hyperedge_node`/`link_hyperedge_member` + 幂等 upsert 模式（`_locked_upsert_node`）
- `core/entity_discovery.py`（v5.8 本体自发现）：`POST /ontology/discover` / `discover/apply`，EntityTypeDef/AttributeDef/EdgeTypeDef（core/ontology_v2.py）
- `core/relation_extractor.py`（v5.8 关系抽取）：10 种谓词正则（FOUNDED/LEADS/ACQUIRED/LOCATED_IN 等），英文+中文双模式，**纯规则无 LLM**
- `core/evidence_tracker.py`（v5.8 置信度）：BLAKE3 内容哈希去重 + evidence_count 累积 + `evidence_tracker.json` 持久化（Kuzu 无 ALTER 时代的做法）
- `core/dream_pipeline.py`：`_persist_entities`（P0-①）在 PERSIST 阶段调用
- 后端：OverGraph v6.0.0（GraphLite 已移除），`dense_vector_dimension=512`，bge-m3 ONNX

## 硬约束（必须遵守）

1. **不依赖 LLM**：属性/关系抽取纯规则 + 正则 + embedding 相似度（v5.8 教训：LLM 不稳定）
2. **证据与置信度**：每个属性/关系必须带证据来源（Episode ids）+ 置信度（evidence_count/来源数）；**证据分区**（不同来源独立计票，禁混池）
3. **OverGraph 契约**：elementKey 幂等（sha1 确定性 key 模式）、MERGE 主键、**无 ALTER TABLE**（新增结构用外部 JSON 或复用 props）
4. **梦境集成**：属性/关系持久化走 write_queue（`_persist_async`），失败降级不阻塞（degraded 自愈语义）
5. **检索通道**：新增数据必须可被检索利用（属性匹配/关系邻居辅助召回），不是只写不读
6. **测试走公共入口**（单测禁直调内部方法）；新增测试用例覆盖：属性提取、关系抽取、置信度累积、幂等重跑、冲突处理

## 输出格式

设计文档（写入 `design_schema_evolution.md`）：
1. 模块/文件结构（新增哪些文件、改动哪些现有文件、函数签名）
2. 存储 schema：EntityNode props 结构、属性/关系证据存储（节点属性 or 外部 JSON）、谓词边 label 设计
3. 属性提取规则设计（哪些模式、如何从 Episode 内容定位实体属性）
4. 关系抽取规则设计（复用 relation_extractor 哪些、新增哪些、中文+英文）
5. 置信度累积策略（阈值、演化规则：出现→确认→强化/修正）
6. 梦境管道接入点（代码位置、函数名、write_queue 顺序）
7. 检索通道设计（如何辅助 `_entity_expansion`/`_scope_retrieve`）
8. API 端点（如 `POST /ontology/evolve`）
9. 测试计划 + AC 验收标准
10. 一句话总结 + 实施量级估算（LOC/小时）

中文输出。
