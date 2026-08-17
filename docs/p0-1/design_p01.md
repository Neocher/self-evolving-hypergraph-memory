# P0-1 设计任务书：实体-属性-时间三维建模（对标 MindMemOS MindSchema）

## 背景

SHM 与 MindMemOS 差距深度研究（2026-08-16）确认三大结构性缺失之一：**实体-属性-时间三维建模**。SHM 当前是消息级扁平存储 + 本体（Ontology v2）+ 超边（Episode/Semantic/Temporal），缺 MindMemOS 的「实体中心 + 属性时间版本链」结构。实测影响：LoCoMo 答案 top-10 召回仅 15.8%（多跳/开放域题需跨消息聚合实体属性）。

MindMemOS MindSchema 四步：episode 分割 → 记忆生成（抽取实体/属性值 + 时间解析）→ 实体融合（等价合并/修订属性/标记过期）→ 图合并。

## 现有基础（SHM 已具备）

- `core/ontology_v2.py`：EntityTypeDef（name/parent/attributes）+ AttributeDef（AttrType 含 DATE/DATETIME）+ EdgeTypeDef
- `core/entity_discovery.py`：NER 模式（EN_PERSON/EN_ORG/CN_ORG...）+ EntityProposal + TypeProposal + 聚类消歧
- `core/entity_resolver.py`：实体等价消歧（342 行）
- `core/relation_extractor.py`：关系抽取（478 行）
- `graph/hyperedge.py`：三种超边（EPISODE/SEMANTIC/TEMPORAL）+ supersession_of 字段
- `graph/graphlite_store.py`：create_episode / create_hyperedge_node
- `retrieval/query_router.py`：`_entity_match`（实体匹配通道）+ `_apply_time_decay` + 三路融合（vector/bm25/entity）
- `core/user_profile.py`：用户画像（240 行，score×1.2 boost 等）

## P0-1 目标（最小正解，Karpathy 简洁优先）

在现有超边/本体基础上，实现「实体-属性-时间」的**属性时间版本链**，不改动消息级存储主路径：

1. **属性时间戳**：AttributeDef 增加可选 `temporal: bool` 标记；写入实体属性值时带 `valid_from`（创建时间）与 `supersedes`（前驱属性 ID）
2. **属性版本链**：同一实体属性更新时创建新版本节点（property_ver 节点），旧版本标记 `expired_at`，通过 `SUPERSEDES` 边连接前驱/后继（复用 graphlite 边）
3. **时间感知检索**：`_entity_match` 查询时对 temporal 属性按时间意图过滤（query 含"现在/最近"取最新版本；含具体时间取对应版本）——EgoCITE 时间意图思路
4. **实体融合增强**：entity_resolver 合并时属性取「最新版本 + 非过期」值（MindMemOS entity fusion 思路）

## 你的任务（CC 设计审查）

只做 read_file/grep 静态分析，**不编译不实测**。审查以下设计决策：

1. **存储模型**：属性版本用「独立 property_ver 节点 + SUPERSEDES 边」vs「EntityNode 属性字段直接覆盖」——哪个更符合 SHM 现有 GraphLite 约束（b64、多语句截断、无 MERGE）？给出取舍。
2. **写入路径**：新版本创建应在哪个模块（entity_resolver 还是 graphlite_store）？现有 write 路径（write_queue/reconciler）如何接入最小化？
3. **检索接入**：`_entity_match` 现有实现如何扩展时间过滤？还是新增独立 `_property_temporal_retrieve` 通道更简洁？
4. **schema 迁移**：AttributeDef 加 `temporal` 字段是向后兼容的吗（dataclass 默认值）？已有 ontology 注册的类型是否需要迁移脚本？
5. **版本约束**：每实体属性版本上限（防无界增长）？过期清理策略（复用 tau 衰减还是独立）？

## 输出格式

- 对 5 个决策点逐一给出：推荐方案 + 理由 + 涉及文件
- 标注风险（🔴 结构性 / 🟠 实现级 / 🟡 轻微）
- 一句话总结：P0-1 最小正解是否可行，改动量估算（文件数/行数）
