# P0-1 实施任务书（OpenCode Phase 2）

基于 CC 设计审查（task_3d6d348b6d89）的 5 个决策点实施。只改以下 5 个文件，每步验证。

## 背景
SHM 与 MindMemOS 差距三大结构性缺失之一：实体-属性-时间三维建模。当前消息级扁平存储缺「实体中心 + 属性时间版本链」。目标：在现有超边/本体基础上实现属性时间版本链，最小正解 ~200-250 行。

## 决策要点（CC 已确认）

### 决策 1：存储模型
- 新增 `PropertyVerNode` 节点：`{id, entity_id, attr_name, value, valid_from, expired_at}`
- 旧版本 `expired_at` 打标 + `SUPERSEDES` 边指向前驱（复用 `archive_node` 血统边范式，graphlite_store.py:552-582）
- `entity_id` 直接落节点字段（查询 `entity_id = $x AND attr_name = $y`）

### 决策 2：写入路径
- **entity_resolver 编排 + graphlite_store 原语**
- 接入点：`api/routes/write.py` 的 `_run_entity_resolver` 闭包（已在写线程 qsubmit 内）
- 属性值来源：**relation_extractor 的 RelationTriple.attributes**（如 attr_amount/attr_year，确定性零 LLM）
- write_reconciler 不碰（语义不同）

### 决策 3：检索接入
- 新增独立 `_property_temporal_retrieve` 通道（仿 `_community_expansion` append 模式）
- 复用 `_time_keywords`（query_router.py:159-175）判定时间意图
- 时间意图：query 含"最近/现在" → 取最新版本（expired_at IS NULL + valid_from 排序）；含具体时间 → 取对应版本
- `_finish` 追加一行

### 决策 4：schema 迁移
- `core/ontology_v2.py` AttributeDef 末尾追加 `temporal: bool = False`
- 零迁移脚本（纯 dataclass + 全关键字构造 + OntologyService 纯内存）

### 决策 5：版本约束
- 每 (entity_id, attr_name) 保留最近 N=8 版，写时惰性裁剪（创建新版本时超限 DETACH DELETE 最旧）
- 不复用 tau 衰减

## 验收标准（AC）

1. `pytest tests/` 全量通过（新增测试 ≥ 10 用例）
2. 新增测试覆盖：版本创建、supersedes 链、时间检索（最近 vs 具体时间）、N=8 裁剪、向后兼容（temporal 默认 False）
3. 语法检查无新错误
4. **版本四处同步**：`shm/_version.py` + `pyproject.toml` + `VERSION` + `README.md` bump 到 **5.47.0**，version_name="Entity-Property-Time"，release_date="2026-08-16"

## 实现顺序
1. ontology_v2.py（+temporal 字段）
2. graphlite_store.py（property CRUD 原语 + SUPERSEDES + 裁剪）
3. entity_resolver.py（版本编排）
4. write.py（接入 _run_entity_resolver）
5. query_router.py（_property_temporal_retrieve + _finish）
6. 测试 + 版本同步

## 必须规避的坑（AGENTS.md 静默失败清单）
- GraphLite b64：中文 value 存 `{b64}` 透明编解码
- 多语句静默截断：GraphLite 一条 execute 多条语句只执行第一条，**必须拆开验证**
- 无 MERGE：幂等靠「查存在→插」两段式
- 单测必须走公共入口（retrieve()/endpoint），不能直调内部方法
- 测试用真实 SDK 异常，不 mock 内置类
