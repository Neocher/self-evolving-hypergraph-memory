# P0-1 终审任务书（Codex Phase 3）

审核 P0-1 实体-属性-时间三维建模的实施质量。这是对标 MindMemOS MindSchema 的重大架构改动（5 文件 ~250 行 + 14 测试）。

## 背景
- CC 设计审查（task_3d6d348b6d89）5 决策点通过
- OpenCode 实施完成，全量 909 passed（基线 894 + 新增 15）
- 版本已 bump 5.47.0（Entity-Property-Time）

## 改动文件
1. `core/ontology_v2.py`：AttributeDef + `temporal: bool = False`（决策 4）
2. `graph/graphlite_store.py`：`create_property_version` / `get_latest_property_version` / `get_property_versions` / `prune_property_versions` + `PROPERTY_MAX_VERSIONS=8`（决策 1+5）
3. `core/entity_resolver.py`：`update_properties_from_triples` / `_update_property_version`（决策 2 编排）
4. `api/routes/write.py`：`_run_entity_resolver` 接入（决策 2）
5. `retrieval/query_router.py`：`_property_temporal_retrieve` / `_extract_query_entities` / `_property_time_mode` / `_pick_property_versions` + `_PROPERTY_BOOST=0.6`（决策 3）
6. `tests/test_property_temporal.py`：14 用例

## 审核重点（AGENTS.md 审计要求）

1. **追踪调用链**：write.py 的 `_run_entity_resolver` → entity_resolver 版本编排 → graphlite_store 原语 → 检索侧 `_property_temporal_retrieve` 完整链路是否通畅
2. **静默失败专项**：
   - 关键字参数失配（改签名后调用方核对）
   - GraphLite b64：中文 value 是否转义（`_gql_value`）
   - 多语句静默截断：`create_property_version` 的 3 条 execute 是否独立
   - 无 MERGE 幂等：查存在→插 两段式是否正确
3. **测试假绿**：单测是否走公共入口（retrieve()/endpoint），还是直调内部方法
4. **时间语义**：`_property_time_mode` 的 latest/at_time/current 三模式是否正确；`_pick_property_versions` 的 at_time 逻辑（valid_from ≤ at_ts）
5. **score 钳制**：`_PROPERTY_BOOST=0.6` 相对尾分缩放后是否突破 EpisodicResult.score le=1.0 约束（v5.39 教训：boost 乘法要 min(1.0) 钳制）
6. **版本四处同步**：5.47.0 == pyproject == VERSION == README
7. **裁剪正确性**：prune 保留最近 N=8 的逻辑（get_property_versions ASC + 删最旧）

## 输出格式
- 问题清单：严重度（🔴/🟠/🟡）+ 文件:行号 + 为什么 + 建议修法
- 终审判定：通过 / 需修改
