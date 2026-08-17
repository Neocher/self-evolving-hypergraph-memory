# P0-1 修复任务书（OpenCode Phase 2b — Codex 终审缺陷修复）

Codex 终审（task_5254a9632ad1）判定需修改，5 个缺陷（2 P1 + 3 P2）。全部实现级，直接修。

## 缺陷清单与修法

### P1-1：时间维生产链路未注入事件时间，at_time 检索失效
- 位置：`core/entity_resolver.py:330/354`、`core/relation_extractor.py:51`
- 问题：`update_properties_from_triples` 只传 subject/attr/value，`_update_property_version` 未传 valid_from 时回落 time.time()。RelationExtractor 的 ACQUIRED 只提金额不提 "in 2014" 时间。生产 valid_from 全是写入时刻，"2021 年的收入"检索全跳过。
- 修法：**在 relation_extractor 解析文本中的年份**（正则 `(?:19|20)\d{2}` 或 "in YYYY"），写入 RelationTriple.attributes（如 `attr_year`），`update_properties_from_triples` 显式传 valid_from 给 `_update_property_version`。
- 验证：HTTP 测试增加 "in 2014" → valid_from≈2014 断言。

### P1-2：实体键精确匹配不一致，属性召回静默漏检
- 位置：`core/relation_extractor.py:51`、`core/entity_resolver.py:332`、`retrieval/query_router.py:1590`、`graph/graphlite_store.py:716`
- 问题：写侧用三元组原始 subject（"Apple Inc"）作 entity_id；读侧 `_extract_query_entities` 只提取大写词序列（"Apple 收入"→"Apple"）。`IN ['Apple']` 精确匹配无法命中 "Apple Inc"，小写 apple 不提。
- 修法：读写两侧共用同一套实体归一化。查询侧候选经规范化（小写化 + 去尾词 Inc/Corp/Ltd/Company 等）后与写入 entity_id 做包含匹配（`CONTAINS` 或正则），或查询候选直接对 store 现有 entity_id 做前缀匹配。
- 验证：补 "写入 Apple Inc → 查询 Apple" 集成测试。

### P2-1：at_time 年份语义取 1 月 1 日，年中生效版本漏掉
- 位置：`retrieval/query_router.py:1631/1660`
- 问题：`datetime(year, 1, 1).timestamp()` 作 at_ts，valid_from > at_ts 全跳过；"2021 年收入"应取 2021 年内最新版。at_time 分支未校验 expired_at，可能返回目标时点已过期的旧版。
- 修法：年份查询用年末时间（Dec 31 23:59:59）或解析完整日期；at_time 分支同时要求 `expired_at IS NULL OR expired_at > at_ts`。

### P2-2：相对时间词被错误归为 latest
- 位置：`retrieval/query_router.py:176/1627`
- 问题：`_time_keywords` 含"昨天/earlier"等相对时间词，但 `_property_time_mode` 命中返回 latest（取当前最新）而非历史版本。
- 修法：相对时间词换算成 at_ts 走 at_time；暂不支持相对时间解析则从 latest 命中集合剔除。

### P2-3：属性版本写入非原子，失败可能留半链
- 位置：`graph/graphlite_store.py:631/637/642`、`api/routes/write.py:227`
- 问题：新版本 INSERT 成功后，旧版本 SET expired_at 或 SUPERSEDES 边插入失败，异常被 `_run_entity_resolver` 吞掉 → 新版本已存在但旧版本未过期/血统边缺失。
- 修法：失败路径补偿（记录已插入 pid，异常时清理或补齐旧版本过期标记）；至少 old_id/new_id 记入告警日志。

## 验收标准
1. 5 个缺陷全部修复
2. 新增/更新测试覆盖：年份解析（in 2014→valid_from）、实体归一化（Apple Inc→Apple）、at_time 年末语义 + expired_at 校验、相对时间词不误归 latest、非原子写补偿
3. `pytest tests/test_property_temporal.py -x -q` 全过
4. 全量 `pytest tests/ -q --deselect tests/test_core_engine.py::TestTauDecay::test_decay_threshold_candidate` 909+ 通过
5. 版本保持 5.47.0（不额外 bump，同一功能迭代）
