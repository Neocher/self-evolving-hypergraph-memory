# P2 R1 复核任务书（Codex — Schema-AttrOps 实施终审）

## 背景

SHM v5.50.0 Schema-AttrOps（P2 Schema 演化深化 P0）实施完成。CC 设计审查（design_p02.md，决策 1-6 + 关键假设 1-6）→ OpenCode 实施（implement_p02.md）→ 本任务：Codex 终审。

**设计核心**：属性别名合并 + 中文映射学习的零回归最小闭环——distinct attr_name 查询喂 LLM → attr_aliases 写入 extended JSON → `_property_temporal_retrieve` 通道内 `_expand_attr_aliases` 归一化 query 扩展 → QueryRouter 注入接线。**砍掉**：属性分裂/废弃/值冲突（无消费方伪需求）。

## 实施内容（待复核）

1. `graph/graphlite_store.py:933` `get_distinct_attr_names()`：只读查询（query_cypher 永不抛异常契约，失败 → []）
2. `core/ontology_evolution.py:225` `_apply_attr_ops(parsed, current, distinct_attrs)`：attr_ops 数组字段处理（max 1/轮 + canonical ∈ distinct_attrs 守卫 + 泛词守卫）+ evolve_once 注入 distinct 清单
3. `retrieval/query_router.py:2335` `_expand_attr_aliases(terms, aliases)` 纯函数 + :2435 通道内消费（_extract_property_terms 后、_attr_name_matches 前插入）
4. `retrieval/query_router.py:264/285` QueryRouter 注入 attr_aliases + `api/app.py:513` 接线（extended JSON 顶层 attr_aliases）
5. `tests/test_schema_attr_ops.py`：新增测试（distinct 查询 / attr_ops 守卫 / alias 扩展 / 通道内消费走公共入口）

## 复核要求（read_file 静态分析，不实测）

按 AGENTS.md 审计要求：

1. **追踪调用链**：`_expand_attr_aliases` 是否在 `_property_temporal_retrieve` 内正确位置插入（_extract_property_terms 后、_attr_name_matches 前）？attr_aliases 注入链（app.py → QueryRouter.__init__ → self._attr_aliases）完整？
2. **零回归契约**：attr_aliases 为空/None 时 `_expand_attr_aliases` 恒等短路？行为与现状逐字节等价？
3. **守卫正确性**：`_apply_attr_ops` 的 canonical ∈ distinct_attrs 守卫、泛词守卫、max 1/轮——实现与 CC 设计一致？
4. **纯增量保证**：`_expand_attr_aliases` 只可能扩出更多 canonical 候选（多命中），不会少命中？
5. **get_distinct_attr_names**：GQL 正确（PropertyVerNode.attr_name distinct）？query_cypher 永不抛异常？
6. **evolve_once 集成**：attr_ops 与类型决策正交（可同轮）？_build_prompt 注入 distinct 清单？
7. **版本四处**：5.50.0 Schema-AttrOps 一致 + VERSION_SUMMARY v5.50.0 段？
8. **全量证据**：1030 passed 是否可信（OpenCode 报告 + 测试日志）？

## 输出格式

- 判定：通过 / 需修改
- 缺陷清单（🔴 P0 / 🟠 P1 / 🟡 P2 / ⚪ P3，含文件:行号 + 证据）
- 修复建议（具体修法）
