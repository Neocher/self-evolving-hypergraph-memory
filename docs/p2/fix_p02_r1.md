# P2 R1 修复任务书（OpenCode — Codex R1 终审缺陷修复）

## 背景

Codex R1 终审判定需修改（1 P0 + 2 P1 + 1 P2 + 2 P3，全部实现级直接修）。修复后须再派 Codex 复核闭环。

## 缺陷清单与修法

### P0-1 🔴：attr_ops 生产链路死代码（OntologyEvolution.evolve 永不传 distinct_attrs）
- 位置：`core/ontology_evolution.py:428/:430`（evolve 调 evolve_once 无 distinct_attrs）+ `core/dream_pipeline.py:796`（生产经 .evolve() 进入）+ `graph/graphlite_store.py:933`（get_distinct_attr_names 无生产调用）
- 问题：`OntologyEvolution.evolve` 永远以 `distinct_attrs=None` 调 `evolve_once`，生产梦境只经 `.evolve()` 进入 → attr_ops 永不触发
- 修法：`OntologyEvolution.__init__` 注入 `graphlite_store`；`evolve()` 内取 `self.graphlite_store.get_distinct_attr_names()` 后传给 `evolve_once(..., distinct_attrs)`（None/无 store → skip attr_ops）

### P1-1 🟠：别名扩展收不到非现存 attr_name 的 term
- 位置：`retrieval/query_router.py:2375/:2382`（_extract_property_terms 只提取硬编码映射/现存 attr_name）+ :2430/:2435（_expand_attr_aliases 在提取后）+ 测试 :284-285（把 alias 同时写为真实 attr_name，绕过缺口）
- 问题：若 alias 不是现存 attr_name（如只存 revenue，查 income）或中文别名未硬编码，`_expand_attr_aliases` 根本收不到该 term → 别名学习不生效
- 修法：把 alias 表传入 `_extract_property_terms`，或在提取前先用 alias 表从 query 识别候选 term（提取阶段做别名匹配）；新增"alias 不是现存 attr_name"的公共入口测试

### P1-2 🟠：顶层 attr_aliases 泄漏进类型决策命名空间
- 位置：`core/ontology_evolution.py:274`（_build_prompt 把 current.items() 全当类型渲染）+ :343（_apply_merge 未阻止 type="attr_aliases"）+ :210/:218（_apply_new_type）
- 问题：`_build_prompt` 渲染伪类型 attr_aliases；`_apply_merge` 可把 alias 表当类型 dict 写入 conflict_keys → 静默污染 extended JSON
- 修法：定义保留键集合 `_RESERVED_KEYS = {"attr_aliases"}`，`_build_prompt` 过滤、`_apply_merge`/`_apply_new_type` 拒绝该目标名（type == reserved → skip）

### P2-1 🟡：集成测试绕过生产入口
- 位置：`tests/test_schema_attr_ops.py:206/:221/:238`（直调内部函数 + 显式传 distinct_attrs）
- 修法：增加走 `OntologyEvolution.evolve` 的集成测试（断言生产路径能取 distinct 清单并落盘）；保留单元测试但补生产入口覆盖

### P3-1 ⚪：canonical 自身进 alias 表
- 位置：`core/ontology_evolution.py:233/:248/:262`（_apply_attr_ops 未过滤 canonical in aliases）
- 修法：`non_generic` 前过滤与 canonical 相等的 alias（可加测试）

### P3-2 ⚪：1030 passed 无独立 pytest 原始日志
- 位置：`/tmp/oc_p02.log:102`（仅 OpenCode 报告内嵌输出）
- 修法：重跑全量测试保留原始输出文件（如 /tmp/pytest_p02_r1.log），作为独立证据

## 验收标准（AC）

1. P0-1：走 OntologyEvolution.evolve 生产路径能取 distinct 清单并触发 attr_ops
2. P1-1：新增"alias 不是现存 attr_name"公共入口测试（query 含 income 命中 canonical revenue）
3. P1-2：attr_aliases 不泄漏进类型决策（_build_prompt 过滤 + merge/new_type 拒绝）
4. P3-1：canonical 自身不进 alias 表
5. 全量测试通过（`-p no:randomly --deselect tests/test_core_engine.py::TestTauDecay::test_decay_threshold_candidate`）+ 独立原始日志

## 关键约束

- 只改任务相关文件，先 read_file 确认实际结构再改
- 版本不 bump（v5.50.0 未发布，同天修复）
