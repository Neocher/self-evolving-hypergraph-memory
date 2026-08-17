# P2 R2 复核任务书（Codex — R1 修复终审）

## 背景

Codex R1 终审判定需修改（1 P0 + 2 P1 + 1 P2 + 2 P3）→ OpenCode R1 修复完成（全量 1039 passed，独立日志 /tmp/pytest_p02_r1.log）。本任务：R2 复核闭环。

## R1 修复内容（待复核）

1. **P0-1** OntologyEvolution 注入 graphlite_store（:431/:434）+ evolve() 取 distinct_attrs 传 evolve_once（:445/:451）——attr_ops 生产链路不再死代码
2. **P1-1** 提取阶段 alias 识别（query_router.py:2444-2445）——alias 不是现存 attr_name 时也能识别 term
3. **P1-2** _RESERVED_KEYS={"attr_aliases"}（:44）+ _apply_new_type 拒绝（:165/:166）+ _apply_merge 拒绝（:220/:221）+ _build_prompt 过滤
4. **P2-1** 新增走 OntologyEvolution.evolve 的集成测试
5. **P3-1** canonical 自身不进 alias 表（non_generic 前过滤）
6. **P3-2** 独立 pytest 原始日志 /tmp/pytest_p02_r1.log（1039 passed）

## 复核要求（read_file 静态分析）

1. P0-1：evolve() 取 distinct_attrs 链路完整（graphlite_store → get_distinct_attr_names → evolve_once）？None store 降级？
2. P1-1：提取阶段 alias 识别逻辑正确（_extract_property_terms 前/中用 alias 表）？纯增量？
3. P1-2：_RESERVED_KEYS 过滤完整（_build_prompt/_apply_new_type/_apply_merge 三处）？attr_aliases 不泄漏进类型决策？
4. P2-1：集成测试走生产入口（OntologyEvolution.evolve）？
5. P3-1：canonical 自身不进 alias 表？
6. 全量证据：/tmp/pytest_p02_r1.log 尾部 1039 passed？
7. 综合判定：是否可闭环发布 v5.50.0？

## 输出格式

- 判定：通过 / 需修改
- 缺陷清单（🔴 P0 / 🟠 P1 / 🟡 P2 / ⚪ P3，含文件:行号 + 证据）
- 修复建议
