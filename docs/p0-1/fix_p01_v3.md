# P0-1 修复任务书 v3（OpenCode Phase 2d — Codex R3 复核缺陷修复）

Codex R3 复核（task_f4db2b93bf12）判定需再修改。1 P1 + 2 P2 + 1 P3。全部实现级，直接修。

## 缺陷清单与修法

### N2-P1 🟠：乱序中段插入破坏 SUPERSEDES 血统链完整性
- 位置：`core/entity_resolver.py:416-432`（`_supersedes_target_for`）
- 问题：`2021→2014→2016→2015` 任意乱序下，历史插入只写 `V2014→V2015` 并覆盖 `V2014.expired_at`，不给 V2015 打 `expired_at=2016`，不建 `V2015→V2016`。SUPERSEDES 血统与 expired_at 语义不一致。
- 修法：历史插入时**同时定位前驱 P 和后继 S**：
  - `P = valid_from < now 的最大 valid_from 版本`（前驱）
  - `S = valid_from > now 的最小 valid_from 版本`（后继）
  - 写 `P.expired_at = now`、`new.expired_at = S.valid_from`、建 `P→new`、`new→S` 边
  - 注意：这是 4 条独立 execute（多语句截断坑），create_property_version 需支持传 superseded_by（后继）+ 双 expired_at 打标，或拆成两个原语调用。

### N2-P2 🟡：同值历史写入被静默丢弃
- 位置：`core/entity_resolver.py:392`
- 问题：只比较最新版 value，先写 `2021=10B` 再写 `2014=10B` 直接 return 0，2014 时点无版本可查。
- 修法：no-op 判定同时纳入 valid_from——`latest.value == value AND 无更早历史` 才 no-op；同值更早 valid_from 应建历史版本（valid_from 不同即不同版本）。

### N2-P3 🟡：部分回归测试直调私有方法
- 位置：`tests/test_property_temporal.py:654`（`_update_property_version`）、`:758`（`_property_time_mode`）
- 问题：绕过生产 retrieve()/HTTP 入口，存在假绿风险。
- 修法：补公共入口用例（走 retrieve() 或 HTTP endpoint 断言时间语义），私有方法直调可保留但必须补公共入口覆盖。

### N5-P3 🟡：英文同义词与生产属性面不匹配
- 位置：`retrieval/query_router.py:61-76`
- 问题：映射含 revenue/market_cap 但生产只生成 acquired_value/invested_value；英文 revenue/sales 查询不被过滤。
- 修法：补充英文同义词映射（acquired→acquisition/bought/took over/acquired；invested→investment/funded/raised/backed），或明确属性词表与可生成 attr_name 对齐。

## 验收标准
1. 4 个缺陷全部修复
2. 新增/更新测试：
   - 乱序中段插入（2021→2014→2016→2015）血统链完整（双方向 SUPERSEDES + expired_at 语义正确）
   - 同值历史写入建版本（2014=10B 在 2021=10B 之后仍建 2014 版本）
   - 公共入口测试（retrieve() 时间语义）
   - 英文属性词过滤
3. `pytest tests/test_property_temporal.py -x -q` 全过
4. 全量 `pytest tests/ -q --deselect tests/test_core_engine.py::TestTauDecay::test_decay_threshold_candidate` 通过
5. 版本保持 5.47.0（不额外 bump）
