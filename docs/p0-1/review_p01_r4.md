# P0-1 终审任务书（Codex R4 — 最终复核 v2）

Codex R3 复核（task_f4db2b93bf12）判定 N2 乱序血统链 P1 未闭环。OpenCode v3 修复不完整（只改了 _supersedes_target_for 为前驱查找），Hermes 按 Codex R3 修法手动补齐：

## Hermes 修复内容（需复核）
1. **`graph/graphlite_store.py` `create_property_version` 新增 `superseded_by` 参数**：乱序中段插入时
   - 新版本打 `expired_at = 后继.valid_from`（读后继 valid_from）
   - 建 `(new)-[:SUPERSEDES]->(succ)` 边
   - 与原 `supersedes_id`（前驱）逻辑组合 → P→new→S 双挂链 + 双向 expired_at
2. **`core/entity_resolver.py` `_supersedes_target_for` → `_chain_neighbors_for`**：返回 (pred, succ) 链邻居
   - pred = valid_from < now 的最新版
   - succ = valid_from > now 的最早版（ASC 序第一个）
   - 传 `supersedes_id=pred, superseded_by=succ`
3. **`core/entity_resolver.py` 同值历史写入判定（N2-P2）**：
   - 同值 + 顺序写入（now >= last_ts）→ 幂等 no-op
   - 同值 + 乱序历史写入（now < last_ts）→ 建历史版本（防 at_time 历史时点查不到）
4. **测试更新**：
   - 旧断言 `test_historical_write_does_not_reverse_chain` 改为期望 2014→2021 边 = 1 + 2014.expired_at≈2021（R3 新语义）
   - 新增 `test_out_of_order_mid_insert_full_chain`（2021→2014→2016→2015 双挂链完整性）
   - 新增 `test_same_value_historical_write_creates_version`（同值乱序建版本）

## 验证（Hermes 实测）
- pytest tests/test_property_temporal.py → 48 passed
- 全量 pytest（排除已知 flaky）→ 运行中

## 审核重点
1. read_file 验证 create_property_version 的 superseded_by 分支：新版本 expired_at 是否正确取后继 valid_from；`(new)-[:SUPERSEDES]->(succ)` 边方向正确（new 被 succ 取代）
2. `_chain_neighbors_for` 的 pred/succ 选择逻辑（ASC 序，pred 取最大 < now，succ 取最小 > now）
3. 同值时序判定：顺序 no-op / 乱序建版本 的分支条件（now >= last_ts vs now < last_ts）
4. 双挂链 5 条 execute 是否各自独立（多语句截断坑）
5. 测试是否走公共入口 + 断言是否覆盖 R3 要求的链完整性语义
6. 修复是否引入新缺陷（如 superseded_by 与 supersedes_id 同时传时的交互、补偿逻辑覆盖）
7. 版本 5.47.0 一致

## 输出格式
- 每修复点：✅ 已闭环 / ❌ 未闭环（+文件行号+修法）
- 新问题（如有）：严重度 + 位置 + 修法
- 终审判定：最终通过 / 需再修改
