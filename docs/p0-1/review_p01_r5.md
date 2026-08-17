# P0-1 终审任务书（Codex R5 — 最终复核 v3）

Codex R4 复核（task_b95f5308f51c）判定需再修改（2 P1 + 2 P2）。Hermes 按 R4 修法手动补齐：

## Hermes 修复内容（需复核）
1. **R4-P1 中段插入删除旧 P→S 边**（`graph/graphlite_store.py`）：
   - `supersedes_id` 与 `superseded_by` 同时非空时，先删除旧的 `(P)-[:SUPERSEDES]->(S)` 边
   - **实测发现 GraphLite 不支持双 MATCH 链式 DELETE**（`MATCH (a),(b) MATCH (a)-[s]->(b) DELETE s` 静默无效，count 仍 1）→ 改为单 MATCH 边模式 `MATCH (a)-[s]->(b) DELETE s`（实测有效，count→0）
2. **R4-P1 superseded_by 失败补偿**：
   - `SET new.expired_at` 失败 → 删新节点 + 恢复 pred expired_at=NULL + 恢复 P→S 边
   - `new→succ` 边失败 → 同上补偿
3. **R4-P2 补偿恢复原值**：supersedes_id 边失败时，中段插入（有 superseded_by）恢复 pred expired_at=succ_ts（非 NULL）
4. **R4-P2 测试负向断言**：新增 2014→2021、2015→2021 边计数 == 0（单链无分支）
5. 保留 `vf == now` 严格比较（上层 now==last_ts 已走 bump 分支，乱序重复时间戳为 P2 边界）

## 验证（Hermes 实测）
- pytest tests/test_property_temporal.py → 48 passed
- 全量 pytest（排除已知 flaky）→ 943 passed
- GraphLite DELETE 语法实测：单 MATCH 边模式有效（count→0）

## 审核重点
1. read_file 验证：DELETE 旧 P→S 边的单 MATCH 语法正确；superseded_by 补偿逻辑（删新节点+恢复 pred+恢复边）完整
2. 补偿路径是否覆盖所有失败点（SET new.expired_at / new→succ 边）；补偿本身失败仅告警是否可接受
3. 测试负向断言（2014→2021==0、2015→2021==0）是否真实覆盖单链无分支
4. 修复是否引入新缺陷（如 DELETE 匹配不到边时静默、补偿顺序）
5. 版本 5.47.0 一致

## 输出格式
- 每修复点：✅ 已闭环 / ❌ 未闭环（+文件行号+修法）
- 新问题（如有）：严重度 + 位置 + 修法
- 终审判定：最终通过 / 需再修改
