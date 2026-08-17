# P0-1 终审任务书（Codex R7 — 最终复核 v5）

Codex R6 复核判定需再修改（2 P1 + 2 P2 + 1 P3）。Hermes 按 R6 修法补齐：

## Hermes 修复内容（需复核）
1. **R6-P1a succ_ts 使用时机**（`graph/graphlite_store.py`）：读取后继 valid_from 的逻辑**提前**到 supersedes_id 块之前（INSERT 新节点后立即读取），所有补偿路径用正确 succ_ts
2. **R6-P1b 后继时间读取失败**：不再 non-fatal 静默（不再以 now 退化），改为回滚新节点后重抛
3. **R6-P2 中段插入失败注入测试**：新增 4 个用例覆盖四个失败点（SET pred.expired_at / P→new 边 / SET new.expired_at / new→succ 边），验证回滚后链完整（P→S 恢复 + pred.expired_at=succ_ts）
4. **R6-P3 注释修正**：tests 739 行注释改为「2014 插入时建过 2014→2021，随后被 2016 插入删除」

## 验证（Hermes 实测）
- pytest tests/test_property_temporal.py → 52 passed（+4 注入用例）
- 全量 pytest（排除已知 flaky）→ 947 passed
- 注入测试发现并修复：`_patch_execute_fail_on("INSERT (a)-[:SUPERSEDES]")` 会误伤补偿恢复边 → 改为计数器只失败第 1 次

## 审核重点
1. read_file 验证 succ_ts 读取位置（应在 supersedes_id 块之前、补偿可及）
2. 读取失败重抛路径：回滚新节点是否执行、异常是否传播
3. 4 个中段插入注入测试的断言（链完整 + pred.expired_at=succ_ts）
4. 是否引入新缺陷（如读取提前后 GET 语义、重复读取删除）
5. 版本 5.47.0 一致

## 输出格式
- 每修复点：✅ 已闭环 / ❌ 未闭环（+文件行号+修法）
- 新问题（如有）：严重度 + 位置 + 修法
- 终审判定：最终通过 / 需再修改
