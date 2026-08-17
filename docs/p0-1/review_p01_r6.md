# P0-1 终审任务书（Codex R6 — 最终复核 v4）

Codex R5 复核判定需再修改（2 P1 + 2 P2）。Hermes 按 R5 修法补齐：

## Hermes 修复内容（需复核）
1. **R5-P1a `SET p.expired_at` 失败补偿**（`graph/graphlite_store.py` ~668）：中段插入时恢复 P→S 边（链不断裂）
2. **R5-P1b `P→new` 边失败补偿**（~697）：中段插入时恢复 P→S 边 + pred.expired_at 恢复 succ_ts
3. **R5-P1c `SET new.expired_at` 失败补偿**（~743）：pred.expired_at 恢复 **succ_ts**（原误为 NULL，破坏「非最新节点必有 expired_at」不变式）+ 恢复 P→S 边
4. **R5-P1d `new→succ` 边失败补偿**（~776）：同上，pred.expired_at 恢复 succ_ts
5. **R5-P2 负向断言修正**（`tests/test_property_temporal.py:737`）：2014→2016==0（真正覆盖「插入 2015 时删除了旧 2014→2016 边」），移除空断言 2015→2021

## 验证（Hermes 实测）
- pytest tests/test_property_temporal.py → 48 passed
- 全量 pytest（排除已知 flaky）→ 943 passed
- 前期实测：GraphLite 双 MATCH DELETE 无效 → 单 MATCH 边模式有效

## 审核重点
1. read_file 验证 4 个补偿路径：每个失败点是否都完整回滚（删新节点 + pred.expired_at 恢复正确值 succ_ts/NULL + P→S 边恢复）
2. 中段插入（superseded_by 非空）与常规更新（仅 supersedes_id）的补偿分支是否正确区分
3. 负向断言 2014→2016==0 是否真实覆盖旧边删除语义
4. 补偿路径是否引入新缺陷（如重复 INSERT P→S 边报错、succ_ts 未定义时使用）
5. 版本 5.47.0 一致

## 输出格式
- 每修复点：✅ 已闭环 / ❌ 未闭环（+文件行号+修法）
- 新问题（如有）：严重度 + 位置 + 修法
- 终审判定：最终通过 / 需再修改
