# P0-1 复核任务书（Codex R2 — 终审缺陷修复复核）

Codex R1 终审判定需修改（2 P1 + 3 P2），OpenCode 已修复。请复核 5 个缺陷是否真正闭环，以及修复是否引入新缺陷。

## 修复内容摘要
1. **P1-1 时间维注入**：`core/relation_extractor.py` 新增 `extract_year_in_sentence`（句子窗口限定的 4 位年份解析）+ `YEAR_RE`；ACQUIRED 等关系提取 `attr_year` → `update_properties_from_triples` 显式传 valid_from
2. **P1-2 实体归一化**：读写两侧共用归一化（具体实现需 read_file 确认）
3. **P2-1 at_time 年末语义**：`_property_time_mode` 年份查询取 Dec 31 23:59:59 + `_pick_property_versions` at_time 分支校验 expired_at
4. **P2-2 相对时间词**：`_relative_time_at_ts` 换算 at_ts 走 at_time，不再误归 latest
5. **P2-3 非原子写补偿**：create_property_version 失败路径补偿/告警日志

## 审核重点
1. read_file 追踪：relation_extractor 的 attr_year → entity_resolver 的 valid_from 传递链是否通畅（含 write.py 闭包）
2. 实体归一化实现是否正确（读写两侧一致性）；`get_property_versions_for_entities` 检索语义
3. `_pick_property_versions` at_time 分支 expired_at 校验是否正确
4. `_relative_time_at_ts` 实现（哪些词换算、换算基准）
5. 非原子写补偿是否真实现（不是只加日志）
6. 修复是否引入新缺陷（年份误抓、归一化过宽/过窄）
7. 版本保持 5.47.0 一致

## 输出格式
- 每缺陷：✅ 已闭环 / ❌ 未闭环（+文件行号+修法）
- 新问题（如有）：严重度 + 位置 + 修法
- 终审判定：最终通过 / 需再修改
