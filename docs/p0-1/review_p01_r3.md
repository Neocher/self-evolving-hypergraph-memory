# P0-1 终审任务书（Codex R3 — 最终复核）

Codex R2 复核判定需再修改（3 P1 + 2 P2 + 1 P3），OpenCode 已修复 v2。请最终复核 6 缺陷是否闭环 + 修复是否引入新缺陷。

## 修复内容摘要（OpenCode 报告）
1. **N1 中文年份**：YEAR_RE 改 `(?<!\d)(?:19|20)\d{2}(?!\d)`，中文 "2021年" → attr_year
2. **N2 乱序时间语义**：仅 `now == last_ts` bump；`now < last_ts` 按历史版本插入（supersedes 目标取 valid_from < now 的前驱）
3. **N3 金额跨句误抓**：属性搜索限定句子窗口（提取公共 `_sentence_window`）
4. **N4 相对时间词**：`_relative_time_at_ts` 数字+单位正则（"5分钟前"/"3 days ago"）；last year/month/week/day 时长映射；"今天"改当前时刻
5. **N5 属性词过滤**：`_PROPERTY_QUERY_TERM_MAP`（收入→revenue 等 14 映射）+ `_extract_property_terms()`；无属性词不过滤（向后兼容）
6. **N6 前缀过宽**：`LOWER(entity_id) = 'apple' OR LOWER(entity_id) LIKE 'apple %'` 词边界

## 验证（OpenCode 报告）
- pytest tests/test_property_temporal.py → 46 passed
- pytest tests/ 全量 → 941 passed, 1 skipped, 1 deselected
- py_compile 5 文件 OK；版本保持 5.47.0

## 审核重点
1. read_file 验证 6 缺陷修复是否真实现（不是只写测试假绿）
2. 追踪调用链：年份 attr_year → valid_from 传递；实体归一化读写一致；属性词过滤在检索链路生效
3. **重点查修复引入的新缺陷**：
   - N2 乱序挂链的 supersedes 目标选择是否正确（valid_from < now 的前驱，不 supersede 最新）
   - N3 句子窗口公共函数是否被年份+金额共用且边界正确
   - N5 属性词映射是否过窄（漏掉常见查询）或误过滤（正常查询被滤掉）
   - N6 词边界 OR 条件是否语义正确（apple % 不含 apple 单名？）
4. 测试是否走公共入口（非直调内部方法）
5. 版本四处 5.47.0 一致

## 输出格式
- 每缺陷：✅ 已闭环 / ❌ 未闭环（+文件行号+修法）
- 新问题（如有）：严重度 + 位置 + 修法
- 终审判定：最终通过 / 需再修改
