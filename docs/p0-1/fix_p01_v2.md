# P0-1 修复任务书 v2（OpenCode Phase 2c — Codex R2 复核缺陷修复）

Codex R2 复核（task_a799947ff6fa）判定需再修改。3 P1 + 3 P2 + 1 P3，全部实现级，直接修。

## 缺陷清单与修法

### N1 🟠 P1：YEAR_RE 用 \b 无法匹配 "2014年"/"2024年度"（中文年份失效）
- 位置：`core/relation_extractor.py:28`
- 问题：`\b(?:19|20)\d{2}\b` 的 \b 在中文语境不生效（"2014年"中 "4年" 无词边界），中文关系抽取拿不到 attr_year，valid_from 静默回落写入时间。
- 修法：改 `(?<!\d)(?:19|20)\d{2}(?!\d)`（与读侧年份正则一致），移除 \b 依赖。

### N2 🟠 P1：乱序写入时历史年份被抬成最新，时间链语义反向
- 位置：`core/entity_resolver.py:390`
- 问题：`if now <= last_ts: now = last_ts + 0.001` 会把历史年份（2014 < 2021）也抬到最新版时间戳。先写 2021 再写 2014，2014 版本被标成最新，supersede 语义反向。
- 修法：仅 `now == last_ts` 时 bump（同微秒防重）；`now < last_ts` 按历史版本插入（不 supersede 最新版，而是按时间序挂链——supersedes_id 取"当前最新但 valid_from < now"的那个版本，若无则无 supersedes 但保留 valid_from）。

### N3 🟠 P1：attr_pattern.search(text) 全文查找金额，跨句误抓
- 位置：`core/relation_extractor.py:248`
- 问题：`attr_pattern.search(text)` 在全文中找金额，同一文档多条 ACQUIRED 时第二条三元组拿到第一条的 value（DeepMind 得到 3B 而非 500M）。
- 修法：将属性搜索限定到当前三元组所在句子窗口（复用 `extract_year_in_sentence` 的句子边界逻辑，或提取公共 `_sentence_window(text, start, end)` 函数）。

### N4 🟡 P2：相对时间词换算基准与匹配范围错误
- 位置：`retrieval/query_router.py:1655`
- 问题：`last`/`previous` 固定换算 1 天前，"last year/month" 语义错误；"今天" 取当日 0 点漏掉当天稍晚生效版本；仅匹配字面"几分钟前"，"5分钟前" 不生效。
- 修法：解析相对单位（last year/month/week/day → 对应时长）；"今天" 用当前时刻；数字正则解析 "N 分钟前/N minutes ago"。

### N5 🟡 P2：属性通道不按查询属性词过滤，返回无关属性
- 位置：`retrieval/query_router.py:1763`
- 问题：查询 "Apple 收入" 可能同时返回 acquired_value 等无关属性，同分后仅靠 _PROPERTY_MAX_RESULTS 截断。
- 修法：按 attr_name/属性同义词做词法或 BM25 过滤后再 append（query 中出现的属性词 → 只保留匹配的 attr_name）。

### N6 🟡 P3：前缀 apple% 过宽命中 Applebee's/Applejack
- 位置：`graph/graphlite_store.py:761`
- 问题：前缀匹配 `apple%` 会命中 Applebee's、Applejack 等无关实体。
- 修法：加词边界后置过滤（实体名后缀精确匹配：apple 精确 OR apple+空格+Inc/Corp 等后缀），或对 LOWER(entity_id) 后缀词做精确/前缀双条件。

## 验收标准
1. 6 个缺陷全部修复
2. 新增/更新测试：中文年份（"2021年"→valid_from）、乱序写入（先 2021 再 2014 不反向）、跨句金额不误抓、相对时间词单位解析、"N 分钟前"、属性词过滤、前缀过宽
3. `pytest tests/test_property_temporal.py -x -q` 全过
4. 全量 `pytest tests/ -q --deselect tests/test_core_engine.py::TestTauDecay::test_decay_threshold_candidate` 通过
5. 版本保持 5.47.0（不额外 bump）
