# P0-2 Agentic 检索 R5 终审 — Codex 任务书

## 背景
Codex R4 复核判定需再修改（1 P2 + 3 P3）→ OpenCode R4 修复已完成。
当前测试：test_agentic_retrieve 30 + test_mcp_session_ts 3 + 新增 3 = 全量 987 passed（1 预存 tau 缺陷无关）。

## R4 修复内容（待复核）
1. **P2-1** `_EN_CONTRACTION_MAP` 补 let's→let us / ain't→is not / o'clock→of the clock / y'all（:117-118）+ _PROPERTY_CANDIDATE_STOPWORDS 加 let/all（:61-62）
2. **P3-1** `_expand_contractions` re.sub 加 \b 词边界（:128）
3. **P3-2** test_mcp_session_ts 补 v2 静态断言
4. **P3-3** _extract_property_terms 调 _expand_contractions（:2012）

## 复核重点
1. R4 4 缺陷是否全部正确修复？
2. P2-1 收缩表完整后，"Let's find Apple" 是否不产出 "Let" 伪实体？let/all 停用是否有副作用（误伤真实实体 "All" 等）？
3. P3-1 \b 词边界是否影响所有格（cache's 中 he's）？
4. 新增测试是否覆盖修复点？
5. 有无引入新缺陷？

## 方法
read_file/grep 静态分析 + 追踪调用链，不编译不实测。

## 输出
每修复点 ✅/❌ + 新问题（按严重度）+ 终审判定（最终通过/需再修改）。
