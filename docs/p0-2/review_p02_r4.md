# P0-2 Agentic 检索 R4 终审 — Codex 任务书

## 背景
Codex R3 复核判定需再修改（2 P2 + 2 P3 + 1 P3 测试）→ OpenCode R3 修复已完成。
当前测试：test_agentic_retrieve.py 30 passed + test_mcp_session_ts.py 3 passed（新增）+ 全量 984 passed（1 预存 tau 缺陷无关）。

## R3 修复内容（待复核）
1. **P2-1** 停用词扩充（:58-60 of/to/has/had）+ 撇号还原表（:104 don't→do not，:119 _normalize_apostrophe）
2. **P2-2** `_extract_property_terms` 词边界（:2220 `re.search(r'\b'+t+r'\b', attr)`）
3. **P3-1** 相对时间锚稳定键（:2119 __time_anchor__:today 语义规范化，绝对时间保留 timestamp）
4. **P3-2** MCP 透传（mcp_server.py:99/:143-144 + mcp_server_v2.py:116）
5. **P3-3** N1 测试补公共入口集成断言

## 复核重点
1. R3 5 缺陷是否全部正确修复？有无引入新缺陷？
2. P2-1 撇号还原表是否完整？还原后停用词匹配是否生效？会不会误伤正常小写词？
3. P2-2 词边界逻辑是否与分类器（N2 已修）一致？下划线归一是否处理 market_cap ↔ market cap？
4. P3-1 稳定键是否影响绝对时间锚（年份/日期）的独立性？
5. P3-2 MCP 透传是否向后兼容（None 默认）？schema 是否与 HTTP 端点一致？
6. 测试覆盖：新增测试是否覆盖各修复点？

## 方法
read_file/grep 静态分析 + 追踪调用链，不编译不实测。

## 输出
每修复点 ✅/❌ + 新问题（按严重度）+ 终审判定（最终通过/需再修改）。
