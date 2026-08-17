# P0-2 Agentic 检索 R3 复核 — Codex 任务书

## 背景
Codex R2 复核判定需再修改（1 P1 + 2 P2 + 3 P3）→ OpenCode R2 修复已完成。

## R2 修复内容（待复核）
1. **N1-P1** 恢复小写英文词提取（query_router.py:1967-1968，"apple 收入" → apple），保留时间词/动词停用词+年份过滤
2. **N2-P2** _classify_property_terms 词边界匹配（:1428-1429，\b 判断；market_cap ↔ market cap 归一）
3. **N3-P2** 时间锚 key 带 timestamp（:96 _TIME_ANCHOR_SENTINEL）
4. **N4-P3** 删年份过滤死代码
5. **N5-P3** seen_anchors lower 归一（:1689/:1693）
6. **N6-P3** MCP/A2A/ACP 透传 session_ts（acp_adapter.py/a2a_server.py）

## 复核重点
1. R2 6 缺陷是否全部正确修复？有无修复不完整/引入新缺陷？
2. N1：小写英文词提取恢复后，是否还正确过滤时间词/动词/年份？会不会引入大量伪实体（R1 的原问题）？
3. N2：词边界匹配是否正确处理 market_cap/market cap？中文属性词是否受影响？
4. N3：时间锚带 timestamp 后，new_anchors 计数是否正确？
5. N6：acp_adapter/a2a_server 透传后，request model 变更是否向后兼容？
6. 测试质量：新增测试是否覆盖 N1（"apple 收入" 提取实体）？

## 方法
read_file/grep 静态分析 + 追踪调用链，不编译不实测。

## 输出
每修复点 ✅/❌ + 新问题（按严重度）+ 终审判定（最终通过/需再修改）。
