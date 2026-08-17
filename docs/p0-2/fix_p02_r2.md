# P0-2 Agentic 检索 R2 修复 — OpenCode 任务书

## 背景
Codex R2 复核判定：需再修改。1 P1 回归 + 2 P2 + 3 P3。

## 修复清单

### N1-P1: 恢复小写英文词提取（P1-3 修复过度回归）
- **问题**：`_extract_query_entities`（query_router.py:1964）删了小写英文词提取，只保留首字母大写词 → `"apple 收入"` 无法提取实体 `apple`，`_property_temporal_retrieve` 静默跳过。修复前代码支持小写 `apple` 对齐写侧 `Apple Inc`
- **修复**：恢复小写英文词提取（query_router.py:2169-2171 相关），但**保留过滤**：时间词/动词停用词 + 年份数字过滤。不要整类删除

### N2-P2: _classify_property_terms 词边界匹配
- **问题**：`:1434-1436` 用 `en in ql` 子串匹配 → "age" 命中 "agent"/"manager"，"sales" 命中 "salesforce" → 误判属性意图
- **修复**：改 token/词边界匹配。`market_cap` 与 `market cap`（空格）归一化后用 `\b` 判断：`re.search(r'\b' + re.escape(en) + r'\b', ql)`；`market cap` 转 `market_cap` 统一

### N3-P2: 时间锚键带 timestamp
- **问题**：`:1572-1573` 所有时间锚统一 `_TIME_ANCHOR_SENTINEL` → 不同时间锚（如 "yesterday" vs "2023"）无法作为新锚点计数
- **修复**：时间锚 key 改 `__time_anchor__:{ats}`（带解析后的 timestamp），保留哨兵区分类型但值唯一

### N4-P3: 删年份过滤死代码
- **问题**：`:1977` `re.fullmatch(r'(?:19|20)\d{2}', c)` 永远匹配不到（c 只来自大小写英文/中文后缀正则）
- **修复**：删死分支，或改为真实数字 token 提取后过滤

### N5-P3: seen_anchors 大小写归一
- **问题**：`:1681` 原始大小写做差集，`Apple` vs `APPLE` 计为不同锚点
- **修复**：seen_anchors 与差集统一 lower 归一

### N6-P3: MCP/A2A/ACP 透传 session_ts
- **问题**：`gateway/acp_adapter.py:90`、`gateway/a2a_server.py:199-204` 未透传
- **修复**：request model + 调用点补齐 session_ts（None 默认，向后兼容）

## 约束
- 遵循 AGENTS.md 编码准则；新方法私有；session_ts 向后兼容
- **版本号不 bump**
- 全部测试通过（962 基线无回归）

## 验收
1. 每个修复点有对应测试（N1 需测试 `"apple 收入"` 能提取实体）
2. 全量 pytest 无回归（用 -p no:randomly 避免已知随机顺序污染）
