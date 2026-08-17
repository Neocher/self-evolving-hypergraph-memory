# P0-2 Agentic 检索 R3 修复 — OpenCode 任务书

## 背景
Codex R3 复核判定：需再修改（2 P2 + 2 P3 + 1 P3 测试）。收敛中。

## 修复清单

### P2-1: 小写英文词伪实体控制
- **问题**：`_extract_query_entities`（:1981 `\b([a-z]{2,})\b`）提取所有小写英文 token，但 `_PROPERTY_CANDIDATE_STOPWORDS` 缺 `of/to/has/had` 等常见词；"don't" 被撇号切开提取到未过滤的 "don"
- **修复**：
  - `_PROPERTY_CANDIDATE_STOPWORDS` 扩充（of/to/has/had/at/by/for/from/with/into/onto/upon 等）
  - 撇号处理：`don't`/`it's` 等先拆成完整词或整体停用（don't 已在停用表则先还原撇号再匹配，或正则排除含撇号 token）

### P2-2: 生产属性过滤链子串匹配
- **问题**：N2 只修了 `_classify_property_terms` 分类器（:1440），但 `_extract_property_terms`（:2156 `w in an`）和最终过滤（:2210 `t in attr_name`）仍子串匹配 → "Apple age" 在 attr_name="manager" 时仍命中
- **修复**：同一套 \b/下划线归一逻辑复用到 :2156 和 :2210——属性 term 与属性名匹配改为词边界比较（`re.search(r'\b'+term+r'\b', attr_name)` 或分词后精确匹配，market_cap ↔ market cap 归一）

### P3-1: 相对时间锚键稳定
- **问题**：无 session_ts 时 `_extract_anchors`（:1558）→ `_relative_time_at_ts`（:2044）用 `time.time()`，每轮不同 timestamp → 同一 "today" 当成新锚点，可能空转到 max_steps
- **修复**：相对时间锚（today/now/yesterday 等）用稳定规范化键（如按语义词规范化 `__time_anchor__:today`），绝对时间（年份/日期）保留 timestamp

### P3-2: MCP 透传 session_ts
- **问题**：`shm/mcp_server_v2.py:111`、`shm/mcp_server.py:91/:135` 检索 schema/调用不接收 session_ts
- **修复**：MCP 检索工具 schema 加可选 session_ts（float|None），调用透传

### P3-3: N1 测试走公共入口
- **问题**：test_agentic_retrieve.py:222 直调 `_extract_query_entities`，未覆盖 retrieve() → _property_temporal_retrieve 实际链路
- **修复**：补一条走 `QueryRouter.retrieve` 的集成断言（"apple 收入" 查询 → 断言属性时间检索被触发或实体被提取）

## 约束
- 遵循 AGENTS.md 编码准则；session_ts 向后兼容
- **版本号不 bump**
- 全部测试通过（971 基线无回归，-p no:randomly）

## 验收
1. 每个修复点有对应测试
2. 全量 pytest 无回归
