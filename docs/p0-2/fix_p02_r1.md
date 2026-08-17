# P0-2 Agentic 检索 R1 修复 — OpenCode 任务书

## 背景
Codex R1 终审判定：需再修改（4 P1 + 2 P2）。均为实现级缺陷，按缺陷分流直接修。

## 修复清单（按 Codex 报告）

### P1-1: agentic_min_new 枯竭保护失效（非增量）
- **问题**：`_extract_anchors` 每轮只返回累计结果的锚点并集，`_agentic_retrieve` 直接覆盖 anchors，无 seen_anchors 差集 → 相同锚点重复满足 min_new
- **修复**：`_agentic_retrieve` 维护 `seen_anchors: set`，每轮 `anchors.new = anchors.all - seen_anchors`，`len(new) < agentic_min_new` 才停；`_extract_anchors` 返回全部锚点（含 old/new 区分或返回集合，由编排器求差）

### P1-2: 证据时间锚未按 session_ts 解析 + time_anchor 不计入 new
- **问题**：`_extract_anchors` 调 `_property_time_mode(text, None)` 用墙钟；time_anchor 不计入 new → 仅时间锚触发下轮被截断；`_property_temporal_retrieve` 从 query 重算时间
- **修复**：
  - `_extract_anchors(top_results, plan, session_ts)` — session_ts 透传，`_property_time_mode(text, session_ts)`
  - time_anchor 计入 new（时间锚也是有效新锚点）
  - `_property_temporal_retrieve` 接受可选的 `at_ts` 参数（从 plan 传入），不重算

### P1-3: 意图分类覆盖不足
- **问题**：`_classify_property_terms` 只识别中文属性词（英文 revenue/market_cap 不识别）；`_extract_query_entities` 把小写动词/时间词当实体（"What happened in 2023" 误判 event）
- **修复**：
  - `_classify_property_terms` 增加英文属性词表（revenue, market_cap, income, age, occupation, salary 等）
  - `_extract_query_entities` 过滤小写英文词 + 时间词（含年份数字）不当实体

### P1-4: 全局开关吞掉 level + session_ts 无生产消费方
- **问题**：`if self.config.agentic_enabled:` 无条件返回 FUSION，劫持 HYPERGRAPH/VECTOR/KEYWORD；API/Gateway/self_evolving 未透传 session_ts，agentic_enabled 未从 settings 注入
- **修复**：
  - 条件改 `if self.config.agentic_enabled and level == RetrievalLevel.FUSION:`
  - `api/routes/search.py` RetrieveRequest 加可选 session_ts（float|None）+ 透传
  - `retrieval/self_evolving.py` retrieve 包装加 session_ts 透传
  - `config/settings.py` + defaults.yaml 加 `retrieval.agentic_enabled` 配置注入（默认 False）
  - gateway/gateway_api.py 若为生产入口同样透传（先 grep 确认调用链）

### P2-1: _hypergraph_supplement 未排序取头尾
- **问题**：融合结果未排序前 `results[:5]` 与 `results[-1]` 不是真实 top/lowest
- **修复**：排序后再取（或按 score 排序后取头尾）

### P2-2: 测试缺公共入口集成断言
- **问题**：无 retrieve(session_ts=...) 验证时间锚透传
- **修复**：新增集成测试 `test_retrieve_session_ts_propagates`——走 retrieve() 公共入口，断言 session_ts 到达 _property_temporal_retrieve（spy 或构造相对时间词验证解析结果）

## 约束
- 遵循 AGENTS.md 编码准则
- 新增方法全私有；session_ts 参数向后兼容（None 回落墙钟）
- **版本号不 bump**（评测通过后统一）
- 全部测试通过（954 基线 + 新增无回归）

## 验收
1. 每个 P1/P2 修复点有对应测试或断言
2. 全量 pytest 无回归
