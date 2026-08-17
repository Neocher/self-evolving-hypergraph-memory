# P0-2 Agentic 检索实施 — OpenCode 任务书

## 背景
CC 设计已完成（方案 B：规则编排 + LLM 可选增强）。核心：
1. **cat=2 时间推理根治 = session_ts 参数注入**（不是多步）——`_relative_time_at_ts` 用墙钟，对历史 session 恒错
2. **cat=1 跨消息关联 = 多步锚点检索**——首轮证据不足才 refine
3. **零回归**：agentic_enabled=False 默认关，第 1 轮 = 现有 FUSION 全路径

## 实施范围（retrieval/query_router.py，最小侵入）

### 1. 配置扩展（QueryRouterConfig ~:115）
新增字段（全部带默认值，向后兼容）：
```python
agentic_enabled: bool = False      # 默认关
agentic_max_steps: int = 3         # 含首轮最多 3 轮
agentic_min_new: int = 3           # 每轮须新增 ≥3 条未见过节点
agentic_score_gap: float = 0.25    # 首轮 top-12 归一化分差 < 该值判证据不足
agentic_top_k: int = 12            # 每轮召回 top-k
```

### 2. 时间锚注入（cat=2 根治）
- `retrieve()` 加 `session_ts: Optional[float] = None`
- `now` 下沉为参数：`_relative_time_at_ts(q, now_ts)`、`_property_time_mode(query, now_ts)`、`_apply_time_decay(results, now_ts)`
- 改 `time.time()` → `now_ts or time.time()`，其余调用点透传
- ⚠️ 这是唯一触及既有函数内部语义的改动，其余全为新增

### 3. 新增规则原语（全私有方法）
```python
_classify_intent(query, session_ts) -> IntentPlan  # {intent, time_mode, at_ts, entities, property_terms}
_route_channels(plan) -> list[str]                  # time→property_temporal; identity→entity+fusion; attribute→property_temporal+fusion; event→fusion+hypergraph; multi_hop→fusion
_sufficiency_check(results, plan) -> bool           # top-12 归一化分差 ≥ gap 且 distinct ≥ 阈值 → 充分
_extract_anchors(top_results, plan) -> AnchorSet    # 证据消息提取实体+时间锚+属性词
```

### 4. 编排器（新增 `_agentic_retrieve`）
```python
_agentic_retrieve(query, raw_query, query_embedding, session_ts, include_archived):
  plan = _classify_intent(query, session_ts)
  seen, results = set(), []
  for step in 1..agentic_max_steps:
      channels = _route_channels(plan) if step==1 else _channels_from_anchors(anchors)
      round = _fusion_retrieve(...) + _finish 补充通道
      round = [r for r in round if r.node_id not in seen]
      results += round; seen |= {r.node_id}
      if step == agentic_max_steps: break
      if _sufficiency_check(results[:12], plan): break
      anchors = _extract_anchors(results[:12], plan)
      if len(anchors.new) < agentic_min_new: break
  return _deduplicate_and_sort(results)
```

### 5. retrieve() 主流程接入
- `if self.config.agentic_enabled: return self._agentic_retrieve(...)`（在 FUSION 分支前）
- `agentic_enabled=False` 时走现有单轮路径，字节级等价

## 约束
- 遵循 AGENTS.md 编码准则：思考优先/简洁/精准修改/验证闭环
- 新增方法全私有，不暴露公共 API（除 session_ts 参数）
- 不动其他文件（除非测试需要）
- **版本号不 bump**（本任务不发布，评测通过后统一 bump）

## 验收
1. 新增测试：`test_agentic_retrieve_disabled_equals_baseline`（默认关字节级等价）
2. 新增测试：`test_agentic_retrieve_time_injection`（session_ts 注入后相对时间词解析正确）
3. 新增测试：`test_agentic_retrieve_max_steps_budget`（死循环防护：max_steps 硬上限）
4. 新增测试：`test_agentic_retrieve_min_new_stop`（锚点枯竭提前停）
5. 全量 pytest 无回归（949 passed + 新增）
