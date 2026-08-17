# P0-2 Agentic 检索终审 — Codex 任务书

## 背景
CC 设计（方案 B：规则编排 + session_ts 时间锚注入）→ OpenCode 已实施（retrieval/query_router.py）。
实施内容：QueryRouterConfig 5 个 agentic_* 字段（默认关）+ retrieve() session_ts 参数 + now 下沉 + 4 个新私有方法 + _agentic_retrieve 编排器 + 5 个新测试（tests/test_agentic_retrieve.py，144 行）。

## 审核重点
1. **零回归保证**：agentic_enabled=False 时走现有单轮路径，是否字节级等价？（检查 _fusion_retrieve 签名变化 now_ts 默认值）
2. **session_ts 注入完整性**：_relative_time_at_ts/_property_time_mode/_apply_time_decay 三处 now 是否都正确下沉？调用点是否全部透传？None 回落墙钟？
3. **_agentic_retrieve 编排器正确性**：三重防死循环（seen 去重/max_steps 硬上限/min_new 枯竭停）是否有效？_sufficiency_check 逻辑？锚点提取？
4. **_classify_intent 意图分类规则**：time/identity/attribute/event/multi_hop 分类是否合理？与 _route_channels 通道映射是否正确？
5. **调用链一致性**：retrieve() 主流程接入点是否正确（FUSION 分支前）？session_ts 是否透传到所有内部检索通道？
6. **引入新缺陷**：now 下沉是否破坏既有时间衰减/相对时间词行为？
7. **测试质量**：5 个测试是否走公共入口（retrieve()）而非直调内部方法？防假绿？

## 方法
read_file/grep 静态分析 + 追踪调用链，不编译不实测。

## 输出
每审核点 ✅/❌ + 新问题（按严重度）+ 终审判定（最终通过/需再修改）。
