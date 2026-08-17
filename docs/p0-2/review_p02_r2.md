# P0-2 Agentic 检索 R2 复核 — Codex 任务书

## 背景
Codex R1 终审判定需再修改（4 P1 + 2 P2）→ OpenCode R1 修复已完成（8 文件：query_router.py/search.py/self_evolving.py/gateway_api.py/settings.py/defaults.yaml/api/app.py/api/models.py）。

## R1 修复内容（待复核）
1. **P1-1** agentic_min_new 增量：_agentic_retrieve 维护 seen_anchors 差集（:1653/:1679/:1681/:1685）
2. **P1-2** session_ts 透传：_extract_anchors(..., session_ts)（:1528/:1678）+ time_anchor 计入 new + _property_temporal_retrieve at_ts 参数
3. **P1-3** 英文属性词表（:77-79 revenue/market_cap/income/age/occupation/salary）
4. **P1-4** level 限定 `agentic_enabled and level == FUSION`（:1279）+ search.py RetrieveRequest.session_ts + self_evolving.py/gateway 透传 + settings.py/defaults.yaml 配置注入
5. **P2-1** _hypergraph_supplement 排序后取头尾
6. **P2-2** 集成测试（test_agentic_retrieve.py 现在 13 个测试）

## 复核重点
1. R1 6 缺陷是否全部正确修复？有无修复不完整/引入新缺陷？
2. seen_anchors 差集逻辑：new_anchors = all - seen_anchors 是否正确？时间锚哨兵处理？
3. session_ts 生产链路完整性：search.py → self_evolving.py → query_router.py → _property_temporal_retrieve 全程透传？cache_key 是否含 session_ts？
4. level==FUSION 限定是否破坏非 FUSION 路径？
5. 英文属性词表是否与 _normalize_query 的中英映射（:62-64）冲突？
6. 配置注入：defaults.yaml 与 settings.py 是否一致？agentic_enabled 默认 False？
7. 测试质量：13 个测试是否覆盖 R1 修复点？走公共入口？

## 方法
read_file/grep 静态分析 + 追踪调用链，不编译不实测。

## 输出
每修复点 ✅/❌ + 新问题（按严重度）+ 终审判定（最终通过/需再修改）。
