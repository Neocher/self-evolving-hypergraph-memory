# 阶段3+4 设计定稿（CC 两轮 + 阶段1-2 实证）

## 阶段3：τ 下沉 + 图作用域
- **D1-D4 τ 下沉否决**（三重错配：检索热路径读静态 tau_initial 无动态衰减可沉 / 引擎无节点级指数衰减原语 / archive≠delete 破坏 archive_supersedes 血统）。τ 保留 Python TauDecayEngine；预算转投图作用域
- **D5 作用域起点 = 首轮种子 EpisodeNode**（实体未物化；OntologyEntity 无连 Episode 边）
- **D6 `_scope_retrieve` 新补充通道，不替代 `_entity_expansion`（CONTAINS）**——精确串命中 vs 邻域语义相似，正交互补
- **D7 scope_max_depth=2、scope_direction 需 PoC 定 "both"**（边方向 hyperedge→episode 单向 + HEBBIAN；共享超边需入+出双向）
- **D8 scope_at_epoch=int(at_ts×1000) 不替代 created_at<=at_ts 节点过滤**（SHM 边无时序）
- **D9 时间单位集中 helper：SHM 秒 → 引擎毫秒（×1000）**
- **D10 空结果/无种子静默返回原 results；graphlite 后端 hasattr 守卫 no-op**
- 实现：overgraph_store.vector_search_scoped（~35 LOC）+ query_router._scope_retrieve（~50 LOC）+ scope_recall 配置 + test_overgraph_scope + test_tau_no_sink（负向钉死）
- **R1 高：scope_direction "both" 实现前 PoC 定标**（2 节点 + 1 共享超边最小用例）

## 阶段4：高阶进化（按性价比排序）
### 4-1 Schema 模式蒸馏（落地，~160 LOC）
- 纯规则复用已丢弃的 llm_patterns → SchemaNode（:Conceptual 标签）→ `_schema_recall` 通道
- 前置"评测前跑一轮蒸馏"（不阻塞在线检索）
- cat1 直接受益（模式节点提供聚合线索）
### 4-2 策略反馈环（落地，~100 LOC）
- core/feedback.py：FeedbackEngine.apply(rewards) → 节点成功计数 → 阈值升级 fact_track='core'（core boost ×1.1）
- **不碰边权重/τ**（防答案泄漏）；只在评测 harness 触发，不进生产在线路径
- AC：幂等 / 阈值边界（1 不升 2 升）/ 生产 retrieve() 零改动
### 4-3 跨语言实体扩召回（**本期降级为接口占位**）
- 根本问题 = embedding 语种不匹配（bge-small-zh 中文查询 vs 英文库语义空间偏移），非实体映射
- LoCoMo 纯英文库 + 无中文查询测试集 → 收益≈0 → 本期不落地完整映射
- 只留：data/entity_aliases.json 接口占位（复制 attr_aliases 范式）+ config 占位，不接线
- 待 ①_scope_retrieve 落地 ②真实跨语种生产数据 再做

## 验收标准（阶段3+4 完成后统一评测）
- test_overgraph_scope（公共入口 FUSION retrieve：scope append ≥1 且分 < 种子；graphlite no-op）
- test_tau_no_sink（负向钉死：引擎边衰减 ≠ TauDecayEngine 节点曲线；PRUNE 走 archive 非 delete）
- test_schema_recall + test_feedback（纯函数/公共入口）
- OverGraph 后端 LoCoMo ≥ 60.5% 基线（统一评测，本轮不单独出分）
- graphlite 后端零回归（全量测试）
