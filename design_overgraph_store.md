# 阶段1 CC 设计任务书 — OverGraphStore 适配层

## 背景
SHM 底层 GraphLite 维护停滞 + 中文触发 Panic（需 b64 绕行，CONTAINS 中文无子串保持性）。方案研究4选定 OverGraph（Rust 内核 + PyO3，Apache-2.0）替代。**PoC 已实证**（2026-08-19，5 轮）：
- ✅ 中文写入零 Panic、中文 CONTAINS 子串直查可用（"上海"命中）——b64 可整体移除
- ✅ 关系/超边映射可行：`MATCH (h:Hyperedge {elementKey:"he_1"}) MATCH (e:EpisodeNode {elementKey:"ep_1"}) CREATE (h)-[rel:MEMBER]->(e)`（**必须重复 MATCH，逗号分隔不支持**）
- ✅ 边权重：`SET rel.weight = 0.8` + `ORDER BY rel.weight DESC`
- ✅ node_id 获取：`RETURN elementKey(e)` / `RETURN id(e)`（elementKey 是引擎级标识非普通属性，`e.elementKey` 属性返回 None）
- ✅ temporal：`SET e.valid_from/valid_to` + `WHERE e.valid_from <= $ts AND e.valid_to >= $ts`
- ✅ 衰减清理：`set_prune_policy` / `prune` / `list_prune_policies`
- ✅ 事务：`begin_write_txn`（乐观冲突检测）
- ✅ 向量：`vector_search(mode, k, dense_query, label_filter, scope_start_node_id, scope_max_depth, scope_at_epoch, ...)` —— **scope_start_node_id 需引擎内部 ID(int)，不是 elementKey 字符串**
- ⚠️ execute_gql 返回 dict（取 `r['rows']`），非 list
- ⚠️ CONTAINS 大小写敏感（与 GraphLite 同——P3c 双变体修复适用）

## 目标
新增 `graph/overgraph_store.py`：**与 GraphLiteStore 同接口契约**的 OverGraph 后端，config `backend: graphlite|overgraph` 切换，SHM 上层业务零改动。**FAISS 同期替换为 OverGraph 内置 HNSW**（用户指令 2026-08-19：一次性利用 OverGraph 换掉 FAISS）——本阶段含向量层迁移，不再后置。

## CC 任务（read_file/grep 核实后设计）
1. **接口契约盘点**：grep 出 `graph/graphlite_store.py` 全部公开方法（create_episode/get_episode/query_cypher/update_with_version/超边/社区/Hebbian 等），列出调用方清单（grep -rn "gstore\.\|graphlite_store\." api/ retrieval/ core/）。**产出接口映射表**：每个方法 → OverGraph GQL/API 实现方式
2. **elementKey 策略**：SHM 现有 node_id（ep_xxx / 超边 he_xxx）→ elementKey 直映？还是引擎内部 ID + 属性映射？给出 node_id 生成/查询/索引策略（OverGraph 的 elementKey 是否有唯一约束？）
3. **b64 移除**：GraphLite 中文 b64 编解码逻辑全部移除（OverGraph 中文原生）——列出 graphlite_store.py 中所有 b64 相关代码位置
4. **事务/乐观锁**：SHM 现有乐观锁版本号机制（update_with_version）→ OverGraph begin_write_txn 如何映射？CAS 语义等价？
5. **超边映射**：SHM Hyperedge 结构（HYPEREDGE_MEMBER 边 + 属性）→ OverGraph 关系 + 权重；社区/共现结构如何迁移
6. **时间锚**：SHM created_at 时间戳属性 vs OverGraph 内置 created_at/updated_at（PoC 节点对象有）——用哪个？
7. **错误处理**：OverGraphError 层级（overgraph.OverGraphError）→ 与 graphlite SDK 异常（QueryError/ConnectionError）映射；熔断器适配
8. **config**：`backend` 开关设计（settings.py + defaults.yaml + 启动加载）
9. **迁移脚本接口**：阶段2 需要的迁移入口（dump GraphLite → load OverGraph），本阶段先设计接口不实现
10. **FAISS 同期替换（用户新增指令）**：设计向量层迁移——
    a. 盘点 `retrieval/vector_store.py`（FaissStore）接口（add/query/dimension/id_map）与调用方（query_router 的 faiss_index 使用点，含 _fusion_retrieve vector 通道、_entity_expansion、_finish）
    b. OverGraph HNSW 替代：embedding 存节点属性（dense 512d）→ `vector_search(mode="dense", k, dense_query, label_filter, scope_start_node_id, scope_max_depth, scope_at_epoch)`；**scope_start_node_id 需引擎内部 ID(int)**——适配层 elementKey→内部 ID 转换
    c. faiss_id_map（ep_xxx → 向量 ID）→ OverGraph elementKey/内部 ID 映射方案
    d. 图作用域搜索强化 _entity_expansion（实体邻域内语义检索）
    e. HNSW vs FAISS FlatL2 召回差异风险（HNSW 近似 vs Flat 精确）——评测等价验证策略（LoCoMo ≥ 60.5%）
    f. 向量写入路径（create_episode 时 embedding 属性）+ 批量重建（启动 FAISS auto-build 等价物）

## 约束
- 不碰 core/llm_client.py；不动 GraphLiteStore 现有实现（并行存在）
- 上层业务代码零改动（只改 config/启动装配）
- 新增测试走公共入口
- 版本：bump v6.0.0（破坏性变更）——本阶段先设计版本方案
- 参考 AGENTS.md 坑清单：SDK 异常类型、静默失败、多语句截断

## 输出格式
接口映射表（方法 → OverGraph 实现 → 风险）+ 设计决策表 + 实现点清单（文件/位置/量级）+ 阶段2 迁移接口设计 + 一句话总结。中文。
