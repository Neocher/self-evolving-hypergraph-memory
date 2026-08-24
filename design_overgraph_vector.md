# OverGraph 迁移设计定稿（CC 两轮 + PoC 8 轮实证）

## A. 基础设计（OverGraphStore 适配层）
1. **接口映射**：高层方法用 typed API（upsert/get_node_by_key/neighbors），裸 GQL 用白名单翻译层（收敛 101 处裸 GQL 面）
2. **elementKey = node_id 直映**（每 label 唯一、一级索引、get_node_by_key O(1)）；`id` 同时落 props 一份保读侧零改动
3. **b64 整体移除**：OverGraphStore 零 b64（中文原生）；读侧 helper 保留兼容 GraphLite 遗留库（startswith 判空 no-op）
4. **事务/乐观锁**：版本号 CAS 用单条 execute_gql 原子 mutation（WHERE version=$v，读 nodes_updated——PoC 已验证重复 CAS 返回 0）；begin_write_txn 只用于多语句原子（create_property_version 补偿链、超边批量）
5. **超边映射**：保留辅助节点模型（HyperedgeNode + HYPEREDGE_MEMBER 边，weight 边属性——PoC 已验证 SET rel.weight + ORDER BY）
6. **时间锚**：business created_at 保留自定义 float 秒属性（引擎内置 int 不可写）；内置仅审计元数据
7. **错误处理**：overgraph.OverGraphError → 熔断器 _INFRA_EXCEPTIONS 唯一成员；写路径 execute_cypher 熔断中立、读路径 query_cypher 永不抛契约不变
8. **config**：Settings.backend（graphlite|overgraph）+ OverGraphConfig(database_path)；make_store(cfg)；属性名仍叫 svc.graphlite_store（duck-typing 零改动）
9. **迁移接口**（阶段2）：StoreMigration.dump_graphlite/load_overgraph/verify；created_at/version/valid_from/tau_initial 原值搬运；遗留 {b64} dump 侧 decode 一次落盘明文
10. **版本**：v6.0.0 四处同步；backend: graphlite 默认使存量零感知

## B. FAISS 同期替换设计（用户指令：一次性）
- **D1**: 新建 `retrieval/vector_index.py`（~180 LOC）：VectorIndexAdapter（faiss.Index 鸭子类型）替代 svc.faiss_index
- **D2**: backend=overgraph → vector_search(mode="dense", k, dense_query, label_filter=["EpisodeNode"]) + `EpisodeNode.dense_vector` 一等字段
- **D3**: uuid5(ep_id) 映射契约原样保留（faiss_id_map）
- **D4**: embedding 存 EpisodeNode.dense_vector（唯一被 HNSW 索引的载体）
- **D5**: 度量优先 l2；否则 cosine 归一化 d=1/s-1（保 1/(1+d) [0,1] 下游）
- **D6**: 写入路径只改 flush_faiss_buffer 分支（ep_id 已在 buffer）
- **D7**: 梦境增量清理 overgraph no-op（节点删即向量删）
- **D8**: 启动重建 batch_upsert dense_vector + Hebbian 改 vector_search（802×1 次）
- **D9**: scope 强化 Phase 1 不接（实体未物化，透传 None）
- **D10**: FAISS 互斥；**视觉 _visual_index 保留**（384d 独立空间，两 backend 都保留，只换主通道）
- **D11**: config `graph.backend` 单开关 + `graph.hnsw.*`
- **R1 高**：VectorHit.score 度量未定 → **实现前 PoC 定标**（2 已知向量节点比对 score 方向/量纲）
- **R3 中**：HNSW 近似召回 < Flat → 802 规模≈精确；ef_search≥k；Jaccard≥0.98 把关
- **R4 中**：分数分布偏移连锁 → 优先 l2 + 归一化 + 端到端比对
- **R8 高**：视觉侧例外（明确 scope 只换主通道）

## C. 实现点清单（合计 ≈560 LOC 生产 + 测试）
| 文件 | 改动 |
|:--|:--|
| graph/overgraph_store.py（新建）| 39 个方法 + _flatten_view + GQL 翻译层 + 向量方法（vector_search_dense/batch_upsert_embeddings/get_episode_keys/get_node_internal_id）+ 复用 CircuitBreaker/EpisodeCache |
| retrieval/vector_index.py（新建）| VectorIndexAdapter + score 映射 |
| config/settings.py + defaults.yaml | GraphConfig.backend/hnsw + OverGraphConfig |
| api/app.py | make_store + backend 分支构造 adapter |
| api/routes/_deps.py | flush_faiss_buffer 分支 + incremental no-op |
| api/routes/system.py | rebuild_index 分支 + Hebbian 近邻 |
| search.py/write.py/dashboard.py/query_router.py | **零改动** |
| tests/test_overgraph_store.py + test_overgraph_vector.py | 走公共入口 + 与 GraphLiteStore 行为对拍 |
| shm/_version.py | v6.0.0 |

## D. 已证语法契约（PoC）
- 中文写入/查询零 Panic；CONTAINS 子串可用但大小写敏感（P3c 双变体适用）
- elementKey(e)/id(e) 取 ID；e.elementKey 属性返回 None
- 重复 MATCH 建关系（逗号分隔不支持）
- CAS 原子（WHERE version SET）；DETACH DELETE ✅；NOT EXISTS ❌ → IS NULL 替代
- archived IS NULL OR = false ✅；SKIP/LIMIT ✅；batch_upsert_nodes/edges ✅
- neighbors_batch(node_ids, direction, edge_label_filter, at_epoch, decay_lambda) ✅
- WriteTxn（stage/commit/rollback）✅；execute_gql 返回 dict（取 rows）
