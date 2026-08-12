# v5.26.0 混合检索设计 v2（CC 审查修订版）

## 背景
`_hypergraph_retrieve` 名为"超图检索"实为 FAISS 向量 + 回查详情。补真正的图支撑。
CC 审查发现 2 个 P0 + 3 项优化，本版吸收。

## 方案 v2（手术刀式）

### 1. P0 修复（独立于扩散的既有 bug）：Hebbian 读端关系名断裂

`graph/graphlite_store.py:485` `get_all_hebbian_connections` 用 `HEBBIAN`，
全库写端（core/hebbian.py、api/routes/system.py）都是 `HEBBIAN_CONNECTION`
→ 恒返回空 → `get_all_connections()` 恒 `{}`。
修复：485 行 `HEBBIAN` → `HEBBIAN_CONNECTION`（1 行）+ 契约测试断言非空。

### 2. 图扩散（纯超边共现，不引入 Hebbian 加权）

**store 新增方法** `get_hypergraph_neighbors(seed_ids: list[str], limit: int) -> dict[str, list[dict]]`：
单条 GQL 两跳一次拿全（防 N+1）：
```
MATCH (e:EpisodeNode {id: $sid})-[:HYPEREDGE_MEMBER]-(h:HyperedgeNode)-[:HYPEREDGE_MEMBER]-(e2:EpisodeNode)
WHERE e2.id <> $sid
RETURN DISTINCT e2.id, e2.content, count(h) AS co_occurrence
ORDER BY co_occurrence DESC LIMIT $limit
```
- 返回 {seed_id: [{id, content, co_occurrence}]}
- 走 query_cypher（含熔断门控 + 异常保护）

**router 新增** `_graph_expansion(seeds, top_k_tail_score)`：
- 种子 = 向量 top-K（≤5）
- 调用 store.get_hypergraph_neighbors
- 扩散新节点（不在向量结果中）分数 = `1/(1+co_occurrence) * 向量尾分 * 0.8`
- 保证带 content（去重 key 依赖）
- 扩散失败/异常 → 返回 [] 静默回退（不破坏纯向量）

**集成**：`_hypergraph_retrieve` 向量结果非空时调用图扩散，融合去重后返回。

### 3. 配置（QueryRouterConfig 新增）

```python
graph_expansion_hop: int = 1        # 固定 1 跳（防图爆炸；hop>1 需 visited 防环）
graph_expansion_max: int = 20       # 扩散补充最大条数
graph_expansion_alpha: float = 0.8  # 扩散新节点分数 = 归一化共现 × 向量尾分 × α
```

### 4. 降级保护
- store 方法走 query_cypher（熔断门控已有）
- router 层包 try/except：任何图异常 → 返回纯向量结果
- FUSION/降级链不变

### 5. 测试（tests/test_graph_expansion.py + test_hebbian_fix.py）
- 扩散召回：mock store 返回邻居 → 断言新节点进入结果
- 融合分数：扩散节点分数 < 向量尾分但 > 0（可插入尾部）
- 降级：store 抛异常 → 返回纯向量
- content 空过滤：无 content 的扩散节点被剔除
- Hebbian 契约：mock 图数据断言 HEBBIAN_CONNECTION 可查

### 6. 版本
bump v5.26.0（四处同步），version_name: "Graph-Expansion-Retrieval"
