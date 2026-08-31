"""
VectorIndexAdapter — OverGraph HNSW 向量索引适配器（faiss.Index 鸭子类型）
==========================================================================
SHM v6.0.0 FAISS 同期替换（design_overgraph_vector.md D1/D2/D3/D5/D7/D10）：
backend=overgraph 时替代 svc.faiss_index（主通道），对上层暴露 faiss.Index
兼容接口（search/ntotal/add_with_ids/remove_ids/dimension），上层零改动。

- D2：检索走 store.vector_search_dense（EpisodeNode.dense_vector 一等字段）
- D3：uuid5(ep_id) 映射契约原样保留（faiss_id_map）
- D5：OverGraph vector_search 恒返回 cosine s∈[-1,1]（R1 PoC 定标，无 L2 可用）
      → distance d = 1/s - 1（s>0）；下游 1/(1+d) = s ∈ (0,1] 保持 [0,1] 契约
- D7：remove_ids no-op（节点删即向量删）；仅清理 faiss_id_map 残留
- D10：视觉 _visual_index 保留 FAISS 独立空间，本适配器只管主通道
"""

import uuid

import numpy as np

# 【v5.10 向量索引退化修复】主通道搜索同时覆盖 EpisodeNode 与 CommunityNode：
# 梦境社区摘要节点（占节点多数）参与向量检索，不再只有 37 个 EpisodeNode。
# Hebbian 建边 / RPE 写门等内部直调 store.vector_search_dense 仍走默认
# EpisodeNode-only（保持既有语义），此处仅放宽主检索通道。
_SEARCH_LABELS = ("EpisodeNode", "CommunityNode")


def faiss_id(ep_id: str) -> int:
    """uuid5(ep_id) 映射契约（D3，与 write.py flush 路径一致）。"""
    return int(uuid.uuid5(uuid.NAMESPACE_OID, str(ep_id)).int & ((1 << 63) - 1))


class VectorIndexAdapter:
    """faiss.Index 鸭子类型 —— OverGraph HNSW 主通道。"""

    def __init__(self, store, dimension: int = 512, faiss_id_map: dict | None = None):
        self._store = store
        self._dim = int(dimension)
        # faiss_id_map 为共享引用（app.py svc.faiss_id_map / query_router 同一对象），
        # 所有更新必须 in-place（clear/update/pop），禁止整体替换破坏引用
        self.faiss_id_map = faiss_id_map if faiss_id_map is not None else {}
        self._count = 0

    # ── faiss.Index 兼容面 ────────────────────────────

    @property
    def index(self):
        """暴露自身（VisualVectorStore.index 等价物）。"""
        return self

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def ntotal(self) -> int:
        """当前索引向量数 = faiss_id_map 规模（flush/rebuild 同步更新）。"""
        return max(self._count, len(self.faiss_id_map))

    @property
    def id_map(self) -> dict[int, str]:
        return self.faiss_id_map

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(distances, indices) 双 (1,k) 数组 —— faiss 语义。

        R1 定标（2026-08-19 overgraph 0.17.0）：store.vector_search_dense 返回
        cosine s∈[-1,1]（相同→1.0，正交→0.0，相反→-1.0），无 L2 距离可用 →
        d = 1/s - 1（D5），下游 1/(1+d) = s ∈ (0,1]；s≤0（正交/相反）视为非近邻
        剔除（FAISS 语义），不足 k 补 -1。search 幂等回填 faiss_id_map
        （冷启动空 map → get 落空防护，D3）。
        """
        if k <= 0:
            return (np.empty((1, 0), dtype=np.float32),
                    np.empty((1, 0), dtype=np.int64))
        query_vec = np.asarray(query, dtype=np.float32).reshape(-1)
        # 【v5.10 向量索引退化修复】OverGraph label_filter 为 AND 语义：
        # {"labels": ["EpisodeNode","CommunityNode"]} 要求节点同时具备两个 label
        # → 恒空。必须按 label 逐次查询后按 score 合并去重（实测 2026-08-31）。
        hits = self._store.vector_search_dense(
            k, query_vec, label_filter=list(_SEARCH_LABELS)
        )
        if not hits and len(_SEARCH_LABELS) > 1:
            merged: dict[str, float] = {}
            for lbl in _SEARCH_LABELS:
                for ep_id, s in self._store.vector_search_dense(
                    k, query_vec, label_filter=[lbl]
                ):
                    if s > 0.0 and (ep_id not in merged or s > merged[ep_id]):
                        merged[ep_id] = s
            hits = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:k]
        distances: list[float] = []
        indices: list[int] = []
        for ep_id, s in hits:
            if s <= 0.0:
                continue
            fid = faiss_id(ep_id)
            self.faiss_id_map.setdefault(fid, ep_id)  # 幂等回填
            # d = 1/s - 1；钳制 d ≥ 0（float32 cosine 可能略 >1 → d<0 →
            # 下游 1/(1+d) >1 破坏 [0,1] 契约）
            distances.append(max(0.0, 1.0 / s - 1.0))
            indices.append(fid)
        n = len(distances)
        if n < k:
            distances.extend([float("inf")] * (k - n))
            indices.extend([-1] * (k - n))
        return (np.array([distances], dtype=np.float32),
                np.array([indices], dtype=np.int64))

    def add_with_ids(self, embeddings: np.ndarray, ids: np.ndarray) -> int:
        """批量添加（防御性）：反查 faiss_id_map → node_id → 写 dense_vector。

        正常路径由 api/routes/_deps.flush_faiss_buffer 的 overgraph 分支直调
        store.batch_upsert_embeddings（buffer 含 ep_id）；此处供 map 已就绪的
        调用兜底（map 无该 id → 跳过，不落库）。

        【v5.10 向量索引退化修复】map 现可含 CommunityNode fid → 按节点实际
        label 写向量（EpisodeNode/CommunityNode），避免把社区节点错写成
        EpisodeNode 产生幻影节点。
        """
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        nodes: list[dict] = []
        for i, fid in enumerate(ids):
            node_id = self.faiss_id_map.get(int(fid))
            if node_id is None:
                continue
            label = None
            for cand in ("EpisodeNode", "CommunityNode"):
                if self._store.get_node_internal_id(node_id, label=cand) is not None:
                    label = cand
                    break
            if label is None:
                continue  # 节点已不存在 → 不写幻影向量
            nodes.append({
                "node_id": node_id,
                "embedding": embeddings[i],
                "label": label,
            })
        if not nodes:
            return 0
        added = self._store.batch_upsert_embeddings(nodes)
        self._count += added
        return added

    def remove_ids(self, ids: np.ndarray) -> int:
        """D7 no-op：OverGraph 节点删即向量删（dense_vector 随节点生命周期）；
        仅 in-place 清理 faiss_id_map 残留。"""
        removed = [int(i) for i in np.asarray(ids).reshape(-1)]
        for fid in removed:
            self.faiss_id_map.pop(fid, None)
        self._count = max(0, self._count - len(removed))
        return len(removed)

    # ── OverGraph 专属 ────────────────────────────────

    def rebuild(self, nodes: list[dict]) -> int:
        """全量重建：覆盖式写 dense_vector + 重建 faiss_id_map（in-place）。

        nodes: [{"node_id": str, "embedding": vec}]（system.py rebuild_index
        overgraph 分支调用）。
        """
        count = self._store.batch_upsert_embeddings(nodes)
        self.faiss_id_map.clear()
        self.faiss_id_map.update(
            {faiss_id(n["node_id"]): n["node_id"] for n in nodes}
        )
        self._count = count
        return count
