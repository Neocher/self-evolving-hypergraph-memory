"""
向量索引退化修复回归测试（v5.10）
================================
问题：node_count=6289 但 vector_index_size=37 —— 5947 个 CommunityNode 无
dense_vector，向量通道只覆盖 EpisodeNode → 检索长期 degraded（graphlite_fallback）。

修复面（三处 + 下游内容回查）：
  A. _rebuild_index_overgraph 同时索引 EpisodeNode + CommunityNode（summary 文本源）
  B. batch_upsert_embeddings label-aware（默认 EpisodeNode 向后兼容）
  C. VectorIndexAdapter.search 主通道同时搜索两种 label（_SEARCH_LABELS）
  D. get_episodes_batch 可解析 CommunityNode key（content 回落 summary）

设计映射：走公共入口（store.batch_upsert_embeddings / adapter.search /
store.get_episodes_batch / _rebuild_index_overgraph）。
"""
import time

import numpy as np
import pytest

pytestmark = pytest.mark.overgraph

# 依赖缺失 → 整模块 skip（不崩收集）
pytest.importorskip("overgraph")

from graph.overgraph_store import OverGraphStore  # noqa: E402
from retrieval.vector_index import VectorIndexAdapter, faiss_id  # noqa: E402

_DIM = 384


class _HashEncoder:
    """确定性编码器：同文本同向量（与 test_overgraph_vector 同构）。"""

    def __init__(self, dim=_DIM):
        self.dim = dim
        self.dimension = dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(self.dim).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


_VEC_FIXTURE = pytest.mark.parametrize(
    "overgraph_store", [{"dimension": _DIM}], indirect=True)


def _make_community(store, summary: str, members: list[str] | None = None) -> str:
    """创建 CommunityNode（id = props['id']，elementKey 一致；无 content 字段）。"""
    import uuid
    cid = str(uuid.uuid4())
    props = {
        "id": cid,
        "name": "dream_test_comm",
        "summary": summary,
        "leiden_score": 0.0,
        "created_at": time.time(),
    }
    # 走 typed upsert（与 dream_candidate_store._persist_community_nodes 等价）
    with store.batch_write_txn() as (txn, db):
        txn.stage([{"op": "upsert_node", "labels": ["CommunityNode"],
                    "key": cid, "props": props}])
        if members:
            for mid in members:
                txn.stage([{"op": "upsert_edge",
                            "from": {"labels": ["CommunityNode"], "key": cid},
                            "to": {"labels": ["EpisodeNode"], "key": mid},
                            "label": "COMMUNITY_MEMBER", "props": {}}])
    return cid


# ─── B: batch_upsert_embeddings label-aware ─────────────────


@_VEC_FIXTURE
def test_batch_upsert_embeddings_community_label(overgraph_store):
    """label 透传：CommunityNode 可写 dense_vector（修复前恒 EpisodeNode）。"""
    store = overgraph_store
    eid = store.create_episode({"content": "episode内容"})
    cid = _make_community(store, "社区摘要文本")
    v_ep = np.zeros(_DIM, dtype=np.float32)
    v_ep[0] = 1.0
    v_cm = np.zeros(_DIM, dtype=np.float32)
    v_cm[1] = 1.0
    added = store.batch_upsert_embeddings([
        {"node_id": eid, "embedding": v_ep},
        {"node_id": cid, "embedding": v_cm, "label": "CommunityNode"},
    ])
    assert added == 2
    # 社区节点经 label 过滤可被检索到（引擎 label_filter 为 AND 语义，多 label
    # 列表恒空 → 必须按单 label 查询；2026-08-31 实测）
    hits = store.vector_search_dense(
        3, v_cm, label_filter=["CommunityNode"])
    hit_ids = {h[0] for h in hits}
    assert cid in hit_ids, hits


@_VEC_FIXTURE
def test_batch_upsert_embeddings_default_label_backward_compat(overgraph_store):
    """不传 label → 默认 EpisodeNode（既有 flush/rebuild 调用零改动）。"""
    store = overgraph_store
    eid = store.create_episode({"content": "默认label"})
    v = np.zeros(_DIM, dtype=np.float32)
    v[0] = 1.0
    added = store.batch_upsert_embeddings([{"node_id": eid, "embedding": v}])
    assert added == 1
    hits = store.vector_search_dense(3, v)  # 默认 EpisodeNode-only
    assert any(h[0] == eid for h in hits)


# ─── C: VectorIndexAdapter.search 主通道覆盖两种 label ──────


@_VEC_FIXTURE
def test_adapter_search_returns_community_hits(overgraph_store):
    """主检索通道：adapter.search 命中 CommunityNode（修复前只查 EpisodeNode）。"""
    store = overgraph_store
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map={})
    eid = store.create_episode({"content": "天气很好"})
    cid = _make_community(store, "这个社区讲天气与气候")
    v_ep = np.zeros(_DIM, dtype=np.float32)
    v_ep[0] = 1.0
    v_cm = np.zeros(_DIM, dtype=np.float32)
    v_cm[0] = 1.0
    v_cm[1] = 0.3
    v_cm /= np.linalg.norm(v_cm)
    adapter.rebuild([
        {"node_id": eid, "embedding": v_ep},
        {"node_id": cid, "embedding": v_cm, "label": "CommunityNode"},
    ])
    assert adapter.ntotal == 2
    D, I = adapter.search(v_cm.reshape(1, -1), 3)
    hit_fids = {int(i) for i in I[0] if int(i) >= 0}
    assert faiss_id(cid) in hit_fids, I[0].tolist()


# ─── D: get_episodes_batch 解析 CommunityNode key ───────────


@_VEC_FIXTURE
def test_get_episodes_batch_resolves_community_summary(overgraph_store):
    """内容回查：CommunityNode key → summary 回落为 content（下游非空）。"""
    store = overgraph_store
    eid = store.create_episode({"content": "ep内容", "created_at": time.time()})
    cid = _make_community(store, "社区摘要:多智能体协作")
    eps = store.get_episodes_batch([eid, cid])
    by_id = {ep.get("id"): ep for ep in eps}
    assert eid in by_id and by_id[eid]["content"] == "ep内容"
    assert cid in by_id, by_id.keys()
    assert by_id[cid]["content"] == "社区摘要:多智能体协作"


# ─── A: _rebuild_index_overgraph 端到端 ────────────────────


@_VEC_FIXTURE
def test_rebuild_index_includes_communities(overgraph_store):
    """_rebuild_index_overgraph：向量索引覆盖 EpisodeNode + CommunityNode。"""
    store = overgraph_store
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map={})
    eid = store.create_episode({"content": "太极多智能体记忆系统", "created_at": time.time()})
    cid = _make_community(store, "社区摘要：多智能体协作记忆", members=[eid])
    encoder = _HashEncoder()

    # 构造最小 Services 鸭子类型（只取 _rebuild_index_overgraph 用到的成员）
    from types import SimpleNamespace
    deps = SimpleNamespace(
        graph_store=store,
        encoder=encoder,
        faiss_index=adapter,
        tfidf_index=None,
    )
    from api.routes.system import _rebuild_index_overgraph
    result = _rebuild_index_overgraph(deps, adapter)
    assert result["status"] == "ok"
    assert result["indexed_count"] == 2, result
    assert result["total_nodes"] == 2, result

    # 修复后：社区摘要可被向量检索命中（node_id=cid；引擎 label_filter AND 语义
    # → 按单 label 查询，2026-08-31 实测）
    hits = store.vector_search_dense(
        3, encoder.embed("多智能体协作记忆"),
        label_filter=["CommunityNode"])
    hit_ids = {h[0] for h in hits}
    assert cid in hit_ids, hits
