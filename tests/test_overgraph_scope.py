"""
图作用域检索测试（阶段3 D5-D10，v6.0.0 overgraph-only）
======================================================
R1 PoC 定标验证（2026-08-19, overgraph 0.17.0）：
  - 共享超边 co-member 两跳可达（HYPEREDGE_MEMBER 单向 → 引擎内 direction=both）
  - scope append 分 < 种子分（max 锚 × boost 0.9）
  - graphlite 后端 hasattr 守卫 no-op（零回归）
  - 开关关闭 bit 级一致
走公共入口 FUSION/VECTOR retrieve（设计验收：test_overgraph_scope）。
"""
import time

import numpy as np
import pytest

pytestmark = pytest.mark.overgraph

# 依赖缺失 → 整模块 skip（不崩收集；统一顶层导入策略）
pytest.importorskip("overgraph")

from graph.overgraph_store import OverGraphStore  # noqa: E402
from retrieval.vector_index import VectorIndexAdapter  # noqa: E402

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


def _build_router(overgraph_store, adapter, fid_map, encoder, scope_cfg=None):
    from config.settings import ScopeRecallConfig
    from retrieval.query_router import QueryRouter, QueryRouterConfig
    cfg = QueryRouterConfig(
        top_k_vector=10, top_k_l1=5, top_k_keyword=10,
        vector_weight=0.6, tau_weight=0.4,
        scope_recall=scope_cfg or ScopeRecallConfig(),
    )
    return QueryRouter(
        graphlite_store=overgraph_store,
        faiss_index=adapter,
        tfidf_index=None,
        encoder=encoder,
        faiss_id_map=fid_map,
        config=cfg,
    )


_VEC_FIXTURE = pytest.mark.parametrize(
    "overgraph_store", [{"dimension": _DIM}], indirect=True)


def _make_scope_graph(store):
    """A—共享超边—B + 孤立 C 拓扑（无 A↔B 直达边，纯超边两跳）。"""
    A = store.create_episode({"content": "Alan Turing was born in London",
                              "created_at": time.time()})
    B = store.create_episode({"content": "He later moved to Princeton",
                              "created_at": time.time()})
    # D 与查询零字符 gram 重叠（中文内容 vs 英文查询，char_wb gram 不相交）
    # → 主通道（BM25/entity/vector）均不召回，仅经共享超边两跳 scope 可达
    D = store.create_episode({"content": "数据模型研究",
                              "created_at": time.time()})
    C = store.create_episode({"content": "The weather in Tokyo is rainy today",
                              "created_at": time.time()})
    h = store.create_hyperedge_node({"id": f"h_{A}", "name": "shared"})
    store.link_hyperedge_member(h, A)
    store.link_hyperedge_member(h, B)
    store.link_hyperedge_member(h, D)
    return store, A, B, C, D


@_VEC_FIXTURE
def test_scope_recall_fusion_shared_hyperedge(overgraph_store):
    """FUSION 公共入口：共享超边 co-member B 经图作用域两跳 append，分 < 种子分。"""
    store, A, B, C, D = _make_scope_graph(overgraph_store)
    encoder = _HashEncoder()
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map={})
    adapter.rebuild([{"node_id": A, "embedding": encoder.embed("Alan Turing was born in London")},
                     {"node_id": B, "embedding": encoder.embed("He later moved to Princeton")},
                     {"node_id": D, "embedding": encoder.embed("数据模型研究")},
                     {"node_id": C, "embedding": encoder.embed("The weather in Tokyo is rainy today")}])
    qr = _build_router(store, adapter, {}, encoder)

    from retrieval.query_router import RetrievalLevel
    results = qr.retrieve(
        "Alan Turing was born in London",
        query_embedding=encoder.embed("Alan Turing was born in London"),
        level=RetrievalLevel.FUSION,
    )
    assert results, results
    seeds = [r for r in results if r.get("level") != "scope"]
    scope = [r for r in results if r.get("_source") == "scope"]
    assert seeds, results
    assert any(r["node_id"] == A for r in seeds), results
    # 共享超边 co-member D 主通道零召回，必须经 scope 两跳命中（direction=both；
    # PoC 定标）。B 的 char_wb gram 与查询重叠会被主通道带分召回，不作 scope 断言
    assert any(r["node_id"] == D for r in scope), results
    # 孤立 C 不得经 scope 进入（scope 过滤生效）
    assert not any(r["node_id"] == C for r in scope), results
    # 扩展分 < 最高种子分（max 锚 × boost 0.9）
    max_seed = max(float(r["score"]) for r in seeds)
    assert all(float(r["score"]) < max_seed for r in scope), (scope, max_seed)


@_VEC_FIXTURE
def test_scope_recall_production_entry_no_embedding(overgraph_store):
    """P1-1 生产路径：公共入口不传 query_embedding → scope 必须真触发。

    生产（self_evolving.py:608 / api/routes/search.py:129）调 retrieve() 均不传
    query_embedding → 修复前 _scope_retrieve 见 None 恒 return，scope 永不 append；
    修复后内部先 _encode_query(query)（归一化 query）再继续。D 主通道零召回
    （中文 vs 英文 char_wb gram 不相交），仅经共享超边两跳 scope 可达。
    """
    store, A, B, C, D = _make_scope_graph(overgraph_store)
    encoder = _HashEncoder()
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map={})
    adapter.rebuild([{"node_id": A, "embedding": encoder.embed("Alan Turing was born in London")},
                     {"node_id": B, "embedding": encoder.embed("He later moved to Princeton")},
                     {"node_id": D, "embedding": encoder.embed("数据模型研究")},
                     {"node_id": C, "embedding": encoder.embed("The weather in Tokyo is rainy today")}])
    qr = _build_router(store, adapter, {}, encoder)

    from retrieval.query_router import RetrievalLevel
    results = qr.retrieve(
        "Alan Turing was born in London",
        level=RetrievalLevel.FUSION,  # 不传 query_embedding（生产形态）
    )
    assert results, results
    scope = [r for r in results if r.get("_source") == "scope"]
    assert scope, results  # 修复前：query_embedding=None 恒 return → scope 恒空
    assert any(r["node_id"] == D for r in scope), results


@_VEC_FIXTURE
def test_scope_recall_disabled_bit_identical(overgraph_store):
    """scope_recall.enabled=False → 关闭时行为 = 现状（零 scope 条目）。"""
    from config.settings import ScopeRecallConfig
    store, A, B, C, D = _make_scope_graph(overgraph_store)
    encoder = _HashEncoder()
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map={})
    adapter.rebuild([{"node_id": A, "embedding": encoder.embed("Alan Turing was born in London")},
                     {"node_id": B, "embedding": encoder.embed("He later moved to Princeton")},
                     {"node_id": D, "embedding": encoder.embed("数据模型研究")},
                     {"node_id": C, "embedding": encoder.embed("The weather in Tokyo is rainy today")}])
    qr = _build_router(store, adapter, {}, encoder,
                       scope_cfg=ScopeRecallConfig(enabled=False))

    from retrieval.query_router import RetrievalLevel
    results = qr.retrieve(
        "Alan Turing was born in London",
        query_embedding=encoder.embed("Alan Turing was born in London"),
        level=RetrievalLevel.FUSION,
    )
    assert results
    assert not any(r.get("_source") == "scope" for r in results)


def test_scope_recall_graphlite_noop(graphlite_store, mock_faiss_index, mock_encoder):
    """graphlite 后端：无 vector_search_scoped → hasattr 守卫假 no-op（零回归）。"""
    from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel

    store = graphlite_store
    A = store.create_episode({"content": "Alan Turing was born in London",
                              "created_at": time.time()})
    B = store.create_episode({"content": "Alan studied mathematics at Cambridge",
                              "created_at": time.time()})
    h = store.create_hyperedge_node({"id": f"h_{A}", "name": "shared"})
    store.link_hyperedge_member(h, A)
    store.link_hyperedge_member(h, B)

    assert not hasattr(store, "vector_search_scoped")
    cfg = QueryRouterConfig(top_k_vector=10, top_k_l1=5, top_k_keyword=10)
    qr = QueryRouter(
        graphlite_store=store,
        faiss_index=mock_faiss_index,
        tfidf_index=None,
        encoder=mock_encoder,
        faiss_id_map={},
        config=cfg,
    )
    results = qr.retrieve("Alan Turing", level=RetrievalLevel.VECTOR)
    assert isinstance(results, list)
    assert not any(r.get("_source") == "scope" for r in results)


@_VEC_FIXTURE
def test_vector_search_scoped_semantics(overgraph_store):
    """store 原语语义（PoC 定标）：depth=2 命中 co-member / depth=1 不命中 / 种子不存在 → []。"""
    store = overgraph_store
    A = store.create_episode({"content": "alpha", "created_at": time.time()})
    B = store.create_episode({"content": "beta", "created_at": time.time()})
    h = store.create_hyperedge_node({"id": f"h_{A}", "name": "shared"})
    store.link_hyperedge_member(h, A)
    store.link_hyperedge_member(h, B)
    enc = _HashEncoder()
    store.batch_upsert_embeddings([
        {"node_id": A, "embedding": enc.embed("alpha")},
        {"node_id": B, "embedding": enc.embed("beta")},
    ])

    vec = enc.embed("beta")
    hits2 = store.vector_search_scoped(A, k=5, query_vec=vec, max_depth=2)
    assert any(eid == B for eid, _ in hits2), hits2
    hits1 = store.vector_search_scoped(A, k=5, query_vec=vec, max_depth=1)
    assert not any(eid == B for eid, _ in hits1), hits1
    assert store.vector_search_scoped("does_not_exist", k=5, query_vec=vec) == []
    # 返回 (ep_id, score) 契约 + score ∈ [0,1]（cosine；HNSW 近似浮点噪声容忍 1e-6）
    for eid, score in hits2:
        assert isinstance(eid, str) and -1.0 - 1e-6 <= float(score) <= 1.0 + 1e-6
