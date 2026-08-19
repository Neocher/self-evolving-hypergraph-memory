"""
VectorIndexAdapter 集成测试（v6.0.0 OverGraph HNSW 主通道）
==========================================================
R1 PoC 定标验证：score 方向/量纲、d=1/s-1 映射、faiss_id_map 幂等回填、
backend 单开关切换。走公共入口（retrieve/_vector_retrieve/search 语义）。

设计映射：D1（faiss.Index 鸭子类型）/ D2（vector_search dense）/
D3（uuid5 契约）/ D5（score 映射）/ D7（remove no-op）/ D10（互斥）。
"""
import time
import uuid

import numpy as np
import pytest

pytestmark = pytest.mark.overgraph

# 依赖缺失 → 整模块 skip（不崩收集；统一顶层导入策略，P2#8）
pytest.importorskip("overgraph")

from graph.overgraph_store import OverGraphStore  # noqa: E402
from retrieval.vector_index import VectorIndexAdapter, faiss_id  # noqa: E402
from config.settings import GraphConfig  # noqa: E402


def _make_adapter(overgraph_store, dimension=512):
    faiss_id_map: dict[int, str] = {}
    adapter = VectorIndexAdapter(
        store=overgraph_store, dimension=dimension, faiss_id_map=faiss_id_map)
    return adapter, faiss_id_map


# ─── score 方向 / 量纲（R1 定标）───────────────────────

_DIM = 384
_VEC_FIXTURE = pytest.mark.parametrize(
    "overgraph_store", [{"dimension": _DIM}], indirect=True)


@_VEC_FIXTURE
def test_search_score_direction_and_mapping(overgraph_store):
    """cosine s → d=1/s-1 → 下游 1/(1+d)=s ∈ (0,1]；s≤0 剔除补 -1。"""
    store = overgraph_store
    adapter, fid_map = _make_adapter(store, dimension=_DIM)
    # 基向量保证精确正交（float 精度下 s 可能残留 1e-5）
    v_anchor = np.zeros(_DIM, dtype=np.float32)
    v_anchor[0] = 1.0
    v_near = v_anchor.copy()
    v_near[1] = 0.5
    v_near /= np.linalg.norm(v_near)
    v_far = v_anchor.copy()
    v_far[1] = 1.0
    v_far[2] = 0.2
    v_far /= np.linalg.norm(v_far)
    v_orth = np.zeros(_DIM, dtype=np.float32)
    v_orth[1] = 1.0  # 与 v_anchor 精确正交（cosine=0）

    eids = [store.create_episode({"content": f"v{i}"}) for i in range(4)]
    adapter.rebuild([
        {"node_id": eids[0], "embedding": v_anchor},
        {"node_id": eids[1], "embedding": v_near},
        {"node_id": eids[2], "embedding": v_far},
        {"node_id": eids[3], "embedding": v_orth},
    ])

    D, I = adapter.search(v_anchor.reshape(1, -1), 5)
    # top1 = 自身（cosine 1.0 → d=0 → 下游 1）
    assert I[0][0] == faiss_id(eids[0])
    assert abs(float(D[0][0])) < 1e-3
    sim_top = 1.0 / (1.0 + float(D[0][0]))
    assert abs(sim_top - 1.0) < 1e-3
    # 排名单调：near > far > orth（orth 被剔除 → -1 填充）
    order = {int(i): rank for rank, i in enumerate(I[0])}
    assert order[faiss_id(eids[1])] < order[faiss_id(eids[2])]
    orth_fid = faiss_id(eids[3])
    assert orth_fid not in order  # s≤0 非近邻剔除
    # 不足 k 补 -1
    assert -1 in I[0].tolist()


@_VEC_FIXTURE
def test_search_score_direction_identical_vs_orthogonal(overgraph_store):
    """相同向量 → score 1.0；正交 → 剔除（FAISS 语义非 top-k）。"""
    store = overgraph_store
    adapter, _ = _make_adapter(store, dimension=_DIM)
    v = np.zeros(_DIM, dtype=np.float32)
    v[0] = 1.0
    v_orth = np.zeros(_DIM, dtype=np.float32)
    v_orth[1] = 1.0
    e1 = store.create_episode({"content": "a"})
    e2 = store.create_episode({"content": "b"})
    adapter.rebuild([{"node_id": e1, "embedding": v},
                     {"node_id": e2, "embedding": v_orth}])
    D, I = adapter.search(v.reshape(1, -1), 3)
    top = I[0].tolist()
    assert top[0] == faiss_id(e1)
    assert faiss_id(e2) not in top  # 正交剔除
    # 相同向量 1/(1+d) ≈ 1.0
    assert abs(1.0 / (1.0 + float(D[0][0])) - 1.0) < 1e-4


# ─── faiss_id_map 回填 / uuid5 契约（D3）───────────────


@_VEC_FIXTURE
def test_faiss_id_map_backfill_on_search(overgraph_store):
    """冷启动空 map → search 幂等回填（map 落空防护）。"""
    store = overgraph_store
    adapter, fid_map = _make_adapter(store)
    e1 = store.create_episode({"content": "回填一"})
    e2 = store.create_episode({"content": "回填二"})
    v1 = np.zeros(_DIM, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(_DIM, dtype=np.float32)
    v2[1] = 1.0
    # 直接写向量（不经 adapter，模拟 map 未初始化）
    store.batch_upsert_embeddings([{"node_id": e1, "embedding": v1},
                                   {"node_id": e2, "embedding": v2}])
    assert len(fid_map) == 0  # 冷启动空 map
    D, I = adapter.search(v1.reshape(1, -1), 3)
    assert fid_map[faiss_id(e1)] == e1  # search 回填
    assert I[0][0] == faiss_id(e1)


def test_faiss_id_uuid5_contract():
    """uuid5(ep_id) 与写路径（write.py flush）契约一致。"""
    ep_id = "ep_test_contract"
    fid = faiss_id(ep_id)
    expected = int(uuid.uuid5(uuid.NAMESPACE_OID, ep_id).int & ((1 << 63) - 1))
    assert fid == expected
    assert 0 <= fid < (1 << 63)


# ─── ntotal / add / remove（D7）────────────────────────


@_VEC_FIXTURE
def test_ntotal_and_add_with_ids(overgraph_store):
    store = overgraph_store
    adapter, fid_map = _make_adapter(store)
    assert adapter.ntotal == 0
    e1 = store.create_episode({"content": "add一"})
    e2 = store.create_episode({"content": "add二"})
    v1 = np.zeros(_DIM, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(_DIM, dtype=np.float32)
    v2[1] = 1.0
    fid_map[faiss_id(e1)] = e1
    fid_map[faiss_id(e2)] = e2
    added = adapter.add_with_ids(np.stack([v1, v2]),
                                 np.array([faiss_id(e1), faiss_id(e2)], dtype=np.int64))
    assert added == 2
    assert adapter.ntotal == 2
    hits = store.vector_search_dense(3, v1)
    assert hits[0][0] == e1


@_VEC_FIXTURE
def test_remove_ids_noop_semantics(overgraph_store):
    """D7：remove_ids no-op（节点删即向量删）+ faiss_id_map 清理。"""
    store = overgraph_store
    adapter, fid_map = _make_adapter(store)
    e1 = store.create_episode({"content": "rm一"})
    e2 = store.create_episode({"content": "rm二"})
    v1 = np.zeros(_DIM, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(_DIM, dtype=np.float32)
    v2[1] = 1.0
    adapter.rebuild([{"node_id": e1, "embedding": v1},
                     {"node_id": e2, "embedding": v2}])
    assert adapter.ntotal == 2
    removed = adapter.remove_ids(np.array([faiss_id(e1)], dtype=np.int64))
    assert removed == 1
    assert faiss_id(e1) not in fid_map
    assert adapter.ntotal == 1
    # 节点删除后向量自动消失（OverGraph 生命周期语义）
    store.execute_cypher(
        "MATCH (e:EpisodeNode {id: $id}) DETACH DELETE e", {"id": e2})
    hits = store.vector_search_dense(3, v2)
    assert all(ep != e2 for ep, _ in hits)


# ─── backend 单开关（D10 互斥）─────────────────────────


def test_backend_switch_make_store():
    """v6.0.0: make_store(cfg) 恒返回 OverGraphStore（GraphLite 分支已移除）。"""
    from api.app import make_store

    class FakeSettings:
        graphlite = type("g", (), {"database_path": "/tmp/gl_db", "max_threads": 4})()
        overgraph = type("o", (), {"database_path": "/tmp/og_db",
                                   "dense_vector_dimension": 512,
                                   "dense_vector_metric": "cosine"})()
        circuit_breaker = type("cb", (), {})()

    # backend 字段保留一个发布周期（回滚保险），但不再影响构造结果
    for backend in ("graphlite", "overgraph"):
        fs = FakeSettings()
        fs.graph = GraphConfig(backend=backend)
        store = make_store(fs)
        assert isinstance(store, OverGraphStore), (backend, type(store))


# ─── 公共检索入口（_vector_retrieve / retrieve VECTOR）──


def _build_router(overgraph_store, adapter, fid_map, encoder):
    from retrieval.query_router import QueryRouter, QueryRouterConfig
    cfg = QueryRouterConfig(
        top_k_vector=10, top_k_l1=5, top_k_keyword=10,
        vector_weight=0.6, tau_weight=0.4,
    )
    qr = QueryRouter(
        graphlite_store=overgraph_store,
        faiss_index=adapter,
        tfidf_index=None,
        encoder=encoder,
        faiss_id_map=fid_map,
        config=cfg,
    )
    return qr


class _HashEncoder:
    """确定性编码器：同文本同向量（与 conftest mock_encoder 语义一致）。"""

    def __init__(self, dim=384):
        self.dim = dim
        self.dimension = dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(self.dim).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


@_VEC_FIXTURE
def test_vector_retrieve_public_entry(overgraph_store):
    """_vector_retrieve（L2 向量通道生产链路）：score 方向 + 内容回查。"""
    store = overgraph_store
    adapter, fid_map = _make_adapter(store, dimension=384)
    encoder = _HashEncoder()
    qr = _build_router(store, adapter, fid_map, encoder)

    target = "上海今天天气不错"
    other = "完全不同的另一条记录"
    e1 = store.create_episode({"content": target, "created_at": time.time()})
    e2 = store.create_episode({"content": other, "created_at": time.time()})
    v1, v2 = encoder.embed(target), encoder.embed(other)
    adapter.rebuild([{"node_id": e1, "embedding": v1},
                     {"node_id": e2, "embedding": v2}])

    results = qr._vector_retrieve("上海今天天气不错",
                                  encoder.embed("上海今天天气不错"))
    assert results, results
    assert results[0]["node_id"] == e1
    assert results[0]["score"] > 0.9
    assert results[0]["level"] == "vector"
    assert results[0]["content"] == target
    # score ∈ [0,1]（R1 定标 d=1/s-1 映射契约）
    assert all(0.0 <= float(r["score"]) <= 1.0 for r in results)


@_VEC_FIXTURE
def test_retrieve_level_vector_public_entry(overgraph_store):
    """retrieve(level=VECTOR) 公共入口：降级链直达向量通道 + _finish 全通道。

    HashEncoder 以整串哈希为种子 → 查询文本须与节点 content 完全一致
    （_normalize_query 不改写纯中文无术语句）。
    """
    store = overgraph_store
    adapter, fid_map = _make_adapter(store, dimension=_DIM)
    encoder = _HashEncoder()
    qr = _build_router(store, adapter, fid_map, encoder)

    target = "上海今天天气不错"
    e1 = store.create_episode({"content": target, "created_at": time.time()})
    e2 = store.create_episode({"content": "完全无关的另一条记录", "created_at": time.time()})
    v1, v2 = encoder.embed(target), encoder.embed("完全无关的另一条记录")
    adapter.rebuild([{"node_id": e1, "embedding": v1},
                     {"node_id": e2, "embedding": v2}])

    from retrieval.query_router import RetrievalLevel
    results = qr.retrieve(target, level=RetrievalLevel.VECTOR)
    assert results, results
    assert results[0]["node_id"] == e1
    assert results[0]["content"] == target
    assert results[0]["score"] > 0.9


# ─── score 量纲端到端（query_router 1/(1+d) 契约）──────


@_VEC_FIXTURE
def test_score_range_after_normalize(overgraph_store):
    """adapter.search 距离经 query_router 1/(1+d) 后 ∈ [0,1] 且 top1≈1。"""
    store = overgraph_store
    adapter, _ = _make_adapter(store, dimension=384)
    encoder = _HashEncoder()
    qr = _build_router(store, adapter, {}, encoder)
    text = "范围测试"
    e1 = store.create_episode({"content": text, "created_at": time.time()})
    adapter.rebuild([{"node_id": e1, "embedding": encoder.embed(text)}])
    D, I = qr.faiss_index.search(encoder.embed(text).reshape(1, -1), 3)
    sim = 1.0 / (1.0 + float(D[0][0]))
    assert 0.0 <= sim <= 1.0
    assert sim > 0.99
