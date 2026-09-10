"""
启动向量加载回归测试（规格: .shm-spec/index-load-spec.md）
==========================================================
背景：服务每次重启都全量重编码建向量索引（6946 节点 GPU fp16 ~2.5 分钟），
但向量已随节点 dense_vector 落库 → 启动优先加载，命中即跳过编码。

覆盖三处交付（均走真实实现，不 mock 被测逻辑本身）：
  A. store.iter_persisted_vectors —— 只回非空 dense_vector；label 过滤；分页
  B. adapter.load_persisted —— 重建 faiss_id_map/计数，绝不调 encoder、不写库
  C. system._load_index_from_store / rebuild_index 路由 —— 命中走 load，
     未命中回落全量编码；env SHM_INDEX_LOAD=0 强制跳过加载

注入点仅在边界：假 db 视图 / 假 encoder / 假 store，被测函数体原样执行。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.overgraph

# 依赖缺失 → 整模块 skip（graph/overgraph_store.py 导入期即要求 overgraph）
pytest.importorskip("overgraph")

from graph.overgraph_store import (  # noqa: E402
    OverGraphStore, LABEL_EPISODE, LABEL_COMMUNITY, VECTOR_LOAD_PAGE,
)
from retrieval.vector_index import VectorIndexAdapter, faiss_id  # noqa: E402

_DIM = 8


# ─── 边界替身：假 db 视图 / 假 store / 哨兵 encoder ──────────


class _View:
    """节点 view 鸭子类型（只暴露加载路径读的 key / dense_vector）。"""

    def __init__(self, key: str, dense_vector=None):
        self.key = key
        self.dense_vector = dense_vector


class _FakeDB:
    """只实现整表 API 的假引擎（无分页 → iter_persisted_vectors 走回落分支）。"""

    def __init__(self, by_label: dict):
        self._by_label = by_label
        self.calls: list[str] = []

    def get_nodes_by_labels(self, label):
        self.calls.append(label)
        return list(self._by_label.get(label, []))


class _PagedFakeDB:
    """实现分页 API 的假引擎（get_nodes_by_labels 置为断言不可达）。"""

    def __init__(self, views: list):
        self._views = views
        self.page_calls: list[tuple[int, int]] = []

    def get_nodes_by_labels_paged(self, label, limit=1000, offset=0):
        self.page_calls.append((int(limit), int(offset)))
        return self._views[int(offset):int(offset) + int(limit)]

    def get_nodes_by_labels(self, label):  # pragma: no cover - 不应到达
        raise AssertionError("分页 API 可用时不得整表读取（iter_persisted_vectors）")


class _ExplodingEncoder:
    """哨兵编码器：任何编码调用即失败（纯加载路径必须零编码）。"""

    def embed(self, *a, **k):
        raise AssertionError("encoder.embed called: load path must not encode")

    def embed_batch(self, *a, **k):
        raise AssertionError("encoder.embed_batch called: load path must not encode")


class _CountingEncoder:
    """计数编码器：返回确定性向量（验证全量重建路径确已执行）。"""

    def __init__(self):
        self.batch_calls = 0

    def embed(self, text: str) -> np.ndarray:
        return np.full(_DIM, 0.5, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self.batch_calls += 1
        return np.tile(np.full(_DIM, 0.5, dtype=np.float32), (len(texts), 1))


class _RouteStore:
    """rebuild/load 路径所需的 store 面（GQL 文本读 + 已存向量读 + 写入计数）。

    query_cypher 只认 _fetch_index_items 的两条 GQL；写入方法计数以便断言
    加载路径零写入。iter_persisted_vectors 挂为实例属性 —— 便于「老 store
    无此方法」的鸭子类型测试直接 del。
    """

    def __init__(self, items: list[tuple[str, str, str]], vectors: list[dict]):
        self._items = items
        self._vectors = vectors
        self.writes = 0

        def _iter_persisted_vectors(labels=None):
            return list(self._vectors)

        self.iter_persisted_vectors = _iter_persisted_vectors

    def query_cypher(self, gql: str, params=None):
        if ":EpisodeNode)" in gql:
            return [{"id": nid, "content": c}
                    for nid, c, lbl in self._items if lbl == LABEL_EPISODE]
        if ":CommunityNode)" in gql:
            return [{"id": nid, "content": c}
                    for nid, c, lbl in self._items if lbl == LABEL_COMMUNITY]
        return []  # Hebbian DELETE 等

    def batch_upsert_embeddings(self, nodes):
        self.writes += 1
        return len(nodes)

    def vector_search_dense(self, k, query_vec, label_filter=None):
        return []


def _vec(fill: float) -> np.ndarray:
    return np.full(_DIM, fill, dtype=np.float32)


def _store_with(db) -> OverGraphStore:
    """真实 OverGraphStore + 注入假 db（不 connect，只测只读映射逻辑）。"""
    store = OverGraphStore()
    store._db = db
    return store


# ─── A: iter_persisted_vectors ─────────────────────────────


def test_iter_persisted_vectors_skips_empty_and_missing_vectors():
    """带/不带 dense_vector 的 view → 只回非空；默认覆盖 Episode+Community。"""
    db = _FakeDB({
        LABEL_EPISODE: [
            _View("ep-1", _vec(1.0)),
            _View("ep-2", None),                      # 无向量 → 跳过
            _View("ep-3", np.zeros(0, dtype=np.float32)),   # 空向量 → 跳过
        ],
        LABEL_COMMUNITY: [_View("cm-1", _vec(2.0))],
    })
    store = _store_with(db)

    rows = store.iter_persisted_vectors()

    assert [r["node_id"] for r in rows] == ["ep-1", "cm-1"]
    assert [r["label"] for r in rows] == [LABEL_EPISODE, LABEL_COMMUNITY]
    assert rows[0]["embedding"].dtype == np.float32
    assert db.calls == [LABEL_EPISODE, LABEL_COMMUNITY], "默认 label 集固定"


def test_iter_persisted_vectors_label_filter():
    """labels 显式传入 → 只读该 label（不触碰其他 label）。"""
    db = _FakeDB({
        LABEL_EPISODE: [_View("ep-1", _vec(1.0))],
        LABEL_COMMUNITY: [_View("cm-1", _vec(2.0))],
    })
    store = _store_with(db)

    rows = store.iter_persisted_vectors(labels=[LABEL_COMMUNITY])

    assert [r["node_id"] for r in rows] == ["cm-1"]
    assert db.calls == [LABEL_COMMUNITY]


def test_iter_persisted_vectors_paged_read():
    """引擎提供分页 API → 分页读取全部节点（大库不一次性物化）。"""
    views = [_View(f"ep-{i}", _vec(1.0)) for i in range(VECTOR_LOAD_PAGE + 5)]
    db = _PagedFakeDB(views)
    store = _store_with(db)

    rows = store.iter_persisted_vectors(labels=[LABEL_EPISODE])

    assert len(rows) == VECTOR_LOAD_PAGE + 5
    assert len(db.page_calls) >= 2, "超过一页必须继续翻页"
    assert db.page_calls[0][1] == 0  # 首页 offset=0


@pytest.mark.parametrize("overgraph_store", [{"dimension": _DIM}], indirect=True)
def test_iter_persisted_vectors_real_engine(overgraph_store):
    """真引擎：dense_vector 落库后可读回（load 路径的数据源）。

    维度须与 _DIM 一致（经 indirect parametrize 注入）——引擎强制
    dense_vector 长度 == 配置维度，默认 fixture 为 512。
    """
    import time
    import uuid

    store = overgraph_store
    eid = store.create_episode({"content": "启动加载测试", "created_at": 1.0})
    cid = str(uuid.uuid4())
    # CommunityNode 走 typed upsert（与 dream 社区摘要落库等价）
    with store.batch_write_txn() as (txn, db):
        txn.stage([{"op": "upsert_node", "labels": [LABEL_COMMUNITY], "key": cid,
                    "props": {"id": cid, "name": "load_test_comm",
                              "summary": "社区摘要:启动加载", "created_at": time.time()}}])
    store.batch_upsert_embeddings([
        {"node_id": eid, "embedding": _vec(1.0)},
        {"node_id": cid, "embedding": _vec(3.0), "label": LABEL_COMMUNITY},
    ])

    rows = store.iter_persisted_vectors()

    by_id = {r["node_id"]: r for r in rows}
    assert eid in by_id and cid in by_id, rows
    assert by_id[eid]["label"] == LABEL_EPISODE
    assert by_id[cid]["label"] == LABEL_COMMUNITY
    assert float(by_id[cid]["embedding"][0]) == pytest.approx(3.0)

    # label 过滤：只取社区节点
    only_comm = store.iter_persisted_vectors(labels=[LABEL_COMMUNITY])
    assert [r["node_id"] for r in only_comm] == [cid]


# ─── B: adapter.load_persisted ─────────────────────────────


def test_load_persisted_rebuilds_map_in_place_without_writing():
    """加载后计数/映射正确；共享 faiss_id_map 原位更新；store 零写入。"""
    db = _FakeDB({LABEL_EPISODE: [
        _View("ep-1", _vec(1.0)), _View("ep-2", _vec(2.0)),
    ]})
    store = _store_with(db)

    def _forbidden(*a, **k):  # 加载路径写库即失败
        raise AssertionError("load_persisted must not write to store")

    store.batch_upsert_embeddings = _forbidden
    shared_map: dict = {}
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map=shared_map)

    count = adapter.load_persisted(store.iter_persisted_vectors())

    assert count == 2
    assert adapter.ntotal == 2
    assert adapter.faiss_id_map is shared_map, "必须原位更新共享引用（不能整体替换）"
    assert shared_map == {faiss_id("ep-1"): "ep-1", faiss_id("ep-2"): "ep-2"}


def test_load_persisted_overwrites_stale_map():
    """加载为覆盖式：旧映射残留被清掉（镜像 rebuild 语义）。"""
    store = _store_with(_FakeDB({}))
    adapter = VectorIndexAdapter(
        store=store, dimension=_DIM,
        faiss_id_map={faiss_id("stale"): "stale"},
    )

    count = adapter.load_persisted([{"node_id": "ep-9", "embedding": _vec(1.0)}])

    assert count == 1
    assert adapter.faiss_id_map == {faiss_id("ep-9"): "ep-9"}
    assert faiss_id("stale") not in adapter.faiss_id_map


# ─── C: _load_index_from_store + 路由接入 ─────────────────


def _items(n_ep: int = 2, n_cm: int = 1) -> list[tuple[str, str, str]]:
    out = [(f"ep-{i}", f"episode 内容 {i}", LABEL_EPISODE) for i in range(n_ep)]
    out += [(f"cm-{i}", f"社区摘要 {i}", LABEL_COMMUNITY) for i in range(n_cm)]
    return out


def _vectors(n: int) -> list[dict]:
    return [{"node_id": f"ep-{i}", "embedding": _vec(1.0), "label": LABEL_EPISODE}
            for i in range(n)]


def test_load_index_from_store_hit_skips_encoding():
    """命中：mode=load + loaded_count，encoder 零调用、store 零写入。"""
    from api.routes.system import _load_index_from_store

    items = _items()                      # 期望 3 个节点
    store = _RouteStore(items, _vectors(3))
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map={})
    deps = SimpleNamespace(graph_store=store, encoder=_ExplodingEncoder(),
                           faiss_index=adapter, tfidf_index=None)

    result = _load_index_from_store(deps, adapter)

    assert result is not None
    assert result["status"] == "ok"
    assert result["mode"] == "load"
    assert result["loaded_count"] == 3
    assert result["total_nodes"] == 3
    assert adapter.ntotal == 3
    assert store.writes == 0, "纯加载路径不得写库"


def test_load_index_from_store_miss_returns_none():
    """未命中（已存向量 < 50% 期望）→ None，调用方回落全量重建。"""
    from api.routes.system import _load_index_from_store

    store = _RouteStore(_items(), _vectors(0))   # 期望 3，阈值 1（max(1,1.5)）
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map={})
    deps = SimpleNamespace(graph_store=store, encoder=_ExplodingEncoder(),
                           faiss_index=adapter, tfidf_index=None)

    assert _load_index_from_store(deps, adapter) is None
    assert adapter.ntotal == 0
    assert store.writes == 0


def test_load_index_from_store_absent_api_returns_none():
    """老 store 无 iter_persisted_vectors → None（不抛异常，回落原路径）。"""
    from api.routes.system import _load_index_from_store

    store = _RouteStore(_items(), _vectors(3))
    del store.iter_persisted_vectors          # 鸭子类型缺方法
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map={})
    deps = SimpleNamespace(graph_store=store, encoder=_ExplodingEncoder(),
                           faiss_index=adapter, tfidf_index=None)

    assert _load_index_from_store(deps, adapter) is None


def test_rebuild_route_default_env_hits_load(monkeypatch):
    """路由默认 SHM_INDEX_LOAD=1：命中即返回 load 结果，完全不触发编码。"""
    import api.routes.system as system

    monkeypatch.delenv("SHM_INDEX_LOAD", raising=False)
    store = _RouteStore(_items(), _vectors(3))
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map={})
    deps = SimpleNamespace(graph_store=store, encoder=_ExplodingEncoder(),
                           faiss_index=adapter, tfidf_index=None)

    result = asyncio.run(system.rebuild_index(deps))

    assert result["mode"] == "load"
    assert result["loaded_count"] == 3
    assert store.writes == 0


def test_rebuild_route_env_zero_forces_full_rebuild(monkeypatch):
    """SHM_INDEX_LOAD=0：强制跳过加载，走原全量编码路径（回滚开关）。"""
    import api.routes.system as system

    monkeypatch.setenv("SHM_INDEX_LOAD", "0")
    store = _RouteStore(_items(), _vectors(3))     # 向量齐备——若未跳过必然命中
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map={})
    encoder = _CountingEncoder()
    deps = SimpleNamespace(graph_store=store, encoder=encoder,
                           faiss_index=adapter, tfidf_index=None)

    result = asyncio.run(system.rebuild_index(deps))

    assert encoder.batch_calls == 1, "env=0 必须走全量编码"
    assert result.get("mode") != "load"
    assert result["indexed_count"] == 3


def test_rebuild_route_miss_falls_back_to_rebuild(monkeypatch):
    """未命中：路由回落全量编码路径（行为与优化前一致）。"""
    import api.routes.system as system

    monkeypatch.delenv("SHM_INDEX_LOAD", raising=False)
    store = _RouteStore(_items(), _vectors(0))
    adapter = VectorIndexAdapter(store=store, dimension=_DIM, faiss_id_map={})
    encoder = _CountingEncoder()
    deps = SimpleNamespace(graph_store=store, encoder=encoder,
                           faiss_index=adapter, tfidf_index=None)

    result = asyncio.run(system.rebuild_index(deps))

    assert encoder.batch_calls == 1
    assert result.get("mode") != "load"
    assert result["indexed_count"] == 3
