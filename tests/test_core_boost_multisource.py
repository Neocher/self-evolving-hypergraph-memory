"""检索 core 优先 + 多源支持度 (v5.35.0) 测试。

覆盖任务书测试清单：
1. _vector_retrieve / _hypergraph_retrieve 组装后含 fact_track 键
2. BM25 旧索引兼容 —— 结果无 _bm25_doc_fact_track 属性不抛、缺省 active
3. _deduplicate_and_sort core 轨 ×1.1 温和 boost
4. EvidenceTracker.is_multi_source（同 source → False；不同 source → True）
5. write.py 写多源 → tau_initial 提升至 0.85
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router, Services, get_services
from core.evidence_tracker import EvidenceTracker
from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel


# ─── 改动 4a：向量/超图组装 fact_track 透传 ───────────────────

class _FakeFaiss:
    """返回固定 (distances, indices) 的假 FAISS 索引。"""

    def __init__(self, distances: np.ndarray, indices: np.ndarray):
        self._distances = distances
        self._indices = indices

    def search(self, query, k):
        return (self._distances, self._indices)


class _FakeStore:
    """仅提供 get_episodes_batch 的假 GraphLiteStore。"""

    def __init__(self, episodes: list[dict]):
        self._episodes = episodes

    def get_episodes_batch(self, node_ids):
        return [e for e in self._episodes if e.get("id") in node_ids]


class _FakeTfidf:
    """返回固定 (doc_id, score, content) 列表的假 TF-IDF 索引。"""

    def __init__(self, items: list[tuple]):
        self._items = items

    def search(self, query, top_k):
        return self._items


class _FallbackStore:
    """仅提供 query_cypher 返回固定行的假 GraphLiteStore（L4 fallback 用）。"""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def query_cypher(self, *args, **kwargs):
        return self._rows


def _make_router(episodes: list[dict]) -> QueryRouter:
    return QueryRouter(
        graphlite_store=_FakeStore(episodes),
        faiss_index=_FakeFaiss(np.array([[0.1]]), np.array([[0]])),
        tfidf_index=None,
        faiss_id_map={0: episodes[0]["id"]},
        config=QueryRouterConfig(),
    )


class TestAssemblyFactTrack:
    def test_vector_retrieve_includes_fact_track(self):
        """清单 1：_vector_retrieve 组装后含 fact_track 键（透传 core）。"""
        router = _make_router([{
            "id": "uuid-1", "content": "我是北京人", "tau_initial": 1.0,
            "archived": False, "fact_track": "core",
        }])
        results = router._vector_retrieve("我是北京人", query_embedding=np.zeros((1, 512)))
        assert results and results[0]["fact_track"] == "core"

    def test_vector_retrieve_fact_track_defaults_active(self):
        """清单 1：回查 dict 无 fact_track 字段 → 缺省 active，不抛错。"""
        router = _make_router([{
            "id": "uuid-1", "content": "今天下午开会", "tau_initial": 1.0,
            "archived": False,
        }])
        results = router._vector_retrieve("今天下午开会", query_embedding=np.zeros((1, 512)))
        assert results and results[0]["fact_track"] == "active"

    def test_hypergraph_retrieve_includes_fact_track(self):
        """清单 1：_hypergraph_retrieve 组装后含 fact_track 键。"""
        router = _make_router([{
            "id": "uuid-1", "content": "我住在北京", "tau_initial": 1.0,
            "fact_track": "core",
        }])
        results = router._hypergraph_retrieve("我住在北京", query_embedding=np.zeros((1, 512)))
        assert results and results[0]["fact_track"] == "core"


# ─── 改动 4a：BM25 旧索引兼容 ────────────────────────────────

class _BM25Store:
    """仅提供 query_cypher 的假 GraphLiteStore。"""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def query_cypher(self, *args, **kwargs):
        return self.rows


def _make_bm25_router(rows: list[dict]) -> QueryRouter:
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router.graphlite_store = _BM25Store(rows)
    router._bm25_built = False
    router._build_bm25_index()
    return router


class TestBM25FactTrackCompat:
    def test_old_index_fact_track_defaults_active(self):
        """清单 2：无 _bm25_doc_fact_track 属性（旧索引缓存）→ 不抛、缺省 active。"""
        router = _make_bm25_router([
            {"node_id": "n0", "content": "我喜欢喝茶", "tau_value": 1.0},
            {"node_id": "n1", "content": "今天下午开会", "tau_value": 1.0},
        ])
        # 模拟旧代码构建的索引：无 fact_track 数组
        del router._bm25_doc_fact_track
        results = router._bm25_search("喝茶")
        assert results
        for r in results:
            assert r.get("fact_track") == "active"

    def test_new_index_transmits_fact_track(self):
        """清单 2：新索引带 fact_track → 结果透传 core。"""
        router = _make_bm25_router([
            {"node_id": "n0", "content": "我喜欢喝茶", "tau_value": 1.0, "fact_track": "core"},
            {"node_id": "n1", "content": "今天下午开会", "tau_value": 1.0, "fact_track": "active"},
        ])
        results = router._bm25_search("喝茶")
        assert results
        core = [r for r in results if r["node_id"] == "n0"]
        assert core and core[0]["fact_track"] == "core"


# ─── 改动 4b：_deduplicate_and_sort core boost ────────────────

class TestDeduplicateCoreBoost:
    def test_core_boost_reorders(self):
        """core 轨 ×1.1 后反超 active，排序结果 core 在前（输入低于 1.0 避免钳制掩盖排序）。"""
        results = [
            {"node_id": "a", "content": "active", "score": 0.8, "fact_track": "active"},
            {"node_id": "b", "content": "core", "score": 0.75, "fact_track": "core"},
        ]
        out = QueryRouter._deduplicate_and_sort(results)
        assert out[0]["node_id"] == "b"
        assert out[0]["score"] == pytest.approx(0.75 * 1.1)
        assert out[1]["node_id"] == "a"
        assert out[1]["score"] == 0.8

    def test_non_core_no_boost(self):
        """active / 缺 fact_track 键 → 不 boost。"""
        results = [
            {"node_id": "a", "content": "active", "score": 1.0, "fact_track": "active"},
            {"node_id": "b", "content": "missing", "score": 0.9},
        ]
        out = QueryRouter._deduplicate_and_sort(results)
        assert out[0]["node_id"] == "a" and out[0]["score"] == 1.0
        assert out[1]["node_id"] == "b" and out[1]["score"] == 0.9


# ─── 缺陷 1：L2 路径经 retrieve() _finish 统一应用 core boost ─────────

class TestVectorPathCoreBoost:
    def test_l2_vector_path_applies_core_boost(self):
        """缺陷 1：L2 向量通道经 retrieve() _finish 统一出口获 core ×1.1 boost。

        输入 FAISS distance=1/9 → score=0.9，×1.1=0.99 < 1.0（越界断言改由
        test_user_profile 钳制用例覆盖，此处验证 boost 数学而非钳制边界）。
        """
        router = _make_router([{
            "id": "uuid-1", "content": "我是北京人", "tau_initial": 1.0,
            "archived": False, "fact_track": "core",
        }])
        router.faiss_index = _FakeFaiss(np.array([[1.0 / 9.0]]), np.array([[0]]))
        router.faiss_id_map = {0: "uuid-1"}
        results = router.retrieve(
            "我是北京人", query_embedding=np.zeros((1, 512)),
            level=RetrievalLevel.VECTOR,
        )
        assert results and results[0]["fact_track"] == "core"
        assert results[0]["score"] == pytest.approx(0.9 * 1.1)


# ─── 缺陷 1b：L3 KEYWORD 路径经 retrieve() 统一出口获 core boost ──────

class TestKeywordPathCoreBoost:
    def test_l3_keyword_path_applies_core_boost(self):
        """缺陷 1b：L3 关键词通道经 retrieve() _finish 统一出口获 core ×1.1 boost。

        输入 TF-IDF score=0.9 → ×1.1=0.99 < 1.0（钳制边界用例见 test_user_profile）。
        """
        router = _make_router([{
            "id": "uuid-1", "content": "deep learning framework", "tau_initial": 1.0,
            "archived": False, "fact_track": "core",
        }])
        router.tfidf_index = _FakeTfidf([("uuid-1", 0.9, "deep learning framework")])
        results = router.retrieve("deep learning", level=RetrievalLevel.KEYWORD)
        assert results and results[0]["fact_track"] == "core"
        assert results[0]["score"] == pytest.approx(0.9 * 1.1)


# ─── 缺陷 1c：L4 GraphLite fallback 路径经 retrieve() 获 core boost ───

class TestFallbackPathCoreBoost:
    def test_l4_fallback_applies_core_boost(self):
        """缺陷 1c：L4 全文兜底（TF-IDF 不可用触发）经 _finish 获 core ×1.1 boost。"""
        router = QueryRouter(
            graphlite_store=_FallbackStore([
                {"node_id": "uuid-1", "content": "deep learning framework",
                 "tau_value": 1.0, "fact_track": "core"},
            ]),
            faiss_index=None,
            tfidf_index=None,
            config=QueryRouterConfig(),
        )
        results = router.retrieve("deep learning", level=RetrievalLevel.KEYWORD)
        assert results and results[0]["fact_track"] == "core"
        assert results[0]["score"] == pytest.approx(0.5 * 1.1)


# ─── 缺陷 2（P2）：_finish 先过滤归档再去重，archived 高分不挤掉 active ──

class TestArchivedDedupOrder:
    def test_archived_high_dup_does_not_drop_active(self):
        """P2：同一 content 的 archived 高分项 + active 低分项，active 项保留。"""
        router = _make_router([
            {"id": "uuid-archived", "content": "shared content", "tau_initial": 1.0,
             "archived": True, "fact_track": "active"},
            {"id": "uuid-active", "content": "shared content", "tau_initial": 1.0,
             "archived": False, "fact_track": "active"},
        ])
        router.tfidf_index = _FakeTfidf([
            ("uuid-archived", 1.0, "shared content"),
            ("uuid-active", 0.5, "shared content"),
        ])
        results = router.retrieve("shared content", level=RetrievalLevel.KEYWORD)
        assert results, "archived 高分项挤掉 active 项导致结果为空"
        assert [r["node_id"] for r in results] == ["uuid-active"]
        assert results[0]["score"] == pytest.approx(0.5)


# ─── 缺陷 2：L1 缓存命中路径 fact_track 透传 ─────────────────────

class TestCacheHitFactTrack:
    def test_cache_hit_core_node_gets_fact_track(self):
        """缺陷 2：flush_faiss_buffer 缓存填充带上 fact_track，缓存命中 core 节点拿到 core。"""
        import threading
        import types

        from api.routes._deps import flush_faiss_buffer

        svc = Services()
        svc._faiss_buffer_lock = threading.Lock()
        svc.faiss_index = types.SimpleNamespace(add_with_ids=lambda v, i: None)
        svc.faiss_id_map = {}
        svc._faiss_buffer = [(1, np.array([0.1, 0.2]), "ep-core")]
        svc.graphlite_store = _FakeStore([
            {"id": "ep-core", "content": "我喜欢喝茶", "fact_track": "core"},
        ])
        assert flush_faiss_buffer(svc) == 1
        assert svc._episode_cache.get("ep-core")["fact_track"] == "core"

        router = QueryRouter.__new__(QueryRouter)
        router.config = QueryRouterConfig()
        router._episode_cache = svc._episode_cache
        router.graphlite_store = svc.graphlite_store
        router._cjk_warned = False
        router.faiss_index = _FakeFaiss(np.array([[-0.9]]), np.array([[1]]))
        router.faiss_id_map = {1: "ep-core"}
        results = router._hypergraph_retrieve("我喜欢喝茶", query_embedding=np.zeros((1, 512)))
        assert results and results[0]["fact_track"] == "core"


# ─── 改动 5：EvidenceTracker.is_multi_source ─────────────────

class TestIsMultiSource:
    def test_single_source_false(self, tmp_path):
        t = EvidenceTracker(data_dir=str(tmp_path))
        t.record("Elon Musk founded SpaceX", source="user")
        assert t.is_multi_source("Elon Musk founded SpaceX") is False

    def test_same_source_twice_false(self, tmp_path):
        t = EvidenceTracker(data_dir=str(tmp_path))
        t.record("Elon Musk founded SpaceX", source="user")
        t.record("Elon Musk founded SpaceX", source="user")
        assert t.is_multi_source("Elon Musk founded SpaceX") is False

    def test_different_source_true(self, tmp_path):
        t = EvidenceTracker(data_dir=str(tmp_path))
        t.record("Elon Musk founded SpaceX", source="user")
        t.record("Elon Musk founded SpaceX", source="agent")
        assert t.is_multi_source("Elon Musk founded SpaceX") is True

    def test_unknown_content_false(self, tmp_path):
        t = EvidenceTracker(data_dir=str(tmp_path))
        assert t.is_multi_source("never recorded") is False


# ─── 改动 5：write.py 写多源 → tau_initial 提升 ────────────────

class TestWriteMultiSourceBoost:
    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(router)

        def _build(svc):
            app.dependency_overrides[get_services] = lambda: svc
            return TestClient(app)

        return _build

    def test_multi_source_write_boosts_tau(self, client, overgraph_store, tmp_path):
        """清单 5：同内容第二次以不同 source 写入 → tau_initial 提升至 0.85。"""
        svc = Services()
        svc.graphlite_store = overgraph_store
        svc.evidence_tracker = EvidenceTracker(data_dir=str(tmp_path))

        content = "我喜欢喝茶"

        r1 = client(svc).post("/memories/episodes", json={"content": content, "source": "user"})
        assert r1.status_code == 200, r1.text
        tau1 = float(overgraph_store.get_episode(r1.json()["episode_id"]).get("tau_initial", 1.0))
        assert abs(tau1 - 0.85) > 1e-6  # 单来源：不提升

        r2 = client(svc).post("/memories/episodes", json={"content": content, "source": "agent"})
        assert r2.status_code == 200, r2.text
        tau2 = float(overgraph_store.get_episode(r2.json()["episode_id"]).get("tau_initial", 1.0))
        assert abs(tau2 - 0.85) < 1e-6  # 多来源：提升持久性
