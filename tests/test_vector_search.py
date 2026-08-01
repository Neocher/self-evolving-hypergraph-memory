"""
向量搜索端点测试
=================
测试 POST /search/vector 的常规路径与降级路径。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router, init_services, Services


# ─── 辅助：创建带深度 mock 的 Services ─────────────────────


def _make_app(svc: Services) -> FastAPI:
    """用给定服务创建 FastAPI app 并初始化路由。"""
    init_services(svc)
    app = FastAPI()
    app.include_router(router)
    return app


def _make_episode(ep_id: str, content: str):
    """模拟 GraphLiteStore.get_episode 返回的 dict。"""
    return {"id": ep_id, "content": content, "created_at": 1000.0, "tau_initial": 1.0}


class MockGraphLiteStore:
    """模拟 GraphLiteStore，只提供 get_episode 方法。"""

    def __init__(self):
        self.episodes: dict[str, dict] = {}

    def get_episode(self, episode_id: str) -> dict | None:
        return self.episodes.get(episode_id)

    def get_episodes_batch(self, node_ids: list[str]) -> list[dict]:
        return [self.episodes.get(pid) for pid in node_ids if pid in self.episodes]

    def query_cypher(self, *args, **kwargs):
        return []

    def close(self):
        pass


class MockFaissIndex:
    """同 conftest.py 的 mock_faiss_index，用于测试路由。"""

    def __init__(self):
        self.vectors: dict[int, np.ndarray] = {}
        self.ntotal: int = 0

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        for vec, fid in zip(vectors, ids):
            self.vectors[int(fid)] = vec.astype(np.float32)
        self.ntotal = len(self.vectors)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.vectors:
            return (np.array([[float("inf")]]), np.array([[-1]]))
        if query.ndim == 1:
            query = query.reshape(1, -1)
        ids_arr = np.array(list(self.vectors.keys()), dtype=np.int64)
        vecs_arr = np.array(list(self.vectors.values()), dtype=np.float32)
        diffs = vecs_arr - query
        distances = np.linalg.norm(diffs, axis=1)
        top_k = min(k, len(distances))
        sorted_idx = np.argsort(distances)[:top_k]
        return (
            distances[sorted_idx].reshape(1, -1),
            ids_arr[sorted_idx].reshape(1, -1),
        )

    def remove_ids(self, id_selector: np.ndarray) -> int:
        remove_set = set(int(x) for x in id_selector)
        removed = 0
        for fid in list(self.vectors.keys()):
            if fid in remove_set:
                del self.vectors[fid]
                removed += 1
        self.ntotal = len(self.vectors)
        return removed


class MockEncoder:
    """模拟 TextEncoder。"""

    def __init__(self):
        self.dim = 384

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(self.dim).astype(np.float32)


# ─── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def mock_graphlite() -> MockGraphLiteStore:
    return MockGraphLiteStore()


@pytest.fixture
def mock_faiss() -> MockFaissIndex:
    return MockFaissIndex()


@pytest.fixture
def mock_enc() -> MockEncoder:
    return MockEncoder()


@pytest.fixture
def populated_faiss(mock_faiss: MockFaissIndex, mock_graphlite: MockGraphLiteStore) -> tuple[MockFaissIndex, dict[int, str], MockGraphLiteStore]:
    """预填充 3 个向量的 FAISS 索引 + GraphLite 数据。"""
    faiss_id_map: dict[int, str] = {}
    rng = np.random.RandomState(42)
    for i in range(3):
        ep_id = f"ep-vector-{i}"
        faiss_id = int(uuid.uuid5(uuid.NAMESPACE_OID, ep_id).int & ((1 << 63) - 1))
        vec = rng.randn(384).astype(np.float32)
        mock_faiss.add_with_ids(vec.reshape(1, -1), np.array([faiss_id], dtype=np.int64))
        faiss_id_map[faiss_id] = ep_id
        mock_graphlite.episodes[ep_id] = _make_episode(ep_id, f"Vector content {i}")
    return mock_faiss, faiss_id_map, mock_graphlite


# ─── 服务构建器 ────────────────────────────────────────────


def _build_svc(
    encoder=None,
    faiss_index=None,
    graphlite_store=None,
    faiss_id_map: dict | None = None,
) -> Services:
    svc = Services(
        encoder=encoder,
        faiss_index=faiss_index,
        graphlite_store=graphlite_store,
    )
    if faiss_id_map is not None:
        svc.faiss_id_map = faiss_id_map
    import threading
    svc._faiss_buffer_lock = threading.Lock()
    return svc


# ─── 测试：正常路径 ────────────────────────────────────────


class TestVectorSearchNormal:
    """正常路径测试：encoder + FAISS + GraphLite 全部可用。"""

    def test_basic_search(self, populated_faiss, mock_enc):
        """常规搜索应返回评分排序结果。"""
        mock_faiss, faiss_id_map, mock_graphlite = populated_faiss
        svc = _build_svc(
            encoder=mock_enc,
            faiss_index=mock_faiss,
            graphlite_store=mock_graphlite,
            faiss_id_map=faiss_id_map,
        )
        app = _make_app(svc)
        client = TestClient(app)
        resp = client.post("/search/vector", json={"query": "test query", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["degraded"] is False
        assert data["total_found"] == 3
        assert len(data["results"]) == 3
        # 结果应按 score 降序
        scores = [r["score"] for r in data["results"]]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        # 验证字段完整性
        for r in data["results"]:
            assert "node_id" in r
            assert "content" in r
            assert "faiss_id" in r
            assert r["node_id"].startswith("ep-vector-")
            assert 0.0 <= r["score"] <= 1.0

    def test_with_limit(self, populated_faiss, mock_enc):
        """limit 参数应限制返回数量。"""
        mock_faiss, faiss_id_map, mock_graphlite = populated_faiss
        svc = _build_svc(
            encoder=mock_enc,
            faiss_index=mock_faiss,
            graphlite_store=mock_graphlite,
            faiss_id_map=faiss_id_map,
        )
        app = _make_app(svc)
        client = TestClient(app)
        resp = client.post("/search/vector", json={"query": "test", "limit": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_found"] == 1
        assert len(data["results"]) == 1

    def test_returns_graphlite_content(self, populated_faiss, mock_enc):
        """返回的 content 应来自 GraphLite。"""
        mock_faiss, faiss_id_map, mock_graphlite = populated_faiss
        svc = _build_svc(
            encoder=mock_enc,
            faiss_index=mock_faiss,
            graphlite_store=mock_graphlite,
            faiss_id_map=faiss_id_map,
        )
        app = _make_app(svc)
        client = TestClient(app)
        resp = client.post("/search/vector", json={"query": "find me", "limit": 10})
        assert resp.status_code == 200
        data = resp.json()
        contents = {r["content"] for r in data["results"]}
        assert "Vector content 0" in contents
        assert "Vector content 1" in contents
        assert "Vector content 2" in contents


# ─── 测试：降级路径 ────────────────────────────────────────


class TestVectorSearchDegraded:
    """降级路径测试：encoder 或 FAISS 不可用。"""

    def test_encoder_none_returns_degraded(self, mock_faiss, mock_graphlite):
        """encoder 为 None 时应降级返回空结果。"""
        svc = _build_svc(
            encoder=None,
            faiss_index=mock_faiss,
            graphlite_store=mock_graphlite,
        )
        app = _make_app(svc)
        client = TestClient(app)
        resp = client.post("/search/vector", json={"query": "hello", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["degraded"] is True
        assert data["total_found"] == 0
        assert data["results"] == []

    def test_faiss_none_returns_degraded(self, mock_enc, mock_graphlite):
        """faiss_index 为 None 时应降级返回空结果。"""
        svc = _build_svc(
            encoder=mock_enc,
            faiss_index=None,
            graphlite_store=mock_graphlite,
        )
        app = _make_app(svc)
        client = TestClient(app)
        resp = client.post("/search/vector", json={"query": "hello", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["degraded"] is True
        assert data["total_found"] == 0

    def test_both_none_returns_degraded(self, mock_graphlite):
        """encoder 和 faiss_index 都为 None 时应降级。"""
        svc = _build_svc(
            encoder=None,
            faiss_index=None,
            graphlite_store=mock_graphlite,
        )
        app = _make_app(svc)
        client = TestClient(app)
        resp = client.post("/search/vector", json={"query": "hello", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["degraded"] is True
        assert data["total_found"] == 0


class TestVectorSearchEdgeCases:
    """边界情况测试。"""

    def test_empty_faiss_returns_empty(self, mock_enc, mock_faiss, mock_graphlite):
        """FAISS 索引无数据时应返回空结果（非降级）。"""
        svc = _build_svc(
            encoder=mock_enc,
            faiss_index=mock_faiss,
            graphlite_store=mock_graphlite,
            faiss_id_map={},
        )
        app = _make_app(svc)
        client = TestClient(app)
        resp = client.post("/search/vector", json={"query": "anything", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["degraded"] is False
        assert data["total_found"] == 0
        assert data["results"] == []

    def test_results_without_graphlite_entry(self, mock_enc, mock_faiss, mock_graphlite):
        """FAISS 结果在 GraphLite 中找不到时应跳过。"""
        # 填充 FAISS 但不填充 GraphLite
        rng = np.random.RandomState(42)
        faiss_id = int(uuid.uuid5(uuid.NAMESPACE_OID, "orphan").int & ((1 << 63) - 1))
        vec = rng.randn(384).astype(np.float32)
        mock_faiss.add_with_ids(vec.reshape(1, -1), np.array([faiss_id], dtype=np.int64))

        svc = _build_svc(
            encoder=mock_enc,
            faiss_index=mock_faiss,
            graphlite_store=mock_graphlite,
            faiss_id_map={faiss_id: "orphan"},
        )
        app = _make_app(svc)
        client = TestClient(app)
        resp = client.post("/search/vector", json={"query": "orphan test", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        # orphan 在 GraphLite 中不存在，但 faiss_id_map 指向它，get_episode 返回 None
        # content 应为空字符串
        assert data["total_found"] == 1
        assert data["results"][0]["content"] == ""

    def test_latency_ms_is_positive(self, populated_faiss, mock_enc):
        """latency_ms 应为正数。"""
        mock_faiss, faiss_id_map, mock_graphlite = populated_faiss
        svc = _build_svc(
            encoder=mock_enc,
            faiss_index=mock_faiss,
            graphlite_store=mock_graphlite,
            faiss_id_map=faiss_id_map,
        )
        app = _make_app(svc)
        client = TestClient(app)
        resp = client.post("/search/vector", json={"query": "test", "limit": 5})
        assert resp.status_code == 200
        assert resp.json()["latency_ms"] >= 0

    def test_invalid_limit_rejected(self):
        """limit 超出范围应返回 422。"""
        svc = _build_svc()
        app = _make_app(svc)
        client = TestClient(app)
        resp = client.post("/search/vector", json={"query": "test", "limit": 999})
        assert resp.status_code == 422

    def test_empty_query_rejected(self):
        """空 query 应返回 422。"""
        svc = _build_svc()
        app = _make_app(svc)
        client = TestClient(app)
        resp = client.post("/search/vector", json={"query": "", "limit": 5})
        assert resp.status_code == 422
