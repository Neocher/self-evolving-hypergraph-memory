"""
P2-a V-Mem 模态路由检索测试
==========================
覆盖：视觉通道端到端（modality=visual）/ 文本通道零回归 / 空库零开销短路 /
CLIP 降级 / 投影一致性（512→384 与 write.py 公式逐元素相等）/ embedding JSON 解析 /
P1-1 写路径增量索引（add_visual_node 惰性引导/幂等）/ P1-2 /memories/visual 写路径
384d CLIP 投影落库 / P2-1 CLIP 冷启动隔离（未加载跳过 + prewarm 预热）/
P2-2 真实 ClipEmbedder 冒烟（模型已缓存时启用）。
R2 新增：P2-1 并发快照一致性（并发 add 无 fid 碰撞 / _visual_snapshot 不暴露
中间态）/ P2-2 写队列超时路径补索引（超时 503 但节点入索引；队列满不产生幽灵节点）。

运行: python -m pytest tests/test_visual_recall.py -v
"""

import asyncio
import base64
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.routes import router, Services, get_services
from core.write_queue import WriteQueueClosedError, WriteQueueFullError
from retrieval.query_router import QueryRouter, QueryRouterConfig
from retrieval.vector_store import FaissStore


# ─── 辅助构造 ──────────────────────────────────────────────


def _write_proj() -> np.ndarray:
    """write.py:437-438 投影公式（seed 42，列归一 512→384）。"""
    rng = np.random.default_rng(42)
    proj = rng.standard_normal((512, 384), dtype=np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    return proj


class FakeClip:
    """可控 CLIP 嵌入器：embed_text 返回固定向量（与图像向量一致 → 距离 0）。"""

    available = True
    dimension = 512

    def __init__(self, seed: int = 7):
        self._vec = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
        self.embed_calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        self.embed_calls += 1
        return self._vec

    def embed_image(self, img_bytes: bytes) -> np.ndarray:
        self.embed_calls += 1
        return self._vec


class UnavailableClip(FakeClip):
    available = False


class MockFaissIndex:
    """同 conftest.mock_faiss：单向量库，search 返回 L2 距离。"""

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
        distances = np.linalg.norm(vecs_arr - query, axis=1)
        top_k = min(k, len(distances))
        sorted_idx = np.argsort(distances)[:top_k]
        return (
            distances[sorted_idx].reshape(1, -1),
            ids_arr[sorted_idx].reshape(1, -1),
        )


class MockEncoder:
    """同 conftest.mock_encoder（确定性随机向量，384d）。"""

    dim = 384

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(self.dim).astype(np.float32)


def _fake_settings(visual_enabled: bool = True) -> SimpleNamespace:
    """查询路由侧配置（同时关闭社区扩召回防干扰）。"""
    return SimpleNamespace(retrieval=SimpleNamespace(
        visual_recall=SimpleNamespace(
            enabled=visual_enabled, boost=0.6, max_results=5, visual_limit=10000,
        ),
        community_expansion=SimpleNamespace(enabled=False),
    ))


def _make_router(store, faiss=None, encoder=None, faiss_id_map=None, services=None,
                 config=None) -> QueryRouter:
    faiss = faiss or MockFaissIndex()
    encoder = encoder or MockEncoder()
    faiss_id_map = faiss_id_map if faiss_id_map is not None else {}
    return QueryRouter(
        graphlite_store=store,
        faiss_index=faiss,
        tfidf_index=MagicMock(),
        encoder=encoder,
        faiss_id_map=faiss_id_map,
        config=config or QueryRouterConfig(),
        services=services,
    )


def _seed_text_channel(faiss: MockFaissIndex, ep_id: str = "ep_b") -> dict:
    """faiss 单向量 + id 映射（文本通道种子）。"""
    vec = np.random.default_rng(11).standard_normal(384).astype(np.float32)
    faiss.add_with_ids(vec.reshape(1, -1), np.array([0], dtype=np.int64))
    return {0: ep_id}


# ─── 端到端 ────────────────────────────────────────────────


class TestVisualRecallEndToEnd:
    """写 VisualNode + 文本种子 → 图像语义 query → modality=visual 结果（HTTP 公共入口）。"""

    @pytest.mark.graphlite  # 【v6.0.0 legacy】GraphLite 专属语义/引擎约束（默认排除，addopts -m 'not graphlite'）
    def test_retrieve_route_returns_visual_modality(self, overgraph_store):

        svc = Services()
        proj = _write_proj()
        clip = FakeClip()

        # 文本种子 episode（与视觉 caption 不同 → 去重键不冲突）
        overgraph_store.create_episode({
            "id": "ep_b", "content": "北京烤鸭很好吃",
            "source": "user", "created_at": 100.0,
        })
        # 视觉节点：embedding = CLIP 图像向量 @ 投影（与 query 向量一致 → 距离 0）
        overgraph_store.create_visual_node({
            "id": "vn1", "caption": "海边日落的照片", "image_path": "/tmp/sea.png",
            "embedding": (clip._vec @ proj).tolist(),
            "source": "user", "created_at": 200.0,
        })

        svc._clip_embedder = clip
        svc._clip_projection = proj
        svc.graphlite_store = overgraph_store
        svc.encoder = MockEncoder()
        svc.quarantine_store = None
        svc.ontology_validator = None
        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss)
        svc.faiss_index = faiss
        svc.faiss_id_map = faiss_id_map
        svc.query_router = _make_router(
            overgraph_store, faiss=faiss, encoder=svc.encoder,
            faiss_id_map=faiss_id_map, services=svc,
        )

        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            asyncio.run(svc.query_router.prewarm_visual())
            assert svc.query_router._visual_index.count == 1

            app = FastAPI()
            app.include_router(router)
            app.dependency_overrides[get_services] = lambda: svc
            resp = TestClient(app).post("/memories/retrieve", json={
                "query": "海边日落", "top_k": 10,
            })
            assert resp.status_code == 200, resp.text
            body = resp.json()
            visual = [r for r in body["results"] if r.get("modality") == "visual"]
            assert visual, "图像语义 query 应召回 modality=visual 结果"
            assert visual[0]["node_id"] == "vn1"
            assert visual[0]["content"] == "海边日落的照片"
            assert visual[0]["source"] == "visual"  # level → EpisodicResult.source
            assert 0.0 <= visual[0]["score"] <= 1.0

    def test_visual_score_strictly_below_text_seed(self, overgraph_store):
        """相对尾分缩放：视觉分 = 1/(1+dist) × min(种子分) × boost < 文本种子分。"""
        svc = Services()
        proj = _write_proj()
        clip = FakeClip()
        overgraph_store.create_episode({
            "id": "ep_b", "content": "北京烤鸭很好吃",
            "source": "user", "created_at": 100.0,
        })
        overgraph_store.create_visual_node({
            "id": "vn1", "caption": "海边日落的照片", "image_path": "/tmp/sea.png",
            "embedding": (clip._vec @ proj).tolist(),
            "source": "user", "created_at": 200.0,
        })
        svc._clip_embedder = clip
        svc._clip_projection = proj
        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss)
        qr = _make_router(overgraph_store, faiss=faiss, faiss_id_map=faiss_id_map, services=svc)
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            asyncio.run(qr.prewarm_visual())
            results = qr.retrieve("海边日落")
        text_scores = [r["score"] for r in results if r.get("modality") != "visual"]
        visual_scores = [r["score"] for r in results if r.get("modality") == "visual"]
        assert text_scores and visual_scores
        assert all(v < min(text_scores) for v in visual_scores)


# ─── 文本通道零回归 ────────────────────────────────────────


class TestTextChannelZeroRegression:
    """视觉通道开启/关闭均不得扰动文本通道结果。"""

    def test_disabled_visual_returns_text_only(self, overgraph_store):
        svc = Services()
        overgraph_store.create_episode({
            "id": "ep_b", "content": "北京烤鸭很好吃",
            "source": "user", "created_at": 100.0,
        })
        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss)
        qr = _make_router(overgraph_store, faiss=faiss, faiss_id_map=faiss_id_map, services=svc)
        with patch("retrieval.query_router.get_settings",
                   side_effect=lambda: _fake_settings(visual_enabled=False)):
            results = qr.retrieve("海边日落")
        assert [r["node_id"] for r in results] == ["ep_b"]
        assert all(r.get("modality") != "visual" for r in results)

    def test_visual_index_present_text_channel_intact(self, overgraph_store):
        """索引已构建但 query 与视觉节点无关 → 文本结果不变（视觉低分追加不顶替）。"""
        svc = Services()
        proj = _write_proj()
        clip = FakeClip()
        overgraph_store.create_episode({
            "id": "ep_b", "content": "北京烤鸭很好吃",
            "source": "user", "created_at": 100.0,
        })
        overgraph_store.create_visual_node({
            "id": "vn1", "caption": "海边日落的照片", "image_path": "",
            "embedding": (clip._vec @ proj).tolist(),
            "source": "user", "created_at": 200.0,
        })
        svc._clip_embedder = clip
        svc._clip_projection = proj
        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss)
        qr = _make_router(overgraph_store, faiss=faiss, faiss_id_map=faiss_id_map, services=svc)
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            asyncio.run(qr.prewarm_visual())
            results = qr.retrieve("北京烤鸭")
        assert results[0]["node_id"] == "ep_b", "文本种子必须仍居首"


# ─── 空库零开销短路 ────────────────────────────────────────


class TestEmptyChannelShortCircuit:
    """_visual_index 为 None 或空 → 直接返回：无 GQL、无 CLIP 实例化。"""

    def _build(self, graphlite_store=None):
        store = graphlite_store or MagicMock()
        store.get_visual_nodes = MagicMock(return_value=[])
        store.get_communities_by_seeds = MagicMock(return_value=[])
        store.get_episodes_batch = MagicMock(return_value=[
            {"id": "ep_a", "content": "text result", "tau_initial": 1.0},
        ])
        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss, "ep_a")
        qr = _make_router(store, faiss=faiss, faiss_id_map=faiss_id_map, services=Services())
        return qr, store

    def test_index_none_no_clip_no_gql(self):
        qr, store = self._build()
        with (
            patch("retrieval.query_router.get_settings", side_effect=_fake_settings),
            patch("multimodal.embedders.ClipEmbedder",
                  side_effect=AssertionError("ClipEmbedder 不应被实例化")),
        ):
            results = qr.retrieve("海边日落")
        assert [r["node_id"] for r in results] == ["ep_a"]
        store.get_visual_nodes.assert_not_called()
        # 恒真断言修正：空通道短路后 services 容器不得被挂载 CLIP 嵌入器
        assert not hasattr(qr._services, "_clip_embedder"), \
            "空通道不得触发 _get_clip_embedder（CLIP 实例化/挂载）"

    def test_index_empty_ntotal_zero_no_clip(self):
        qr, store = self._build()
        qr._visual_index = FaissStore(dimension=384)  # 空索引 ntotal=0
        qr._visual_id_map = {}
        qr._visual_meta = {}
        with (
            patch("retrieval.query_router.get_settings", side_effect=_fake_settings),
            patch("multimodal.embedders.ClipEmbedder",
                  side_effect=AssertionError("ClipEmbedder 不应被实例化")),
        ):
            results = qr.retrieve("海边日落")
        assert [r["node_id"] for r in results] == ["ep_a"]
        assert all(r.get("modality") != "visual" for r in results)


# ─── CLIP 降级 ─────────────────────────────────────────────


class TestClipDegradation:
    """CLIP 不可用 → 视觉通道静默跳过，文本结果零回归。"""

    def test_clip_unavailable_returns_unchanged(self, overgraph_store):
        svc = Services()
        overgraph_store.create_episode({
            "id": "ep_b", "content": "北京烤鸭很好吃",
            "source": "user", "created_at": 100.0,
        })
        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss)
        # 索引已构建但 CLIP 不可用（模型加载失败降级）
        qr = _make_router(overgraph_store, faiss=faiss, faiss_id_map=faiss_id_map, services=svc)
        qr._visual_index = FaissStore(dimension=384)
        qr._visual_index.add(
            np.random.default_rng(5).standard_normal((1, 384)).astype(np.float32),
            np.array([0], dtype=np.int64),
        )
        qr._visual_id_map = {0: "vn1"}
        qr._visual_meta = {"vn1": {"caption": "海边日落", "created_at": 200.0, "image_path": ""}}
        svc._clip_embedder = UnavailableClip()
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            results = qr.retrieve("海边日落")
        assert [r["node_id"] for r in results] == ["ep_b"]
        assert all(r.get("modality") != "visual" for r in results)

    def test_clip_embed_failure_returns_unchanged(self, overgraph_store):
        """embed_text 返回 None（编码失败）→ 静默跳过。"""
        svc = Services()
        overgraph_store.create_episode({
            "id": "ep_b", "content": "北京烤鸭很好吃",
            "source": "user", "created_at": 100.0,
        })

        class FailingClip(FakeClip):
            def embed_text(self, text: str):
                return None

        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss)
        qr = _make_router(overgraph_store, faiss=faiss, faiss_id_map=faiss_id_map, services=svc)
        qr._visual_index = FaissStore(dimension=384)
        qr._visual_index.add(
            np.random.default_rng(5).standard_normal((1, 384)).astype(np.float32),
            np.array([0], dtype=np.int64),
        )
        qr._visual_id_map = {0: "vn1"}
        qr._visual_meta = {"vn1": {"caption": "x", "created_at": 1.0, "image_path": ""}}
        svc._clip_embedder = FailingClip()
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            results = qr.retrieve("海边日落")
        assert [r["node_id"] for r in results] == ["ep_b"]


# ─── 投影一致性 ────────────────────────────────────────────


class TestProjectionConsistency:
    """512→384 投影与 write.py:437-438 公式逐元素相等（seed 42）；services 复用。"""

    def test_matches_write_formula_elementwise(self):
        qr = _make_router(MagicMock(), services=None)
        qr._visual_projection = None
        proj = qr._get_projection()
        expected = _write_proj()
        assert proj.shape == (512, 384)
        assert proj.dtype == np.float32
        assert np.array_equal(proj, expected), "投影矩阵必须与写路径逐元素相等"

    def test_reuses_services_projection_identity(self):
        """services._clip_projection 已存在 → 直接复用（同一对象，不重算）。"""
        svc = Services()
        svc._clip_projection = _write_proj()
        qr = _make_router(MagicMock(), services=svc)
        proj = qr._get_projection()
        assert proj is svc._clip_projection

    def test_creates_and_backfills_services(self):
        """services 无投影 → 创建后回写 services._clip_projection（写路径复用）。"""
        svc = Services()
        qr = _make_router(MagicMock(), services=svc)
        proj = qr._get_projection()
        assert svc._clip_projection is proj
        assert np.array_equal(proj, _write_proj())


# ─── embedding JSON 解析 ───────────────────────────────────


class TestEmbeddingJsonParse:
    """prewarm_visual 解析 GraphLite 落库的 JSON 字符串 embedding；非 384d 防御性跳过。"""

    def test_keeps_384_skips_512(self, overgraph_store):
        proj = _write_proj()
        clip_vec = np.random.default_rng(7).standard_normal(512).astype(np.float32)
        overgraph_store.create_visual_node({
            "id": "vn384", "caption": "384d 节点", "image_path": "/tmp/a.png",
            "embedding": (clip_vec @ proj).tolist(),
            "source": "user", "created_at": 100.0,
        })
        overgraph_store.create_visual_node({
            "id": "vn512", "caption": "512d 节点（应跳过）", "image_path": "",
            "embedding": np.random.default_rng(3).standard_normal(512).astype(np.float32).tolist(),
            "source": "user", "created_at": 200.0,
        })
        svc = Services()
        svc._clip_embedder = FakeClip()
        svc._clip_projection = proj
        qr = _make_router(overgraph_store, services=svc)
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            asyncio.run(qr.prewarm_visual())
        assert qr._visual_index is not None
        assert qr._visual_index.count == 1, "512d 节点必须被跳过"
        assert qr._visual_id_map == {0: "vn384"}
        assert qr._visual_meta["vn384"]["caption"] == "384d 节点"

    def test_embedding_roundtrip_exact(self, overgraph_store):
        """embedding 经 JSON 落库/读回后与源向量逐元素相等（float32 无损）。"""
        proj = _write_proj()
        clip_vec = np.random.default_rng(7).standard_normal(512).astype(np.float32)
        stored = (clip_vec @ proj).astype(np.float32)
        overgraph_store.create_visual_node({
            "id": "vn1", "caption": "c", "image_path": "",
            "embedding": stored.tolist(),
            "source": "user", "created_at": 100.0,
        })
        svc = Services()
        svc._clip_embedder = FakeClip()
        svc._clip_projection = proj
        qr = _make_router(overgraph_store, services=svc)
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            asyncio.run(qr.prewarm_visual())
        # 精确命中：query 向量 == 存储向量 → 距离 ≈ 0
        distances, indices = qr._visual_index.search(stored[None], 1)
        assert float(distances[0][0]) < 1e-5
        assert int(indices[0][0]) == 0

    def test_empty_db_no_index(self, overgraph_store):
        """无 VisualNode → prewarm 后 _visual_index 保持 None（短路前提）。"""
        svc = Services()
        qr = _make_router(overgraph_store, services=svc)
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            asyncio.run(qr.prewarm_visual())
        assert qr._visual_index is None


# ─── P1-1 增量索引 ─────────────────────────────────────────


class TestIncrementalVisualIndex:
    """【P1-1】prewarm 后写入的 VisualNode 经 add_visual_node 增量入索引，无需重启/prewarm。"""

    def test_add_after_prewarm_immediately_searchable(self, overgraph_store):
        svc = Services()
        proj = _write_proj()
        clip = FakeClip()
        overgraph_store.create_visual_node({
            "id": "vn1", "caption": "海边日落的照片", "image_path": "/tmp/a.png",
            "embedding": (clip._vec @ proj).tolist(),
            "source": "user", "created_at": 100.0,
        })
        svc._clip_embedder = clip
        svc._clip_projection = proj
        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss)
        qr = _make_router(overgraph_store, faiss=faiss, faiss_id_map=faiss_id_map, services=svc)
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            asyncio.run(qr.prewarm_visual())
            assert qr._visual_index.count == 1
            # 写路径增量（模拟 visual.py / write.py 创建后调用）
            vn2 = {
                "id": "vn2", "caption": "雪山星空的照片", "image_path": "/tmp/b.png",
                "embedding": (clip._vec @ proj).tolist(),
                "source": "user", "created_at": 200.0,
            }
            assert qr.add_visual_node(vn2) is True
            assert qr._visual_index.count == 2
            assert qr._visual_id_map.get(1) == "vn2"
            assert qr._visual_meta["vn2"]["caption"] == "雪山星空的照片"
            # 幂等：重复调用不重复入索引
            assert qr.add_visual_node(vn2) is False
            assert qr._visual_index.count == 2
            # 增量节点立即可检索（无需重启/重 prewarm）
            results = qr.retrieve("海边日落")
        visual_ids = [r["node_id"] for r in results if r.get("modality") == "visual"]
        assert "vn1" in visual_ids
        assert "vn2" in visual_ids

    def test_lazy_bootstrap_when_index_none(self, overgraph_store):
        """prewarm 未构建（空库/未跑）→ add_visual_node 惰性引导构建，检索即命中。"""
        svc = Services()
        proj = _write_proj()
        clip = FakeClip()
        svc._clip_embedder = clip
        svc._clip_projection = proj
        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss)
        qr = _make_router(overgraph_store, faiss=faiss, faiss_id_map=faiss_id_map, services=svc)
        vn1 = {
            "id": "vn1", "caption": "海边日落的照片", "image_path": "",
            "embedding": (clip._vec @ proj).tolist(),
            "source": "user", "created_at": 100.0,
        }
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            assert qr._visual_index is None
            assert qr.add_visual_node(vn1) is True
            assert qr._visual_index is not None
            assert qr._visual_index.count == 1
            results = qr.retrieve("海边日落")
        assert [r["node_id"] for r in results if r.get("modality") == "visual"] == ["vn1"]

    def test_rejects_non_384d(self, overgraph_store):
        """维度不符（旧 bge 512d 直落形态）→ 拒绝入索引（索引空间纯净）。"""
        qr = _make_router(overgraph_store, services=Services())
        bad = {
            "id": "vn512", "caption": "旧 512d 节点", "image_path": "",
            "embedding": np.random.default_rng(3).standard_normal(512).astype(np.float32).tolist(),
            "source": "user", "created_at": 1.0,
        }
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            assert qr.add_visual_node(bad) is False
        assert qr._visual_index is None


# ─── P1-2 /memories/visual 写路径（HTTP 公共入口）──────────


class TestVisualRouteWritePath:
    """【P1-2】/memories/visual 写路径落 384d CLIP 投影空间 + 增量入索引（修复 512d 缺陷）。"""

    @pytest.mark.graphlite  # 【v6.0.0 legacy】384d CLIP 视觉向量 vs OverGraph 512d 引擎约束（默认排除）
    def test_route_writes_384d_and_immediately_searchable(self, overgraph_store):
        svc = Services()
        proj = _write_proj()
        clip = FakeClip()
        svc._clip_embedder = clip
        svc._clip_projection = proj
        svc.graphlite_store = overgraph_store
        svc.encoder = MockEncoder()
        svc.quarantine_store = None
        svc.ontology_validator = None
        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss)
        svc.faiss_index = faiss
        svc.faiss_id_map = faiss_id_map
        svc.query_router = _make_router(
            overgraph_store, faiss=faiss, faiss_id_map=faiss_id_map, services=svc,
        )
        image_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128).decode()
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_services] = lambda: svc
        resp = TestClient(app).post("/memories/visual", json={
            "image_base64": image_b64, "caption": "海边日落的照片", "source": "user",
        })
        assert resp.status_code == 200, resp.text
        vid = resp.json()["visual_id"]
        # 落库形态：384d（CLIP 投影空间），不再是 bge 512d 直落
        node = overgraph_store.get_visual_node(vid)
        emb = QueryRouter._parse_visual_embedding(node.get("embedding"))
        assert emb is not None and emb.shape[0] == 384, "写路径必须落 384d（P1-2 修复）"
        # 增量入索引：无需 prewarm，检索即命中（P1-1）
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            assert svc.query_router._visual_index is not None
            assert svc.query_router._visual_index.count == 1
            # query 用唯一串避开 search.py 模块级 _result_cache 与旧用例同 key 碰撞
            results = TestClient(app).post("/memories/retrieve", json={
                "query": "海边日落 P1-2 集成用例", "top_k": 10,
            }).json()["results"]
        visual = [r for r in results if r.get("modality") == "visual"]
        assert visual, "写路径节点应立即可检索（modality=visual）"
        assert visual[0]["node_id"] == vid
        assert visual[0]["content"] == "海边日落的照片"


# ─── P2-1 CLIP 冷启动隔离 ─────────────────────────────────


class TestClipColdStartIsolation:
    """【P2-1】CLIP 未加载（冷启动）→ 检索跳过视觉通道，不触发模型加载（3s 预算保护）。"""

    def test_real_clip_unloaded_skips_channel(self, overgraph_store):
        from multimodal.embedders import ClipEmbedder
        svc = Services()
        overgraph_store.create_episode({
            "id": "ep_b", "content": "北京烤鸭很好吃",
            "source": "user", "created_at": 100.0,
        })
        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss)
        qr = _make_router(overgraph_store, faiss=faiss, faiss_id_map=faiss_id_map, services=svc)
        qr._visual_index = FaissStore(dimension=384)
        qr._visual_index.add(
            np.random.default_rng(5).standard_normal((1, 384)).astype(np.float32),
            np.array([0], dtype=np.int64),
        )
        qr._visual_id_map = {0: "vn1"}
        qr._visual_meta = {"vn1": {"caption": "海边日落", "created_at": 200.0, "image_path": ""}}
        clip = ClipEmbedder()  # 真实实例，模型未加载（_model is None）
        assert clip._model is None
        svc._clip_embedder = clip
        with (
            patch("retrieval.query_router.get_settings", side_effect=_fake_settings),
            # 冷启动守卫缺失时若触发 embed_text → 断言失败（模型加载被阻止）
            patch.object(ClipEmbedder, "embed_text",
                         side_effect=AssertionError("CLIP 未加载时检索不得调用 embed_text")),
        ):
            results = qr.retrieve("海边日落")
        assert [r["node_id"] for r in results] == ["ep_b"]
        assert all(r.get("modality") != "visual" for r in results)

    def test_prewarm_warms_clip(self, overgraph_store):
        """prewarm 构建索引后预热 CLIP（首次检索不触发模型加载）。"""
        svc = Services()
        proj = _write_proj()
        clip = FakeClip()
        overgraph_store.create_visual_node({
            "id": "vn1", "caption": "海边日落的照片", "image_path": "",
            "embedding": (clip._vec @ proj).tolist(),
            "source": "user", "created_at": 100.0,
        })
        svc._clip_embedder = clip
        svc._clip_projection = proj
        qr = _make_router(overgraph_store, services=svc)
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            asyncio.run(qr.prewarm_visual())
        assert clip.embed_calls >= 1, "prewarm 后必须预热 CLIP（至少一次 embed_text）"


# ─── P2-2 真实 CLIP 冒烟（模型可下载/已缓存时启用）─────────


def _real_clip_usable() -> bool:
    """真实 CLIP 冒烟前提：模型已缓存（离线无缓存 → skip，不触发下载）。"""
    if os.environ.get("SHM_TEST_REAL_CLIP") == "1":
        return True
    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32-multilingual-v1"
    )
    return os.path.isdir(cache)


@pytest.mark.skipif(not _real_clip_usable(), reason="CLIP 模型未缓存（离线/无网络），冒烟跳过")
class TestRealClipSmoke:
    """真实 ClipEmbedder 端到端冒烟：文本/图像 512d + 投影后 FAISS 命中。"""

    def test_real_clip_text_projection_roundtrip(self):
        from multimodal.embedders import ClipEmbedder
        clip = ClipEmbedder()
        vec = clip.embed_text("海边日落的照片")
        assert vec is not None
        emb_512 = np.asarray(vec, dtype=np.float32).reshape(-1)
        assert emb_512.shape[0] == 512
        proj = _write_proj()
        emb_384 = emb_512 @ proj
        assert emb_384.shape == (384,)
        index = FaissStore(dimension=384)
        index.add(emb_384[None].astype(np.float32), np.array([0], dtype=np.int64))
        distances, indices = index.search(emb_384[None].astype(np.float32), 1)
        assert float(distances[0][0]) < 1e-3, "投影后同源向量应自命中（距离≈0）"
        assert int(indices[0][0]) == 0


# ─── P2-1 并发快照一致性 ─────────────────────────────────


class TestVisualIndexAtomicSnapshot:
    """【P2-1】add_visual_node 并发增量 + _visual_snapshot 读侧：
    _visual_lock 串行化保证 fid 无碰撞、跨结构（index/map/meta）不暴露中间态。"""

    def test_concurrent_adds_distinct_fids(self, overgraph_store):
        """80 节点 8 线程并发入索引：无 fid 碰撞、count 精确、map/meta 同步。"""
        proj = _write_proj()
        clip = FakeClip()
        qr = _make_router(MagicMock(), services=Services())
        nodes = [
            {
                "id": f"vn{i}", "caption": f"照片 {i}", "image_path": "",
                "embedding": (clip._vec @ proj).tolist(),
                "source": "user", "created_at": float(i),
            }
            for i in range(80)
        ]
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            with ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(qr.add_visual_node, nodes))
            assert qr._visual_index.count == 80
            assert len(qr._visual_id_map) == 80
            assert len(set(qr._visual_id_map.values())) == 80, "fid 必须互不碰撞"
            assert set(qr._visual_id_map.values()) == set(qr._visual_meta.keys())
            for i in range(80):
                assert f"vn{i}" in qr._visual_meta

    def test_snapshot_never_exposes_partial_swap(self, overgraph_store):
        """并发写线程 + 读线程反复 _visual_snapshot：快照永不暴露 id_map/meta 中间态。"""
        proj = _write_proj()
        clip = FakeClip()
        qr = _make_router(MagicMock(), services=Services())
        stop = threading.Event()
        errors: list[str] = []

        def writer(offset: int):
            try:
                for i in range(50):
                    qr.add_visual_node({
                        "id": f"vn{offset}_{i}", "caption": "c", "image_path": "",
                        "embedding": (clip._vec @ proj).tolist(),
                        "source": "user", "created_at": float(offset * 50 + i),
                    })
            except Exception as exc:  # pragma: no cover
                errors.append(f"writer: {exc!r}")

        def reader():
            try:
                while not stop.is_set():
                    _index, id_map, meta = qr._visual_snapshot()
                    # id_map 与 meta 同一临界区 swap：读侧快照必须二者同步
                    if id_map and set(id_map.values()) != set(meta.keys()):
                        errors.append("snapshot: id_map/meta 不一致")
                        return
            except Exception as exc:  # pragma: no cover
                errors.append(f"reader: {exc!r}")

        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            threads = [threading.Thread(target=writer, args=(k,)) for k in range(2)]
            threads.append(threading.Thread(target=reader))
            for t in threads:
                t.start()
            for t in threads[:2]:
                t.join()
            stop.set()
            threads[2].join()
        assert not errors, errors
        assert qr._visual_index.count == 100
        assert len(set(qr._visual_id_map.values())) == 100


# ─── P2-2 写队列超时路径补索引 ────────────────────────────


class _RejectingQueue:
    """fake write queue：submit 固定抛指定异常（模拟超时/队列满/关闭）。"""

    def __init__(self, exc):
        self.exc = exc

    async def submit(self, fn, *args, **kwargs):
        raise self.exc


class TestVisualWriteQueueTimeoutPath:
    """【P2-2】qsubmit 超时（已入队迟到完成，DB 将落库）→ 补索引；
    队列满/关闭（未入队，DB 未落库）→ 不补（防幽灵节点）。"""

    def _svc(self, overgraph_store, queue):
        proj = _write_proj()
        clip = FakeClip()
        svc = Services()
        svc._clip_embedder = clip
        svc._clip_projection = proj
        svc.graphlite_store = overgraph_store
        svc.encoder = MockEncoder()
        svc.quarantine_store = None
        svc.ontology_validator = None
        faiss = MockFaissIndex()
        faiss_id_map = _seed_text_channel(faiss)
        svc.faiss_index = faiss
        svc.faiss_id_map = faiss_id_map
        svc.write_queue = queue
        svc.query_router = _make_router(
            overgraph_store, faiss=faiss, faiss_id_map=faiss_id_map, services=svc,
        )
        return svc

    def _post(self, svc):
        image_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128).decode()
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app).post("/memories/visual", json={
            "image_base64": image_b64, "caption": "海边日落的照片", "source": "user",
        })

    def test_timeout_returns_503_but_node_indexed(self, overgraph_store):
        """超时 ≠ 失败：任务已入队将迟到完成（DB 仍会落库）→ 必须补索引。"""
        svc = self._svc(overgraph_store, _RejectingQueue(asyncio.TimeoutError()))
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            resp = self._post(svc)
        assert resp.status_code == 503, resp.text
        assert svc.query_router._visual_index is not None
        assert svc.query_router._visual_index.count == 1, "超时路径必须补索引"
        assert len(svc.query_router._visual_id_map) == 1

    def test_queue_full_no_phantom_index(self, overgraph_store):
        """队列满：任务未入队（DB 未落库）→ 不补索引（防幽灵节点）。"""
        svc = self._svc(overgraph_store, _RejectingQueue(WriteQueueFullError("full")))
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            resp = self._post(svc)
        assert resp.status_code == 503, resp.text
        assert svc.query_router._visual_index is None, "未入队不得产生幽灵索引"

    def test_closed_no_phantom_index(self, overgraph_store):
        """队列关闭：任务未入队 → 不补索引。"""
        svc = self._svc(overgraph_store, _RejectingQueue(WriteQueueClosedError("closed")))
        with patch("retrieval.query_router.get_settings", side_effect=_fake_settings):
            resp = self._post(svc)
        assert resp.status_code == 503, resp.text
        assert svc.query_router._visual_index is None
