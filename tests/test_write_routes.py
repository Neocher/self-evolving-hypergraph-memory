"""
写入路由测试（Defense 超时降级）
===============================
覆盖：【M2】pre_check 超时 → QUARANTINE（fail-closed 而非 fail-open）。

修复前超时降级为 ALLOW（fail-open）：高并发写入下 pre_check 内
asyncio.Lock 串行化 R1/R3/R4/R5，锁等待超时即放行 → 攻击者可绕过
R1 限流 / R4 防重复 / R5 信任衰减。修复后超时降级为 QUARANTINE。

运行: python -m pytest tests/test_write_routes.py -v
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router, Services, get_services


class HangingDefenseEngine:
    """pre_check 永远挂起（模拟锁等待/编码器阻塞超时）。"""

    config = SimpleNamespace(enabled=True, silent=True)

    async def pre_check(self, **kwargs):
        await asyncio.sleep(30)


class BlockingDefenseEngine:
    """pre_check 在获得 asyncio.Lock 后挂起（模拟锁内死锁/慢规则）。"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.config = SimpleNamespace(enabled=True, silent=True)

    async def pre_check(self, **kwargs):
        async with self._lock:
            await asyncio.sleep(30)


def _make_svc(**overrides) -> Services:
    svc = Services()
    # 【P0-2】显式 mock 替代裸 MagicMock：裸 MagicMock 对任意未声明方法（如
    # get_or_create_session）自动返回 truthy → `if session_node_id:` 恒真假绿。
    # 显式覆盖关键方法 return_value=""，防止未声明方法自动返回 truthy。
    gstore = MagicMock()
    gstore.create_episode = MagicMock(return_value=None)
    gstore.execute_cypher = MagicMock(return_value=False)
    gstore.ensure_session = MagicMock()
    gstore.link_to_session = MagicMock()
    gstore.get_episode = MagicMock(return_value=None)
    # 显式覆盖：任何未 mock 的方法调用返回空字符串/None（非 truthy）
    gstore.get_or_create_session = MagicMock(return_value="")
    svc.graphlite_store = gstore
    svc.quarantine_store = MagicMock()
    svc.quarantine_store.quarantine = MagicMock()
    for k, v in overrides.items():
        setattr(svc, k, v)
    return svc


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    def _build(svc):
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app)

    return _build


class TestDefenseTimeoutFailClosed:
    """【M2】pre_check 超时必须 QUARANTINE（fail-closed），不得 ALLOW"""

    def test_timeout_quarantines_not_allows(self, client):
        svc = _make_svc(defense_engine=HangingDefenseEngine())
        with patch("api.routes.write._EXTERNAL_CALL_TIMEOUT", 0.1):
            resp = client(svc).post("/memories/episodes", json={
                "content": "test content",
                "source": "agent_x",
            })

        assert resp.status_code == 200, resp.text
        # 写入照常完成（隔离标记在写入后执行）
        assert svc.graphlite_store.create_episode.called
        # 关键断言：超时 → QUARANTINE（quarantine() 被调用），而非 ALLOW
        assert svc.quarantine_store.quarantine.called, (
            "pre_check 超时应降级为 QUARANTINE（fail-closed），而不是 ALLOW"
        )
        reason = svc.quarantine_store.quarantine.call_args[0][1]
        assert "defense_timeout" in reason

    def test_lock_wait_timeout_quarantines(self, client):
        """锁内阻塞导致超时同样降级为 QUARANTINE（覆盖 asyncio.Lock 串行化场景）"""
        svc = _make_svc(defense_engine=BlockingDefenseEngine())
        with patch("api.routes.write._EXTERNAL_CALL_TIMEOUT", 0.1):
            resp = client(svc).post("/memories/episodes", json={
                "content": "test content",
                "source": "agent_x",
            })

        assert resp.status_code == 200
        assert svc.quarantine_store.quarantine.called, (
            "锁等待超时也应降级为 QUARANTINE（fail-closed）"
        )

    def test_normal_allow_does_not_quarantine(self, client):
        """对照组：防御判定 ALLOW 时不应隔离（证明 QUARANTINE 来自超时降级）"""

        class AllowingDefenseEngine:
            config = SimpleNamespace(enabled=True, silent=True)

            async def pre_check(self, **kwargs):
                from core.defense import MemoryDefenseVerdict
                return MemoryDefenseVerdict.ALLOW, "all rules passed"

        svc = _make_svc(defense_engine=AllowingDefenseEngine())
        resp = client(svc).post("/memories/episodes", json={
            "content": "test content",
            "source": "agent_x",
        })

        assert resp.status_code == 200
        assert not svc.quarantine_store.quarantine.called


class TestMultimodalTimeoutKeepsFile:
    """【M3-a】媒体嵌入超时后保留文件（不删除）+ 标记未嵌入"""

    def test_image_embed_timeout_keeps_file_marks_unembedded(self, client, tmp_path):
        """嵌入超时：文件保留在磁盘与 media_paths，并标记 unembedded_paths。

        修复前超时会 _remove_media_file 删除已落盘的用户媒体文件 → 瞬时故障
        导致真实数据丢失。修复后仅标记"未嵌入"，文件保留。
        """
        import base64 as _b64

        class SlowClip:
            def embed_image(self, img_bytes):
                time.sleep(5)  # 模拟模型卡死/冷启动加载超时

        class FakeMediaStore:
            base_dir = str(tmp_path)

            def save_image(self, img_bytes):
                (tmp_path / "saved_img.png").write_bytes(img_bytes)
                return "saved_img.png"

        svc = _make_svc(
            _clip_embedder=SlowClip(),
            _media_store=FakeMediaStore(),
            _whisper_embedder=MagicMock(),
            encoder=MagicMock(),
        )
        img_b64 = _b64.b64encode(b"fake-image-bytes").decode()

        with (
            patch("api.routes.write._EXTERNAL_CALL_TIMEOUT", 0.1),
            patch("api.routes.write._MEDIA_WARMUP_TIMEOUT", 0.05),
        ):
            resp = client(svc).post("/memories/multimodal", json={
                "text": "with image",
                "images": [img_b64],
                "source": "agent_x",
            })

        assert resp.status_code == 200
        body = resp.json()
        # 文件保留：media_paths 含该文件，且磁盘文件未被删除
        assert "saved_img.png" in body["media_paths"]
        assert (tmp_path / "saved_img.png").exists(), "超时后用户媒体文件不应被删除"
        # 标记未嵌入
        assert "saved_img.png" in body["unembedded_paths"]

    def test_image_embed_success_not_unembedded(self, client, tmp_path):
        """对照组：嵌入成功 → 文件保留且不标记未嵌入"""
        import base64 as _b64

        class FastClip:
            def embed_image(self, img_bytes):
                return np.zeros(512, dtype=np.float32)

        class FakeMediaStore:
            base_dir = str(tmp_path)

            def save_image(self, img_bytes):
                (tmp_path / "ok_img.png").write_bytes(img_bytes)
                return "ok_img.png"

        svc = _make_svc(
            _clip_embedder=FastClip(),
            _media_store=FakeMediaStore(),
            _whisper_embedder=MagicMock(),
            encoder=MagicMock(),
        )
        img_b64 = _b64.b64encode(b"fake-image-bytes").decode()

        resp = client(svc).post("/memories/multimodal", json={
            "text": "with image",
            "images": [img_b64],
            "source": "agent_x",
        })

        assert resp.status_code == 200
        body = resp.json()
        assert "ok_img.png" in body["media_paths"]
        assert "ok_img.png" not in body["unembedded_paths"]
        assert body["visual_node_id"] is not None


class TestMediaWarmupBudget:
    """【B-复审】warmup 预算仅在真实尝试媒体嵌入/转录后消耗。

    修复前 L222 无条件置位 _media_warmup_done=True：冷启动首个请求若为
    纯文本 multimodal（无 images/audio），首个真实媒体写入失去 30s
    warmup 预算退回 10s 常规超时（冷启动模型加载常 >10s → 误超时）。
    """

    def test_pure_text_request_does_not_consume_warmup(self, client):
        """纯文本 multimodal：_media_warmup_done 仍为 False，预算保留"""
        svc = _make_svc(
            _clip_embedder=MagicMock(),
            _whisper_embedder=MagicMock(),
            encoder=MagicMock(),
        )
        resp = client(svc).post("/memories/multimodal", json={
            "text": "纯文本记忆",
            "source": "agent_x",
        })

        assert resp.status_code == 200
        assert getattr(svc, "_media_warmup_done", False) is False, (
            "纯文本请求未尝试任何媒体嵌入，不应消耗 warmup 预算"
        )

    def test_media_request_consumes_warmup(self, client, tmp_path):
        """带媒体请求：实际尝试嵌入后 _media_warmup_done 置位为 True"""
        import base64 as _b64

        class FastClip:
            def embed_image(self, img_bytes):
                return np.zeros(512, dtype=np.float32)

        class FakeMediaStore:
            base_dir = str(tmp_path)

            def save_image(self, img_bytes):
                (tmp_path / "warmup_img.png").write_bytes(img_bytes)
                return "warmup_img.png"

        svc = _make_svc(
            _clip_embedder=FastClip(),
            _media_store=FakeMediaStore(),
            _whisper_embedder=MagicMock(),
            encoder=MagicMock(),
        )
        img_b64 = _b64.b64encode(b"fake-image-bytes").decode()

        resp = client(svc).post("/memories/multimodal", json={
            "text": "with image",
            "images": [img_b64],
            "source": "agent_x",
        })

        assert resp.status_code == 200
        assert getattr(svc, "_media_warmup_done", False) is True, (
            "带媒体请求实际尝试嵌入后应置位 _media_warmup_done"
        )


class TestForcePromoteProtectedFlagRoute:
    """【v5.27.0】force_promote=true 路由级测试（走生产链路）。

    覆盖 POST /memories/episodes → create_episode → 落库 全链路：
    断言落库节点的 protected 标记。修复前路由直调层仅写入 τ 值，未在
    episode dict 打 protected 标记（与 gateway/store_episode 的 A2A/ACP
    旁路同构缺陷），本测试证明 force_promote=true 时标记真实落库。
    """

    def test_force_promote_route_persists_protected_flag(self, client, graphlite_store):
        """POST force_promote=true → 落库节点 protected in (True, "true", 1)。

        真实 GraphLite 存储读回，兼容 _flatten_row 还原的 Python True
        与 GraphLite 原生 "true"/1 形态。
        """
        svc = Services()
        svc.graphlite_store = graphlite_store

        resp = client(svc).post("/memories/episodes", json={
            "content": "重要记忆 force promote 路由级测试",
            "source": "test",
            "force_promote": True,
        })

        assert resp.status_code == 200, resp.text
        episode_id = resp.json()["episode_id"]
        got = graphlite_store.get_episode(episode_id)
        assert got is not None
        assert got["content"] == "重要记忆 force promote 路由级测试"
        assert got.get("protected") in (True, "true", 1), (
            "force_promote=true 应经生产链路打 protected 标记并落库，"
            f"实际读回 protected={got.get('protected')!r}"
        )
