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

    def test_force_promote_route_persists_protected_flag(self, client, overgraph_store):
        """POST force_promote=true → 落库节点 protected in (True, "true", 1)。

        真实 GraphLite 存储读回，兼容 _flatten_row 还原的 Python True
        与 GraphLite 原生 "true"/1 形态。
        """
        svc = Services()
        svc.graphlite_store = overgraph_store

        resp = client(svc).post("/memories/episodes", json={
            "content": "重要记忆 force promote 路由级测试",
            "source": "test",
            "force_promote": True,
        })

        assert resp.status_code == 200, resp.text
        episode_id = resp.json()["episode_id"]
        got = overgraph_store.get_episode(episode_id)
        assert got is not None
        assert got["content"] == "重要记忆 force promote 路由级测试"
        assert got.get("protected") in (True, "true", 1), (
            "force_promote=true 应经生产链路打 protected 标记并落库，"
            f"实际读回 protected={got.get('protected')!r}"
        )


class TestP2EmbedBatchSingleConnScan:
    """P2：embed 批扫 get_all_connections 提到批循环外（每批恰 1 次全表 MATCH 读）。

    修复前：_process_embed_queue 循环内每个 episode 都调一次 get_all_connections
    （N 条批 → N 次全表扫描）；修复后提到批循环外（to_thread 不阻塞 loop），
    循环内复用。update 原地 mutate，逐条语义不变。
    """

    def _svc(self):
        svc = Services()
        svc.encoder = MagicMock()
        svc.encoder.embed.return_value = np.zeros(384, dtype=np.float32)
        svc.faiss_index = MagicMock()
        svc._faiss_buffer_lock = __import__("threading").Lock()
        svc._faiss_buffer = []
        svc.hebbian_updater = MagicMock()
        svc.graphlite_store = MagicMock()
        svc.graphlite_store.get_all_connections = MagicMock(return_value={"a": {"b": 0.5}})
        svc.quarantine_store = MagicMock()
        svc.quarantine_store.get_quarantined_ids.return_value = set()
        return svc

    @patch("api.routes.write._embed_queue", [("e1", "c1", 1.0), ("e2", "c2", 1.0)])
    def test_get_all_connections_called_once_per_batch(self):
        """N 条批 → get_all_connections 恰 1 次（修复前 N 次）。"""
        import asyncio
        from api.routes.write import _process_embed_queue
        svc = self._svc()

        async def run():
            count = await _process_embed_queue(svc)
            return count
        count = asyncio.run(run())

        assert count == 2, f"2 条批应都处理: {count}"
        assert svc.graphlite_store.get_all_connections.call_count == 1, (
            "P2: get_all_connections 应提到批循环外（每批 1 次），"
            f"实际 {svc.graphlite_store.get_all_connections.call_count} 次"
        )

    @patch("api.routes.write._embed_queue", [("e1", "c1", 1.0), ("e2", "c2", 1.0)])
    def test_conns_reused_across_batch_items(self):
        """循环内复用同一 conns dict（update 原地 mutate，逐条语义不变）。"""
        import asyncio
        from unittest.mock import MagicMock
        from api.routes.write import _process_embed_queue
        svc = self._svc()
        hebbian_calls: list = []
        svc.hebbian_updater.update = MagicMock(side_effect=lambda *a, **k: hebbian_calls.append(a))

        async def run():
            await _process_embed_queue(svc)
        asyncio.run(run())

        assert len(hebbian_calls) == 2, f"2 条批应触发 2 次 hebbian update: {len(hebbian_calls)}"
        # 每次 update 收到的是同一 conns 对象（引用复用，而非每批重取）
        assert all(call[1] is svc.graphlite_store.get_all_connections.return_value
                   for call in hebbian_calls), "循环内应复用同一 conns dict"


class TestP7SingleWriteSingleWindowScan:
    """P7：单条写超边检测 2 条窗口 GQL 合并为 1 条（3600s 窗 LIMIT 20）。

    修复前：_auto_create_hyperedges 对单条写入发 2 次全表 MATCH
    （300s LIMIT5 + 3600s LIMIT20），且 loop 上持锁同步读。
    修复后：合并为 1 条（RETURN e.id, e.created_at ... LIMIT 20），
    Python 侧按 created_at >= now-300 过滤出 recent；包 asyncio.to_thread。
    批量路径（_auto_create_hyperedges_batch）已各自合并，不动。
    """

    def _svc(self, rows: list[list]) -> Services:
        from api.routes.write import _auto_create_hyperedges

        svc = Services()
        gstore = MagicMock()
        gstore.query_cypher = MagicMock(return_value=rows)
        svc.graphlite_store = gstore
        svc.hyperedge_manager = MagicMock()
        return svc

    def test_single_write_single_cypher_query(self):
        """单条写：query_cypher 恰 1 次（修复前 2 次）。"""
        from api.routes.write import _auto_create_hyperedges

        svc = self._svc([])
        asyncio.run(_auto_create_hyperedges("e1", "src", "c", svc))
        assert svc.graphlite_store.query_cypher.call_count == 1, (
            "P7: 单条写两条窗口查询应合并为 1 条 GQL，"
            f"实际 {svc.graphlite_store.query_cypher.call_count} 次"
        )

    def test_merged_query_uses_3600s_window_limit20(self):
        """合并后查询使用 3600s 窗 + LIMIT 20，且返回 id + created_at。"""
        from api.routes.write import _auto_create_hyperedges

        svc = self._svc([])
        asyncio.run(_auto_create_hyperedges("e1", "src", "c", svc))
        gql = svc.graphlite_store.query_cypher.call_args[0][0]
        assert "created_at >= $cutoff" in gql
        assert "LIMIT 20" in gql
        assert "e.created_at" in gql, "合并查询应 RETURN e.created_at 供 Python 侧过滤"
        params = svc.graphlite_store.query_cypher.call_args[0][1]
        assert params["cutoff"] <= time.time(), "cutoff 应为 3600s 窗（now-3600）"

    def test_recent_filtered_in_python(self):
        """Python 侧 300s 过滤：300s 内节点触发时态超边，更旧节点只进情节窗。"""
        from api.routes.write import _auto_create_hyperedges

        now = time.time()
        # 行序按 created_at DESC：fresh(50s 前) → old(2000s 前)
        rows = [
            ["e2", now - 50],
            ["e3", now - 2000],
        ]
        svc = self._svc(rows)
        created = asyncio.run(_auto_create_hyperedges("e1", "src", "c", svc))

        # recent_ids = [e2]（len==1 → 时态 pair 超边）+ window_ids = [e2, e3]
        # （len==2 < 4 → 不创建情节超边）
        assert created == 1, f"应创建 1 条时态 pair 超边: {created}"
        temporal_calls = [
            c for c in svc.hyperedge_manager.create_temporal_hyperedge.call_args_list
        ]
        assert len(temporal_calls) == 1
        member_ids = temporal_calls[0].kwargs["member_ids"]
        assert member_ids == ["e1", "e2"], (
            f"时态超边成员应为 [e1, e2]（300s 内过滤），实际 {member_ids}"
        )
        svc.hyperedge_manager.create_episode_hyperedge.assert_not_called()

    def test_episode_window_uses_full_pool(self):
        """情节窗使用全部 3600s 内节点（≥4 → 创建情节超边）。"""
        from api.routes.write import _auto_create_hyperedges

        now = time.time()
        rows = [
            ["e2", now - 100],
            ["e3", now - 1000],
            ["e4", now - 2000],
            ["e5", now - 3000],
        ]
        svc = self._svc(rows)
        created = asyncio.run(_auto_create_hyperedges("e1", "src", "c", svc))
        assert created == 2, f"应创建时态超边 + 情节超边: {created}"
        episode_calls = [
            c for c in svc.hyperedge_manager.create_episode_hyperedge.call_args_list
        ]
        assert len(episode_calls) == 1
        member_ids = episode_calls[0].kwargs["member_ids"]
        assert member_ids == ["e1"] + [f"e{i}" for i in range(2, 6)], (
            f"情节超边成员应为 [e1, e2..e5]，实际 {member_ids}"
        )

    def test_handles_dict_rows(self):
        """兼容 dict 行（扁平 dict 解析）：e.id / e.created_at。"""
        from api.routes.write import _auto_create_hyperedges

        now = time.time()
        rows = [
            {"id": "e2", "created_at": now - 50},
        ]
        svc = self._svc(rows)
        created = asyncio.run(_auto_create_hyperedges("e1", "src", "c", svc))
        assert created == 1
        member_ids = svc.hyperedge_manager.create_temporal_hyperedge.call_args.kwargs["member_ids"]
        assert member_ids == ["e1", "e2"]
