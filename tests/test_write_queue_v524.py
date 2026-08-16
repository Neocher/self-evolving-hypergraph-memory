"""
v5.24 写路径全量接入 WriteQueue 测试
====================================
覆盖（.trio-task-v524.md 验收 §3/§4）:
  · gateway_api write_sensory / store_episode / store_multimodal — 写调用经队列
  · gateway 并发压测 — 8 并发 write_sensory 期间事件循环心跳无大间隔（验收 #3）
  · visual 路由 — create_visual_node 经队列
  · communities 路由 — resolve / resolve-all / reconcile update_with_version 经队列
  · ontology batch_relations — 6 次 execute_cypher 组闭包整体经队列
  · hyperedges 路由 — create_hyperedge 三分支经队列
  · write.py create_episode — entity_resolver / extract_and_relate 闭包经队列
  · _process_embed_queue — hebbian update 经队列, poll loop 不被写阻塞
  · app._persist_dream_state — 调度器同步回调经 async 闭包 + create_task 入队
  · 降级 — 无队列同步直调; 队列满/关闭 → 503 语义降级不崩

运行: python -m pytest tests/test_write_queue_v524.py -v
"""

import asyncio
import base64
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.app import _persist_dream_state
from api.routes import router, Services, get_services
from api.routes._deps import _embed_queue, _embed_queue_lock
from api.routes.write import _process_embed_queue
from core.write_queue import (
    WriteQueue,
    WriteQueueFullError,
    WriteQueueClosedError,
)
from gateway.gateway_api import GatewayAPI
from graph.hyperedge import HyperedgeType as CoreHyperedgeType


# ─── 共享构建 ──────────────────────────────────────────────


def _make_svc(**overrides) -> Services:
    svc = Services()
    gstore = MagicMock()
    gstore.create_episode = MagicMock(return_value=None)
    gstore.execute_cypher = MagicMock(return_value=False)
    gstore.ensure_session = MagicMock()
    gstore.link_to_session = MagicMock()
    gstore.get_episode = MagicMock(return_value=None)
    gstore.get_or_create_session = MagicMock(return_value="")
    svc.graphlite_store = gstore
    for k, v in overrides.items():
        setattr(svc, k, v)
    return svc


def _make_gateway_svc(**overrides) -> Services:
    svc = Services()
    gstore = MagicMock()
    gstore._sensory_buffer = None  # 触发无环形缓冲区的兜底写路径
    gstore.create_episode = MagicMock(return_value=None)
    gstore.ensure_session = MagicMock()
    gstore.link_to_session = MagicMock()
    svc.graphlite_store = gstore
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


@pytest.fixture(autouse=True)
def _clear_embed_queue():
    """隔离 _embed_queue 全局状态（跨测试不污染）。"""
    with _embed_queue_lock:
        _embed_queue.clear()
    yield
    with _embed_queue_lock:
        _embed_queue.clear()


# ─── gateway_api 写路径 ────────────────────────────────────


class TestGatewayWriteQueue:
    def test_write_sensory_via_queue(self):
        """write_sensory 无环形缓冲区兜底：create → ensure → link 均经队列且有序。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_gateway_svc(write_queue=q)
            api = GatewayAPI(svc)
            resp = asyncio.run(api.write_sensory("hello world", source="tester", namespace="ns1"))
            assert resp.record_id
            svc.graphlite_store.create_episode.assert_called_once()
            payload = svc.graphlite_store.create_episode.call_args[0][0]
            assert payload["content"] == "hello world"
            svc.graphlite_store.ensure_session.assert_called_once_with("ns1")
            svc.graphlite_store.link_to_session.assert_called_once()
            assert svc.graphlite_store.link_to_session.call_args[0][0] == "ns1"
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_store_episode_via_queue(self):
        """store_episode 持久化 + 命名空间链接经队列。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_gateway_svc(write_queue=q)
            api = GatewayAPI(svc)
            resp = asyncio.run(api.store_episode("episode content", source="user", namespace="ns2"))
            assert resp.episode_id
            payload = svc.graphlite_store.create_episode.call_args[0][0]
            assert payload["content"] == "episode content"
            svc.graphlite_store.ensure_session.assert_called_once_with("ns2")
            svc.graphlite_store.link_to_session.assert_called_once()
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_store_multimodal_via_queue(self):
        """store_multimodal 纯文本：EpisodeNode 写经队列。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_gateway_svc(write_queue=q)
            api = GatewayAPI(svc)
            resp = asyncio.run(api.store_multimodal(text="multimodal text", namespace="ns3"))
            assert resp.episode_id
            payload = svc.graphlite_store.create_episode.call_args[0][0]
            assert payload["content"] == "multimodal text"
            svc.graphlite_store.ensure_session.assert_called_once_with("ns3")
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_gateway_write_fallback_sync_without_queue(self):
        """无写队列（测试/降级）→ 同步直调, 行为与改造前一致。"""
        svc = _make_gateway_svc()
        api = GatewayAPI(svc)
        resp = asyncio.run(api.write_sensory("no queue", source="tester"))
        assert resp.record_id
        svc.graphlite_store.create_episode.assert_called_once()


class TestGatewayConcurrency:
    def test_8_concurrent_sensory_writes_loop_stays_responsive(self):
        """验收 #3: 并发压测走 gateway 路由, 读（loop 心跳）不受写阻塞影响。"""
        q = WriteQueue(wait_timeout=30.0)
        try:
            svc = Services()
            gstore = MagicMock()
            gstore._sensory_buffer = None
            create_calls = []

            def _slow_create(payload):
                time.sleep(0.02)  # 模拟 GraphLite 写耗时
                create_calls.append(payload)
                return payload.get("id", "")
            gstore.create_episode = _slow_create
            gstore.ensure_session = MagicMock()
            gstore.link_to_session = MagicMock()
            svc.graphlite_store = gstore
            svc.write_queue = q
            api = GatewayAPI(svc)

            ticks = []

            async def heartbeat():
                for _ in range(120):
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.01)

            async def run():
                hb = asyncio.create_task(heartbeat())
                await asyncio.gather(
                    *[api.write_sensory(f"load-{i}", source="load") for i in range(8)]
                )
                await hb

            asyncio.run(run())
            gaps = [b - a for a, b in zip(ticks, ticks[1:])]
            # 若写阻塞 loop，gap ≈ 0.02s×并发串行；经队列后 gap ≈ 定时器间隔
            assert max(gaps) < 0.12, f"event loop stalled during gateway writes: {max(gaps):.3f}s"
            assert len(create_calls) == 8
            assert q.pending_count() == 0
        finally:
            q.shutdown()


# ─── visual 路由 ───────────────────────────────────────────


class TestVisualQueue:
    def test_create_visual_node_via_queue(self, client, monkeypatch, tmp_path):
        """POST /memories/visual 的 VisualNode INSERT 经写队列。

        【P1-2】写路径改 CLIP 投影 384d（原 bge encoder 512d 直落缺陷修复）：
        注入 FakeClip 共享实例，断言落库 embedding 为 384d。
        """
        import api.routes.visual as visual_mod
        monkeypatch.setattr(visual_mod, "VISUALS_DIR", str(tmp_path))

        class _FakeClip:
            available = True
            dimension = 512

            def embed_text(self, text):
                return np.zeros(512, dtype=np.float32)

        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_svc(write_queue=q)
            svc._clip_embedder = _FakeClip()
            image_b64 = base64.b64encode(b"fake-image-bytes").decode()
            resp = client(svc).post("/memories/visual", json={
                "image_base64": image_b64, "caption": "a cat", "source": "tester",
            })
            assert resp.status_code == 200, resp.text
            svc.graphlite_store.create_visual_node.assert_called_once()
            payload = svc.graphlite_store.create_visual_node.call_args[0][0]
            assert payload["caption"] == "a cat"
            assert len(payload["embedding"]) == 384, "写路径必须落 384d（CLIP 投影空间）"
            assert q.pending_count() == 0
        finally:
            q.shutdown()


# ─── communities / conflicts 路由 ──────────────────────────


class TestCommunitiesQueue:
    def test_resolve_conflict_via_queue(self, client):
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_svc(write_queue=q)
            resp = client(svc).post("/conflicts/c1/resolve")
            assert resp.status_code == 200, resp.text
            svc.graphlite_store.execute_cypher.assert_called_once()
            cypher = svc.graphlite_store.execute_cypher.call_args[0][0]
            assert "SET c.resolved = true" in cypher
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_resolve_all_conflicts_via_queue(self, client):
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_svc(write_queue=q)
            resp = client(svc).post("/conflicts/resolve-all")
            assert resp.status_code == 200, resp.text
            svc.graphlite_store.execute_cypher.assert_called_once()
            cypher = svc.graphlite_store.execute_cypher.call_args[0][0]
            assert "SET c.resolved = true" in cypher
            assert "c.resolved = false" in cypher
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_reconcile_update_with_version_via_queue(self, client):
        """OCC 检测 resolve() 留 loop；update_with_version 复合闭包整体入队。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_svc(write_queue=q)
            svc.write_reconciler = MagicMock()
            svc.write_reconciler.resolve.return_value = {
                "conflict": False, "resolved": True, "strategy": "lww",
                "data": {"content": "resolved", "source": "s", "visibility": "v",
                         "unrelated": 1},
                "expected_version": 1, "current_version": 1,
            }
            resp = client(svc).post("/conflicts/reconcile", json={
                "node_id": "n1", "data": {"content": "old"},
                "expected_version": 1, "strategy": "lww",
            })
            assert resp.status_code == 200, resp.text
            svc.graphlite_store.update_with_version.assert_called_once()
            kwargs = svc.graphlite_store.update_with_version.call_args[1]
            assert kwargs["node_id"] == "n1"
            assert kwargs["updates"]["content"] == "resolved"
            assert "unrelated" not in kwargs["updates"]  # 只写允许字段
            assert kwargs["expected_version"] == 1
            assert q.pending_count() == 0
        finally:
            q.shutdown()


# ─── ontology / hyperedges / batch 路由 ─────────────────────


class TestOntologyBatchQueue:
    def test_batch_relations_closure_via_queue(self, client):
        """每个三元组的 6 次 execute_cypher 组闭包整体经队列（写线程内原子）。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_svc(write_queue=q)
            resp = client(svc).post("/batch/relations", json={
                "relations": [
                    {"subject": "shm", "relation": "uses", "object": "queue"},
                    {"subject": "a", "relation": "r", "object": "b",
                     "edge_type": "RELATES_TO"},
                ],
                "source": "tester",
            })
            assert resp.status_code == 200, resp.text
            assert resp.json()["created"] == 2
            assert svc.graphlite_store.execute_cypher.call_count == 12  # 2 三元组 × 6 次
            assert q.pending_count() == 0
        finally:
            q.shutdown()


class TestHyperedgesQueue:
    def test_create_episode_hyperedge_via_queue(self, client):
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_svc(write_queue=q)
            svc.hyperedge_manager = MagicMock()
            svc.hyperedge_manager.create_episode_hyperedge.return_value = SimpleNamespace(
                id="h1", type=CoreHyperedgeType.EPISODE, member_ids=["a", "b"],
                created_at=1.0, gate_value=1.0, metadata={},
            )
            resp = client(svc).post("/hyperedges", json={
                "type": "episode", "member_ids": ["a", "b"], "topic": "t",
            })
            assert resp.status_code == 200, resp.text
            assert resp.json()["id"] == "h1"
            svc.hyperedge_manager.create_episode_hyperedge.assert_called_once()
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_create_temporal_hyperedge_via_queue(self, client):
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_svc(write_queue=q)
            svc.hyperedge_manager = MagicMock()
            svc.hyperedge_manager.create_temporal_hyperedge.return_value = SimpleNamespace(
                id="h2", type=CoreHyperedgeType.TEMPORAL, member_ids=["a", "b"],
                created_at=1.0, gate_value=1.0, metadata={},
            )
            resp = client(svc).post("/hyperedges", json={
                "type": "temporal", "member_ids": ["a", "b"],
                "start_time": 1.0, "end_time": 2.0,
            })
            assert resp.status_code == 200, resp.text
            svc.hyperedge_manager.create_temporal_hyperedge.assert_called_once()
            assert q.pending_count() == 0
        finally:
            q.shutdown()


# ─── write.py create_episode 随主写入队 ────────────────────


class TestWriteRouteSubProcesses:
    def test_entity_resolver_and_extract_relate_via_queue(self, client, monkeypatch):
        """create_episode 内 entity_resolver.process / extract_and_relate 闭包经队列。

        长内容（>80 字符）触发 Step3 实体消歧；relation extractor 返回空 →
        Phase3 fallback 走 extract_and_relate。两者整体在写线程执行。
        """
        q = WriteQueue(wait_timeout=10.0)
        try:
            svc = _make_svc(write_queue=q)
            svc.ontology_validator = MagicMock()
            svc.ontology_validator.write_validate.return_value = SimpleNamespace(
                passed=True, conflict_count=0, contradictions=[],
                ontology_type=None, entity_name=None, entity_value=None,
            )
            svc.ontology_validator.extract_and_relate.return_value = 3
            fake_extractor = MagicMock()
            fake_extractor.extract.return_value = None  # → triples=None → fallback
            monkeypatch.setattr(
                "core.relation_extractor.RelationExtractor", lambda: fake_extractor)
            fake_resolver = MagicMock()
            fake_resolver.process.return_value = {"alias_count": 2, "entities": ["e1"]}
            monkeypatch.setattr(
                "core.entity_resolver.EntityResolver", lambda **kw: fake_resolver)

            long_content = "这是一段超过八十个字符的测试内容用于触发实体消歧与关系抽取的完整链路验证路径扩展测试" * 2
            resp = client(svc).post("/memories/episodes", json={
                "content": long_content, "source": "tester",
            })
            assert resp.status_code == 200, resp.text
            fake_resolver.process.assert_called_once_with(long_content)
            svc.ontology_validator.extract_and_relate.assert_called_once_with(long_content)
            assert q.pending_count() == 0
        finally:
            q.shutdown()


# ─── _process_embed_queue：hebbian 入队 ─────────────────────


def _make_embed_svc(write_queue=None, hebbian_delay: float = 0.0) -> Services:
    svc = Services()
    svc.encoder = MagicMock()
    svc.encoder.embed.return_value = np.zeros(64, dtype=np.float32)
    svc.faiss_index = MagicMock()
    svc._faiss_buffer = []
    svc._faiss_buffer_lock = threading.Lock()
    svc.quarantine_store = None
    if hebbian_delay:
        def _slow_update(active, conns):
            time.sleep(hebbian_delay)
        svc.hebbian_updater = MagicMock()
        svc.hebbian_updater.update.side_effect = _slow_update
    else:
        svc.hebbian_updater = MagicMock()
    svc.graphlite_store = MagicMock()
    svc.graphlite_store.get_all_connections.return_value = {}
    svc.write_queue = write_queue
    return svc


class TestEmbedQueueHebbian:
    def test_hebbian_update_via_queue(self):
        """_process_embed_queue 内 hebbian update 经写队列（读留 loop）。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_embed_svc(write_queue=q)
            with _embed_queue_lock:
                _embed_queue.append(("ep1", "hello world", time.time()))
            count = asyncio.run(_process_embed_queue(svc))
            assert count == 1
            svc.hebbian_updater.update.assert_called_once()
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_hebbian_poll_loop_not_blocked_by_write(self):
        """hebbian 写（慢）在写线程执行：消费 loop 心跳无大间隔（poll 不卡 loop）。"""
        q = WriteQueue(wait_timeout=30.0)
        try:
            svc = _make_embed_svc(write_queue=q, hebbian_delay=0.05)
            ticks = []

            async def heartbeat():
                for _ in range(120):
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.01)

            async def run():
                hb = asyncio.create_task(heartbeat())
                with _embed_queue_lock:
                    _embed_queue.append(("ep1", "x", time.time()))
                await _process_embed_queue(svc)
                await hb

            asyncio.run(run())
            gaps = [b - a for a, b in zip(ticks, ticks[1:])]
            assert max(gaps) < 0.12, f"loop stalled by hebbian write: {max(gaps):.3f}s"
        finally:
            q.shutdown()

    def test_fallback_sync_without_queue(self):
        svc = _make_embed_svc(write_queue=None)
        with _embed_queue_lock:
            _embed_queue.append(("ep1", "x", time.time()))
        count = asyncio.run(_process_embed_queue(svc))
        assert count == 1
        svc.hebbian_updater.update.assert_called_once()

    def test_queue_busy_degrades_gracefully(self):
        """poll loop 内队列满 → 503 语义无意义，降级记 WARNING 不崩。"""

        class _BusyQueue:
            async def submit(self, *a, **k):
                raise WriteQueueFullError("full")

        svc = _make_embed_svc(write_queue=_BusyQueue())
        with _embed_queue_lock:
            _embed_queue.append(("ep1", "x", time.time()))
        # 不抛异常：hebbian 被降级跳过，embed 仍计数
        count = asyncio.run(_process_embed_queue(svc))
        assert count == 1
        svc.hebbian_updater.update.assert_not_called()


# ─── app._persist_dream_state 入队 ─────────────────────────


class TestPersistDreamState:
    def test_via_queue_no_running_loop(self):
        """无 running loop（同步上下文）→ asyncio.run 桥接，MATCH+SET/INSERT 在写线程。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = Services()
            gstore = MagicMock()
            gstore.execute_cypher = MagicMock(return_value=False)  # MATCH 空 → INSERT
            svc.graphlite_store = gstore
            svc.write_queue = q
            _persist_dream_state(svc, {"status": "running", "n": 1})
            assert gstore.execute_cypher.call_count == 2  # MATCH + INSERT
            insert_cypher = gstore.execute_cypher.call_args_list[1].args[0]
            assert "INSERT (s:SystemNode" in insert_cypher
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_via_queue_with_running_loop(self):
        """调度器在 loop 线程同步回调 → create_task 桥接，写经队列完成。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = Services()
            gstore = MagicMock()
            gstore.execute_cypher = MagicMock(return_value=False)
            svc.graphlite_store = gstore
            svc.write_queue = q

            async def run():
                _persist_dream_state(svc, {"status": "running"})
                # fire-and-forget 任务：轮询等待写线程完成
                for _ in range(100):
                    if gstore.execute_cypher.call_count >= 2:
                        break
                    await asyncio.sleep(0.02)
                assert gstore.execute_cypher.call_count >= 2

            asyncio.run(run())
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_fallback_sync_without_queue(self):
        """无写队列 → 同步直调（与原实现一致）。"""
        svc = Services()
        gstore = MagicMock()
        gstore.execute_cypher = MagicMock(return_value=False)
        svc.graphlite_store = gstore
        svc.write_queue = None
        _persist_dream_state(svc, {"status": "x"})
        assert gstore.execute_cypher.call_count == 2

    def test_queue_closed_degrades_gracefully(self):
        """shutdown 竞态（队列已关闭）→ 记 WARNING 降级，不抛异常。"""

        class _ClosedQueue:
            async def submit(self, *a, **k):
                raise WriteQueueClosedError("closed")

        svc = Services()
        gstore = MagicMock()
        gstore.execute_cypher = MagicMock(return_value=False)
        svc.graphlite_store = gstore
        svc.write_queue = _ClosedQueue()
        _persist_dream_state(svc, {"status": "x"})
        assert gstore.execute_cypher.call_count == 0
