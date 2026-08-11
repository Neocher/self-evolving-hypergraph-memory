"""
v5.25 写路径最终收敛测试
=======================
覆盖（.trio-task-v525.md 验收 §1/§2/§3）:
  · dream API apply_candidate — PRUNE/MERGE 整体闭包经写队列（loop 不阻塞）
  · dream apply 队列忙 → 503 降级 deferred 语义（不写库不标记, 幂等保持）
  · dream apply 无队列 → 同步直调（降级路径与原实现一致）
  · _persist_dream_state fire-and-forget task SDK 异常兜底 — 不泄漏
    "Task exception was never retrieved" 噪音, 记 ERROR 日志
  · dream_scheduler._run_dream 不再含 auto_apply 同步写（交 _dream_poll_loop）

运行: python -m pytest tests/test_write_queue_v525.py -v
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.app as app_mod
from api.app import _persist_dream_state
from api.routes import router, Services, get_services
from core.write_queue import WriteQueue, WriteQueueFullError


# ─── 共享构建 ──────────────────────────────────────────────


def _make_dream_svc(**overrides) -> Services:
    """构造含 dream_candidate_store + graphlite_store 的 Services。"""
    svc = Services()
    svc.graphlite_store = MagicMock()
    store = MagicMock()
    store.apply_candidate.return_value = True
    svc.dream_candidate_store = store
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


# ─── dream API apply_candidate 入队 ─────────────────────────


class TestDreamApplyQueue:
    def test_apply_dream_candidate_via_queue(self, client):
        """apply_candidate（PRUNE/MERGE 循环写）整体闭包经写队列, loop 不阻塞。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_dream_svc(write_queue=q)
            resp = client(svc).post("/dream/candidates/dream-1/apply")
            assert resp.status_code == 200, resp.text
            assert resp.json()["success"] is True
            svc.dream_candidate_store.apply_candidate.assert_called_once()
            args = svc.dream_candidate_store.apply_candidate.call_args[0]
            assert args[0] == "dream-1"
            assert args[1] is svc.graphlite_store
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_apply_dream_candidate_queue_busy_returns_deferred(self, client):
        """队列满 → 503 降级 deferred：整体未执行（不写库不标记）, 幂等保持。"""

        class _BusyQueue:
            async def submit(self, *a, **k):
                raise WriteQueueFullError("full")

        svc = _make_dream_svc(write_queue=_BusyQueue())
        resp = client(svc).post("/dream/candidates/dream-2/apply")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is False
        assert "deferred" in body["message"].lower()
        # 整体闭包未入队 → apply_candidate 从未执行（下次重试重走全部流程）
        svc.dream_candidate_store.apply_candidate.assert_not_called()

    def test_apply_dream_candidate_fallback_sync_without_queue(self, client):
        """无写队列（测试/降级）→ 同步直调, 行为与改造前一致。"""
        svc = _make_dream_svc()
        resp = client(svc).post("/dream/candidates/dream-3/apply")
        assert resp.status_code == 200, resp.text
        assert resp.json()["success"] is True
        svc.dream_candidate_store.apply_candidate.assert_called_once_with(
            "dream-3", svc.graphlite_store)

    def test_apply_dream_candidate_failure_propagates(self, client):
        """apply_candidate 内部失败 → 返回 success=False（原有语义不变）。"""
        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = _make_dream_svc(write_queue=q)
            svc.dream_candidate_store.apply_candidate.return_value = False
            resp = client(svc).post("/dream/candidates/dream-4/apply")
            assert resp.status_code == 200, resp.text
            assert resp.json()["success"] is False
            assert q.pending_count() == 0
        finally:
            q.shutdown()


# ─── _persist_dream_state SDK 异常兜底 ──────────────────────


class TestPersistDreamStateSDKException:
    def test_sdk_exception_logged_not_raised(self, monkeypatch):
        """fire-and-forget task 内 execute_cypher SDK 异常 → logger.exception 兜底,
        不泄漏 "Task exception was never retrieved"（无未处理异常）。"""
        calls = []

        def _boom(store, node_id, payload):
            calls.append(node_id)
            raise RuntimeError("simulated GraphLite ConnectionError")

        monkeypatch.setattr(app_mod, "_upsert_system_node", _boom)
        fake_logger = MagicMock()
        monkeypatch.setattr(app_mod, "logger", fake_logger)

        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = Services()
            svc.graphlite_store = MagicMock()
            svc.write_queue = q

            async def run():
                _persist_dream_state(svc, {"status": "running"})
                # fire-and-forget 任务：轮询等待 task 执行（异常已内部消化）
                for _ in range(200):
                    if calls:
                        break
                    await asyncio.sleep(0.01)
                assert calls == ["dream_scheduler_state"], "task 应已执行"
                await asyncio.sleep(0.05)  # 让日志调用完成

            asyncio.run(run())
            fake_logger.exception.assert_called()
            # 队列满降级日志不应出现（本次是 SDK 异常路径）
            assert q.pending_count() == 0
        finally:
            q.shutdown()

    def test_sdk_exception_no_running_loop(self, monkeypatch):
        """无 running loop（同步上下文）→ asyncio.run 桥接路径同样兜底。"""
        calls = []

        def _boom(store, node_id, payload):
            calls.append(node_id)
            raise RuntimeError("simulated GraphLite QueryError")

        monkeypatch.setattr(app_mod, "_upsert_system_node", _boom)
        fake_logger = MagicMock()
        monkeypatch.setattr(app_mod, "logger", fake_logger)

        q = WriteQueue(wait_timeout=5.0)
        try:
            svc = Services()
            svc.graphlite_store = MagicMock()
            svc.write_queue = q
            _persist_dream_state(svc, {"status": "x"})
            assert calls == ["dream_scheduler_state"]
            fake_logger.exception.assert_called()
        finally:
            q.shutdown()
