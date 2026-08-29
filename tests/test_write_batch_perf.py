"""
P1-1 批量写路径优化测试
======================
覆盖:
  · 批量超边创建按 source 分组 — 每 source 2 次 MATCH (n 项不再 2n 次)
  · 批量返回结构不变 (全部 created)
  · 批量写入通知梦境调度器 (写压力/累积计数)
运行: python -m pytest tests/test_write_batch_perf.py -v
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router, Services, get_services


def _make_svc(**overrides) -> Services:
    svc = Services()
    gstore = MagicMock()
    gstore.create_episode = MagicMock(return_value=None)
    gstore.execute_cypher = MagicMock(return_value=False)
    gstore.ensure_session = MagicMock()
    gstore.link_to_session = MagicMock()
    gstore.query_cypher = MagicMock(return_value=[])
    svc.graph_store = gstore
    svc.hyperedge_manager = MagicMock()
    # 路由对 on_activity/on_node_created 使用 await → 必须 async mock
    svc.dream_scheduler = AsyncMock()
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


class TestBatchHyperedgeMerge:
    def test_hyperedges_grouped_by_source_one_window_query_each(self, client):
        svc = _make_svc()

        resp = client(svc).post("/memories/episodes/batch", json=[
            {"content": f"batch item {i}", "source": "src_a"} for i in range(4)
        ])

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 4
        assert all(r["status"] == "created" for r in body["results"])
        assert svc.graph_store.create_episode.call_count == 4
        # 超边查询: 1 source × 2 窗口 = 2 次 (原逐条实现为 4 项 × 2 = 8 次)
        query_calls = [c for c in svc.graph_store.query_cypher.call_args_list]
        assert len(query_calls) == 2, f"expected 2 MATCH queries, got {len(query_calls)}"
        # 批次节点已落库 → recent 窗口含本批, 时态超边应创建 (4 成员 ≥ 2)
        svc.hyperedge_manager.create_temporal_hyperedge.assert_called_once()
        # 梦境通知: 每项 on_activity + on_node_created
        assert svc.dream_scheduler.on_activity.call_count == 4
        assert svc.dream_scheduler.on_node_created.call_count == 4

    def test_batch_no_recent_still_creates_temporal_hyperedge(self, client):
        svc = _make_svc()

        resp = client(svc).post("/memories/episodes/batch", json=[
            {"content": f"item {i}", "source": "src_b"} for i in range(2)
        ])
        assert resp.status_code == 200
        # 2 本批成员 ≥ 2 → 时态超边创建; 2 < 4 → 无情节超边
        svc.hyperedge_manager.create_temporal_hyperedge.assert_called_once()
        svc.hyperedge_manager.create_episode_hyperedge.assert_not_called()

    def test_batch_error_item_skipped_from_hyperedges(self, client):
        svc = _make_svc()
        # 仅第一条失败, 第二条成功 (side_effect 列表按调用顺序消费)
        svc.graph_store.create_episode.side_effect = [RuntimeError("boom"), None]

        resp = client(svc).post("/memories/episodes/batch", json=[
            {"content": "fail", "source": "src_c"},
            {"content": "ok", "source": "src_c"},
        ])
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"][0]["status"] == "error"
        assert body["results"][1]["status"] == "created"
        svc.hyperedge_manager.create_temporal_hyperedge.assert_not_called()  # 仅 1 项成功 < 2
