"""τ 自适应衰减接线回归测试（审计 P0-4a / P0-4b / P0-3）

背景：v2.0 自适应衰减引擎（register_node / update_importance / refresh_tau）
在生产链路零调用 → _node_info 恒空，importance 调制与访问频次增强全部无效
（审计 P0-4）。且写路径 τ 门卫恒真（created_at=_now() → dt≈0 → τ≡1.0 >
decay_threshold），拒绝分支与 force_promote 全是死代码（审计 P0-3）。

修复后：
- 写路径（POST /memories/episodes）先 register_node 再 compute_strength
  —— 节点注册进引擎，importance 反映内容长度启发式；
- 读路径（POST /memories/retrieve）命中节点 update_importance + refresh_tau
  —— access_count 递增、created_at 衰减基准前移（refresh_tau 修复 P2-12）。

全部走公共入口，不直调内部方法。
"""
from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from api.routes import router, Services, get_services
from core.tau_decay import TauDecayEngine


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    def _build(svc):
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app)

    return _build


class TestWritePathRegistersNode:
    """P0-4a：写路径必须把节点注册进自适应衰减引擎。"""

    def test_create_episode_registers_node(self, client, overgraph_store):
        """POST /memories/episodes → register_node 已调用且 importance 生效。"""
        svc = Services()
        svc.graphlite_store = overgraph_store
        svc.tau_engine = TauDecayEngine()

        content = "一条用于 τ 接线回归的记忆"
        resp = client(svc).post("/memories/episodes", json={
            "content": content,
            "source": "user",
        })
        assert resp.status_code == 200, resp.text
        episode_id = resp.json()["episode_id"]

        # 节点必须已注册（旧实现 _node_info 恒空）
        assert episode_id in svc.tau_engine._node_info
        info = svc.tau_engine._node_info[episode_id]
        assert info.importance == pytest.approx(min(1.0, len(content) / 1000.0))
        # created_at 被记录（refresh_tau 需它做衰减基准）
        assert info.created_at > 0

    def test_create_episode_no_longer_gated_by_tau(self, client, overgraph_store):
        """P0-3：正常写入不再被恒真 τ 门卫拒绝（400 分支已删除）。"""
        svc = Services()
        svc.graphlite_store = overgraph_store
        svc.tau_engine = TauDecayEngine()

        resp = client(svc).post("/memories/episodes", json={
            "content": "短内容也应该正常写入",
            "source": "user",
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "created"


class TestRetrievalPathRefreshesTau:
    """P0-4b：检索命中节点必须更新 importance 并 refresh（再巩固）。"""

    def _make_svc_with_results(self, results):
        svc = Services()
        svc.query_router = MagicMock()
        svc.query_router.retrieve.return_value = results
        svc.graphlite_store = MagicMock()
        svc.quarantine_store = None
        svc.ontology_validator = None
        svc.tau_engine = TauDecayEngine()
        return svc

    def test_retrieve_hit_refreshes_tau(self, client):
        """命中结果 → update_importance + refresh_tau 已调用（access_count 递增）。"""
        nid = "hit_node_1"
        svc = self._make_svc_with_results([
            {"node_id": nid, "content": "命中内容", "score": 0.85, "level": "hypergraph"},
        ])
        # 预注册该节点（生产路径由写路径注册）
        svc.tau_engine.register_node(nid, time.time() - 100, importance=0.3)

        resp = client(svc).post("/memories/retrieve", json={
            "query": "命中",
            "top_k": 5,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["results"], "检索应返回结果"

        info = svc.tau_engine._node_info[nid]
        # 分数 0.85 → 平滑调制后 importance 上升（IMP_SMOOTH_ALPHA=0.8）
        assert info.importance > 0.3
        # refresh_tau 递增 access_count
        assert info.access_count >= 1

    def test_retrieve_empty_results_no_tau_calls(self, client):
        """空结果 → 不触发 tau 更新（无节点可刷新，不引入副作用）。"""
        svc = self._make_svc_with_results([])
        svc.tau_engine = MagicMock()

        resp = client(svc).post("/memories/retrieve", json={
            "query": "空查询",
            "top_k": 5,
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["results"] == []
        # 空结果时 update_importance / refresh_tau 不应被调用
        svc.tau_engine.update_importance.assert_not_called()
        svc.tau_engine.refresh_tau.assert_not_called()
