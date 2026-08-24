"""命名空间隔离 fail-closed 回归测试（审计 P0-1）

背景：P0-1 命名空间预取失败时原实现跳过过滤（fail-open），放行全量结果，
跨 Agent 私有记忆可泄漏。修复后：预取超时/异常 → ns_set=set() → 全量
结果被过滤为空（fail-closed），不放行任何跨命名空间结果。

覆盖两条失败路径：
1. query_cypher 抛真实 SDK 异常（overgraph.OverGraphError）→ 空结果
2. query_cypher 返回空行（无 SessionNode 关联）→ 空结果

全部走公共入口 POST /memories/retrieve。
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from api.routes import router, Services, get_services

try:
    from overgraph import OverGraphError
    HAS_OVERGRAPH = True
except Exception:  # pragma: no cover - 环境无 overgraph 时退化
    HAS_OVERGRAPH = False


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    def _build(svc):
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app)

    return _build


def _make_svc() -> Services:
    """构造一个命中命名空间预取分支的服务容器。

    query_router 返回一条结果（模拟检索命中），graphlite_store.query_cypher
    由具体测试决定是抛异常还是返回空——从而命中 P0-1 的两条失败路径。
    """
    svc = Services()
    svc.query_router = MagicMock()
    svc.query_router.retrieve.return_value = [
        {"node_id": "ns_node_a", "content": "private memory of ns", "score": 0.9,
         "level": "hypergraph"},
    ]
    svc.graphlite_store = MagicMock()
    svc.quarantine_store = None
    svc.ontology_validator = None
    svc.tau_engine = None
    return svc


def test_namespace_prefetch_raises_sdk_error_fail_closed(client):
    """预取抛真实 OverGraphError → 结果必须为空（不放行全量）。"""
    svc = _make_svc()
    svc.graphlite_store.query_cypher.side_effect = (
        OverGraphError("ns query failed") if HAS_OVERGRAPH else RuntimeError("ns query failed")
    )

    resp = client(svc).post("/memories/retrieve", json={
        "query": "find something",
        "top_k": 5,
        "namespace": "agent_alpha",
    })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["results"] == [], f"fail-closed: 应返回空结果而非跨命名空间泄露, got {body['results']}"


def test_namespace_prefetch_empty_rows_fail_closed(client):
    """预取返回空（该命名空间无任何会话节点）→ 结果必须为空。"""
    svc = _make_svc()
    svc.graphlite_store.query_cypher.return_value = []

    resp = client(svc).post("/memories/retrieve", json={
        "query": "find something",
        "top_k": 5,
        "namespace": "agent_beta",
    })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["results"] == [], f"fail-closed: ns_set 为空集应过滤全部, got {body['results']}"
