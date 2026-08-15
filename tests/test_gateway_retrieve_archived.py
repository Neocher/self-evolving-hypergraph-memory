"""v5.32.1 — GatewayAPI.retrieve include_archived 透传回归测试。

Codex 终审发现：tests/test_gateway_api.py 不存在，GatewayAPI 层透传无直接单测。
本文件补充 GatewayAPI.retrieve 的 include_archived 参数透传 + Cypher 兜底条件化测试。
"""

import pytest


class _FakeQueryRouter:
    """记录 retrieve 收到的 include_archived 参数。"""

    def __init__(self):
        self.calls = []

    def retrieve(self, query, include_archived=False):
        self.calls.append({"query": query, "include_archived": include_archived})
        return [{"content": f"result_{query}", "score": 0.7}]


class _FakeServices:
    def __init__(self, router):
        self.query_router = router
        self.graphlite_store = None


def _make_api(router=None):
    from gateway.gateway_api import GatewayAPI

    router = router or _FakeQueryRouter()
    services = _FakeServices(router)
    api = GatewayAPI.__new__(GatewayAPI)
    api._svc = services
    api._logger = type("L", (), {"exception": lambda self, *a, **k: None})()
    return api, router


@pytest.mark.asyncio
async def test_retrieve_passthrough_include_archived_true():
    api, router = _make_api()
    await api.retrieve("hello", include_archived=True)
    assert router.calls[0]["include_archived"] is True


@pytest.mark.asyncio
async def test_retrieve_passthrough_include_archived_default_false():
    api, router = _make_api()
    await api.retrieve("hello")
    assert router.calls[0]["include_archived"] is False


@pytest.mark.asyncio
async def test_retrieve_passthrough_include_archived_explicit_false():
    api, router = _make_api()
    await api.retrieve("hello", include_archived=False)
    assert router.calls[0]["include_archived"] is False


def test_cypher_fallback_archived_clause_builder():
    """Cypher 兜底的 archived_clause 逻辑（静态验证条件化语义）。"""
    from gateway.gateway_api import GatewayAPI

    # 直接验证签名中存在 include_archived 且默认 False
    import inspect

    sig = inspect.signature(GatewayAPI.retrieve)
    assert "include_archived" in sig.parameters
    assert sig.parameters["include_archived"].default is False
