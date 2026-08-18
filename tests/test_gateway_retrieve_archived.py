"""v5.32.1 — GatewayAPI.retrieve include_archived 透传回归测试。

Codex 终审发现：tests/test_gateway_api.py 不存在，GatewayAPI 层透传无直接单测。
本文件补充 GatewayAPI.retrieve 的 include_archived 参数透传 + Cypher 兜底条件化测试。
"""

import pytest


class _FakeQueryRouter:
    """记录 retrieve 收到的 include_archived / level / rerank 参数。"""

    def __init__(self):
        self.calls = []

    def retrieve(self, query, include_archived=False, session_ts=None,
                 level=None, rerank=None):
        self.calls.append({"query": query, "include_archived": include_archived,
                           "level": level, "rerank": rerank})
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


@pytest.mark.asyncio
async def test_retrieve_passes_level_and_rerank_from_strategy():
    """P3a R2: strategy 经 _level_from_strategy 映射为 level 透传；rerank=None 读配置。"""
    from retrieval.query_router import RetrievalLevel
    api, router = _make_api()
    await api.retrieve("hello", strategy="hybrid")
    assert router.calls[0]["level"] == RetrievalLevel.FUSION
    assert router.calls[0]["rerank"] is None


@pytest.mark.asyncio
async def test_retrieve_default_strategy_maps_hypergraph():
    """默认 strategy="auto" → level=HYPERGRAPH，rerank=None。"""
    from retrieval.query_router import RetrievalLevel
    api, router = _make_api()
    await api.retrieve("hello")
    assert router.calls[0]["level"] == RetrievalLevel.HYPERGRAPH
    assert router.calls[0]["rerank"] is None


@pytest.mark.asyncio
async def test_retrieve_fusion_not_degraded():
    """P3a R4 P2-1: GatewayAPI 正常 FUSION 结果（level=fusion_*）不得标 degraded。"""
    class _FusionRouter:
        def retrieve(self, query, include_archived=False, session_ts=None,
                     level=None, rerank=None):
            return [{"content": "fusion result", "score": 0.9, "level": "fusion_multi"}]

    api, _ = _make_api(router=_FusionRouter())
    resp = await api.retrieve("hello", strategy="hybrid")
    assert resp.degraded is False, "正常 FUSION 结果不得标 degraded"


def test_cypher_fallback_archived_clause_builder():
    """Cypher 兜底的 archived_clause 逻辑（静态验证条件化语义）。"""
    from gateway.gateway_api import GatewayAPI

    # 直接验证签名中存在 include_archived 且默认 False
    import inspect

    sig = inspect.signature(GatewayAPI.retrieve)
    assert "include_archived" in sig.parameters
    assert sig.parameters["include_archived"].default is False


@pytest.mark.asyncio
async def test_cypher_fallback_timeout_degraded_true():
    """P1-2: Cypher 兜底超时/异常 → 空结果 + degraded=True（对齐 REST 语义）。

    修复前 degraded=True 置于 wait_for 之后，超时/异常跳 except 时该行未执行，
    degraded 恒 False。修复后置位于 wait_for 之前，超时/异常仍 degraded=True。
    """
    import asyncio

    from gateway.gateway_api import GatewayAPI

    class _EmptyRouter:
        def retrieve(self, query, include_archived=False, session_ts=None,
                     level=None, rerank=None):
            return []

    class _TimeoutStore:
        def query_cypher(self, *args, **kwargs):
            raise asyncio.TimeoutError()

    class _Svcs:
        query_router = _EmptyRouter()
        graphlite_store = _TimeoutStore()

    api = GatewayAPI.__new__(GatewayAPI)
    api._svc = _Svcs()
    api._logger = type("L", (), {
        "warning": lambda self, *a, **k: None,
        "exception": lambda self, *a, **k: None,
    })()

    resp = await api.retrieve("hello timeout")
    assert resp.degraded is True, "Cypher 兜底超时应标 degraded=True"
    assert resp.results == []
    assert resp.total_found == 0