"""R2 N6-P3: A2A / ACP 网关 session_ts 透传测试。

网关层补齐 session_ts 透传（None 默认向后兼容），时间锚下沉到网关：
  1. A2A /memory/retrieve：RetrieveRequest.session_ts → api.retrieve(session_ts=...)
  2. ACP shm:retrieve：params["session_ts"] → gateway.retrieve(session_ts=...)
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.a2a_server import register_routes
from gateway.acp_adapter import SHMACPAdapter


def _resp(**kw) -> SimpleNamespace:
    base = dict(
        query="q", strategy_used="auto", results=[], total_found=0,
        latency_ms=0.0, degraded=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestA2ASessionTs:

    def _client(self, api) -> TestClient:
        app = FastAPI()
        register_routes(app, api)
        return TestClient(app)

    def test_retrieve_passes_session_ts(self):
        api = MagicMock()
        api.retrieve = AsyncMock(return_value=_resp())
        client = self._client(api)
        session_ts = 1_700_000_000.0
        resp = client.post("/memory/retrieve", json={"query": "q", "session_ts": session_ts})
        assert resp.status_code == 200
        api.retrieve.assert_awaited_once()
        assert api.retrieve.call_args.kwargs["session_ts"] == session_ts

    def test_retrieve_session_ts_defaults_none(self):
        api = MagicMock()
        api.retrieve = AsyncMock(return_value=_resp())
        client = self._client(api)
        resp = client.post("/memory/retrieve", json={"query": "q"})
        assert resp.status_code == 200
        assert api.retrieve.call_args.kwargs["session_ts"] is None


class TestACPSessionTs:

    def test_retrieve_passes_session_ts(self):
        gateway = MagicMock()
        gateway.retrieve = AsyncMock(return_value=_resp())
        adapter = SHMACPAdapter(MagicMock(), gateway)
        session_ts = 1_700_000_000.0
        result = asyncio.run(
            adapter._handle_retrieve({"query": "q", "session_ts": session_ts})
        )
        assert result["status"] == "ok"
        gateway.retrieve.assert_awaited_once()
        assert gateway.retrieve.call_args.kwargs["session_ts"] == session_ts

    def test_retrieve_session_ts_defaults_none(self):
        gateway = MagicMock()
        gateway.retrieve = AsyncMock(return_value=_resp())
        adapter = SHMACPAdapter(MagicMock(), gateway)
        result = asyncio.run(adapter._handle_retrieve({"query": "q"}))
        assert result["status"] == "ok"
        assert gateway.retrieve.call_args.kwargs["session_ts"] is None
