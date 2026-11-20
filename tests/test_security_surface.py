"""第二轮安全面修复（组 β）回归测试

- G1: A2A 非 loopback 监听 + 未设 A2A_API_KEY → 拒绝启动（SystemExit(2)）
- G2: /batch/relations edge_type 白名单：非法值 400，不进查询
- G4: TokenManager.validate 快照迭代：过期 token 连续两次请求稳定 401（非 500）

全部走公共入口（main()/路由/中间件），不直调被测私有逻辑以外的内部方法。
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


# ─── G1: A2A 启动守卫 ────────────────────────────────────────────────


class TestG1A2AStartupGuard:

    def _argv(self, monkeypatch, *args):
        import sys
        monkeypatch.setattr(sys, "argv", ["a2a_server", *args])

    def test_non_loopback_without_key_refuses(self, monkeypatch):
        """--host 0.0.0.0 且无 A2A_API_KEY → SystemExit(2)，不进 create_app。"""
        import gateway.a2a_server as a2a
        monkeypatch.delenv("A2A_API_KEY", raising=False)
        monkeypatch.setattr(a2a, "A2A_API_KEY", "")
        self._argv(monkeypatch, "--host", "0.0.0.0", "--port", "18001")
        with pytest.raises(SystemExit) as ei:
            a2a.main()
        assert ei.value.code == 2

    def test_loopback_without_key_passes_guard(self, monkeypatch):
        """默认 loopback 无 key → 过守卫继续启动（向后兼容本地开发）。"""
        import gateway.a2a_server as a2a
        import uvicorn
        monkeypatch.delenv("A2A_API_KEY", raising=False)
        monkeypatch.setattr(a2a, "A2A_API_KEY", "")
        monkeypatch.setattr(a2a, "create_app", lambda: object())
        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
        self._argv(monkeypatch, "--host", "127.0.0.1", "--port", "18001")
        a2a.main()  # 不抛 SystemExit

    def test_non_loopback_with_key_passes_guard(self, monkeypatch):
        """0.0.0.0 + 显式 key → 行为不变（向后兼容）。"""
        import gateway.a2a_server as a2a
        import uvicorn
        monkeypatch.setenv("A2A_API_KEY", "sekret")
        monkeypatch.setattr(a2a, "A2A_API_KEY", "sekret")
        monkeypatch.setattr(a2a, "create_app", lambda: object())
        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
        self._argv(monkeypatch, "--host", "0.0.0.0", "--port", "18001")
        a2a.main()


# ─── G2: /batch/relations edge_type 白名单 ────────────────────────────


class TestG2BatchRelationEdgeType:

    def _client(self, svc):
        from api.routes import router, get_services
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app)

    def _svc(self):
        svc = MagicMock()
        svc.write_queue = None  # qsubmit 降级同步直调
        return svc

    def test_invalid_edge_type_rejected_400(self):
        svc = self._svc()
        resp = self._client(svc).post("/batch/relations", json={"relations": [
            {"subject": "a", "relation": "r", "object": "b",
             "edge_type": "X; DETACH DELETE (n)"},
        ]})
        assert resp.status_code == 400
        assert "edge_type" in resp.text
        svc.graph_store.execute_cypher.assert_not_called()

    def test_non_string_edge_type_rejected_400(self):
        svc = self._svc()
        resp = self._client(svc).post("/batch/relations", json={"relations": [
            {"subject": "a", "relation": "r", "object": "b", "edge_type": 123},
        ]})
        assert resp.status_code == 400

    def test_default_edge_type_accepted(self):
        svc = self._svc()
        resp = self._client(svc).post("/batch/relations", json={"relations": [
            {"subject": "a", "relation": "r", "object": "b"},
        ]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["created"] == 1
        svc.graph_store.execute_cypher.assert_called()

    def test_lowercase_edge_type_rejected_400(self):
        svc = self._svc()
        resp = self._client(svc).post("/batch/relations", json={"relations": [
            {"subject": "a", "relation": "r", "object": "b", "edge_type": "relates_to"},
        ]})
        assert resp.status_code == 400


# ─── G4: 过期 token 连续两次稳定 401 ──────────────────────────────────


class TestG4ExpiredToken:

    def _manager(self, tmp_path):
        from gateway.auth import TokenManager
        tm = TokenManager(token_path=str(tmp_path / "tokens.json"))
        # 过期 token 放首位（非末位）：覆盖"删除后迭代器是否继续前进"的场景
        expired = tm.create_key("expired", ttl=-10)
        tm.create_key("valid", ttl=3600)
        return tm, expired

    def test_validate_expired_twice_no_error(self, tmp_path):
        tm, expired = self._manager(tmp_path)
        assert tm.validate(expired) is None  # 首击：命中并撤销过期 token
        assert tm.validate(expired) is None  # 二击：稳定 None，不抛错
        assert tm.count() == 1
        assert tm.validate(expired) is None  # 已撤销的 token 仍是 None

    def test_middleware_expired_token_401_twice(self, tmp_path):
        """连续两次携带过期 token 的请求都应 401（回归：不得 500）。"""
        from gateway.auth import create_auth_middleware
        tm, expired = self._manager(tmp_path)
        app = FastAPI()

        @app.post("/x")
        async def _x():
            return {"ok": True}

        app.middleware("http")(create_auth_middleware(dev_mode=False, token_manager=tm))
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {expired}"}
        r1 = client.post("/x", headers=headers)
        r2 = client.post("/x", headers=headers)
        assert r1.status_code == 401, r1.text
        assert r2.status_code == 401, r2.text

    def test_valid_token_still_works(self, tmp_path):
        from gateway.auth import create_auth_middleware, TokenManager
        tm = TokenManager(token_path=str(tmp_path / "tokens.json"))
        good = tm.create_key("good", ttl=3600)
        app = FastAPI()

        @app.post("/x")
        async def _x():
            return {"ok": True}

        app.middleware("http")(create_auth_middleware(dev_mode=False, token_manager=tm))
        client = TestClient(app)
        resp = client.post("/x", headers={"Authorization": f"Bearer {good}"})
        assert resp.status_code == 200
