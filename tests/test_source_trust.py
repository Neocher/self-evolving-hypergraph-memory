"""
来源信任分级接通测试（P3）
========================
覆盖：写时 source_type 落库 + 防洗白降级 + promote 硬编码 inferred。

- create_episode 落库后 _flatten_row 回读 source_type=="direct" 默认值
- 写 source="hermes" 且 source_type="direct" → 被降级为 inferred（防洗白）
- promote_to_episode 落库 source_type=="inferred"

走生产链路（真实 GraphLiteStore + HTTP 路由 / GatewayAPI），
断言落库后 get_episode 读回的 source_type 字段。
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.models import resolve_source_type
from api.routes import router, Services, get_services


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    def _build(svc):
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app)

    return _build


def _svc(graphlite_store) -> Services:
    svc = Services()
    svc.graphlite_store = graphlite_store
    return svc


class TestResolveSourceType:
    """防洗白规则单元测试（纯函数，无依赖）。"""

    def test_user_direct_allowed(self):
        assert resolve_source_type("user", "direct") == "direct"

    def test_agent_direct_downgraded(self):
        for src in ("hermes", "codex", "claude", "opencode", "api"):
            assert resolve_source_type(src, "direct") == "inferred", src

    def test_tool_not_downgraded(self):
        assert resolve_source_type("hermes", "tool") == "tool"

    def test_inferred_passthrough(self):
        assert resolve_source_type("hermes", "inferred") == "inferred"

    def test_enum_coercion(self):
        from api.models import SourceType
        assert resolve_source_type("hermes", SourceType.DIRECT) == "inferred"
        assert resolve_source_type("user", SourceType.DIRECT) == "direct"


class TestSourceTypeDefaultDirect:
    """清单 1：create_episode 落库后回读 source_type=="direct" 默认值。"""

    def test_create_episode_defaults_direct(self, graphlite_store):
        eid = graphlite_store.create_episode({
            "id": "ep_default_direct",
            "content": "默认来源分级测试",
            "source": "user",
        })
        got = graphlite_store.get_episode(eid)
        assert got is not None
        assert got.get("source_type") == "direct", (
            f"未显式传 source_type 时应默认 direct，实际 {got.get('source_type')!r}"
        )


class TestAntiLaunderingRoute:
    """清单 2：写 source="hermes" 且 source_type="direct" → 降级 inferred。"""

    def test_agent_declared_direct_downgraded(self, client, graphlite_store):
        svc = _svc(graphlite_store)
        resp = client(svc).post("/memories/episodes", json={
            "content": "agent 直述防洗白测试",
            "source": "hermes",
            "source_type": "direct",
        })
        assert resp.status_code == 200, resp.text
        episode_id = resp.json()["episode_id"]
        got = graphlite_store.get_episode(episode_id)
        assert got is not None
        assert got.get("source_type") == "inferred", (
            f"agent 来源声明 direct 应降级 inferred，实际 {got.get('source_type')!r}"
        )

    def test_user_direct_kept(self, client, graphlite_store):
        svc = _svc(graphlite_store)
        resp = client(svc).post("/memories/episodes", json={
            "content": "用户直述不降级测试",
            "source": "user",
            "source_type": "direct",
        })
        assert resp.status_code == 200, resp.text
        got = graphlite_store.get_episode(resp.json()["episode_id"])
        assert got.get("source_type") == "direct"


class TestPromoteInferred:
    """清单 3：promote_to_episode 落库 source_type=="inferred"。"""

    def test_promote_writes_inferred(self, client, graphlite_store):
        svc = _svc(graphlite_store)
        resp = client(svc).post("/memories/promote", json={
            "sensory_record_id": "nonexistent_record",
        })
        assert resp.status_code == 200, resp.text
        episode_id = resp.json()["episode_id"]
        got = graphlite_store.get_episode(episode_id)
        assert got is not None
        assert got.get("source_type") == "inferred", (
            f"promote 落库应写死 inferred，实际 {got.get('source_type')!r}"
        )


class TestGatewayStoreEpisodeSourceType:
    """gateway_api 直调 create_episode 同步 source_type（堵 A2A 洗白漏洞）。"""

    @pytest.mark.asyncio
    async def test_agent_source_inferred(self, graphlite_store):
        from gateway.gateway_api import GatewayAPI
        api = GatewayAPI(_svc(graphlite_store))
        resp = await api.store_episode("A2A agent 写入", source="hermes")
        got = graphlite_store.get_episode(resp.episode_id)
        assert got.get("source_type") == "inferred"

    @pytest.mark.asyncio
    async def test_user_source_direct(self, graphlite_store):
        from gateway.gateway_api import GatewayAPI
        api = GatewayAPI(_svc(graphlite_store))
        resp = await api.store_episode("A2A 用户写入", source="user")
        got = graphlite_store.get_episode(resp.episode_id)
        assert got.get("source_type") == "direct"
