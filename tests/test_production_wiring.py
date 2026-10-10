"""生产路径接线验收测试（P1 世界模型落库 + P2 schema 版本化门控）。

覆盖 AC1–AC6：默认关闭零回归 / P1 落库 / P1 消费 / P2 四动作接线 /
P2 不覆盖旧 schema / 迟滞在接线后仍生效。
"""
import math

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import Services, get_services, router
from config.settings import (
    RetrievalConfig,
    SchemaVersioningConfig,
    Settings,
    WriteGateConfig,
)
from core.dream_pipeline import DreamPipeline
from retrieval.query_router import QueryRouter, QueryRouterConfig


# ════════════════════════════════════════════════════════════════
# 公共辅助
# ════════════════════════════════════════════════════════════════

class _FakeEncoder:
    """确定性假 encoder（与 test_world_model_predict_sigma 同构）。"""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2 ** 31))
        v = rng.randn(self.dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


def _svc(overgraph_store) -> Services:
    """Services 容器：真实 OverGraphStore + 假 encoder + 可选服务置 None。"""
    svc = Services()
    svc.graph_store = overgraph_store
    svc.encoder = _FakeEncoder()
    svc.tau_engine = None
    svc.ssm_gate = None
    svc.defense_engine = None
    svc.quarantine_store = None
    svc.ontology_validator = None
    svc.ontology_v2 = None
    svc.evidence_tracker = None
    svc.dream_scheduler = None
    svc.write_queue = None
    svc.query_router = None
    svc.hyperedge_manager = None
    return svc


def _enable_world_model_gate(monkeypatch, alpha: float = 0.5) -> WriteGateConfig:
    wg = WriteGateConfig(
        enabled=True,
        surprise_deep=0.45,
        surprise_cache=0.25,
        utility_min=0.5,
        cache_tau=0.5,
        world_model_alpha=alpha,
    )
    s = Settings()
    s.write_gate = wg
    monkeypatch.setattr("config.settings._settings", s)
    return wg


def _enable_schema_versioning(monkeypatch) -> None:
    s = Settings()
    s.schema_versioning = SchemaVersioningConfig(enabled=True)
    monkeypatch.setattr("config.settings._settings", s)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    def _build(svc):
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app)

    return _build


# 内容足够长 + 含大写 → utility ≥ 0.5 → deep 路由落库 rpe_surprise。
_CONTENT = ("Alice visited Paris in 2023 for a long research trip across "
            "Europe to study the history of modern art museums and their collections.")

# alpha=0（默认）deep 路由 episode 节点的字段集合（改动前基线，逐字段一致）。
_BASELINE_EPISODE_KEYS = frozenset({
    "archived", "content", "created_at", "fact_track", "id",
    "rpe_surprise", "source", "source_type", "tau_initial",
    "version", "visibility",
})


# ════════════════════════════════════════════════════════════════
# AC1-A / AC2 —— P1 写路径落库（默认关零回归 + α>0 落库）
# ════════════════════════════════════════════════════════════════

class TestWorldModelWritePersistence:
    def test_alpha_zero_no_new_fields(self, client, overgraph_store, monkeypatch):
        """AC1-A：α=0（默认）→ episode_data 与改动前逐字段一致（无新字段）。"""
        _enable_world_model_gate(monkeypatch, alpha=0.0)
        svc = _svc(overgraph_store)
        resp = client(svc).post("/memories/episodes", json={
            "content": _CONTENT, "source": "test",
        })
        assert resp.status_code == 200, resp.text
        got = overgraph_store.get_episode(resp.json()["episode_id"])
        assert got is not None
        assert "rpe_surprise_sigma" not in got
        assert "pred_entropy" not in got
        # 逐字段一致：字段集合 == 改动前基线（无任何新字段）
        assert set(got.keys()) == _BASELINE_EPISODE_KEYS, set(got.keys())

    def test_alpha_positive_writes_sigma_fields(self, client, overgraph_store, monkeypatch):
        """AC2：α>0 → episode_data 出现 rpe_surprise_sigma 与 pred_entropy（有限浮点）。"""
        _enable_world_model_gate(monkeypatch, alpha=0.5)
        overgraph_store.create_entity("Alice", entity_type="Person")
        # 两类 (predicate, object) 各 1 条 → 分布熵 ln(2)，预测残差 0.5
        overgraph_store.create_atomic_fact("Alice", "visited", "Paris")
        overgraph_store.create_atomic_fact("Alice", "works_at", "MIT")
        svc = _svc(overgraph_store)
        resp = client(svc).post("/memories/episodes", json={
            "content": _CONTENT, "source": "test",
        })
        assert resp.status_code == 200, resp.text
        got = overgraph_store.get_episode(resp.json()["episode_id"])
        assert "rpe_surprise_sigma" in got
        assert "pred_entropy" in got
        sigma = got["rpe_surprise_sigma"]
        entropy = got["pred_entropy"]
        # 均为有限浮点数（非 nan/inf），且由输入计算（非硬编码常量）
        assert isinstance(sigma, float) and math.isfinite(sigma)
        assert isinstance(entropy, float) and math.isfinite(entropy)
        assert sigma == pytest.approx(0.5)
        assert entropy == pytest.approx(math.log(2))


# ════════════════════════════════════════════════════════════════
# AC3 / A3 —— P1 消费端 rpe_surprise_sigma 独立阈值重排
# ════════════════════════════════════════════════════════════════

def _qr(config=None) -> QueryRouter:
    return QueryRouter(None, None, None, config=config)


def _sigma_candidates() -> list[dict]:
    return [
        {"id": "a", "score": 0.50, "rpe_surprise_sigma": 0.9},  # 高残差 → boost
        {"id": "b", "score": 0.55, "rpe_surprise_sigma": 0.1},  # 低残差 → dampen
        {"id": "c", "score": 0.60},                             # 字段缺失 → 不变
    ]


class TestRpeSigmaConsumption:
    def test_switch_off_unchanged(self):
        """AC3：开关关 → 同一输入返回不变（分数与顺序都不动）。"""
        inp = _sigma_candidates()
        out = _qr(QueryRouterConfig(rpe_sigma_enabled=False))._rpe_rerank_results(inp)
        assert out == inp
        assert [r["score"] for r in out] == [0.50, 0.55, 0.60]

    def test_field_missing_unchanged(self):
        """AC3：字段缺失（开关开）→ 分数不修改，仅按原分数自然降序。"""
        inp = [
            {"id": "a", "score": 0.50},
            {"id": "b", "score": 0.55},
            {"id": "c", "score": 0.60},
        ]
        out = _qr(QueryRouterConfig(rpe_sigma_enabled=True))._rpe_rerank_results(inp)
        assert [r["score"] for r in out] == [0.60, 0.55, 0.50]

    def test_enabled_changes_order(self):
        """AC3：开关开 + 字段存在 → 分数被修改且排序改变（非恒真断言）。"""
        off = _qr(QueryRouterConfig(rpe_sigma_enabled=False))._rpe_rerank_results(
            _sigma_candidates())
        on = _qr(QueryRouterConfig(rpe_sigma_enabled=True))._rpe_rerank_results(
            _sigma_candidates())
        assert [r["id"] for r in off] == ["a", "b", "c"]
        # a 升 0.525（boost）、b 降 0.5225（dampen）、c 不变 0.60 → 排序改变
        assert [r["id"] for r in on] == ["c", "a", "b"]
        on_scores = {r["id"]: r["score"] for r in on}
        assert on_scores["a"] == pytest.approx(0.525)
        assert on_scores["b"] == pytest.approx(0.5225)
        assert on_scores["c"] == pytest.approx(0.60)

    def test_boost_clamped_to_original_max(self):
        """A3：重排后分数不得超出原最高分（钳制语义与既有 rpe_rerank 一致）。"""
        inp = [
            {"id": "top", "score": 0.90, "rpe_surprise_sigma": 0.9},
            {"id": "low", "score": 0.10},
        ]
        out = _qr(QueryRouterConfig(rpe_sigma_enabled=True))._rpe_rerank_results(inp)
        scores = {r["id"]: r["score"] for r in out}
        assert scores["top"] == pytest.approx(0.90)  # 0.9×1.05=0.945 → 钳制到 0.90


# ════════════════════════════════════════════════════════════════
# AC1-B / AC4 / AC5 —— P2 schema 版本化门控接线
# ════════════════════════════════════════════════════════════════

def _seed_engineer_schema(pipe: DreamPipeline, n: int = 3) -> None:
    for i in range(n):
        pipe._persist_schema_versioning([{
            "id": "c1",
            "episode_id": f"ep{i}",
            "content": "alice engineer acme",
            "slots": [("role", "engineer")],
        }])


class TestSchemaVersioningWiring:
    def test_disabled_skips_gate(self, overgraph_store, monkeypatch):
        """AC1-B：enabled=False（默认）→ _persist_schema_evolution 副作用与改动前一致。"""
        monkeypatch.setattr("config.settings._settings", Settings())
        pipe = DreamPipeline()
        evolved = pipe._persist_schema_evolution(
            overgraph_store, [{"id": "c1", "entity_links": []}])
        assert evolved == 0
        # 门控被完全跳过：未建注册表、未累积 episode（零副作用）
        assert pipe._schema_registry is None
        assert pipe._schema_episodes == set()

    def test_supersede_on_sustained_contradiction(self, monkeypatch):
        """AC4：持续异常证据（r≥2）→ 产生 SUPERSEDE 后继版本。"""
        _enable_schema_versioning(monkeypatch)
        pipe = DreamPipeline()
        _seed_engineer_schema(pipe, n=3)
        executed = pipe._persist_schema_versioning([{
            "id": "c1",
            "episode_id": "epX",
            "content": "alice astronaut acme",
            "slots": [("role", "astronaut")],
            "schema_signals": {"r": 2.0},
        }])
        assert executed == 1
        reg = pipe._schema_registry
        assert "c1" in reg.schemas
        head = reg.head("c1")
        assert head.id != "c1"               # 新版本 id ≠ 旧 id
        assert head.predecessor == "c1"      # predecessor 链存在
        assert head.version == 2

    def test_supersede_preserves_old_schema(self, monkeypatch):
        """AC5：SUPERSEDE 后旧 schema 仍可查到且标记 historical。"""
        _enable_schema_versioning(monkeypatch)
        pipe = DreamPipeline()
        _seed_engineer_schema(pipe, n=3)
        pipe._persist_schema_versioning([{
            "id": "c1",
            "episode_id": "epX",
            "content": "alice astronaut acme",
            "slots": [("role", "astronaut")],
            "schema_signals": {"r": 2.0},
        }])
        reg = pipe._schema_registry
        old = reg.schemas["c1"]
        assert old.active is False           # historical
        assert old.successor is not None
        assert old.successor == reg.head("c1").id
        assert old.id == "c1"                # 旧 id 未变、仍可查到

    def test_isolated_contradiction_no_new_version(self, monkeypatch):
        """AC4/B3：孤立强矛盾（r=0,x=0）→ 迟滞捷径，不新建版本。"""
        _enable_schema_versioning(monkeypatch)
        pipe = DreamPipeline()
        _seed_engineer_schema(pipe, n=3)
        before = set(pipe._schema_registry.schemas.keys())
        pipe._persist_schema_versioning([{
            "id": "c1",
            "episode_id": "epX",
            "content": "alice astronaut acme",
            "slots": [("role", "astronaut")],
        }])
        after = set(pipe._schema_registry.schemas.keys())
        assert before == after               # 保持不新建版本
        assert pipe._schema_registry.schemas["c1"].active is True

    def test_config_defaults_off(self):
        """新开关默认 False/0（零回归基线）。"""
        assert RetrievalConfig().rpe_sigma_enabled is False
        assert QueryRouterConfig().rpe_sigma_enabled is False
        assert SchemaVersioningConfig().enabled is False
        assert Settings().schema_versioning.enabled is False
        assert WriteGateConfig().world_model_alpha == 0.0
