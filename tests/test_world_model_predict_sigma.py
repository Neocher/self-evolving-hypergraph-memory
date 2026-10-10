"""SHM 世界模型最小可行预测器 PredictSigma 测试（P1/Phase-1）。

覆盖：零训练频率预测 / 预测残差 surprise_Σ / α 融合零回归 / 配置默认值，
以及真实 OverGraphStore 集成（AtomicFact 统计读取）。
"""
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import Services, get_services, router
from config.settings import WriteGateConfig
from core.world_model import PredictSigma, SigmaPrediction, fuse_surprise


class _FakeStore:
    """仅暴露 PredictSigma 所需只读统计接口的假 store（单元测试用）。"""

    def __init__(self, facts=None, entities=None):
        self._facts = list(facts or [])
        self._entities = list(entities or [])

    def get_atomic_facts_by_subject(self, subject, limit=200):
        return [f for f in self._facts if f.get("subject") == subject][:limit]

    def get_entities(self, limit=200):
        return self._entities[:limit]


def _fact(subject, predicate, object_):
    return {"subject": subject, "predicate": predicate, "object": object_}


# ════════════════════════════════════════════════════════════════
# 一、零训练预测 + 预测残差（假 store）
# ════════════════════════════════════════════════════════════════

class TestPredictSigmaUnit:
    def test_predict_empty_history(self):
        p = PredictSigma(store=_FakeStore())
        got = p.predict("Alice")
        assert isinstance(got, SigmaPrediction)
        assert got.relation == ""
        assert got.entity == ""
        assert got.probability == 0.0

    def test_predict_most_frequent_pair(self):
        store = _FakeStore(facts=[
            _fact("Alice", "visited", "Paris"),
            _fact("Alice", "visited", "Paris"),
            _fact("Alice", "works_at", "MIT"),
        ])
        got = PredictSigma(store=store).predict("Alice")
        assert got.relation == "visited"
        assert got.entity == "Paris"
        assert abs(got.probability - 2 / 3) < 1e-9

    def test_surprise_no_history_is_novel(self):
        assert PredictSigma(store=_FakeStore()).surprise("Alice") == 1.0

    def test_surprise_deterministic_is_zero(self):
        store = _FakeStore(facts=[
            _fact("Alice", "visited", "Paris"),
            _fact("Alice", "visited", "Paris"),
        ])
        assert PredictSigma(store=store).surprise("Alice") == 0.0

    def test_surprise_actual_seen(self):
        store = _FakeStore(facts=[
            _fact("Alice", "visited", "Paris"),
            _fact("Alice", "visited", "Paris"),
            _fact("Alice", "works_at", "MIT"),
        ])
        p = PredictSigma(store=store)
        assert abs(p.surprise("Alice", "visited", "Paris") - 1 / 3) < 1e-9
        assert abs(p.surprise("Alice", "works_at", "MIT") - 2 / 3) < 1e-9

    def test_surprise_actual_unseen_entity_falls_back_to_relation(self):
        store = _FakeStore(facts=[
            _fact("Alice", "visited", "Paris"),
            _fact("Alice", "visited", "Rome"),
        ])
        p = PredictSigma(store=store)
        # (visited, London) 未观测 → 退化为关系边缘 P(visited)=1 → 残差 0
        assert p.surprise("Alice", "visited", "London") == 0.0
        # 完全未观测关系 → 残差 1
        assert p.surprise("Alice", "lives_in", "London") == 1.0

    def test_surprise_actual_relation_only_uses_marginal(self):
        store = _FakeStore(facts=[
            _fact("Alice", "visited", "Paris"),
            _fact("Alice", "visited", "Rome"),
            _fact("Alice", "works_at", "MIT"),
        ])
        p = PredictSigma(store=store)
        assert abs(p.surprise("Alice", "visited") - 1 / 3) < 1e-9
        assert p.surprise("Alice", "unknown") == 1.0


class TestPredictSigmaContentMatch:
    def test_longest_entity_wins(self):
        store = _FakeStore(
            facts=[_fact("MIT", "located_in", "Cambridge")],
            entities=[{"name": "MIT"}, {"name": "MIT Media Lab"}],
        )
        got = PredictSigma(store=store).surprise_from_content("The MIT Media Lab opened.")
        # 最长实体 "MIT Media Lab" 无事实 → 1.0；若误匹配 "MIT" 则会有历史 → 0
        assert got == 1.0

    def test_no_match_is_novel(self):
        got = PredictSigma(
            store=_FakeStore(entities=[{"name": "Alice"}])
        ).surprise_from_content("Nothing relevant here")
        assert got == 1.0


# ════════════════════════════════════════════════════════════════
# 二、α 融合（零回归）
# ════════════════════════════════════════════════════════════════

class TestFuseSurprise:
    def test_alpha_zero_is_identity(self):
        assert fuse_surprise(0.3, 0.9, 0.0) == 0.3
        assert fuse_surprise(1.0, 0.0) == 1.0  # 默认 α=0

    def test_convex_combination(self):
        assert abs(fuse_surprise(0.2, 0.8, 0.5) - 0.5) < 1e-9
        assert abs(fuse_surprise(0.2, 0.8, 0.25) - 0.35) < 1e-9

    def test_alpha_clamped(self):
        assert fuse_surprise(0.2, 0.8, 2.0) == 0.8
        assert fuse_surprise(0.2, 0.8, -1.0) == 0.2


# ════════════════════════════════════════════════════════════════
# 三、配置默认值（零回归开关）
# ════════════════════════════════════════════════════════════════

class TestWorldModelConfig:
    def test_default_alpha_zero(self):
        assert WriteGateConfig().world_model_alpha == 0.0

    def test_alpha_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            WriteGateConfig(world_model_alpha=1.5)
        with pytest.raises(ValueError):
            WriteGateConfig(world_model_alpha=-0.1)


# ════════════════════════════════════════════════════════════════
# 四、真实 OverGraphStore 集成（AtomicFact 统计读取）
# ════════════════════════════════════════════════════════════════

class TestPredictSigmaRealStore:
    def test_predict_and_surprise(self, overgraph_store):
        # 真实库 create_atomic_fact 幂等（同 (s,p,o,vt) 覆盖同一节点）且
        # supersession（同 (s,p) 异 object → 旧值作废）。用不同 valid_time 造
        # 同 (s,p,o) 的多个 fact 节点，演示 pair 频率统计路径。
        s = overgraph_store
        s.create_atomic_fact("Alice", "visited", "Paris", valid_time="2023")
        s.create_atomic_fact("Alice", "visited", "Paris", valid_time="2024")
        s.create_atomic_fact("Alice", "works_at", "MIT")
        p = PredictSigma(store=s)
        got = p.predict("Alice")
        assert got.relation == "visited"
        assert got.entity == "Paris"
        assert abs(got.probability - 2 / 3) < 1e-9
        assert abs(p.surprise("Alice") - 1 / 3) < 1e-9
        assert abs(p.surprise("Alice", "visited", "Paris") - 1 / 3) < 1e-9
        assert abs(p.surprise("Alice", "works_at", "MIT") - 2 / 3) < 1e-9

    def test_surprise_from_content(self, overgraph_store):
        s = overgraph_store
        s.create_entity("Alice", entity_type="Person")
        s.create_atomic_fact("Alice", "visited", "Paris", valid_time="2023")
        s.create_atomic_fact("Alice", "visited", "Paris", valid_time="2024")
        s.create_atomic_fact("Alice", "works_at", "MIT")
        p = PredictSigma(store=s)
        assert abs(p.surprise_from_content("Alice went to Paris") - 1 / 3) < 1e-9
        assert p.surprise_from_content("no matching entity here") == 1.0

    def test_empty_store_is_novel(self, overgraph_store):
        p = PredictSigma(store=overgraph_store)
        assert p.surprise("Alice") == 1.0
        assert p.predict("Alice").probability == 0.0


# ════════════════════════════════════════════════════════════════
# 五、α>0 写路径融合（真实路由端到端）
# ════════════════════════════════════════════════════════════════

class _FakeEncoder:
    """确定性假 encoder（与 test_memory_accuracy 同构）。"""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2 ** 31))
        v = rng.randn(self.dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


def _svc(overgraph_store) -> Services:
    """Services 容器：真实 OverGraphStore + 假 encoder + 下游可选服务置 None。"""
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
    from config.settings import Settings

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


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    def _build(svc):
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app)

    return _build


# 内容足够长 + 含大写 → 长期效用 utility ≥ 0.5 → deep 路由落库 rpe_surprise。
_CONTENT = ("Alice visited Paris in 2023 for a long research trip across "
            "Europe to study the history of modern art museums and their collections.")


class TestWorldModelWritePathFusion:
    def test_alpha_zero_preserves_legacy_surprise(self, client, overgraph_store, monkeypatch):
        """α=0 → rpe_surprise 恒等于既有 1-max_sim（空向量库 max_sim=0 → 1.0）。"""
        _enable_world_model_gate(monkeypatch, alpha=0.0)
        svc = _svc(overgraph_store)
        resp = client(svc).post("/memories/episodes", json={
            "content": _CONTENT, "source": "test",
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "created", body
        got = overgraph_store.get_episode(body["episode_id"])
        assert got.get("rpe_surprise") == 1.0

    def test_alpha_half_fuses_sigma(self, client, overgraph_store, monkeypatch):
        """α=0.5、surprise_Σ=0（subject 完全可预测）→ rpe_surprise=0.5。"""
        _enable_world_model_gate(monkeypatch, alpha=0.5)
        overgraph_store.create_entity("Alice", entity_type="Person")
        overgraph_store.create_atomic_fact("Alice", "visited", "Paris")
        svc = _svc(overgraph_store)
        resp = client(svc).post("/memories/episodes", json={
            "content": _CONTENT, "source": "test",
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "created", body
        got = overgraph_store.get_episode(body["episode_id"])
        # 空向量库 max_sim=0 → sim_surprise=1.0；surprise_Σ=0 → 0.5·1.0+0.5·0.0
        assert abs(got.get("rpe_surprise") - 0.5) < 1e-9
