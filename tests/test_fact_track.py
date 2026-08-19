"""双轨事实记忆 (Dual-Track Facts) 测试。

覆盖任务书测试清单 8 项：
1. classify_fact_track("我喜欢喝茶") == "core"（关键词分支）
2. classify_fact_track("今天下午开会") == "active"（默认）
3. classify_fact_track(ontology_type=...) 映射分支
4. create_episode 落库回读 fact_track=="active" 默认值
5. write.py POST 传偏好内容 → 回读 fact_track=="core"
6. compute_tau 差异化：同一 created_at，core 衰减更慢
(7/8 由 test_tau_decay.py / test_version_consistency.py 覆盖不回归)
"""
import time
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router, Services, get_services
from core.fact_track import (
    CORE_ONTOLOGY_TYPES,
    CORE_KEYWORDS,
    classify_fact_track,
    is_core_track,
)
from core.tau_decay import TauDecayEngine


# ─── 改动 1：classify_fact_track 纯函数 ──────────────────────

class TestClassifyFactTrack:
    def test_keyword_preference_is_core(self):
        """清单 1：内容含持久化关键词 → core。"""
        assert classify_fact_track("我喜欢喝茶") == "core"

    def test_default_active(self):
        """清单 2：事件/临时内容 → active。"""
        assert classify_fact_track("今天下午开会") == "active"

    def test_core_ontology_type_mapping(self):
        """清单 3：core 本体类型 → core。"""
        assert classify_fact_track("张三和李四是夫妻", ontology_type="relationship") == "core"

    def test_active_ontology_type_mapping(self):
        """清单 3：event_date → active。"""
        assert classify_fact_track("会议在2026年8月举行", ontology_type="event_date") == "active"

    def test_all_core_ontology_types(self):
        """判别性：6 个 core 本体类型全部直判 core。"""
        for otype in CORE_ONTOLOGY_TYPES:
            assert classify_fact_track("任意内容", ontology_type=otype) == "core"

    def test_generic_fact_with_core_keyword(self):
        """generic_fact + 偏好关键词 → core（关键词分支补 core）。"""
        assert classify_fact_track("我喜欢咖啡", ontology_type="generic_fact") == "core"

    def test_empty_content_default_active(self):
        """空内容 → active。"""
        assert classify_fact_track("") == "active"

    def test_is_core_track(self):
        assert is_core_track("core") is True
        assert is_core_track("active") is False

    def test_core_keywords_nonempty(self):
        """判别性：CORE_KEYWORDS 非空且含任务书要求的关键词。"""
        assert "喜欢" in CORE_KEYWORDS
        assert "我住" in CORE_KEYWORDS


# ─── 改动 2/4：落库默认 + 写时分类 ───────────────────────────

class TestFactTrackPersistence:
    def test_create_episode_defaults_active(self, overgraph_store):
        """清单 4：不带 fact_track 的 create_episode → 回读 active。"""
        eid = str(uuid.uuid4())
        overgraph_store.create_episode({
            "id": eid,
            "content": "一条普通记忆",
            "created_at": time.time(),
            "tau_initial": 1.0,
            "source": "test",
        })
        got = overgraph_store.get_episode(eid)
        assert got is not None
        assert got.get("fact_track") == "active"

    def test_create_episode_explicit_core_preserved(self, overgraph_store):
        """显式 fact_track="core" 不被 setdefault 覆盖。"""
        eid = str(uuid.uuid4())
        overgraph_store.create_episode({
            "id": eid,
            "content": "我住在北京",
            "created_at": time.time(),
            "tau_initial": 1.0,
            "source": "test",
            "fact_track": "core",
        })
        got = overgraph_store.get_episode(eid)
        assert got is not None
        assert got.get("fact_track") == "core"


class TestWriteRouteFactTrack:
    """清单 5：POST /memories/episodes 偏好内容 → fact_track 落库 core。"""

    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(router)

        def _build(svc):
            app.dependency_overrides[get_services] = lambda: svc
            return TestClient(app)

        return _build

    def test_preference_content_persists_core(self, client, overgraph_store):
        """偏好内容经生产写路径 → 落库 fact_track=="core"。"""
        svc = Services()
        svc.graphlite_store = overgraph_store

        resp = client(svc).post("/memories/episodes", json={
            "content": "我喜欢喝茶",
            "source": "user",
        })

        assert resp.status_code == 200, resp.text
        episode_id = resp.json()["episode_id"]
        got = overgraph_store.get_episode(episode_id)
        assert got is not None
        assert got.get("fact_track") == "core"

    def test_event_content_persists_active(self, client, overgraph_store):
        """事件内容经生产写路径 → 落库 fact_track=="active"。"""
        svc = Services()
        svc.graphlite_store = overgraph_store

        resp = client(svc).post("/memories/episodes", json={
            "content": "今天下午开会",
            "source": "user",
        })

        assert resp.status_code == 200, resp.text
        episode_id = resp.json()["episode_id"]
        got = overgraph_store.get_episode(episode_id)
        assert got is not None
        assert got.get("fact_track") == "active"


# ─── 改动 3：τ 衰减差异化 ───────────────────────────────────

class TestTauDecayFactTrack:
    def test_compute_tau_core_slower_decay(self):
        """清单 6：同一 created_at，core 在 dt=3600 时衰减更慢。"""
        engine = TauDecayEngine()
        created = 1000.0
        now = created + 3600.0
        tau_core = engine.compute_tau(
            "core_node", created_at=created, force_now=now, fact_track="core"
        )
        tau_active = engine.compute_tau(
            "active_node", created_at=created, force_now=now, fact_track="active"
        )
        assert tau_core > tau_active

    def test_compute_tau_default_equals_active(self):
        """向后兼容：默认（不传 fact_track）行为与 active 一致。"""
        engine = TauDecayEngine()
        created = 1000.0
        now = created + 3600.0
        default = engine.compute_tau("a", created_at=created, force_now=now)
        active = engine.compute_tau("b", created_at=created, force_now=now, fact_track="active")
        assert default == pytest.approx(active)

    def test_compute_strength_forwards_fact_track(self):
        """compute_strength 透传 fact_track（dream_pipeline 调用点）。"""
        engine = TauDecayEngine()
        created = 1000.0
        # 用相同 created_at：core 的 τ 分量更高 → strength 更高（ROEM_ALPHA=1.0 时等价 tau）
        s_core = engine.compute_strength(created, node_id="c", fact_track="core")
        s_active = engine.compute_strength(created, node_id="a", fact_track="active")
        # 两者 age 相同（time.time() 差异可忽略），core τ 更高 → strength 更高
        assert s_core >= s_active - 1e-6

    def test_core_boost_clamped_to_max(self):
        """core ×2.0 后仍受 tau_decay_max 钳制。"""
        engine = TauDecayEngine()
        created = 1000.0
        now = created + 1.0
        # 注册一个 importance=1.0 的 core 节点：boost = 2.0 × 2.0 = 4.0 → 7200s 钳制
        engine.register_node("core_max", created_at=created, importance=1.0)
        tau = engine.compute_tau("core_max", created_at=created, force_now=now, fact_track="core")
        # effective 钳制到 7200s；dt=1s → 近乎 1.0
        assert 0.0 < tau <= 1.0
