"""
P1 显式冲突撤销测试（TEPA 对标）
============================
覆盖：_decide_winner 裁决逻辑（来源分级/信任 tie-break/τ 衰减/NULL 保守）、
     写路径自动归档 + SUPERSEDES 血统边时序、弱匹配不归档、restore 可翻转软删。

走生产链路（真实 GraphLiteStore + HTTP 路由 + 真实 OntologyValidator），
qsubmit 的 write_queue 为 None 时同步直调（与测试环境一致）。

P1-1/P1-2 后集成测试一律用生产字段形态（create_episode 写路径字段：
tau_initial/source_type 等，不手工注入 trust_score/tau_value）。

运行: python -m pytest tests/test_conflict_revocation.py -v
"""
from __future__ import annotations

import uuid
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router, Services, get_services
from api.routes.write import _decide_winner
from core.ontology_validator import OntologyValidator, OntologyConfig


# ─── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    def _build(svc):
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app)

    return _build


def _svc(overgraph_store, validator=None) -> Services:
    svc = Services()
    svc.graph_store = overgraph_store
    svc.ontology_validator = validator
    return svc


def _create_old_episode(store, *, content: str, source_type: str = "tool",
                        ontology_type: str = "person_birth",
                        entity_name: str = "张三", entity_value: str = "1990",
                        protected: bool = False,
                        fact_track: Optional[str] = None) -> str:
    """用 GraphLiteStore.create_episode 创建旧节点（生产字段形态）。

    P1-1/P1-2: 生产写路径不落库 trust_score/tau_value（实测 'Null'），裁决只读
    tau_initial + source_type——测试不得手工注入 trust_score（防假绿）。
    P1-R2: protected/fact_track 由调用方显式传（create_episode 默认 active，
    与生产分类路径一致；protected 仅 force_promote 打标记）。
    """
    ep_id = str(uuid.uuid4())
    data = {
        "id": ep_id,
        "content": content,
        "source": "test",
        "source_type": source_type,
        "created_at": 1.0,
        "tau_initial": 1.0,
        "ontology_type": ontology_type,
        "entity_name": entity_name,
        "entity_value": entity_value,
    }
    if protected:
        data["protected"] = True
    if fact_track is not None:
        data["fact_track"] = fact_track
    store.create_episode(data)
    return ep_id


def _count_supersedes(store, old_id: str, new_id: str) -> int:
    rows = store.execute_cypher(
        "MATCH (a:EpisodeNode {id: $old_id})-[s:SUPERSEDES]->"
        "(b:EpisodeNode {id: $new_id}) RETURN s",
        {"old_id": old_id, "new_id": new_id},
    )
    return len(rows)


# ─── _decide_winner 裁决单测（纯逻辑，不依赖图存储）─────────────


class TestDecideWinner:

    def test_new_wins_higher_source_level(self):
        """新值级 > 旧值级 → 新胜（direct 1.0 > tool 0.7，信任不参与）。"""
        assert _decide_winner("direct", "tool", 1.0, 0.9, 1.0) is True

    def test_old_wins_higher_old_source_level(self):
        """新值级 < 旧值级 → 旧胜（tool 0.7 < direct 1.0，不归档）。"""
        assert _decide_winner("tool", "direct", 0.7, 0.3, 1.0) is False

    def test_same_level_old_protected_not_archived(self):
        """P1-R2 防护：旧值 protected=True → 绝不自动归档（即使同级 recency 新胜语义）。

        P2-1 重写：原 test_same_level_old_higher_trust_old_wins 直传生产不可达的
        old_trust=0.9（生产旧信任 = source_type 权重代理，同级恒相等），改为生产
        字段形态（0.7 == 0.7）+ 防护语义断言。"""
        assert _decide_winner("tool", "tool", 0.7, 0.7, 1.0, old_protected=True) is False

    def test_same_level_old_core_stable_not_archived(self):
        """P1-R2 防护：旧值 fact_track=core（稳定事实）且新源未严格更高 → 不自动归档。"""
        assert _decide_winner("tool", "tool", 0.7, 0.7, 1.0, old_fact_track="core") is False

    def test_core_old_strictly_higher_new_source_archives(self):
        """P1-R2 防护边界：core 稳定事实遇严格更高源（direct 1.0 > tool 0.7）→ 允许归档。"""
        assert _decide_winner("direct", "tool", 1.0, 0.7, 1.0, old_fact_track="core") is True

    def test_protected_old_wins_even_against_higher_source(self):
        """P1-R2 防护优先：protected=True 即使新源级更高也绝不自动归档（只待人工）。"""
        assert _decide_winner("direct", "tool", 1.0, 0.7, 1.0,
                              old_protected=True, old_fact_track="core") is False

    def test_same_level_new_higher_trust_new_wins(self):
        """同级 tie-break：新值信任更高 → 新胜。"""
        assert _decide_winner("tool", "tool", 0.7, 0.4, 1.0) is True

    def test_same_level_tie_recency_new_wins(self):
        """同级且信任相近（差 < ε）→ 新胜（TEPA recency 兜底）。"""
        assert _decide_winner("tool", "tool", 0.7, 0.71, 1.0) is True

    def test_old_missing_source_type_proxy_trust_recency_new_wins(self):
        """P2-1 生产形态：旧值缺 source_type（None）按 inferred 最低处理，旧信任代理
        同为 0.5（无 trust_score 落库，不再直传不可达的 0.9）→ 同级相等 → recency 新胜。"""
        assert _decide_winner("inferred", None, 0.5, 0.5, 1.0) is True

    def test_old_missing_source_type_low_trust_new_wins(self):
        """旧值缺 source_type 且信任低 → 新胜（保守处理不瘫痪撤销）。"""
        assert _decide_winner("inferred", None, 0.5, 0.3, 1.0) is True

    def test_old_decayed_tau_attenuates_trust(self):
        """同级且 τ 大幅衰减折损有效信任 → 新胜（时间久者主张弱）。"""
        assert _decide_winner("tool", "tool", 0.7, 0.9, 0.2) is True

    def test_old_tau_missing_no_attenuation(self):
        """旧值缺 τ（None）按 1.0 不折损信任（防旧数据缺字段被误杀）。"""
        assert _decide_winner("tool", "tool", 0.7, 0.9, None) is False

    def test_old_tau_zero_real_decay_new_wins(self):
        """P2-1: τ=0.0 是真实完全衰减（非缺失），有效信任归零 → 新胜（is None 区分）。"""
        assert _decide_winner("tool", "tool", 0.7, 0.9, 0.0) is True

    def test_old_trust_zero_real_zero_new_wins(self):
        """P2-1: trust=0.0 是真实零信任（非缺失），按 0.0 参与比较 → 新胜。"""
        assert _decide_winner("tool", "tool", 0.7, 0.0, 1.0) is True

    def test_same_level_equal_trust_proxy_recency_new_wins(self):
        """P2-2/P1-1: 同级新旧信任代理相等（生产 source_type 权重形态）→ recency 新胜。"""
        assert _decide_winner("tool", "tool", 0.7, 0.7, 1.0) is True


# ─── 写路径自动归档（真实 GraphLite + HTTP 全链路）──────────────


class TestWritePathRevocation:

    def test_new_wins_archives_old_with_supersedes(self, overgraph_store, client):
        """新值来源级更高 → 自动归档旧记忆，SUPERSEDES 边存在（新节点落库后归档）。"""
        old_id = _create_old_episode(
            overgraph_store, content="张三出生于1990年", source_type="tool",
            entity_name="张三", entity_value="1990")
        validator = OntologyValidator(graphlite_store=overgraph_store,
                                      config=OntologyConfig(enabled=True))

        resp = client(_svc(overgraph_store, validator)).post("/memories/episodes", json={
            "content": "张三出生于2000年", "source": "user",
        })

        assert resp.status_code == 200, resp.text
        new_id = resp.json()["episode_id"]
        # 旧记忆被归档、新记忆保持 active
        assert overgraph_store.get_episode(old_id)["archived"] is True
        assert overgraph_store.get_episode(new_id)["archived"] is False
        # SUPERSEDES 血统边：证明归档发生在新节点落库之后
        assert _count_supersedes(overgraph_store, old_id, new_id) == 1

    def test_old_wins_no_archive(self, overgraph_store, client):
        """agent 声明 direct 被防洗白降级 inferred → 新值级更低 → 旧胜不归档。"""
        old_id = _create_old_episode(
            overgraph_store, content="张三出生于1990年", source_type="direct",
            entity_name="张三", entity_value="1990")
        validator = OntologyValidator(graphlite_store=overgraph_store,
                                      config=OntologyConfig(enabled=True))

        resp = client(_svc(overgraph_store, validator)).post("/memories/episodes", json={
            "content": "张三出生于2000年", "source": "hermes", "source_type": "direct",
        })

        assert resp.status_code == 200, resp.text
        assert overgraph_store.get_episode(old_id)["archived"] is False

    def test_weak_match_contradictory_claim_no_archive(self, overgraph_store, client):
        """CONTAINS 弱匹配（contradictory_claim）即使新值来源更高也不自动归档。"""
        old_id = _create_old_episode(
            overgraph_store, content="2024年实验证明吸烟导致肺癌", source_type="tool",
            ontology_type="scientific_claim",
            entity_name="吸烟", entity_value="2024")
        validator = OntologyValidator(graphlite_store=overgraph_store,
                                      config=OntologyConfig(enabled=True))

        resp = client(_svc(overgraph_store, validator)).post("/memories/episodes", json={
            "content": "2025年证明吸烟导致肺癌死亡率上升", "source": "user",
        })

        assert resp.status_code == 200, resp.text
        assert overgraph_store.get_episode(old_id)["archived"] is False

    def test_same_level_recency_new_wins_integration(self, overgraph_store, client):
        """P1-1 同级 tie-break 走全链路（生产字段形态）：同级 tool/tool 新旧信任
        代理相等（0.7 == 0.7），τ 未衰减（tau_initial=1.0）→ |diff| < ε → recency 新胜归档。
        旧版退化为常量 0.5 的 trust_score 注入已从 helper 移除（防假绿）。"""
        old_id = _create_old_episode(
            overgraph_store, content="张三出生于1990年", source_type="tool",
            entity_name="张三", entity_value="1990")
        validator = OntologyValidator(graphlite_store=overgraph_store,
                                      config=OntologyConfig(enabled=True))

        resp = client(_svc(overgraph_store, validator)).post("/memories/episodes", json={
            "content": "张三出生于2000年", "source": "hermes", "source_type": "tool",
        })

        assert resp.status_code == 200, resp.text
        # 同级 tool(0.7) == 旧代理 0.7，τ=1.0 → 有效信任相等 → recency → 新胜归档
        assert overgraph_store.get_episode(old_id)["archived"] is True

    def test_protected_old_not_archived_same_level(self, overgraph_store, client):
        """P1-R2 全链路：旧值 protected=True + 同级 tool/tool（recency 本应新胜）
        → 不自动归档（ConflictNode 仍建，待人工裁决）。"""
        old_id = _create_old_episode(
            overgraph_store, content="张三出生于1990年", source_type="tool",
            entity_name="张三", entity_value="1990", protected=True)
        validator = OntologyValidator(graphlite_store=overgraph_store,
                                      config=OntologyConfig(enabled=True))

        resp = client(_svc(overgraph_store, validator)).post("/memories/episodes", json={
            "content": "张三出生于2000年", "source": "hermes", "source_type": "tool",
        })

        assert resp.status_code == 200, resp.text
        assert overgraph_store.get_episode(old_id)["archived"] is False

    def test_core_old_not_archived_same_level(self, overgraph_store, client):
        """P1-R2 全链路：旧值 fact_track=core（稳定事实）+ 同级 tool/tool
        （新源未严格更高）→ 不自动归档。修复前同级恒新胜 → 稳定事实被静默归档。"""
        old_id = _create_old_episode(
            overgraph_store, content="张三出生于1990年", source_type="tool",
            entity_name="张三", entity_value="1990", fact_track="core")
        validator = OntologyValidator(graphlite_store=overgraph_store,
                                      config=OntologyConfig(enabled=True))

        resp = client(_svc(overgraph_store, validator)).post("/memories/episodes", json={
            "content": "张三出生于2000年", "source": "hermes", "source_type": "tool",
        })

        assert resp.status_code == 200, resp.text
        assert overgraph_store.get_episode(old_id)["archived"] is False

    def test_core_old_archived_by_strictly_higher_source(self, overgraph_store, client):
        """P1-R2 全链路：旧值 fact_track=core 但新源严格更高（user→direct 1.0 >
        tool 0.7）→ 允许归档（稳定事实可被更高信任源取代）。"""
        old_id = _create_old_episode(
            overgraph_store, content="张三出生于1990年", source_type="tool",
            entity_name="张三", entity_value="1990", fact_track="core")
        validator = OntologyValidator(graphlite_store=overgraph_store,
                                      config=OntologyConfig(enabled=True))

        resp = client(_svc(overgraph_store, validator)).post("/memories/episodes", json={
            "content": "张三出生于2000年", "source": "user",
        })

        assert resp.status_code == 200, resp.text
        new_id = resp.json()["episode_id"]
        assert overgraph_store.get_episode(old_id)["archived"] is True
        assert _count_supersedes(overgraph_store, old_id, new_id) == 1

    def test_conflict_node_still_created(self, overgraph_store, client):
        """即使自动归档，ConflictNode 审计节点仍落库（冲突可追溯）。"""
        old_id = _create_old_episode(
            overgraph_store, content="张三出生于1990年", source_type="tool",
            entity_name="张三", entity_value="1990")
        validator = OntologyValidator(graphlite_store=overgraph_store,
                                      config=OntologyConfig(enabled=True))

        resp = client(_svc(overgraph_store, validator)).post("/memories/episodes", json={
            "content": "张三出生于2000年", "source": "user",
        })

        assert resp.status_code == 200, resp.text
        new_id = resp.json()["episode_id"]
        rows = overgraph_store.execute_cypher(
            "MATCH (c:ConflictNode {id: $cid}) RETURN c",
            {"cid": f"conflict_{new_id}_{old_id}"})
        assert len(rows) == 1


# ─── restore 端点 ────────────────────────────────────────────


class TestRestoreEndpoint:

    def test_restore_flips_archived_keeps_supersedes(self, overgraph_store, client):
        """归档 → restore → archived=false，SUPERSEDES 血统边保留（可翻转软删）。"""
        ep_id = overgraph_store.create_episode({
            "id": str(uuid.uuid4()),
            "content": "待恢复的记忆",
            "source": "user",
            "source_type": "direct",
            "created_at": 1.0,
            "tau_initial": 1.0,
        })
        replacement_id = overgraph_store.create_episode({
            "id": str(uuid.uuid4()),
            "content": "取代它的新记忆",
            "source": "user",
            "source_type": "direct",
            "created_at": 1.0,
            "tau_initial": 1.0,
        })
        assert overgraph_store.archive_node(ep_id, replacement_id=replacement_id) is True
        assert overgraph_store.get_episode(ep_id)["archived"] is True

        resp = client(_svc(overgraph_store)).post(f"/episodes/{ep_id}/restore")

        assert resp.status_code == 200, resp.text
        assert resp.json()["archived"] is False
        assert overgraph_store.get_episode(ep_id)["archived"] is False
        assert _count_supersedes(overgraph_store, ep_id, replacement_id) == 1

    def test_restore_missing_episode_404(self, overgraph_store, client):
        resp = client(_svc(overgraph_store)).post("/episodes/nonexistent_id/restore")
        assert resp.status_code == 404
