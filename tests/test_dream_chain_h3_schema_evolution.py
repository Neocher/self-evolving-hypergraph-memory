"""[H3/B-P0-3] Schema 自演化主链恒 no-op。

基线缺陷（B-core 审计 P0-3，实跑验证）：
    生产 CLUSTER 从不给 community 写 "content"/"summary" 键（只写
    episodes/facts/report/entity_links/...），而 `_persist_schema_evolution`
    只读 `comm.get("content")`、`_build_schema_versioning_input` 只读
    content/summary → 两条自演化链（P0-② 属性/关系演化、P2 schema 版本化）
    在 100% 的生产 community 上静默空转，且无任何日志。

本测试证明修复后：
    H3-1 给定生产形态 community（仅 episodes 原文，无 content 键），
         自演化阶段真正执行：get_entity 被调、属性 sidecar 出现
         company 候选（EMERGE 计票落 sidecar）；
    H3-2 无内容 community（旧行为 100% 命中场景）打 warning 计数，不静默；
    H3-3 _build_schema_versioning_input 对生产形态 community 可构造
         （修复前恒 None）。
"""
from __future__ import annotations

import logging

from core.dream_pipeline import DreamPipeline
from core.tau_decay import TauDecayEngine, TauDecayConfig


class _StubEntityStore:
    """最小实体存储桩：P0-② 链需要的 5 个接口。"""

    def __init__(self, entities: list[dict]):
        self._entities = {e["name"].lower(): e for e in entities}
        self.props: dict[str, dict] = {e["id"]: {} for e in entities}
        self.get_entity_calls: list[str] = []

    def get_entity(self, name: str):
        self.get_entity_calls.append(name)
        return self._entities.get(str(name).lower())

    def locked_update_entity_props(self, eid: str, fn):
        props = self.props.setdefault(eid, {})
        return fn(props)

    def _decode_sidecar(self, props: dict, key: str) -> dict:
        v = props.get(key)
        return v if isinstance(v, dict) else {}

    def create_property_version(self, eid, attr, value, supersedes_id=None):
        return "pvn_test"

    def create_rel_edge(self, src, dst, pred, confidence=0.0):
        pass


def _pipe() -> DreamPipeline:
    return DreamPipeline(tau_engine=TauDecayEngine(TauDecayConfig()))


def _entity():
    return [{"id": "ent_alice", "name": "Alice", "entity_type": "Person"}]


# ─── H3-1 ─────────────────────────────────────────────────────


def test_schema_evolution_executes_with_episodes_content(tmp_path):
    """H3-1（关键回归）：生产形态 community（episodes 有原文、无 content 键）
    → 属性演化链真正执行，sidecar 出现 company 候选（EMERGE）。"""
    store = _StubEntityStore(_entity())
    pipe = _pipe()
    comm = {
        "id": "comm1",
        "episodes": [{"id": "ep1", "content": "Alice works at Acme Corp."}],
        "entity_links": [{"entity": "Alice", "occurrences": ["ep1"], "count": 1}],
    }

    pipe._persist_schema_evolution(store, [comm])

    assert store.get_entity_calls, "get_entity 从未被调 → 演化链未执行"
    sidecar = store.props["ent_alice"].get("attrs_json") or {}
    assert "company" in sidecar, f"属性 sidecar 未写入: {sidecar!r}"
    cands = sidecar["company"].get("candidates") or {}
    assert any(c.get("value") == "Acme Corp" for c in cands.values()), \
        f"company 候选缺失: {cands!r}"


# ─── H3-2 ─────────────────────────────────────────────────────


def test_schema_evolution_no_content_logged_not_silent(caplog):
    """H3-2：无内容 community（修复前 100% 场景）→ warning 计数可见。"""
    store = _StubEntityStore(_entity())
    pipe = _pipe()
    comm = {
        "id": "comm2",
        "episodes": [],  # 无原文、无 content/report → 无法抽取
        "entity_links": [{"entity": "Alice", "occurrences": [], "count": 1}],
    }

    with caplog.at_level(logging.WARNING, logger="core.dream_pipeline"):
        pipe._persist_schema_evolution(store, [comm])

    assert any(
        "schema-evolution" in r.message and "no_content=1" in r.message
        for r in caplog.records
    ), [r.message for r in caplog.records]


# ─── H3-3 ─────────────────────────────────────────────────────


def test_schema_versioning_input_built_from_episodes():
    """H3-3：P2 输入构造对生产形态 community 不再恒 None。"""
    pipe = _pipe()
    comm = {
        "id": "comm3",
        "episodes": [{"id": "ep9", "content": "some schema text content"}],
    }

    built = pipe._build_schema_versioning_input(comm)

    assert built is not None, "仅 episodes 原文的 community 必须可构造（修复前恒 None）"
    schema, episode, signals, schema_id = built
    assert episode.raw_text == "some schema text content"
    assert schema_id == "comm3"
