"""
策略反馈环测试（阶段4-2，v6.0.0）
====================================
覆盖设计 AC：计数阈值（1 不升 2 升）/ 幂等 / 升级落库（fact_track='core'）/
生产 retrieve() 零改动（core ×1.1 既有机制验证）/ 负反馈不计 / 边权重 τ 不碰。
"""
import time

import numpy as np
import pytest

from core.feedback import FeedbackEngine


def test_threshold_one_no_upgrade_two_upgrade(overgraph_store):
    """计数阈值：1 次正确 → 不升；2 次正确 → 升级 fact_track='core'。"""
    store = overgraph_store
    eid = store.create_episode({"content": "事实 A", "created_at": time.time()})
    engine = FeedbackEngine(store)

    upgraded = engine.apply([("q1", [eid], True)])
    assert upgraded == []
    assert store.get_episode(eid)["fact_track"] == "active"  # 1 不升

    upgraded = engine.apply([("q2", [eid], True)])
    assert upgraded == [eid]
    assert store.get_episode(eid)["fact_track"] == "core"  # 2 升

    # core 轨检索 ×1.1 既有机制（v5.35.0）：升级后检索权重生效
    from retrieval.query_router import QueryRouter, QueryRouterConfig
    qr = QueryRouter(store, None, None, config=QueryRouterConfig())
    seeded = [{"node_id": eid, "content": "事实 A", "score": 0.8, "fact_track": "core"}]
    out = QueryRouter._deduplicate_and_sort(seeded)
    assert out[0]["score"] == pytest.approx(0.88)  # 0.8 × 1.1


def test_apply_idempotent_and_upgraded_skipped(overgraph_store):
    """幂等：已升级节点不再重复升级/计数（upgraded 集合守卫）。"""
    store = overgraph_store
    eid = store.create_episode({"content": "事实 B", "created_at": time.time()})
    engine = FeedbackEngine(store)

    engine.apply([("q1", [eid], True)])
    engine.apply([("q2", [eid], True)])
    assert eid in engine.upgraded
    counts_after_upgrade = engine.counts[eid]

    # 升级后再 apply（计数继续累加）→ 不重复出现在 upgraded 返回、不重复升级
    more = engine.apply([("q3", [eid], True)])
    assert more == []
    assert store.get_episode(eid)["fact_track"] == "core"
    assert engine.counts[eid] == counts_after_upgrade  # 已升级节点停止计数


def test_negative_feedback_not_counted(overgraph_store):
    """correct=False → 不计成功计数，永不触发升级。"""
    store = overgraph_store
    eid = store.create_episode({"content": "事实 C", "created_at": time.time()})
    engine = FeedbackEngine(store)

    engine.apply([("q1", [eid], False), ("q2", [eid], False)])
    assert engine.counts.get(eid, 0) == 0
    assert store.get_episode(eid)["fact_track"] == "active"


def test_multi_node_rewards_and_threshold_per_node(overgraph_store):
    """多节点奖励：升级判定按节点独立计数（一个达阈值不连带其他）。"""
    store = overgraph_store
    e1 = store.create_episode({"content": "事实 D", "created_at": time.time()})
    e2 = store.create_episode({"content": "事实 E", "created_at": time.time()})
    engine = FeedbackEngine(store)

    engine.apply([("q1", [e1, e2], True), ("q2", [e1], True)])
    assert engine.upgraded == {e1}
    assert store.get_episode(e1)["fact_track"] == "core"
    assert store.get_episode(e2)["fact_track"] == "active"


def test_upgrade_does_not_touch_edge_or_tau(overgraph_store):
    """铁律：升级只改 fact_track，不碰边权重/τ（防答案泄漏）。"""
    store = overgraph_store
    a = store.create_episode({"content": "节点 A", "created_at": time.time()})
    b = store.create_episode({"content": "节点 B", "created_at": time.time()})
    store.link_hyperedge_member(
        store.create_hyperedge_node({"id": f"h_{a}", "name": "ha"}), a
    )
    store.link_hyperedge_member(
        store.create_hyperedge_node({"id": f"h_{b}", "name": "hb"}), b
    )
    engine = FeedbackEngine(store)
    before = store.get_episode(a)
    engine.apply([("q1", [a], True), ("q2", [a], True)])
    assert store.get_episode(a)["fact_track"] == "core"
    after = store.get_episode(a)
    # τ 不变（升级只改 fact_track，不碰 τ/边权重）
    assert after.get("tau_initial") == before.get("tau_initial")
    assert after.get("archived") == before.get("archived")
