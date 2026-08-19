"""
τ 下沉负向钉死测试（阶段3 D1-D4，v6.0.0 overgraph-only）
======================================================
设计定稿 D1-D4：τ 保留 Python TauDecayEngine（检索热路径读静态 tau_initial
无动态衰减可沉 / 引擎无节点级指数衰减原语 / archive≠delete 破坏血统）。

钉死两点（实证，2026-08-19 overgraph 0.17.0）：
  1. neighbors_batch(decay_lambda) 边衰减 ≠ TauDecayEngine 节点 τ 曲线——
     decay_lambda 对返回边权重恒等（引擎无衰减原语），更无节点级 τ；
     τ 下沉不可行 → τ 必须留在 Python 引擎。
  2. 梦境 PRUNE 走 archive_node（archived=true）非物理删除——
     get_episode 返回 dict（archived=true）而非 None（血统/审计可追溯）。
"""
import time

import numpy as np
import pytest

pytestmark = pytest.mark.overgraph

pytest.importorskip("overgraph")


@pytest.mark.parametrize("overgraph_store", [{"dimension": 32}], indirect=True)
def test_neighbors_batch_decay_not_node_tau_curve(overgraph_store):
    """decay_lambda 不改边权重 → 引擎无节点 τ 曲线（τ 下沉不可行，负向钉死）。"""
    store = overgraph_store
    db = store.conn
    A = store.create_episode({"content": "a", "created_at": time.time()})
    B = store.create_episode({"content": "b", "created_at": time.time()})
    C = store.create_episode({"content": "c", "created_at": time.time()})
    ia = store.get_node_internal_id(A)
    ib = store.get_node_internal_id(B)
    ic = store.get_node_internal_id(C)
    db.upsert_edge(ia, ib, "HEBBIAN_CONNECTION", weight=0.8)
    db.upsert_edge(ia, ic, "HEBBIAN_CONNECTION", weight=0.4)

    def edge_weights(decay_lambda):
        kw = {} if decay_lambda is None else {"decay_lambda": decay_lambda}
        nb = db.neighbors_batch(
            [ia], direction="outgoing",
            edge_label_filter=["HEBBIAN_CONNECTION"], **kw,
        )
        return sorted((e.node_id, round(float(e.weight), 4)) for e in nb[ia])

    w0 = edge_weights(None)
    assert w0 == [(ib, 0.8), (ic, 0.4)]
    # 【实证】decay_lambda 对返回边权重恒等（引擎无衰减原语）→ 边权重不随
    # lambda 变化，更不存在节点级 τ 指数衰减曲线可沉
    for lam in (0.01, 0.1, 0.5):
        assert edge_weights(lam) == w0, f"decay_lambda={lam} 应恒等"

    # TauDecayEngine 节点 τ 曲线独立存在（Python 侧）：τ(t)=τ₀·exp(-t/τ_decay)，
    # 与引擎边衰减无关 → τ 下沉到引擎不可行（D1 实证）
    from core.tau_decay import TauDecayEngine, TauDecayConfig
    engine = TauDecayEngine(config=TauDecayConfig(
        tau_initial=1.0, tau_decay_seconds=1800, decay_threshold=0.1,
    ))
    created = time.time() - 3600  # 1 小时前创建
    tau_a = engine.compute_tau(A, created_at=created, force_now=time.time())
    expected = 1.0 * np.exp(-3600.0 / 1800.0)
    assert abs(tau_a - expected) < 1e-9
    # 边权重（0.8/0.4）与 τ 曲线（≈0.135）量纲/语义均不同：边衰减不构成节点 τ
    assert all(w != tau_a for _, w in w0)


@pytest.mark.parametrize("overgraph_store", [{"dimension": 32}], indirect=True)
def test_dream_prune_archives_not_deletes(overgraph_store):
    """梦境 PRUNE 走 archive_node → get_episode 返回 archived=true（非 None）。

    钉死 archive≠delete（D3）：τ 衰减节点归档而非物理删除，血统/审计可追溯；
    若实现误用 DETACH DELETE，get_episode 返回 None → 本测试失败。
    """
    store = overgraph_store
    eid = store.create_episode({"content": "prunable", "created_at": time.time()})
    assert store.get_episode(eid) is not None

    archived = store.archive_node(eid)
    assert archived is True

    ep = store.get_episode(eid)
    assert ep is not None, "PRUNE 后节点应存在（归档非删除）"
    assert ep.get("archived") in (True, "true", 1), ep
