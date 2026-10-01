"""P3 AtomicFact bi-temporal supersession 测试。

覆盖任务书判据 1-8 + 2 条边界补强（_fact_supersede_enabled 解析变体、
get_active_facts_by_sp 空入参守卫）。全部走 store 公共入口（create_atomic_fact /
get_active_facts_by_sp / get_atomic_facts_by_subject），复用 conftest 的
overgraph_store fixture（真实临时 OverGraph 库，connect/close 托管）。

判据对照：
  1. 写 1080→2080 → active 只回 2080（1 条）；include_superseded 回 2 条且旧值
     superseded_at 非空、superseded_by=新 fid
  2. 幂等：同 object 连写两次 → active 仍 1 条、无 superseded_at
  3. 同 S 不同 P 隔离：同 P 不同 object 会 supersede；不同 P 不影响
  4. SHM_FACT_SUPERSEDE=0 关 → 新旧两条均 active、无 superseded_at
  5. 改旧 fact 后字段完整（全量覆盖坑没踩）
  6. get_atomic_facts_by_subject 默认只回 2080（active only）
  7. valid_time 空串场景 supersede 仍生效（不依赖 valid_time）
  8. 新 fact 的 supersedes 指向旧 fid（血缘可回溯）
"""

from __future__ import annotations

import pytest

from graph.overgraph_store import _fact_supersede_enabled

# 依赖缺失 → 整模块 skip（不崩收集；与 test_overgraph_store 同策略）
pytest.importorskip("overgraph")


# ─── 判据 1：写侧作废 ───────────────────────────────────

def test_supersede_active_only_new_value(overgraph_store):
    store = overgraph_store
    fid1 = store.create_atomic_fact("代理端口", "是", "1080")
    fid2 = store.create_atomic_fact("代理端口", "是", "2080")

    active = store.get_active_facts_by_sp("代理端口", "是")
    assert len(active) == 1
    assert active[0]["object"] == "2080"
    assert active[0]["id"] == fid2

    all_facts = store.get_active_facts_by_sp("代理端口", "是", include_superseded=True)
    assert len(all_facts) == 2
    old = [f for f in all_facts if f["object"] == "1080"]
    assert len(old) == 1
    assert old[0]["id"] == fid1
    assert old[0]["superseded_at"] is not None
    assert old[0]["superseded_by"] == fid2


# ─── 判据 2：幂等 ───────────────────────────────────────

def test_supersede_idempotent_same_object(overgraph_store):
    store = overgraph_store
    store.create_atomic_fact("代理端口", "是", "1080")
    store.create_atomic_fact("代理端口", "是", "1080")

    active = store.get_active_facts_by_sp("代理端口", "是")
    assert len(active) == 1
    assert active[0]["object"] == "1080"
    assert active[0].get("superseded_at") is None

    all_facts = store.get_active_facts_by_sp("代理端口", "是", include_superseded=True)
    assert len(all_facts) == 1
    assert all(f.get("superseded_at") is None for f in all_facts)


# ─── 判据 3：同 S 不同 P 隔离 ────────────────────────────

def test_supersede_same_subject_diff_predicate_isolated(overgraph_store):
    store = overgraph_store
    store.create_atomic_fact("服务版本", "是", "1.0")
    store.create_atomic_fact("服务版本", "是", "2.0")
    store.create_atomic_fact("服务版本", "部署于", "北京")

    # 同 P（是）1.0 → 2.0 被作废，active 只回 2.0
    active_shi = store.get_active_facts_by_sp("服务版本", "是")
    assert len(active_shi) == 1
    assert active_shi[0]["object"] == "2.0"

    # 不同 P（部署于）不受影响，且不影响（服务版本/是/*）
    active_deploy = store.get_active_facts_by_sp("服务版本", "部署于")
    assert len(active_deploy) == 1
    assert active_deploy[0]["object"] == "北京"


# ─── 判据 4：开关关闭 ───────────────────────────────────

def test_supersede_disabled_by_env(overgraph_store, monkeypatch):
    store = overgraph_store
    monkeypatch.setenv("SHM_FACT_SUPERSEDE", "0")
    store.create_atomic_fact("代理端口", "是", "1080")
    store.create_atomic_fact("代理端口", "是", "2080")

    # 关 → 两条均 active，无 superseded_at
    active = store.get_active_facts_by_sp("代理端口", "是")
    assert len(active) == 2
    assert all(f.get("superseded_at") is None for f in active)
    assert {f["object"] for f in active} == {"1080", "2080"}


# ─── 判据 5：全量覆盖坑不踩 ─────────────────────────────

def test_supersede_preserves_old_fact_fields(overgraph_store):
    store = overgraph_store
    fid1 = store.create_atomic_fact(
        "代理端口", "是", "1080", valid_time="", source_episode="ep1")
    fid2 = store.create_atomic_fact(
        "代理端口", "是", "2080", valid_time="", source_episode="ep2")

    all_facts = store.get_active_facts_by_sp("代理端口", "是", include_superseded=True)
    old = [f for f in all_facts if f["id"] == fid1]
    assert len(old) == 1
    o = old[0]
    assert o["subject"] == "代理端口"
    assert o["predicate"] == "是"
    assert o["object"] == "1080"
    assert o["valid_time"] == ""
    assert o["source_episode"] == "ep1"
    assert o["superseded_at"] is not None
    assert o["superseded_by"] == fid2


# ─── 判据 6：get_atomic_facts_by_subject 默认 active only ──

def test_get_atomic_facts_by_subject_active_only(overgraph_store):
    store = overgraph_store
    store.create_atomic_fact("代理端口", "是", "1080")
    store.create_atomic_fact("代理端口", "是", "2080")

    rows = store.get_atomic_facts_by_subject("代理端口")
    assert len(rows) == 1
    assert rows[0]["object"] == "2080"


# ─── 判据 7：valid_time 空串场景 supersede 仍生效 ─────────

def test_supersede_works_with_empty_valid_time(overgraph_store):
    store = overgraph_store
    store.create_atomic_fact("数据库", "是", "MySQL", valid_time="")
    store.create_atomic_fact("数据库", "是", "PostgreSQL", valid_time="")

    active = store.get_active_facts_by_sp("数据库", "是")
    assert len(active) == 1
    assert active[0]["object"] == "PostgreSQL"
    assert active[0]["valid_time"] == ""
    # 旧值 valid_time 同样为空串，但已因 object 不同被作废（不依赖 valid_time）
    all_facts = store.get_active_facts_by_sp("数据库", "是", include_superseded=True)
    old = [f for f in all_facts if f["object"] == "MySQL"]
    assert len(old) == 1
    assert old[0]["superseded_at"] is not None


# ─── 判据 8：新 fact supersedes 指向旧 fid ───────────────

def test_new_fact_supersedes_points_to_old_fid(overgraph_store):
    store = overgraph_store
    fid1 = store.create_atomic_fact("代理端口", "是", "1080")
    fid2 = store.create_atomic_fact("代理端口", "是", "2080")

    active = store.get_active_facts_by_sp("代理端口", "是")
    assert len(active) == 1
    assert active[0]["id"] == fid2
    assert active[0]["supersedes"] == fid1


# ─── 边界补强：开关解析变体 ──────────────────────────────

def test_fact_supersede_enabled_variants(monkeypatch):
    monkeypatch.delenv("SHM_FACT_SUPERSEDE", raising=False)
    assert _fact_supersede_enabled() is True

    for v in ("0", "false", "False", ""):
        monkeypatch.setenv("SHM_FACT_SUPERSEDE", v)
        assert _fact_supersede_enabled() is False

    monkeypatch.setenv("SHM_FACT_SUPERSEDE", "1")
    assert _fact_supersede_enabled() is True


# ─── 边界补强：空入参守卫 ────────────────────────────────

def test_get_active_facts_by_sp_empty_guard(overgraph_store):
    store = overgraph_store
    store.create_atomic_fact("代理端口", "是", "1080")
    assert store.get_active_facts_by_sp("", "是") == []
    assert store.get_active_facts_by_sp("代理端口", "") == []
    assert store.get_active_facts_by_sp("", "") == []
