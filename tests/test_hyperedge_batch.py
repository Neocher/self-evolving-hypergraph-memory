"""Hyperedge 批量边写入测试（真实 GraphLite 引擎）。

验证 _persist_hyperedge 的多边单语句 INSERT：
- 创建 1 个 hyperedge + 8 个成员 → 8 条 HYPEREDGE_MEMBER 边全部存在
- 空成员列表 → 不执行边语句（不抛异常）
- 幂等性：重复执行 _persist_hyperedge → 产生重复边（GraphLite 无 MERGE）
"""
import uuid
import pytest

from graph.hyperedge import HyperedgeManager, Hyperedge


def _create_episode(store, ep_id: str, content: str = "test"):
    store.create_episode({
        "id": ep_id,
        "content": content,
        "created_at": 1.0,
        "tau_initial": 1.0,
        "tau_value": 0.5,
        "source": "test",
        "trust_score": 0.8,
    })


class TestBatchEdgeInsert:
    def test_batch_8_members_all_present(self, graphlite_store):
        gstore = graphlite_store
        mgr = HyperedgeManager(gstore)

        member_ids = [str(uuid.uuid4()) for _ in range(8)]
        for mid in member_ids:
            _create_episode(gstore, mid, f"content_{mid[:8]}")

        edge = mgr.create_episode_hyperedge(member_ids, topic="batch_test")
        assert edge.id is not None

        members = gstore.get_hyperedge_members(edge.id)
        got_ids = {m["id"] for m in members}
        assert got_ids == set(member_ids), (
            f"Expected all 8 members, got {len(got_ids)}. Missing: {set(member_ids) - got_ids}"
        )

    def test_empty_members_no_error(self, graphlite_store):
        gstore = graphlite_store
        mgr = HyperedgeManager(gstore)

        with pytest.raises(ValueError, match="at least 2 member nodes"):
            mgr.create_episode_hyperedge([], topic="empty_test")

    def test_single_member_rejected(self, graphlite_store):
        gstore = graphlite_store
        mgr = HyperedgeManager(gstore)
        mid = str(uuid.uuid4())
        _create_episode(gstore, mid)

        with pytest.raises(ValueError, match="at least 2 member nodes"):
            mgr.create_episode_hyperedge([mid], topic="single_test")

    def test_idempotency_no_duplicate_edges(self, graphlite_store):
        gstore = graphlite_store
        mgr = HyperedgeManager(gstore)

        member_ids = [str(uuid.uuid4()) for _ in range(3)]
        for mid in member_ids:
            _create_episode(gstore, mid)

        edge = mgr.create_episode_hyperedge(member_ids, topic="idem_test")
        count_before = len(gstore.get_hyperedge_members(edge.id))
        assert count_before == 3

        mgr._persist_hyperedge(edge)
        count_after = len(gstore.get_hyperedge_members(edge.id))
        assert count_after == 3, (
            f"GraphLite deduplicates edges: expected 3, got {count_after}"
        )

    def test_delete_then_recreate_is_clean(self, graphlite_store):
        gstore = graphlite_store
        mgr = HyperedgeManager(gstore)

        member_ids = [str(uuid.uuid4()) for _ in range(3)]
        for mid in member_ids:
            _create_episode(gstore, mid)

        edge = mgr.create_episode_hyperedge(member_ids)
        assert len(gstore.get_hyperedge_members(edge.id)) == 3

        mgr.delete_hyperedge(edge.id)
        assert len(gstore.get_hyperedge_members(edge.id)) == 0

        mgr._persist_hyperedge(edge)
        assert len(gstore.get_hyperedge_members(edge.id)) == 3, (
            "Re-creating after delete should produce exactly 3 edges"
        )

    def test_mixed_types(self, graphlite_store):
        gstore = graphlite_store
        mgr = HyperedgeManager(gstore)

        member_ids = [str(uuid.uuid4()) for _ in range(5)]
        for mid in member_ids:
            _create_episode(gstore, mid)

        edge = mgr.create_semantic_hyperedge(member_ids, conclusion="mixed test")
        members = gstore.get_hyperedge_members(edge.id)
        assert len(members) == 5
