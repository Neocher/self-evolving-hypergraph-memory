"""
梦境候选孤儿社区修复测试
========================
验证 _persist_community_nodes 不再全删 CommunityNode，
正确创建 COMMUNITY_MEMBER 边，兼容旧格式候选。
"""
from __future__ import annotations

import time
import uuid

from core.dream_candidate_store import DreamCandidate, DreamCandidateStore


def _make_candidate(
    dream_id: str = "test-dream-001",
    community_summaries: list[dict] | None = None,
) -> DreamCandidate:
    """构造 DreamCandidate 测试对象。"""
    if community_summaries is None:
        community_summaries = []
    return DreamCandidate(
        dream_id=dream_id,
        created_at=time.time(),
        trigger_mode="test",
        stats={"created": 0, "updated": 0, "deleted": 0},
        community_count=len(community_summaries),
        prune_count=0,
        conflict_count=0,
        community_summaries=community_summaries,
        prune_ops=[],
        merge_ops=[],
    )


def _create_episode(graphlite_store, ep_id: str, content: str = "test content") -> None:
    """在 GraphLite 中创建一个 EpisodeNode。"""
    graphlite_store.create_episode({
        "id": ep_id,
        "content": content,
        "created_at": time.time(),
        "tau_initial": 1.0,
        "tau_value": 0.6,
        "source": "test",
        "trust_score": 0.8,
    })


class TestPersistCommunityNodes:
    """_persist_community_nodes 修复验证。"""

    def test_preserves_external_communities(self, graphlite_store):
        """预置外部 CommunityNode → persist 后仍存在（验证不全删）。"""
        store = DreamCandidateStore()
        external_cid = f"ext-comm-{uuid.uuid4().hex[:8]}"

        # 预置一个外部 CommunityNode（不是 dream 创建的）
        graphlite_store.execute_cypher(
            "INSERT (c:CommunityNode {id: $id, name: $name, "
            "summary: $summary, leiden_score: $score, "
            "created_at: $created_at})",
            {
                "id": external_cid,
                "name": "external_community",
                "summary": "pre-existing community",
                "score": 0.5,
                "created_at": time.time(),
            },
        )

        # 创建一个 dream 候选（不同 community，无成员）
        candidate = _make_candidate(
            community_summaries=[{
                "id": f"dream-comm-{uuid.uuid4().hex[:8]}",
                "member_count": 3,
                "member_ids": [],
                "report": "new community from dream",
                "keywords": [],
                "topics": [],
                "patterns": [],
                "contradictions": [],
            }],
        )

        store._persist_community_nodes(candidate, graphlite_store)

        # 外部社区仍存在（不会被 DETACH DELETE 全删）
        result = graphlite_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $id}) RETURN c",
            {"id": external_cid},
        )
        assert result, f"External community {external_cid} was deleted"

    def test_creates_member_edges(self, graphlite_store):
        """persist 后新社区有 COMMUNITY_MEMBER 边（验证建边 + 两阶段）。"""
        store = DreamCandidateStore()
        comm_id = f"comm-{uuid.uuid4().hex[:8]}"
        ep1_id = f"ep-{uuid.uuid4().hex[:8]}"
        ep2_id = f"ep-{uuid.uuid4().hex[:8]}"

        # 先创建 EpisodeNode（否则 MATCH 无行不执行 INSERT 边）
        _create_episode(graphlite_store, ep1_id, "episode 1")
        _create_episode(graphlite_store, ep2_id, "episode 2")

        candidate = _make_candidate(
            community_summaries=[{
                "id": comm_id,
                "member_count": 2,
                "member_ids": [ep1_id, ep2_id],
                "report": "test community with members",
                "keywords": [],
                "topics": [],
                "patterns": [],
                "contradictions": [],
            }],
        )

        store._persist_community_nodes(candidate, graphlite_store)

        # 验证社区节点存在
        comm_result = graphlite_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $id}) RETURN c",
            {"id": comm_id},
        )
        assert comm_result, f"Community node {comm_id} not found"

        # 验证 COMMUNITY_MEMBER 边存在
        for ep_id in [ep1_id, ep2_id]:
            edge_result = graphlite_store.execute_cypher(
                "MATCH (c:CommunityNode {id: $cid})"
                "-[:COMMUNITY_MEMBER]->"
                "(e:EpisodeNode {id: $eid}) RETURN e",
                {"cid": comm_id, "eid": ep_id},
            )
            assert edge_result, (
                f"COMMUNITY_MEMBER edge missing: {comm_id} -> {ep_id}"
            )

    def test_old_format_candidate_compat(self, graphlite_store):
        """无 member_ids 的旧格式候选不崩、只建节点不建边。"""
        store = DreamCandidateStore()
        comm_id = f"old-comm-{uuid.uuid4().hex[:8]}"

        # 旧格式：没有 member_ids 字段
        candidate = _make_candidate(
            community_summaries=[{
                "id": comm_id,
                "member_count": 5,
                # 故意不包含 "member_ids"
                "report": "old format community without member_ids",
                "keywords": [],
                "topics": [],
                "patterns": [],
                "contradictions": [],
            }],
        )

        # 不应崩溃
        created = store._persist_community_nodes(candidate, graphlite_store)
        assert created == 1

        # 节点已创建
        comm_result = graphlite_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $id}) RETURN c",
            {"id": comm_id},
        )
        assert comm_result, (
            f"Old-format community node {comm_id} not created"
        )

        # 无边（member_ids 缺失 → 跳过边创建，仅 logger.warning）
        edge_result = graphlite_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $cid})-[r:COMMUNITY_MEMBER]->() RETURN r",
            {"cid": comm_id},
        )
        assert edge_result == [], (
            f"Expected no edges for old-format community, got {edge_result}"
        )

    def test_shared_member_keeps_largest_community(self, graphlite_store):
        """共享成员 persist 后只保留最大社区的边，验证不湮灭。

        构造 C3(member_count=2, [epX, epA]) + C4(member_count=3, [epX, epB, epC])。
        epX 共享 → persist 后只属于 C4（member_count 更大）。
        """
        store = DreamCandidateStore()
        c3_id = f"c3-{uuid.uuid4().hex[:8]}"
        c4_id = f"c4-{uuid.uuid4().hex[:8]}"
        epX = f"epX-{uuid.uuid4().hex[:8]}"
        epA = f"epA-{uuid.uuid4().hex[:8]}"
        epB = f"epB-{uuid.uuid4().hex[:8]}"
        epC = f"epC-{uuid.uuid4().hex[:8]}"

        # 创建所有 EpisodeNode
        for ep_id, label in [(epX, "shared"), (epA, "C3-only"),
                              (epB, "C4-only"), (epC, "C4-only")]:
            _create_episode(graphlite_store, ep_id, label)

        candidate = _make_candidate(
            community_summaries=[
                {
                    "id": c3_id,
                    "member_count": 2,
                    "member_ids": [epX, epA],
                    "report": "C3: smaller community sharing epX",
                    "keywords": [],
                    "topics": [],
                    "patterns": [],
                    "contradictions": [],
                },
                {
                    "id": c4_id,
                    "member_count": 3,
                    "member_ids": [epX, epB, epC],
                    "report": "C4: larger community sharing epX",
                    "keywords": [],
                    "topics": [],
                    "patterns": [],
                    "contradictions": [],
                },
            ],
        )

        store._persist_community_nodes(candidate, graphlite_store)

        # epX 只属于 C4（更大社区）
        c3_edge = graphlite_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $cid})"
            "-[:COMMUNITY_MEMBER]->"
            "(e:EpisodeNode {id: $eid}) RETURN e",
            {"cid": c3_id, "eid": epX},
        )
        c4_edge = graphlite_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $cid})"
            "-[:COMMUNITY_MEMBER]->"
            "(e:EpisodeNode {id: $eid}) RETURN e",
            {"cid": c4_id, "eid": epX},
        )
        assert not c3_edge, (
            f"epX should NOT belong to smaller C3, but edge exists: {c3_edge}"
        )
        assert c4_edge, (
            f"epX should belong to larger C4, but edge missing"
        )

        # C3 独有成员 epA 仍属于 C3
        c3_epA = graphlite_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $cid})"
            "-[:COMMUNITY_MEMBER]->"
            "(e:EpisodeNode {id: $eid}) RETURN e",
            {"cid": c3_id, "eid": epA},
        )
        assert c3_epA, f"C3 exclusive member epA should still belong to C3"

    def test_idempotent_double_apply(self, graphlite_store):
        """同一候选 persist 两次 → 节点 1 行、边不重复（防回归）。
        """
        store = DreamCandidateStore()
        comm_id = f"comm-{uuid.uuid4().hex[:8]}"
        ep1_id = f"ep-{uuid.uuid4().hex[:8]}"
        ep2_id = f"ep-{uuid.uuid4().hex[:8]}"

        _create_episode(graphlite_store, ep1_id, "episode 1")
        _create_episode(graphlite_store, ep2_id, "episode 2")

        candidate = _make_candidate(
            community_summaries=[{
                "id": comm_id,
                "member_count": 2,
                "member_ids": [ep1_id, ep2_id],
                "report": "test community for idempotency",
                "keywords": [],
                "topics": [],
                "patterns": [],
                "contradictions": [],
            }],
        )

        # 第一次 persist
        store._persist_community_nodes(candidate, graphlite_store)

        # 第二次 persist（幂等）
        store._persist_community_nodes(candidate, graphlite_store)

        # 社区节点只有 1 行
        comm_rows = graphlite_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $id}) RETURN c.id AS id",
            {"id": comm_id},
        )
        assert comm_rows is not None, "Community node should exist"
        assert len(comm_rows) == 1, (
            f"Expected 1 CommunityNode row, got {len(comm_rows)}: {comm_rows}"
        )

        # 边不重复：每个成员只有一条 COMMUNITY_MEMBER
        for ep_id in [ep1_id, ep2_id]:
            edge_rows = graphlite_store.execute_cypher(
                "MATCH (c:CommunityNode {id: $cid})"
                "-[r:COMMUNITY_MEMBER]->"
                "(e:EpisodeNode {id: $eid}) RETURN r",
                {"cid": comm_id, "eid": ep_id},
            )
            assert edge_rows is not None
            assert len(edge_rows) == 1, (
                f"Expected exactly 1 edge for {comm_id}->{ep_id}, "
                f"got {len(edge_rows)}: {edge_rows}"
            )
