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


def _create_episode(overgraph_store, ep_id: str, content: str = "test content") -> None:
    """在 GraphLite 中创建一个 EpisodeNode。"""
    overgraph_store.create_episode({
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

    def test_preserves_external_communities(self, overgraph_store):
        """预置外部 CommunityNode → persist 后仍存在（验证不全删）。"""
        store = DreamCandidateStore()
        external_cid = f"ext-comm-{uuid.uuid4().hex[:8]}"

        # 预置一个外部 CommunityNode（不是 dream 创建的）
        overgraph_store.execute_cypher(
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

        store._persist_community_nodes(candidate, overgraph_store)

        # 外部社区仍存在（不会被 DETACH DELETE 全删）
        result = overgraph_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $id}) RETURN c",
            {"id": external_cid},
        )
        assert result, f"External community {external_cid} was deleted"

    def test_creates_member_edges(self, overgraph_store):
        """persist 后新社区有 COMMUNITY_MEMBER 边（验证建边 + 两阶段）。"""
        store = DreamCandidateStore()
        comm_id = f"comm-{uuid.uuid4().hex[:8]}"
        ep1_id = f"ep-{uuid.uuid4().hex[:8]}"
        ep2_id = f"ep-{uuid.uuid4().hex[:8]}"

        # 先创建 EpisodeNode（否则 MATCH 无行不执行 INSERT 边）
        _create_episode(overgraph_store, ep1_id, "episode 1")
        _create_episode(overgraph_store, ep2_id, "episode 2")

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

        store._persist_community_nodes(candidate, overgraph_store)

        # 验证社区节点存在
        comm_result = overgraph_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $id}) RETURN c",
            {"id": comm_id},
        )
        assert comm_result, f"Community node {comm_id} not found"

        # 验证 COMMUNITY_MEMBER 边存在
        for ep_id in [ep1_id, ep2_id]:
            edge_result = overgraph_store.execute_cypher(
                "MATCH (c:CommunityNode {id: $cid})"
                "-[:COMMUNITY_MEMBER]->"
                "(e:EpisodeNode {id: $eid}) RETURN e",
                {"cid": comm_id, "eid": ep_id},
            )
            assert edge_result, (
                f"COMMUNITY_MEMBER edge missing: {comm_id} -> {ep_id}"
            )

    def test_old_format_candidate_compat(self, overgraph_store):
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
        created = store._persist_community_nodes(candidate, overgraph_store)
        assert created == 1

        # 节点已创建
        comm_result = overgraph_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $id}) RETURN c",
            {"id": comm_id},
        )
        assert comm_result, (
            f"Old-format community node {comm_id} not created"
        )

        # 无边（member_ids 缺失 → 跳过边创建，仅 logger.warning）
        edge_result = overgraph_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $cid})-[r:COMMUNITY_MEMBER]->() RETURN r",
            {"cid": comm_id},
        )
        assert edge_result == [], (
            f"Expected no edges for old-format community, got {edge_result}"
        )

    def test_shared_member_keeps_largest_community(self, overgraph_store):
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
            _create_episode(overgraph_store, ep_id, label)

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

        store._persist_community_nodes(candidate, overgraph_store)

        # epX 只属于 C4（更大社区）
        c3_edge = overgraph_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $cid})"
            "-[:COMMUNITY_MEMBER]->"
            "(e:EpisodeNode {id: $eid}) RETURN e",
            {"cid": c3_id, "eid": epX},
        )
        c4_edge = overgraph_store.execute_cypher(
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
        c3_epA = overgraph_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $cid})"
            "-[:COMMUNITY_MEMBER]->"
            "(e:EpisodeNode {id: $eid}) RETURN e",
            {"cid": c3_id, "eid": epA},
        )
        assert c3_epA, f"C3 exclusive member epA should still belong to C3"

    def test_idempotent_double_apply(self, overgraph_store):
        """同一候选 persist 两次 → 节点 1 行、边不重复（防回归）。
        """
        store = DreamCandidateStore()
        comm_id = f"comm-{uuid.uuid4().hex[:8]}"
        ep1_id = f"ep-{uuid.uuid4().hex[:8]}"
        ep2_id = f"ep-{uuid.uuid4().hex[:8]}"

        _create_episode(overgraph_store, ep1_id, "episode 1")
        _create_episode(overgraph_store, ep2_id, "episode 2")

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
        store._persist_community_nodes(candidate, overgraph_store)

        # 第二次 persist（幂等）
        store._persist_community_nodes(candidate, overgraph_store)

        # 社区节点只有 1 行
        comm_rows = overgraph_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $id}) RETURN c.id AS id",
            {"id": comm_id},
        )
        assert comm_rows is not None, "Community node should exist"
        assert len(comm_rows) == 1, (
            f"Expected 1 CommunityNode row, got {len(comm_rows)}: {comm_rows}"
        )

        # 边不重复：每个成员只有一条 COMMUNITY_MEMBER
        for ep_id in [ep1_id, ep2_id]:
            edge_rows = overgraph_store.execute_cypher(
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


class TestAutoApplyHeartbeat:
    """auto_apply_candidates 分块心跳：长写（PRUNE + _persist_community_nodes）
    期间按批调用 heartbeat_fn，不改变写入逻辑/返回语义。"""

    @staticmethod
    def _store_with_candidates(tmp_path, monkeypatch, candidate) -> DreamCandidateStore:
        """构造 storage_dir 含 20 个占位候选文件的 store，并注入唯一候选。"""
        store = DreamCandidateStore(storage_dir=str(tmp_path))
        # 候选文件数阈值 >= 20（内容无关：_load_all_candidates 被替换）
        for i in range(20):
            (tmp_path / f"cand-{i:02d}.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(store, "_load_all_candidates", lambda: [candidate])
        return store

    @staticmethod
    def _comm(i: int) -> dict:
        return {
            "id": f"comm-{i:02d}",
            "member_count": 5,
            "member_ids": [f"ep-{i:02d}-{j}" for j in range(3)],
            "report": "x" * 50,
            "keywords": [],
            "topics": [],
            "patterns": [],
            "contradictions": [],
        }

    def test_auto_apply_calls_heartbeat_per_batch(self, tmp_path, monkeypatch):
        """PRUNE 每 ~10 个操作、persist 每 ~10 个社区 touch 一次心跳。"""
        comms = [self._comm(i) for i in range(25)]
        candidate = DreamCandidate(
            dream_id="hb-test-001",
            created_at=time.time(),
            trigger_mode="test",
            stats={"created": 0, "updated": 0, "deleted": 0},
            community_count=len(comms),
            prune_count=25,
            conflict_count=0,
            community_summaries=comms,
            prune_ops=[{"node_id": f"node-{i}"} for i in range(25)],
            merge_ops=[],
        )
        store = self._store_with_candidates(tmp_path, monkeypatch, candidate)

        class FakeTxn:
            def stage(self, ops):
                self.staged = getattr(self, "staged", 0) + len(ops)

            def commit(self):
                pass

            def rollback(self):
                pass

        class FakeDb:
            """模拟原生 OverGraph：无任何 EpisodeNode → 成员边全部跳过。"""

            def get_node_by_key(self, label, key):
                return None

            def get_edge_by_triple(self, frm, to, label):
                return None

        class FakeStore:
            def __init__(self):
                self.cypher_calls = []
                self.staged_ops = 0

            def query_cypher(self, statement, params=None):
                self.cypher_calls.append(("query", statement, params))
                return []

            def execute_cypher(self, statement, params=None):
                self.cypher_calls.append(("execute", statement, params))
                return []

            def batch_write_txn(self):
                from contextlib import contextmanager

                @contextmanager
                def _cm():
                    txn = FakeTxn()
                    yield txn, FakeDb()
                    self.staged_ops += getattr(txn, "staged", 0)

                return _cm()

        fake = FakeStore()
        hb_calls = []
        applied, created, deleted, summaries = store.auto_apply_candidates(
            fake, heartbeat_fn=lambda: hb_calls.append(1),
        )
        # 返回语义不变：1 个候选应用、25 个社区、文件删除标记 1、25 条摘要
        assert (applied, created, deleted) == (1, 25, 1)
        assert len(summaries) == 25
        # PRUNE 25 ops → 2 批；persist 25 comms → 2 批；合计 >= 4 次心跳
        assert len(hb_calls) >= 4, f"heartbeat called only {len(hb_calls)}x"

    def test_auto_apply_heartbeat_none_unchanged(self, tmp_path, monkeypatch):
        """heartbeat_fn=None（默认）→ 行为与旧版一致，返回语义不变。"""
        comms = [self._comm(0)]
        candidate = DreamCandidate(
            dream_id="hb-test-002",
            created_at=time.time(),
            trigger_mode="test",
            stats={"created": 0, "updated": 0, "deleted": 0},
            community_count=1,
            prune_count=0,
            conflict_count=0,
            community_summaries=comms,
            prune_ops=[],
            merge_ops=[],
        )
        store = self._store_with_candidates(tmp_path, monkeypatch, candidate)

        class FakeTxn:
            def stage(self, ops):
                pass

            def commit(self):
                pass

            def rollback(self):
                pass

        class FakeDb:
            def get_node_by_key(self, label, key):
                return None

            def get_edge_by_triple(self, frm, to, label):
                return None

        class FakeStore:
            def query_cypher(self, statement, params=None):
                return []

            def execute_cypher(self, statement, params=None):
                return []

            def batch_write_txn(self):
                from contextlib import contextmanager

                @contextmanager
                def _cm():
                    yield FakeTxn(), FakeDb()

                return _cm()

        result = store.auto_apply_candidates(FakeStore())
        assert result == (1, 1, 1, comms)
