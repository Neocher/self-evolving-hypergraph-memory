"""
OverGraph 批量写入优化测试
==========================
验证 _persist_community_nodes / _persist_one_community / _persist_hyperedges
改用批量事务（batch_write_txn + WriteTxn.stage）后：
1. 批量 API 被调用而非 execute_cypher（热路径零 Cypher）
2. 返回语义不变（created 数）
3. 批量写入中途失败 → rollback，无部分写入残留
4. 至少一处批量化有可测量提速证据（基准 vs 优化计时）
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager

from core.dream_candidate_store import DreamCandidate, DreamCandidateStore
from core.dream_pipeline import DreamPipeline


def _make_candidate(
    dream_id: str = "batch-test-001",
    community_summaries: list[dict] | None = None,
) -> DreamCandidate:
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


def _create_episode(store, ep_id: str, content: str = "test content") -> None:
    store.create_episode({
        "id": ep_id,
        "content": content,
        "created_at": time.time(),
        "tau_initial": 1.0,
        "tau_value": 0.6,
        "source": "test",
        "trust_score": 0.8,
    })


def _comm(i: int, n_members: int = 3) -> dict:
    return {
        "id": f"batch-comm-{i:03d}",
        "member_count": n_members,
        "member_ids": [f"batch-ep-{i:03d}-{j}" for j in range(n_members)],
        "report": f"batch test community {i} with a sufficiently long report body",
        "keywords": [],
        "topics": [],
        "patterns": [],
        "contradictions": [],
    }


class _CountedTxn:
    """包装真实 WriteTxn：计数 commit/rollback，透传 stage（代理计数用）。"""

    def __init__(self, txn, counters: dict):
        self._txn = txn
        self._counters = counters

    def stage(self, ops):
        return self._txn.stage(ops)

    def commit(self):
        self._counters["commits"] += 1
        return self._txn.commit()

    def rollback(self):
        self._counters["rollbacks"] += 1
        return self._txn.rollback()


def _replace_batch_write_txn(store, monkeypatch, counters: dict,
                             fail_stage: bool = False):
    """用「真实 begin_write_txn + 计数代理」替换 store.batch_write_txn。

    自管 commit/rollback（计数 + 透传真实 txn），不依赖原 CM 内部路径。
    """
    real_db = store._db

    @contextmanager
    def counted_batch_write_txn():
        with store._session_lock:
            txn = real_db.begin_write_txn()
            counters["entered"] += 1
            if fail_stage:
                class _Failing(_CountedTxn):
                    def stage(self, ops):
                        raise RuntimeError(
                            "simulated batch write failure mid-stage")

                proxy = _Failing(txn, counters)
            else:
                proxy = _CountedTxn(txn, counters)
            try:
                yield proxy, real_db
            except BaseException:
                try:
                    proxy.rollback()
                except Exception:
                    pass
                raise
            else:
                proxy.commit()

    monkeypatch.setattr(store, "batch_write_txn", counted_batch_write_txn)


class _ForbidCypher:
    """execute_cypher 守卫：persist 期间调用即断言失败，之后放行验证查询。"""

    def __init__(self, store):
        self._orig = store.execute_cypher
        self.forbid = False

    def __call__(self, query, params=None):
        if self.forbid:
            raise AssertionError(
                f"execute_cypher called in batch persist path! {query[:60]}")
        return self._orig(query, params)


class TestBatchPersistCommunityNodes:
    """_persist_community_nodes 批量化验收。"""

    def test_uses_batch_txn_not_execute_cypher(self, overgraph_store, monkeypatch):
        """热路径零 execute_cypher：persist 期间任何 Cypher 调用即回归。"""
        guard = _ForbidCypher(overgraph_store)
        monkeypatch.setattr(overgraph_store, "execute_cypher", guard)
        counters = {"entered": 0, "commits": 0, "rollbacks": 0}
        _replace_batch_write_txn(overgraph_store, monkeypatch, counters)

        comms = [_comm(i) for i in range(25)]  # 25 社区 × 3 成员
        for c in comms:
            for mid in c["member_ids"]:
                _create_episode(overgraph_store, mid)

        store = DreamCandidateStore()
        guard.forbid = True
        created = store._persist_community_nodes(
            _make_candidate(community_summaries=comms), overgraph_store,
        )
        guard.forbid = False

        assert created == 25, f"created={created}"
        # 2 个社区块（10+10+5 → 3 块）+ 阶段3 清理 = >= 3 次批量事务
        assert counters["entered"] >= 3, f"batch txns entered: {counters}"
        assert counters["commits"] >= 3, f"commits={counters}"
        assert counters["rollbacks"] == 0, f"rollbacks={counters}"
        # 边落库（行为等价实证）
        rows = overgraph_store.execute_cypher(
            "MATCH (:CommunityNode)-[r:COMMUNITY_MEMBER]->(:EpisodeNode) "
            "RETURN count(r) AS n",
        )
        assert rows and rows[0]["n"] == 75, f"edges={rows}"

    def test_return_semantics_unchanged(self, overgraph_store):
        """返回 created = 成功建节点社区数（含无成员旧格式社区）。"""
        comms = [_comm(0), _comm(1)]
        # 旧格式社区：无 member_ids → 只建节点不建边
        comms.append({
            "id": "batch-legacy-001",
            "member_count": 0,
            "member_ids": [],
            "report": "legacy candidate without member ids",
            "keywords": [], "topics": [], "patterns": [], "contradictions": [],
        })
        for c in comms:
            for mid in c.get("member_ids", []):
                _create_episode(overgraph_store, mid)

        created = DreamCandidateStore()._persist_community_nodes(
            _make_candidate(community_summaries=comms), overgraph_store,
        )
        assert created == 3, f"created={created}"
        legacy = overgraph_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $id}) RETURN c",
            {"id": "batch-legacy-001"},
        )
        assert legacy, "legacy (no-member) community node must exist"

    def test_failure_rolls_back_no_partial_writes(self, overgraph_store, monkeypatch):
        """批量写入中途失败 → rollback 被调用，无部分写入残留。"""
        comms = [_comm(0, n_members=2)]
        for c in comms:
            for mid in c["member_ids"]:
                _create_episode(overgraph_store, mid)

        counters = {"entered": 0, "commits": 0, "rollbacks": 0}
        _replace_batch_write_txn(
            overgraph_store, monkeypatch, counters, fail_stage=True)

        store = DreamCandidateStore()
        created = store._persist_community_nodes(
            _make_candidate(community_summaries=comms), overgraph_store,
        )
        # 失败回滚 → 不计数、不产生部分写入；按约定降级（warning + 继续）
        assert created == 0, f"created={created}"
        assert counters["rollbacks"] >= 1, f"rollbacks={counters}"
        assert counters["commits"] == 0, f"commits={counters}"
        # 无部分写入残留
        comm_rows = overgraph_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $id}) RETURN c",
            {"id": comms[0]["id"]},
        )
        assert not comm_rows, f"partial CommunityNode left: {comm_rows}"
        edge_rows = overgraph_store.execute_cypher(
            "MATCH (:CommunityNode)-[r:COMMUNITY_MEMBER]->(:EpisodeNode) "
            "RETURN count(r) AS n",
        )
        assert edge_rows and edge_rows[0]["n"] == 0, f"partial edges left: {edge_rows}"
        # 引擎未损坏，后续写入可用
        _create_episode(overgraph_store, "post-failure-ep")
        assert overgraph_store.execute_cypher(
            "MATCH (e:EpisodeNode {id: $id}) RETURN e",
            {"id": "post-failure-ep"},
        ), "store must remain usable after rollback"


class TestBatchPipelinePersist:
    """dream_pipeline 直连模式批量机会（_persist_one_community/_persist_hyperedges）。"""

    def test_persist_one_community_uses_batch_not_cypher(self, overgraph_store, monkeypatch):
        guard = _ForbidCypher(overgraph_store)
        monkeypatch.setattr(overgraph_store, "execute_cypher", guard)
        ep1, ep2 = "p1-ep-a", "p1-ep-b"
        _create_episode(overgraph_store, ep1)
        _create_episode(overgraph_store, ep2)

        pipeline = DreamPipeline()
        guard.forbid = True
        created, member_set = pipeline._persist_one_community(
            overgraph_store,
            {"id": "p1-comm", "members": [ep1, ep2], "report": "p1 report"},
            "p1", idx=0,
        )
        guard.forbid = False
        assert created == 1
        assert member_set == {ep1, ep2}
        edge_rows = overgraph_store.execute_cypher(
            "MATCH (c:CommunityNode {id: $cid})"
            "-[:COMMUNITY_MEMBER]->(e:EpisodeNode) RETURN count(e) AS n",
            {"cid": "p1-comm"},
        )
        assert edge_rows and edge_rows[0]["n"] == 2

    def test_persist_hyperedges_uses_batch_not_cypher(self, overgraph_store, monkeypatch):
        def _forbid(*args, **kwargs):
            raise AssertionError("query_cypher called in _persist_hyperedges!")

        monkeypatch.setattr(overgraph_store, "query_cypher", _forbid)
        eps = ["ph-ep-a", "ph-ep-b", "ph-ep-c"]
        for e in eps:
            _create_episode(overgraph_store, e)

        created = DreamPipeline()._persist_hyperedges(
            overgraph_store,
            [{"id": "ph-comm", "members": eps, "keywords": ["k"]}],
            "ph",
        )
        assert created == 1
        rows = overgraph_store.execute_cypher(
            "MATCH (:HyperedgeNode)-[r:HYPEREDGE_MEMBER]->(:EpisodeNode) "
            "RETURN count(r) AS n",
        )
        assert rows and rows[0]["n"] == 3


class TestBatchWritePerf:
    """至少一处批量化有可测量提速证据：同一真实库，逐条 execute_cypher
    （原实现等价路径） vs 批量事务（新实现）。"""

    @staticmethod
    def _persist_baseline_execute_cypher(store, candidate) -> int:
        """原实现语义复刻（逐社区/逐成员 execute_cypher，DELETE+INSERT 建边）。"""
        created = 0
        for comm in candidate.community_summaries:
            comm_id = comm["id"]
            comm_vals = {
                "id": comm_id,
                "name": f"dream_{candidate.dream_id[:8]}_comm_{created}",
                "summary": (comm.get("report", "") or "")[:800],
                "score": 0.0,
                "created_at": time.time(),
            }
            if store.execute_cypher(
                "MATCH (c:CommunityNode {id: $id}) RETURN c", {"id": comm_id},
            ):
                store.execute_cypher(
                    "MATCH (c:CommunityNode {id: $id}) "
                    "SET c.name = $name, c.summary = $summary, "
                    "c.leiden_score = $score, c.created_at = $created_at",
                    comm_vals,
                )
            else:
                store.execute_cypher(
                    "INSERT (c:CommunityNode {id: $id, name: $name, "
                    "summary: $summary, leiden_score: $score, "
                    "created_at: $created_at})",
                    comm_vals,
                )
            for mid in comm["member_ids"]:
                try:
                    store.execute_cypher(
                        "MATCH (c:CommunityNode {id: $cid})"
                        "-[r:COMMUNITY_MEMBER]->"
                        "(e:EpisodeNode {id: $mid}) DELETE r",
                        {"cid": comm_id, "mid": mid},
                    )
                except Exception:
                    pass
                try:
                    store.execute_cypher(
                        "MATCH (c:CommunityNode {id: $cid}), "
                        "(e:EpisodeNode {id: $mid}) "
                        "INSERT (c)-[:COMMUNITY_MEMBER]->(e)",
                        {"cid": comm_id, "mid": mid},
                    )
                except Exception:
                    pass
            created += 1
        return created

    def test_batch_persist_measurably_faster(self, overgraph_store):
        n_comms, n_members = 30, 20
        comms = [_comm(i, n_members=n_members) for i in range(n_comms)]
        for c in comms:
            for mid in c["member_ids"]:
                _create_episode(overgraph_store, mid)
        candidate = _make_candidate(dream_id=f"perf-{uuid.uuid4().hex[:6]}",
                                    community_summaries=comms)

        # 基准：逐条 execute_cypher（原实现路径）
        t0 = time.perf_counter()
        base_created = self._persist_baseline_execute_cypher(
            overgraph_store, candidate)
        t_base = time.perf_counter() - t0

        # 优化：批量事务（新实现，二次 apply 同时验证幂等提速）
        t0 = time.perf_counter()
        batch_created = DreamCandidateStore()._persist_community_nodes(
            candidate, overgraph_store)
        t_batch = time.perf_counter() - t0

        assert base_created == n_comms
        assert batch_created == n_comms
        speedup = t_base / t_batch if t_batch > 0 else float("inf")
        print(
            f"\n[perf] persist {n_comms} communities x {n_members} members: "
            f"baseline(execute_cypher)={t_base:.3f}s, "
            f"batch_txn={t_batch:.3f}s, speedup={speedup:.1f}x"
        )
        # 量级差异显著，0.5 倍阈值留足余量防抖动
        assert t_batch < t_base * 0.5, (
            f"batch not faster: base={t_base:.3f}s batch={t_batch:.3f}s"
        )
