"""
v6.28 检索超时修复（社区成员批量查询缓存 + 扩展通道独立预算熔断）验收测试
========================================================================
验证（对齐任务书 AC1..AC6）：
  AC1 缓存生效：相同 community_ids 连续调 2 次 → 底层 GQL 只发生 1 次（计数断言）
  AC2 失效正确：invalidate_community_members_cache() 后再次查询；
      且 dream_candidate_store 落库新成员后能拿到新成员
  AC3 语义等价：缓存命中与未命中返回的成员表内容+顺序逐字段一致
  AC4 熔断：慢 store 第 1 次调用后熔断打开，后续调用 <50ms 直接返回原 results
  AC5 半开：熔断窗口过后 store 恢复正常 → 扩展通道重新生效（再次调用底层）
  AC6 端到端：真实 store + level=FUSION ≥5 次，中位耗时 < 3.0s

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_retrieval_timeout_fix.py -q
"""
from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock

import pytest

from core.dream_candidate_store import DreamCandidate, DreamCandidateStore
from core.dream_pipeline import DreamPipeline


def _create_episode(store, eid: str, content: str, tau: float = 1.0) -> None:
    """创建 EpisodeNode（tau_initial 用于 get_community_members_batch 的 ORDER BY）。"""
    store.create_episode({
        "id": eid, "content": content, "tau_initial": tau,
        "fact_track": "active", "archived": False,
    })


def _persist_community(store, cid: str, members: list[str], report: str) -> None:
    """用生产 _persist_one_community 造真实社区边（社区→成员）。"""
    DreamPipeline()._persist_one_community(
        store, {"id": cid, "members": members, "report": report}, "eval", 0
    )


def _dream_candidate(cid: str, member_ids: list[str]) -> DreamCandidate:
    """构造 DreamCandidate（供 dream_candidate_store 落库路径测试失效）。"""
    return DreamCandidate(
        dream_id=f"dream-{uuid.uuid4().hex[:8]}",
        created_at=time.time(),
        trigger_mode="test",
        stats={},
        community_count=1,
        prune_count=0,
        conflict_count=0,
        community_summaries=[{
            "id": cid,
            "member_count": len(member_ids),
            "member_ids": member_ids,
            "report": "K8s 集群网络 flannel calico 多会话排障 记忆",
            "keywords": [],
            "topics": [],
            "patterns": [],
            "contradictions": [],
        }],
        prune_ops=[],
        merge_ops=[],
    )


def _make_router(store, seed_results: list[dict]):
    """构造 QueryRouter：mock _hypergraph_retrieve 控制种子；社区走真实 store。"""
    from retrieval.query_router import QueryRouter, QueryRouterConfig
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router._zh_en_tech_map = {}
    router._time_keywords = set()
    router.graph_store = store
    router.encoder = None  # __new__ 绕过 __init__：_encode_query 需显式存在（None → 静默降级）
    router._hypergraph_retrieve = MagicMock(return_value=seed_results)
    return router


def _seed_result(nid: str, content: str, score: float = 0.9) -> dict:
    return {
        "node_id": nid, "content": content, "score": score,
        "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss",
    }


class TestCommunityMembersCache:
    """Part A — 社区成员批量查询进程内缓存。"""

    def test_cache_avoids_second_gql_call(self, overgraph_store, monkeypatch):
        """AC1：相同 community_ids 连续调 2 次 → 底层 GQL 只发生 1 次。"""
        _create_episode(overgraph_store, "ep1", "K8s 集群 flannel 问题")
        _create_episode(overgraph_store, "ep2", "calico 替换 flannel 网络")
        _persist_community(overgraph_store, "comm_a", ["ep1", "ep2"],
                           "K8s 集群网络 flannel calico 排障")

        orig = overgraph_store.query_cypher
        wrapped = MagicMock(wraps=orig)
        monkeypatch.setattr(overgraph_store, "query_cypher", wrapped)

        first = overgraph_store.get_community_members_batch(["comm_a"], limit=10)
        second = overgraph_store.get_community_members_batch(["comm_a"], limit=10)

        assert first == second
        assert wrapped.call_count == 1, (
            f"缓存未命中：相同输入第 2 次仍查库（GQL 调用 {wrapped.call_count} 次）"
        )

    def test_invalidate_forces_requery(self, overgraph_store, monkeypatch):
        """AC2a：invalidate_community_members_cache() 后再查 → 底层再次查询。"""
        _create_episode(overgraph_store, "ep1", "K8s 集群 flannel 问题")
        _persist_community(overgraph_store, "comm_a", ["ep1"],
                           "K8s 集群网络 flannel 排障")

        orig = overgraph_store.query_cypher
        wrapped = MagicMock(wraps=orig)
        monkeypatch.setattr(overgraph_store, "query_cypher", wrapped)

        overgraph_store.get_community_members_batch(["comm_a"], limit=10)  # miss
        overgraph_store.get_community_members_batch(["comm_a"], limit=10)  # hit
        assert wrapped.call_count == 1

        overgraph_store.invalidate_community_members_cache()
        overgraph_store.get_community_members_batch(["comm_a"], limit=10)  # miss again
        assert wrapped.call_count == 2

    def test_dream_persist_invalidates_and_reveals_new_member(self, overgraph_store):
        """AC2b：dream_candidate_store 落库新成员后，再查能拿到新成员。"""
        _create_episode(overgraph_store, "ep1", "K8s 集群 flannel 问题", tau=1.0)
        candidate_store = DreamCandidateStore()

        # 先落库社区（仅 ep1），并查询一次以填充缓存
        candidate_store._persist_community_nodes(
            _dream_candidate("comm_a", ["ep1"]), overgraph_store)
        members = overgraph_store.get_community_members_batch(["comm_a"], limit=10)
        assert {m["member_id"] for m in members["comm_a"]} == {"ep1"}

        # 新增 ep2 并再次经 dream_candidate_store 落库（触发缓存失效）
        _create_episode(overgraph_store, "ep2", "calico 替换 flannel 网络", tau=1.0)
        candidate_store._persist_community_nodes(
            _dream_candidate("comm_a", ["ep1", "ep2"]), overgraph_store)

        members = overgraph_store.get_community_members_batch(["comm_a"], limit=10)
        assert {m["member_id"] for m in members["comm_a"]} == {"ep1", "ep2"}, (
            "dream_candidate_store 落库后缓存未失效，仍返回过期成员"
        )

    def test_cache_hit_matches_miss_field_by_field(self, overgraph_store):
        """AC3：缓存命中与首次未命中的成员表内容+顺序逐字段一致。"""
        _create_episode(overgraph_store, "ep_a", "A 内容", tau=0.5)
        _create_episode(overgraph_store, "ep_b", "B 内容", tau=0.9)
        _create_episode(overgraph_store, "ep_c", "C 内容", tau=0.7)
        _persist_community(overgraph_store, "comm_x", ["ep_a", "ep_b", "ep_c"],
                           "K8s 集群网络 flannel calico 排障")

        first = overgraph_store.get_community_members_batch(["comm_x"], limit=10)
        second = overgraph_store.get_community_members_batch(["comm_x"], limit=10)  # hit

        # 内容 + 顺序完全一致
        assert first == second
        assert list(first.keys()) == list(second.keys())
        for cid in first:
            a, b = first[cid], second[cid]
            assert [m["member_id"] for m in a] == [m["member_id"] for m in b]
            for m1, m2 in zip(a, b):
                assert m1 == m2  # 逐字段
                assert list(m1.keys()) == list(m2.keys())

        # 顺序 = tau_initial DESC（0.9 > 0.7 > 0.5）
        assert [m["member_id"] for m in first["comm_x"]] == ["ep_b", "ep_c", "ep_a"]


class TestCommunityExpansionBreaker:
    """Part B — 扩展通道独立预算熔断。"""

    def _slow_store(self, member_dicts: list[dict], sleep_s: float):
        store = MagicMock()
        store.get_communities_by_seeds.return_value = [{
            "community_id": "comm_slow",
            "summary": "K8s 集群网络 flannel calico 多会话排障 记忆",
            "member_ids": ["ep1"],
        }]

        def batch(cids, limit):
            time.sleep(sleep_s)
            return {"comm_slow": member_dicts}

        store.get_community_members_batch.side_effect = batch
        return store

    def test_breaker_skips_after_slow_call(self, monkeypatch):
        """AC4：慢 store 第 1 次调用后熔断打开，后续调用 <50ms 直接返回原 results。"""
        from config.settings import Settings, CommunityExpansionConfig
        s = Settings()
        s.retrieval.community_expansion = CommunityExpansionConfig(
            budget_s=0.2, breaker_s=60.0)
        monkeypatch.setattr("retrieval.query_router.get_settings", lambda: s)

        store = self._slow_store([], sleep_s=0.6)
        seeds = [_seed_result("ep1", "K8s 集群 flannel 问题")]
        router = _make_router(store, seeds)
        query = "多会话 K8s 集群 flannel 网络问题"

        # 第 1 次：慢调用（≥0.5s），打开熔断
        t0 = time.monotonic()
        out1 = router._community_expansion(seeds, query, None, query)
        first_elapsed = time.monotonic() - t0
        assert first_elapsed >= 0.5, f"第 1 次应走慢调用: {first_elapsed:.3f}s"
        assert store.get_community_members_batch.call_count == 1
        # 熔断确实打开
        assert getattr(router, "_community_expansion_breaker_open_until", 0.0) \
            > time.monotonic()

        # 后续：熔断窗口内直接跳过（不再次调用底层）
        t1 = time.monotonic()
        out2 = router._community_expansion(seeds, query, None, query)
        second_elapsed = time.monotonic() - t1
        assert second_elapsed < 0.05, f"熔断未跳过，耗时 {second_elapsed:.3f}s"
        assert store.get_community_members_batch.call_count == 1
        # 结果 = 原 results（fail-soft，静默返回）
        assert [r["node_id"] for r in out2] == ["ep1"]
        assert [r["node_id"] for r in out1] == ["ep1"]

    def test_breaker_half_open_recovers(self, monkeypatch):
        """AC5：熔断窗口过后 store 恢复正常 → 扩展通道重新生效（再次调用底层）。"""
        from config.settings import Settings, CommunityExpansionConfig
        s = Settings()
        s.retrieval.community_expansion = CommunityExpansionConfig(
            budget_s=0.1, breaker_s=0.5)
        monkeypatch.setattr("retrieval.query_router.get_settings", lambda: s)

        member = [{
            "member_id": "ep2", "content": "另一会话用 calico 解决了 flannel 网络问题",
            "archived": False, "fact_track": "active", "tau_value": 1.0,
        }]
        store = self._slow_store(member, sleep_s=0.3)
        # 先慢后快
        state = {"slow": True}

        def batch(cids, limit):
            if state["slow"]:
                time.sleep(0.3)
            return {"comm_slow": member}

        store.get_community_members_batch.side_effect = batch
        seeds = [_seed_result("ep1", "K8s 集群 flannel 问题")]
        router = _make_router(store, seeds)
        query = "多会话 K8s 集群 flannel 网络问题"

        # 第 1 次：慢 → 打开熔断（0.5s 窗口）
        router._community_expansion(seeds, query, None, query)
        assert store.get_community_members_batch.call_count == 1
        # 窗口内：跳过
        router._community_expansion(seeds, query, None, query)
        assert store.get_community_members_batch.call_count == 1

        # 等熔断窗口过期 → 半开重试；store 已恢复正常（快）
        time.sleep(0.6)
        state["slow"] = False
        out3 = router._community_expansion(seeds, query, None, query)
        assert store.get_community_members_batch.call_count == 2, (
            "半开未重试底层（call_count 仍为 1）"
        )
        exp = [r for r in out3 if r["level"] == "community_expansion"]
        assert any(e["node_id"] == "ep2" for e in exp), "扩展通道未重新生效"


class TestEndToEndFusionLatency:
    """AC6 — 端到端提速（真实 store + FUSION 中位耗时 < 3.0s）。"""

    def test_fusion_median_under_3s(self, overgraph_store, monkeypatch):
        from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel
        from unittest.mock import MagicMock as _Mock

        # 生产同构数据：多成员社区 + 成员 τ 分化
        _create_episode(overgraph_store, "ep0", "多会话记忆里 K8s 集群遇到 flannel 问题", tau=1.0)
        _create_episode(overgraph_store, "ep1", "另一会话用 calico 解决了 flannel 网络问题", tau=0.9)
        _create_episode(overgraph_store, "ep2", "补充会话里 calico 配置了 BGP 网络", tau=0.8)
        _persist_community(overgraph_store, "comm_prod", ["ep0", "ep1", "ep2"],
                           "K8s 集群网络 flannel calico 多会话排障 记忆")

        # 复现生产慢遍历：只放慢「成员批量拉取」GQL（get_community_members_batch；
        # get_communities_by_seeds 无 ORDER BY 不受影响）。
        orig = overgraph_store.query_cypher

        def slow_gql(query, params=None):
            q = query or ""
            if "COMMUNITY_MEMBER" in q and "ORDER BY e.tau_initial" in q:
                time.sleep(1.2)
            return orig(query, params)

        monkeypatch.setattr(overgraph_store, "query_cypher", slow_gql)

        # 真实 store + 完整 __init__；FUSION 候选生成 mock 为确定性种子，
        # 社区扩召回（真实 store + 缓存）才是被测链路。
        router = QueryRouter(
            graphlite_store=overgraph_store,
            faiss_index=MagicMock(),
            tfidf_index=None,
            encoder=MagicMock(),
            faiss_id_map={},
            episode_cache={},
            config=QueryRouterConfig(rerank_enabled=False),
        )
        router._fusion_retrieve = _Mock(return_value=[
            _seed_result("ep0", "多会话记忆里 K8s 集群遇到 flannel 问题"),
        ])

        query = "多会话 K8s 集群 flannel 网络问题"
        times = []
        for _ in range(5):
            t0 = time.monotonic()
            router.retrieve(query, level=RetrievalLevel.FUSION)
            times.append(time.monotonic() - t0)

        median = sorted(times)[len(times) // 2]
        assert median < 3.0, (
            f"FUSION 中位耗时 {median:.3f}s >= 3.0s: {[round(t, 3) for t in times]}"
        )
