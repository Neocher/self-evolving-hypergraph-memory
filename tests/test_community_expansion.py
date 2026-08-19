"""
v5.41.0 社区扩召回（Community-Expansion）测试
============================================
覆盖（任务书 10 用例）：
  1. 边方向正确（真实 GraphLite：种子→社区反查命中）
  2. 社区不相关 → 不加分不召回
  3. 相关社区 → 成员召回排除种子
  4. 假阳性护栏：扩展分 ≤ min(种子分)
  5. GraphLite 失败 → 静默降级返回 []，主检索零回归
  6. 开关关闭 → 结果 bit 级一致
  7. 与画像叠加不双重放大
  8. 3s 超时预算内（压测不触发 _RETRIEVE_TIMEOUT）

运行: python -m pytest tests/test_community_expansion.py -v
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from core.dream_pipeline import DreamPipeline


def _persist_community(store, cid: str, members: list[str], report: str, idx: int = 0):
    """用生产 _persist_one_community 造真实社区边（COMMUNITY_MEMBER，社区→成员）。"""
    return DreamPipeline()._persist_one_community(
        store, {"id": cid, "members": members, "report": report}, "eval", idx
    )


def _insert_episode(store, nid: str, content: str, fact_track: str = "active"):
    store.create_episode({
        "id": nid, "content": content, "tau_initial": 1.0,
        "fact_track": fact_track, "archived": False,
    })


def _make_router(store, hypergraph_results: list[dict]):
    """构造 QueryRouter：mock _hypergraph_retrieve 控制种子；社区走真实 store。"""
    from retrieval.query_router import QueryRouter, QueryRouterConfig
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router._zh_en_tech_map = {}
    router._time_keywords = set()
    router.graphlite_store = store
    router._hypergraph_retrieve = MagicMock(return_value=hypergraph_results)
    return router


def _seed_result(nid: str, content: str, score: float = 0.9,
                 fact_track: str = "active") -> dict:
    return {
        "node_id": nid, "content": content, "score": score,
        "fact_track": fact_track, "tau_value": 1.0, "level": "l1_faiss",
    }


class TestStorePrimitives:
    """改动 1：批量社区查询原语（真实 GraphLite）"""

    def test_config_loaded_from_yaml(self):
        """改动 3 验收：defaults.yaml 的 community_expansion 正确接线为 dataclass 实例。"""
        from config.settings import load_settings
        ce = load_settings().retrieval.community_expansion
        assert ce.enabled is True
        assert ce.boost == 0.6
        assert ce.threshold == 0.5
        assert ce.max_members == 10

    def test_edge_direction_seed_to_community(self, overgraph_store):
        """边方向：(c:CommunityNode)-[:COMMUNITY_MEMBER]->(e:EpisodeNode)。

        种子→社区反查命中（种子属于社区）；非社区成员不命中；成员批量取回。
        """
        _insert_episode(overgraph_store, "ep1", "K8s 集群搭建 flannel 网络问题")
        _insert_episode(overgraph_store, "ep2", "用 calico 替换 flannel 解决网络")
        _insert_episode(overgraph_store, "ep3", "无关买菜清单")
        _persist_community(overgraph_store, "comm_net", ["ep1", "ep2"],
                           "K8s 集群网络 flannel calico 排障 多会话")
        # 种子 → 社区反查命中
        comms = overgraph_store.get_communities_by_seeds(["ep1"])
        assert len(comms) == 1
        assert comms[0]["community_id"] == "comm_net"
        assert set(comms[0]["member_ids"]) == {"ep1"}
        assert "flannel" in comms[0]["summary"]
        # 非成员种子 → 不命中
        assert overgraph_store.get_communities_by_seeds(["ep3"]) == []
        # 空输入 → []
        assert overgraph_store.get_communities_by_seeds([]) == []
        # 成员批量取回（含 content，供检索复用）
        members = overgraph_store.get_community_members("comm_net", limit=10)
        mids = {m["member_id"] for m in members}
        assert mids == {"ep1", "ep2"}
        assert all(m["content"] for m in members)


class TestCommunityExpansion:
    """改动 2：检索层社区扩召回（_finish 去重前 append）"""

    def test_irrelevant_community_no_expansion(self, overgraph_store):
        """社区不相关（summary 与查询词法无重叠）→ 不加分不召回。"""
        _insert_episode(overgraph_store, "ep1", "今天讨论 K8s 集群搭建")
        _insert_episode(overgraph_store, "ep2", "买菜清单 西红柿 鸡蛋")
        _persist_community(overgraph_store, "comm_unrel", ["ep1", "ep2"],
                           "菜谱烹饪 家常菜 西红柿炒蛋")
        router = _make_router(overgraph_store, [
            _seed_result("ep1", "今天讨论 K8s 集群搭建"),
        ])
        out = router.retrieve("K8s 集群网络排障")
        assert all(r["level"] != "community_expansion" for r in out)
        assert [r["node_id"] for r in out] == ["ep1"]

    def test_relevant_community_recalls_members_excluding_seed(self, overgraph_store):
        """相关社区 → 成员召回（level=community_expansion），种子被排除。"""
        _insert_episode(overgraph_store, "ep1", "多会话记忆里 K8s 集群遇到 flannel 问题")
        _insert_episode(overgraph_store, "ep2", "另一会话用 calico 解决了 flannel 网络问题")
        _persist_community(overgraph_store, "comm_k8s", ["ep1", "ep2"],
                           "K8s 集群网络 flannel calico 多会话排障 记忆")
        router = _make_router(overgraph_store, [
            _seed_result("ep1", "多会话记忆里 K8s 集群遇到 flannel 问题"),
        ])
        out = router.retrieve("多会话 K8s 集群 flannel 网络问题")
        exp = [r for r in out if r["level"] == "community_expansion"]
        assert exp, "相关社区应召回跨会话成员"
        assert exp[0]["node_id"] == "ep2"
        assert exp[0]["_source"] == "community"
        # 种子（已在主检索中）被排除，不重复召回
        assert "ep1" not in {e["node_id"] for e in exp}

    def test_expansion_score_below_min_seed(self, overgraph_store):
        """假阳性护栏：扩展分 = relevance × min(种子分) × 0.6，严格低于种子分。"""
        _insert_episode(overgraph_store, "ep1", "多会话记忆里 K8s 集群遇到 flannel 问题")
        _insert_episode(overgraph_store, "ep2", "另一会话用 calico 解决了 flannel 网络问题")
        _insert_episode(overgraph_store, "ep3", "补充会话里 calico 配置了 BGP 网络")
        _persist_community(overgraph_store, "comm_k8s", ["ep1", "ep2", "ep3"],
                           "K8s 集群网络 flannel calico 多会话排障 记忆")
        # 种子分 [0.9, 0.6] → min=0.6；ep3 非种子成员 → 被召回
        router = _make_router(overgraph_store, [
            _seed_result("ep1", "多会话记忆里 K8s 集群遇到 flannel 问题", score=0.9),
            _seed_result("ep2", "另一会话用 calico 解决了 flannel 网络问题", score=0.6),
        ])
        out = router.retrieve("多会话 K8s 集群 flannel 网络问题")
        exp = [r for r in out if r["level"] == "community_expansion"]
        assert exp and exp[0]["node_id"] == "ep3"
        assert all(e["score"] < 0.6 for e in exp), \
            f"扩展分应 < min(种子分)=0.6: {[e['score'] for e in exp]}"
        assert all(e["score"] > 0.0 for e in exp)

    def test_graphlite_failure_silent_degrade(self):
        """GraphLite 查询失败 → 静默降级返回原 results，主检索零回归。"""
        store = MagicMock()
        store.get_communities_by_seeds.side_effect = RuntimeError("graphlite down")
        router = _make_router(store, [_seed_result("s1", "content 1")])
        out = router.retrieve("some query")
        assert [r["node_id"] for r in out] == ["s1"]
        assert all(r["level"] != "community_expansion" for r in out)

    def test_disabled_bit_identical(self, overgraph_store, monkeypatch):
        """开关关闭 → 结果与现状 bit 级一致（无社区扩召回项，种子结果原样）。"""
        from config.settings import Settings, CommunityExpansionConfig
        _insert_episode(overgraph_store, "ep1", "多会话记忆里 K8s 集群遇到 flannel 问题")
        _insert_episode(overgraph_store, "ep2", "另一会话用 calico 解决了 flannel 网络问题")
        _persist_community(overgraph_store, "comm_k8s", ["ep1", "ep2"],
                           "K8s 集群网络 flannel calico 多会话排障 记忆")
        s = Settings()
        s.retrieval.community_expansion = CommunityExpansionConfig(enabled=False)
        monkeypatch.setattr("retrieval.query_router.get_settings", lambda: s)
        router = _make_router(overgraph_store, [
            _seed_result("ep1", "多会话记忆里 K8s 集群遇到 flannel 问题"),
        ])
        out = router.retrieve("多会话 K8s 集群 flannel 网络问题")
        assert all(r["level"] != "community_expansion" for r in out)
        assert [r["node_id"] for r in out] == ["ep1"]
        assert out[0]["score"] == 0.9  # 种子分未被扩召回影响

    def test_profile_boost_not_double_applied(self, overgraph_store):
        """与画像叠加：core ×1.1 + 画像 ×1.2 在 _deduplicate_and_sort 单点各一次，
        扩召回内不加分 → 不双重放大。"""
        from retrieval.query_router import set_user_profile
        set_user_profile({"preferences": {"flannel": {"weight": 1.0, "sources": 1}}})
        try:
            _insert_episode(overgraph_store, "ep1", "K8s 集群 flannel 问题", fact_track="core")
            _insert_episode(overgraph_store, "ep2", "flannel 被 calico 替换", fact_track="core")
            _persist_community(overgraph_store, "comm_f", ["ep1", "ep2"],
                               "K8s 集群 flannel calico 网络 排障")
            router = _make_router(overgraph_store, [
                _seed_result("ep1", "K8s 集群 flannel 问题", score=0.8, fact_track="core"),
            ])
            out = router.retrieve("flannel calico 网络")
            exp = [r for r in out if r["level"] == "community_expansion"]
            assert exp and exp[0]["node_id"] == "ep2"
            e = exp[0]
            assert e["fact_track"] == "core"
            # 单次 boost 链：最终分 = 扩召回原始分 × 1.1(core) × 1.2(画像)
            rel = router._community_relevance(
                "flannel calico 网络", ["K8s 集群 flannel calico 网络 排障"])[0]
            base = round(rel * 0.8 * 0.6, 6)
            expected = min(1.0, base * 1.1 * 1.2)
            assert e["score"] == pytest.approx(expected, rel=1e-6), \
                f"score={e['score']} expected(单次 boost)={expected}"
        finally:
            set_user_profile({})

    def test_community_relevance_multi_community(self):
        """P1 回归：多社区 relevance 行索引独立（CSR 列切片 indices 是列号非行号）。

        修复前 col.indices 恒 0 → 所有社区 BM25 分累加到 summaries[0]，
        其余社区恒 0（_community_relevance("K8s 网络", ["", "K8s 集群网络 flannel"])
        -> [0.92, 0.0] 错误）。修复后各社区分独立落位。
        """
        router = _make_router(MagicMock(), [])
        rel = router._community_relevance(
            "K8s 集群网络 flannel",
            ["", "K8s 集群网络 flannel calico 排障"],
        )
        assert rel[0] < 0.1, f"无词法重叠社区应 ~0: {rel[0]}"
        assert rel[1] > 0.5, f"强匹配社区应高相关: {rel[1]}"
        assert rel[1] > rel[0]

    @pytest.mark.graphlite  # 【v6.0.0 legacy】GraphLite 专属语义/引擎约束（默认排除，addopts -m 'not graphlite'）
    def test_within_timeout_budget(self, overgraph_store):

        """3s 超时预算内：5 次带社区扩召回的检索总耗时 < 3s（单次远低于 _RETRIEVE_TIMEOUT）。"""
        _insert_episode(overgraph_store, "ep1", "多会话记忆里 K8s 集群遇到 flannel 问题")
        _insert_episode(overgraph_store, "ep2", "另一会话用 calico 解决了 flannel 网络问题")
        _persist_community(overgraph_store, "comm_k8s", ["ep1", "ep2"],
                           "K8s 集群网络 flannel calico 多会话排障 记忆")
        router = _make_router(overgraph_store, [
            _seed_result("ep1", "多会话记忆里 K8s 集群遇到 flannel 问题"),
        ])
        t0 = time.monotonic()
        for _ in range(5):
            out = router.retrieve("多会话 K8s 集群 flannel 网络问题")
            assert any(r["level"] == "community_expansion" for r in out)
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, f"5 次扩召回检索耗时 {elapsed:.2f}s 超预算"
