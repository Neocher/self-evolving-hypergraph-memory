"""
v5.49.0 MESA 记忆增强检索（Mesa-Synthesis）测试
==============================================
覆盖（任务书 6 类用例）：
  1. 默认关：mesa_enabled=False → _mesa_synthesis 返回原 results（bit 级等价）
  2. 开启：mock GraphLite 返回社区 → 合成节点 append（level=mesa_synthesis）
  3. score 数学保证：合成节点 < 社区成员 < 种子（0.4 < 0.6 < 1）
  4. 阈值：relevance < threshold → 丢弃
  5. max_nodes 限制 append 数
  6. GraphLite 异常 → 静默降级返回原 results

运行: python -m pytest tests/test_mesa_synthesis.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.dream_pipeline import DreamPipeline


def _insert_episode(store, nid: str, content: str, fact_track: str = "active"):
    store.create_episode({
        "id": nid, "content": content, "tau_initial": 1.0,
        "fact_track": fact_track, "archived": False,
    })


def _persist_community(store, cid: str, members: list[str], report: str, idx: int = 0):
    return DreamPipeline()._persist_one_community(
        store, {"id": cid, "members": members, "report": report}, "eval", idx
    )


def _make_router(store, hypergraph_results: list[dict], mesa_enabled: bool = False,
                 **mesa_cfg):
    """构造 QueryRouter：mock _hypergraph_retrieve 控制种子；mesa 配置可控。"""
    from retrieval.query_router import QueryRouter, QueryRouterConfig
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig(mesa_enabled=mesa_enabled, **mesa_cfg)
    router._zh_en_tech_map = {}
    router._time_keywords = set()
    router.graphlite_store = store
    router._hypergraph_retrieve = MagicMock(return_value=hypergraph_results)
    return router


def _seed_result(nid: str, content: str, score: float = 0.9) -> dict:
    return {
        "node_id": nid, "content": content, "score": score,
        "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss",
    }


def _community_row(cid: str, summary: str) -> dict:
    return {"community_id": cid, "summary": summary, "member_ids": ["ep1"]}


class TestMesaConfig:
    def test_config_loaded_from_yaml(self):
        """defaults.yaml 的 mesa 段正确接线为 MesaConfig 实例（默认关）。"""
        from config.settings import load_settings
        mesa = load_settings().retrieval.mesa
        assert mesa.enabled is False
        assert mesa.boost == 0.4
        assert mesa.threshold == 0.5
        assert mesa.max_nodes == 5

    def test_boost_above_max_rejected(self):
        """P2-3：mesa.boost 上界 0.59（严格 < community boost 0.6），超界配置拒绝。"""
        from config.settings import MesaConfig
        with pytest.raises(ValueError):
            MesaConfig(boost=0.7)
        assert MesaConfig(boost=0.59).boost == 0.59
        assert MesaConfig(boost=0.4).boost == 0.4


class TestMesaSynthesis:
    def test_disabled_returns_original(self):
        """默认关：mesa_enabled=False → 返回原 results（同一对象，bit 级等价）。"""
        store = MagicMock()
        store.get_communities_by_seeds.return_value = [
            _community_row("c1", "K8s 集群网络 flannel calico")
        ]
        router = _make_router(store, [], mesa_enabled=False)
        seeds = [_seed_result("ep1", "content 1")]
        out = router._mesa_synthesis(seeds, "K8s 网络", "K8s 网络")
        assert out is seeds
        assert all(r["level"] != "mesa_synthesis" for r in out)

    def test_enabled_synthesis_node_appended(self):
        """开启：mock GraphLite 返回社区 → 合成节点 append（level=mesa_synthesis，
        score=rel×min_seed×0.4）。"""
        store = MagicMock()
        store.get_communities_by_seeds.return_value = [
            _community_row("c1", "K8s 集群网络 flannel calico 排障")
        ]
        router = _make_router(store, [], mesa_enabled=True)
        seeds = [_seed_result("ep1", "多会话记忆里 K8s 集群遇到 flannel 问题", score=0.9)]
        query = "K8s 集群网络排障"
        out = router._mesa_synthesis(seeds, query, query)
        syn = [r for r in out if r["level"] == "mesa_synthesis"]
        assert len(syn) == 1
        s = syn[0]
        assert s["node_id"] == "c1"
        assert s["content"] == "K8s 集群网络 flannel calico 排障"
        assert s["_source"] == "mesa"
        assert s["fact_track"] == "active"
        assert "archived" not in s  # 无 archived 字段（_filter_archived 恒保留）
        rel = router._community_relevance(
            query, ["K8s 集群网络 flannel calico 排障"])[0]
        assert s["score"] == pytest.approx(round(rel * 0.9 * 0.4, 6))

    def test_threshold_discards_low_relevance(self):
        """阈值：relevance < threshold(0.5) → 丢弃（不合成节点）。"""
        store = MagicMock()
        store.get_communities_by_seeds.return_value = [
            _community_row("c1", "买菜清单 西红柿 鸡蛋 家常菜")
        ]
        router = _make_router(store, [], mesa_enabled=True)
        seeds = [_seed_result("ep1", "K8s 集群搭建", score=0.9)]
        out = router._mesa_synthesis(seeds, "K8s 集群网络排障", "K8s 集群网络排障")
        assert all(r["level"] != "mesa_synthesis" for r in out)
        assert len(out) == len(seeds)

    def test_max_nodes_limit(self):
        """max_nodes 限制 append 数：7 个相关社区 → 只合成前 max_nodes=3 个。"""
        store = MagicMock()
        store.get_communities_by_seeds.return_value = [
            _community_row(f"c{i}", f"K8s 集群网络 flannel calico 排障 {i}")
            for i in range(7)
        ]
        router = _make_router(store, [], mesa_enabled=True, mesa_max_nodes=3)
        seeds = [_seed_result("ep1", "K8s 集群搭建", score=0.9)]
        out = router._mesa_synthesis(seeds, "K8s 集群网络排障", "K8s 集群网络排障")
        syn = [r for r in out if r["level"] == "mesa_synthesis"]
        assert len(syn) == 3

    def test_max_nodes_zero_synthesizes_nothing(self):
        """P2-2：max_nodes=0 → 不合成任何节点（修复前先 append 再 break 仍合成 1 条）。"""
        store = MagicMock()
        store.get_communities_by_seeds.return_value = [
            _community_row("c1", "K8s 集群网络 flannel calico 排障")
        ]
        router = _make_router(store, [], mesa_enabled=True, mesa_max_nodes=0)
        seeds = [_seed_result("ep1", "K8s 集群搭建", score=0.9)]
        out = router._mesa_synthesis(seeds, "K8s 集群网络排障", "K8s 集群网络排障")
        assert all(r["level"] != "mesa_synthesis" for r in out)
        assert len(out) == len(seeds)

    def test_graphlite_exception_silent_degrade(self):
        """GraphLite 异常 → 静默降级返回原 results（主检索零回归）。"""
        store = MagicMock()
        store.get_communities_by_seeds.side_effect = RuntimeError("graphlite down")
        router = _make_router(store, [], mesa_enabled=True)
        seeds = [_seed_result("ep1", "content 1", score=0.9)]
        out = router._mesa_synthesis(seeds, "some query", "some query")
        assert out is seeds
        assert all(r["level"] != "mesa_synthesis" for r in out)

    def test_boost_runtime_clamp_below_community(self):
        """P2-3：mesa_boost=0.7 配置时运行时 clamp 到 community_boost*0.95=0.57，
        合成节点严格低于社区成员分（0.57 < 0.6）。"""
        store = MagicMock()
        store.get_communities_by_seeds.return_value = [
            _community_row("c1", "K8s 集群网络 flannel calico 排障")
        ]
        router = _make_router(store, [], mesa_enabled=True, mesa_boost=0.7)
        seeds = [_seed_result("ep1", "多会话记忆里 K8s 集群遇到 flannel 问题", score=0.9)]
        query = "K8s 集群网络排障"
        out = router._mesa_synthesis(seeds, query, query)
        syn = [r for r in out if r["level"] == "mesa_synthesis"]
        assert len(syn) == 1
        rel = router._community_relevance(
            query, ["K8s 集群网络 flannel calico 排障"])[0]
        # clamp = min(0.7, 0.6*0.95=0.57) = 0.57；社区成员分 = rel*0.9*0.6
        assert syn[0]["score"] == pytest.approx(round(rel * 0.9 * 0.57, 6))
        assert syn[0]["score"] < round(rel * 0.9 * 0.6, 6)

    def test_community_boost_lowered_synthesis_below_member(self, monkeypatch):
        """P2-3：community_expansion.boost 调低到 0.5 时，mesa 合成分仍低于社区成员
        （clamp=min(0.7, 0.5*0.95=0.475) < 0.5）。"""
        from config.settings import Settings, RetrievalConfig, CommunityExpansionConfig
        fake = Settings()
        fake.retrieval = RetrievalConfig(
            community_expansion=CommunityExpansionConfig(boost=0.5))
        monkeypatch.setattr("retrieval.query_router.get_settings", lambda: fake)
        store = MagicMock()
        store.get_communities_by_seeds.return_value = [
            _community_row("c1", "K8s 集群网络 flannel calico 排障")
        ]
        router = _make_router(store, [], mesa_enabled=True, mesa_boost=0.7)
        seeds = [_seed_result("ep1", "多会话记忆里 K8s 集群遇到 flannel 问题", score=0.9)]
        query = "K8s 集群网络排障"
        out = router._mesa_synthesis(seeds, query, query)
        syn = [r for r in out if r["level"] == "mesa_synthesis"]
        assert len(syn) == 1
        rel = router._community_relevance(
            query, ["K8s 集群网络 flannel calico 排障"])[0]
        assert syn[0]["score"] == pytest.approx(round(rel * 0.9 * 0.475, 6))
        assert syn[0]["score"] < round(rel * 0.9 * 0.5, 6)


class TestMesaIntegration:
    def test_retrieve_default_off_no_mesa_nodes(self, overgraph_store):
        """默认关零回归：mesa_enabled=False 走生产 retrieve，无 mesa 节点，种子原样。"""
        _insert_episode(overgraph_store, "ep1", "多会话记忆里 K8s 集群遇到 flannel 问题")
        _insert_episode(overgraph_store, "ep2", "另一会话用 calico 解决了 flannel 网络问题")
        _persist_community(overgraph_store, "comm_k8s", ["ep1", "ep2"],
                           "K8s 集群网络 flannel calico 多会话排障 记忆")
        router = _make_router(overgraph_store, [
            _seed_result("ep1", "多会话记忆里 K8s 集群遇到 flannel 问题", score=0.9),
        ], mesa_enabled=False)
        out = router.retrieve("多会话 K8s 集群 flannel 网络问题")
        assert all(r["level"] != "mesa_synthesis" for r in out)
        assert out[0]["node_id"] == "ep1"
        assert out[0]["score"] == 0.9  # 种子分未被 mesa 影响

    def test_score_math_mesa_below_member_below_seed(self, overgraph_store):
        """score 数学保证：合成节点 < 社区成员 < 种子（mesa_boost 0.4 < community 0.6 < 1）。"""
        _insert_episode(overgraph_store, "ep1", "多会话记忆里 K8s 集群遇到 flannel 问题")
        _insert_episode(overgraph_store, "ep2", "另一会话用 calico 解决了 flannel 网络问题")
        _persist_community(overgraph_store, "comm_k8s", ["ep1", "ep2"],
                           "K8s 集群网络 flannel calico 多会话排障 记忆")
        router = _make_router(overgraph_store, [
            _seed_result("ep1", "多会话记忆里 K8s 集群遇到 flannel 问题", score=0.9),
        ], mesa_enabled=True)
        out = router.retrieve("多会话 K8s 集群 flannel 网络问题")
        seeds = [r for r in out if r["level"] == "l1_faiss"]
        members = [r for r in out if r["level"] == "community_expansion"]
        mesa = [r for r in out if r["level"] == "mesa_synthesis"]
        assert seeds and members and mesa, \
            f"三种节点都应存在: seeds={seeds} members={members} mesa={mesa}"
        # 数学保证：合成节点(0.4) < 社区成员(0.6) < 种子(1.0)
        assert all(m["score"] < mb["score"] for m in mesa for mb in members), \
            f"合成节点应低于社区成员: {[m['score'] for m in mesa]} vs {[mb['score'] for mb in members]}"
        assert all(mb["score"] < s["score"] for mb in members for s in seeds), \
            f"社区成员应低于种子: {[mb['score'] for mb in members]} vs {[s['score'] for s in seeds]}"


class TestMesaAgenticPath:
    """P1-1：agentic 路径下 MESA 生效（_agentic_round 补 _mesa_synthesis）。"""

    def test_agentic_round_synthesizes_mesa_nodes(self):
        from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel

        store = MagicMock()
        store.get_communities_by_seeds.return_value = [
            _community_row("c1", "K8s 集群网络 flannel calico 排障")
        ]
        router = QueryRouter.__new__(QueryRouter)
        router.config = QueryRouterConfig(
            agentic_enabled=True, agentic_max_steps=1,
            mesa_enabled=True, mesa_boost=0.4, mesa_threshold=0.5, mesa_max_nodes=5,
        )
        router._zh_en_tech_map = {}
        router._time_keywords = set()
        router._cjk_warned = False
        router.graphlite_store = store
        router._episode_cache = {}
        router._services = None
        router.faiss_index = None
        router.faiss_id_map = {}
        router.encoder = None
        # 基础/补充通道 mock 恒等，仅 _mesa_synthesis 走真实实现
        router._fusion_retrieve = MagicMock(
            return_value=[_seed_result("ep1", "多会话记忆里 K8s 集群遇到 flannel 问题", score=0.9)]
        )
        router._community_expansion = MagicMock(side_effect=lambda r, q, qe, rq: r)
        router._visual_recall = MagicMock(side_effect=lambda r, q, rq: r)
        router._property_temporal_retrieve = MagicMock(
            side_effect=lambda r, q, rq, now_ts=None, at_ts=None: r
        )
        router._hypergraph_supplement = MagicMock(side_effect=lambda r: r)

        out = router.retrieve(
            "多会话 K8s 集群 flannel 网络问题", level=RetrievalLevel.FUSION,
        )
        mesa = [r for r in out if r.get("level") == "mesa_synthesis"]
        assert mesa, "agentic 路径应合成 MESA 节点（P1-1）"
        assert mesa[0]["node_id"] == "c1"
        assert mesa[0]["_source"] == "mesa"


class TestMesaConfigPassthrough:
    """P1-2：api/app.py 透传 mesa 配置——and/or 不吞合法零值。"""

    def test_zero_values_not_swallowed(self):
        from api.app import _build_router_config

        class FakeMesa:
            enabled = False
            boost = 0.0
            threshold = 0.0
            max_nodes = 0

        class FakeRcfg:
            tau_weight = 0.4
            vector_weight = 0.6
            top_k_l1 = 5
            top_k_vector = 20
            top_k_keyword = 20
            mesa = FakeMesa()

        qcfg = _build_router_config(FakeRcfg())
        assert qcfg.mesa_boost == 0.0, "boost=0.0 不应被 or 吞成 0.4"
        assert qcfg.mesa_threshold == 0.0
        assert qcfg.mesa_max_nodes == 0
        assert qcfg.mesa_enabled is False

    def test_missing_mesa_defaults(self):
        from api.app import _build_router_config

        class NoMesa:
            tau_weight = 0.4
            vector_weight = 0.6
            top_k_l1 = 5
            top_k_vector = 20
            top_k_keyword = 20

        qcfg = _build_router_config(NoMesa())
        assert qcfg.mesa_boost == 0.4
        assert qcfg.mesa_threshold == 0.5
        assert qcfg.mesa_max_nodes == 5
        assert qcfg.mesa_enabled is False
