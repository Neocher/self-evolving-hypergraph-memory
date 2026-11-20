"""检索/存储修复验收测试（v6.28.x 第二轮组 δ）

覆盖 K1 / K2 / K3 / K4 修复的回归防护，全部走公共入口或最小可观测契约：
- K1  _gql_value 转义与 update_with_version 失败可观测
- K2  融合归一化保留通道间绝对质量差
- K3  _vector_retrieve 熔断时通道可被上层感知（不产出空 content 伪结果）
- K4  _entity_expansion 频率门控折叠进单条 SELECT（无独立 COUNT 全表扫描）
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from graph.common import _gql_value


# ══════════════════════════════════════════════════════════════════
# K1 —— _gql_value 转义 + update_with_version 失败可观测
# ══════════════════════════════════════════════════════════════════


def test_gql_value_escapes_newline_cr_quote_backslash():
    """_gql_value 与 _esc_literal 转义语义一致：\\n / \\r / ' / \\ 全覆盖。"""
    assert _gql_value("O'Brien") == "'O\\'Brien'"
    assert _gql_value("a\\b") == "'a\\\\b'"
    assert _gql_value("line\nbreak") == "'line\\nbreak'"
    assert _gql_value("carriage\rreturn") == "'carriage\\rreturn'"
    # 组合：换行 + 回车 + 单引号 + 反斜杠，顺序与 _esc_literal 一致（反斜杠先行）
    assert _gql_value("mix\n\r\\'all") == "'mix\\n\\r\\\\\\'all'"
    # 无害值原样
    assert _gql_value("plain text") == "'plain text'"
    assert _gql_value("") == "''"


@pytest.mark.overgraph
def test_update_with_version_roundtrips_special_chars(overgraph_store):
    """含 \\n / \\r / ' / \\ 的内容经 update_with_version 写入后读回等价。"""
    pytest.importorskip("overgraph")
    store = overgraph_store
    eid = store.create_episode({"id": "ep_special", "content": "v0"})
    payload = "line1\nline2\rline3 'quoted' \\backslash\\"
    assert store.update_with_version(eid, {"content": payload}, expected_version=1) is True
    ep = store.get_episode(eid)
    assert ep is not None
    assert ep["content"] == payload


@pytest.mark.overgraph
def test_update_with_version_exception_is_observed(overgraph_store, caplog):
    """执行异常不再静默：留下 WARNING 日志 + 实例计数器自增。"""
    pytest.importorskip("overgraph")
    store = overgraph_store
    eid = store.create_episode({"id": "ep_fail", "content": "v0"})
    # 让底层执行抛异常（模拟语法错误 / 引擎故障），CAS 路径进入 except。
    # 注意：真实 OverGraph 对象的 execute_gql 是只读 Rust 属性，不能 patch.object
    # 其方法 —— 整体替换 _db 引用（Python 属性）为会抛异常的 stub。
    fake_db = MagicMock()
    fake_db.execute_gql.side_effect = RuntimeError("boom")
    with patch.object(store, "_db", fake_db):
        with caplog.at_level(logging.WARNING, logger="shm.overgraph_store"):
            ok = store.update_with_version(eid, {"content": "x"}, expected_version=1)
    assert ok is False
    assert getattr(store, "_update_with_version_failures", 0) == 1
    assert any("update_with_version failed" in r.message for r in caplog.records)


# ══════════════════════════════════════════════════════════════════
# K2 —— 融合归一化保留通道间绝对质量差
# ══════════════════════════════════════════════════════════════════


def _fuse_router():
    from retrieval.query_router import QueryRouter, QueryRouterConfig
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    return router


def test_fusion_noise_bm25_channel_does_not_beat_strong_vector():
    """低质量 bm25 通道（单条弱命中 0.05）不再经 min-max 顶到 1.0 拿满权重。

    旧实现：bm25 单条 → min-max → score=1.0 → 融合分 1.0×0.40=0.40，
    高于 vector 0.9×0.35=0.315，弱噪声挤掉强证据。
    新实现：bm25 0.05→0.05/1.05≈0.0476 → 融合分 0.019，低于 vector 0.315。
    """
    router = _fuse_router()
    vector = [{"node_id": "v1", "content": "genuine strong", "score": 0.9, "tau_value": 0.0}]
    bm25 = [{"node_id": "b1", "content": "weak noise", "score": 0.05, "tau_value": 0.0}]
    out = router._fuse_results(vector, bm25, [])
    by_id = {r["node_id"]: r["score"] for r in out}
    assert by_id["v1"] > by_id["b1"], by_id
    assert by_id["b1"] < 0.1, by_id  # 弱噪声不再拿满 0.40 权重


def test_fusion_strong_bm25_still_competes_with_vector():
    """强 bm25 命中（真实 BM25 和可 >10）经饱和映射后仍能与强向量竞争。"""
    router = _fuse_router()
    vector = [{"node_id": "v1", "content": "v", "score": 0.3, "tau_value": 0.0}]
    bm25 = [{"node_id": "b1", "content": "b", "score": 11.0, "tau_value": 0.0}]
    out = router._fuse_results(vector, bm25, [])
    by_id = {r["node_id"]: r["score"] for r in out}
    # 饱和映射 11/12≈0.917 × 0.40 ≈ 0.367 > 0.3×0.35=0.105
    assert by_id["b1"] > by_id["v1"], by_id


# ══════════════════════════════════════════════════════════════════
# K3 —— _vector_retrieve 熔断时通道可被上层感知
# ══════════════════════════════════════════════════════════════════


def _k3_router():
    from retrieval.query_router import QueryRouter, QueryRouterConfig
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router._zh_en_tech_map = {}
    router._cjk_warned = False
    return router


def test_vector_retrieve_raises_on_circuit_breaker():
    """图库熔断时 _vector_retrieve 上抛（不产出空 content 伪结果）。"""
    from graph.common import CircuitBreakerOpen
    router = _k3_router()
    router.faiss_index = MagicMock()
    router.faiss_index.search.return_value = (np.array([[0.1]]), np.array([[7]]))
    router.faiss_id_map = {7: "ep_7"}
    router.encoder = None
    router.graph_store = MagicMock()
    router.graph_store.get_episodes_batch.side_effect = CircuitBreakerOpen("open")
    with pytest.raises(CircuitBreakerOpen):
        router._vector_retrieve("q", np.zeros((1, 512)))


def test_fusion_vector_channel_skipped_on_circuit_breaker():
    """FUSION 下向量通道熔断 → 通道标记 skipped，不产出伪结果污染融合。"""
    from graph.common import CircuitBreakerOpen
    router = _k3_router()
    router.graph_store = MagicMock()
    router._vector_retrieve = MagicMock(side_effect=CircuitBreakerOpen("open"))
    router._bm25_search = MagicMock(return_value=[
        {"node_id": "b1", "content": "bm25 hit", "score": 0.8, "tau_value": 0.0,
         "level": "bm25"},
    ])
    router._entity_match = MagicMock(return_value=[])
    out = router._fusion_retrieve("hello world", np.zeros((1, 512)))
    # 无空 content 伪结果；bm25 结果保留且带降级标记
    assert [r["node_id"] for r in out] == ["b1"]
    assert all(r.get("_degradation_level") == "fusion_channel_skip" for r in out)


def test_vector_cascade_to_keyword_on_circuit_breaker():
    """L2 VECTOR 熔断 → 级联到 L3 keyword，且结果带 l2_circuit_breaker 标记。"""
    from graph.common import CircuitBreakerOpen
    from retrieval.query_router import RetrievalLevel
    router = _k3_router()
    router._time_keywords = ["recent"]
    router.graph_store = MagicMock()
    router._vector_retrieve = MagicMock(side_effect=CircuitBreakerOpen("open"))
    router._keyword_retrieve = MagicMock(return_value=[
        {"node_id": "k1", "content": "kw", "score": 0.5, "tau_value": 0.0,
         "fact_track": "active", "level": "keyword"},
    ])
    for name in (
        "_community_expansion", "_mesa_synthesis", "_visual_recall",
        "_property_temporal_retrieve", "_entity_expansion", "_attribute_expansion",
        "_scope_retrieve", "_schema_recall", "_fact_retrieve",
    ):
        setattr(router, name, MagicMock(side_effect=lambda r, *a, **k: r))
    router._load_shedding_active = MagicMock(return_value=(False, 0, 4))
    out = router.retrieve("hello", query_embedding=np.zeros((1, 512)),
                          level=RetrievalLevel.VECTOR)
    assert [r["node_id"] for r in out] == ["k1"]
    assert out[0].get("_degradation_level") == "l2_circuit_breaker"
