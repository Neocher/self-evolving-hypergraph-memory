"""达摩院 P0-a 引擎会话作用域 — QueryRouter.retrieve(scope=) 集成测试
===================================================================
会话作用域下沉引擎 (round3 研究 §5 + §11 红线): QueryRouter 原生支持按
conversation_idx (灌库 episode.session_id 同坐标) 限定检索, 过滤点在引擎
统一出口 (_finish / agentic 每轮), 不在评测脚本。

覆盖验收:
  AC1 (引擎): retrieve(scope=N) 返回结果 100% session_id==N;
              scope=None 行为与不带 scope 全库逐字节等价 (回归基线)。
  AC2 (引擎): scope 过滤 (FUSION 真实融合) + 会话内 hit@k 不降 +
              _agentic round2 追加检索带 scope (mock fusion 验证 pool_k 透传
              与轮内过滤)。
  - scope 支持 int 与数字字符串 (官方 conversation_idx 形态, _norm_session)。
  - 会话路由只消费节点 session 归属, 不读题面/evidence (§11 红线 7)。

运行: python -m pytest tests/test_query_router_scope.py -v
"""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel


# ─── 假件（镜像 test_fusion_channel_topk，episode 带 session_id 归属）───────

class _FakeFaiss:
    """固定 (distances, indices)；距离随 index 递增 → score 递减，顺序确定。"""

    def __init__(self, n_items: int):
        self._distances = np.linspace(0.05, 3.0, n_items).reshape(1, n_items).astype(np.float32)
        self._indices = np.arange(n_items).reshape(1, n_items)
        self.search_k_history: list[int] = []

    def search(self, query, k):
        self.search_k_history.append(int(k))
        return self._distances[:, :k], self._indices[:, :k]


class _FakeStore:
    """仅提供 get_episodes_batch + 空 query_cypher 的假 OverGraphStore。"""

    def __init__(self, episodes: list[dict]):
        self._episodes = episodes

    def get_episodes_batch(self, node_ids):
        return [e for e in self._episodes if e.get("id") in set(node_ids)]

    def query_cypher(self, *args, **kwargs):
        return []


def _contents(n: int) -> list[str]:
    """每条内容彼此高区分（防近重复去重折叠），含稳定可检索词。"""
    import random as _r

    vocab = ("apple banana cherry dragon elephant forest grape hotel island jungle "
             "kitchen lemon mountain notebook ocean planet quartz rocket silver "
             "tiger umbrella valley window yellow zebra anchor bridge candle "
             "diamond engine factory garden hammer igloo jacket kettle ladder "
             "mirror needle olive pillow quilt ribbon scissors tulip uranium "
             "vessel walrus xylophone yarn zephyr").split()
    out = []
    for i in range(n):
        rng = _r.Random(900 + i)
        out.append(" ".join(rng.choice(vocab) for _ in range(16)) + f" sess msg{i}")
    return out


def _make_scoped_router(
    n_items: int = 40,
    session_of=None,
    **cfg_kwargs,
) -> tuple[QueryRouter, _FakeFaiss]:
    """零依赖 QueryRouter：episode 会话归属 = ep_i → session_of(ep_i)。

    session_of: callable(int idx) -> session_id | None。缺省 = idx // 20（前 20 条
    会话 0，后 20 条会话 1），构造双会话小语料。
    """
    session_of = session_of or (lambda i: i // 20)
    contents = _contents(n_items)
    episodes = [{"id": f"ep_{i}", "content": contents[i], "tau_initial": 1.0,
                 "archived": False, "fact_track": "active",
                 "session_id": session_of(i)} for i in range(n_items)]
    faiss = _FakeFaiss(n_items)
    cfg = QueryRouterConfig(rerank_enabled=False, fusion_dedup_enabled=False, **cfg_kwargs)
    qr = QueryRouter(
        graphlite_store=_FakeStore(episodes),
        faiss_index=faiss,
        tfidf_index=None,
        faiss_id_map={i: f"ep_{i}" for i in range(n_items)},
        episode_cache={},
        session_index={f"ep_{i}": session_of(i) for i in range(n_items)},
        config=cfg,
    )
    qr._build_bm25_index = lambda: None
    stack = ExitStack()
    for name in (
        "_community_expansion", "_mesa_synthesis", "_visual_recall",
        "_property_temporal_retrieve", "_entity_expansion", "_attribute_expansion",
        "_scope_retrieve", "_schema_recall", "_fact_retrieve",
    ):
        stack.enter_context(patch.object(qr, name, side_effect=lambda results, *a, **k: results))
    return qr, faiss


def _session_of_result(r: dict) -> int | None:
    nid = r.get("node_id", "")
    try:
        return int(nid.split("_")[1]) // 20
    except Exception:
        return None


# ─── AC1: scope=None 全库零回归 + scope=N 100% 会话内 ─────────────

def test_scope_none_equals_no_scope_full_library():
    """AC1 回归: scope=None 与不带 scope 调用逐字节等价（全库返回全部 40 条）。"""
    qr, faiss = _make_scoped_router(n_items=40)
    baseline = qr.retrieve("sess msg", query_embedding=np.zeros((1, 512)),
                           level=RetrievalLevel.FUSION)
    scoped_none = qr.retrieve("sess msg", query_embedding=np.zeros((1, 512)),
                              level=RetrievalLevel.FUSION, scope=None)
    assert scoped_none == baseline, "scope=None 必须与不带 scope 完全一致"
    assert len(baseline) == 40, f"全库应返回 40 条, 实际 {len(baseline)}"
    assert {_session_of_result(r) for r in baseline} == {0, 1}


def test_scope_n_returns_only_session_n_episodes():
    """AC1: retrieve(scope=N) 返回 100% session_id==N（FUSION 真实融合 + 引擎出口过滤）。"""
    qr, faiss = _make_scoped_router(n_items=40)
    out0 = qr.retrieve("sess msg", query_embedding=np.zeros((1, 512)),
                       level=RetrievalLevel.FUSION, scope=0)
    assert out0, "会话 0 应有结果"
    assert all(_session_of_result(r) == 0 for r in out0), \
        f"scope=0 结果含异会话: {[r['node_id'] for r in out0]}"
    assert len(out0) == 20, f"会话 0 应返回 20 条, 实际 {len(out0)}"

    out1 = qr.retrieve("sess msg", query_embedding=np.zeros((1, 512)),
                       level=RetrievalLevel.FUSION, scope=1)
    assert all(_session_of_result(r) == 1 for r in out1), \
        f"scope=1 结果含异会话: {[r['node_id'] for r in out1]}"
    assert len(out1) == 20


def test_scope_accepts_digit_string_conversation_idx():
    """scope 支持数字字符串（官方 conversation_idx 形态, _norm_session 归一）。"""
    qr, _ = _make_scoped_router(n_items=40)
    out_int = qr.retrieve("sess msg", query_embedding=np.zeros((1, 512)),
                          level=RetrievalLevel.FUSION, scope=1)
    out_str = qr.retrieve("sess msg", query_embedding=np.zeros((1, 512)),
                          level=RetrievalLevel.FUSION, scope="1")
    assert out_str == out_int, "scope='1' 与 scope=1 应等价"


def test_scope_filter_keeps_in_session_recall():
    """AC2: scope 过滤后会话内证据不丢（小语料集成）——pool 加深后过滤,
    全库可见的本会话证据在 scoped 结果中仍全部在场 (hit@k 会话内不降)。"""
    # 40 条双会话 (ep_0..19 = 会话 0, ep_20..39 = 会话 1)
    # FUSION 向量通道默认深度 100 ≥ 40 → 全库两会话候选全部可见
    qr, _ = _make_scoped_router(n_items=40)
    full = qr.retrieve("sess msg", query_embedding=np.zeros((1, 512)),
                       level=RetrievalLevel.FUSION)
    assert len(full) == 40, f"全库应全量可见, 实际 {len(full)}"
    in_sess0_full = {r["node_id"] for r in full if _session_of_result(r) == 0}
    assert len(in_sess0_full) == 20, "会话 0 全部 20 条应进入全库候选"
    scoped0 = qr.retrieve("sess msg", query_embedding=np.zeros((1, 512)),
                          level=RetrievalLevel.FUSION, scope=0)
    scoped0_ids = {r["node_id"] for r in scoped0}
    # 会话内 hit@k 不降: 全库可见的会话 0 证据全部仍在 scoped 结果
    assert in_sess0_full <= scoped0_ids, \
        f"scope 后丢会话内证据: {in_sess0_full - scoped0_ids}"
    assert len(scoped0) == 20


# ─── AC2: agentic round2 追加检索带 scope ──────────────────────

def _make_agentic_scope_router(fusion_rounds: list[list[dict]], **cfg) -> QueryRouter:
    """真实编排器 + mock fusion（可指定多轮结果），带 node→session 归属索引。

    fusion_rounds[k] = 第 k 次 _fusion_retrieve 调用返回的 episode 列表；
    node_id 形如 ep_<i>，归属按 _SESSION_OF 判定（i<20 → 会话 0）。
    """
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig(
        agentic_enabled=True, rerank_enabled=False,
        fusion_dedup_enabled=False, **cfg,
    )
    router._zh_en_tech_map = {}
    router._time_keywords = set()
    router.graph_store = None
    router._cjk_warned = False
    router._episode_cache = {}
    router._session_index = {f"ep_{i}": 0 if i < 20 else 1 for i in range(40)}
    router._services = None
    router.faiss_index = None
    router.faiss_id_map = {}
    router.encoder = None
    calls: list[dict] = []

    def _fusion(q, qe=None, rq=None, now_ts=None, pool_k=None, **kw):
        calls.append({"pool_k": pool_k})
        idx = len(calls) - 1
        if idx < len(fusion_rounds):
            return [dict(r) for r in fusion_rounds[idx]]
        return []

    router._fusion_retrieve = MagicMock(side_effect=_fusion)
    router._fusion_calls = calls
    router._community_expansion = MagicMock(side_effect=lambda r, q, qe, rq: r)
    router._visual_recall = MagicMock(side_effect=lambda r, q, rq: r)
    router._property_temporal_retrieve = MagicMock(
        side_effect=lambda r, q, rq, now_ts=None, at_ts=None: r)
    router._hypergraph_supplement = MagicMock(side_effect=lambda r: r)
    router._mesa_synthesis = MagicMock(side_effect=lambda r, q, rq: r)
    router._entity_expansion = MagicMock(side_effect=lambda r, q, rq, now_ts=None: r)
    router._attribute_expansion = MagicMock(side_effect=lambda r, q, rq, now_ts=None: r)
    router._schema_recall = MagicMock(side_effect=lambda r, q: r)
    router._fact_retrieve = MagicMock(side_effect=lambda r, q, rq, now_ts=None: r)
    return router


def _ep_session(node_id: str) -> int:
    return 0 if int(node_id.split("_")[1]) < 20 else 1


def test_agentic_rounds_respect_scope():
    """AC2: agentic 追加检索（多轮）带 scope — 每轮 fusion 传 pool_k,
    轮内/合并结果 100% session_id==N。"""
    # 首轮: 会话 0 + 会话 1 混杂（触发追加轮）; round2: 补更多会话 1 证据
    rounds = [
        [{"node_id": "ep_0", "content": "apple revenue 10", "score": 0.5},
         {"node_id": "ep_25", "content": "apple revenue 99", "score": 0.9}],
        [{"node_id": "ep_30", "content": "apple growth 88", "score": 0.8}],
    ]
    router = _make_agentic_scope_router(rounds, agentic_max_steps=2, agentic_min_new=1,
                                        agentic_top_k=12)
    out = router.retrieve("apple revenue", level=RetrievalLevel.FUSION, scope=1)
    node_ids = [r["node_id"] for r in out]
    assert node_ids, "scoped agentic 应有结果"
    assert all(_ep_session(nid) == 1 for nid in node_ids), \
        f"agentic scope=1 混入会话 0: {node_ids}"
    # round2 追加检索确实发生（≥2 次 fusion 调用）
    assert router._fusion_retrieve.call_count >= 2, \
        f"应触发追加检索, 实际 {router._fusion_retrieve.call_count} 次 fusion"
    # 每轮 fusion 都带 pool_k（scope 非 None → 候选池加深透传）
    assert all(c["pool_k"] is not None for c in router._fusion_calls), \
        f"scoped agentic 每轮 fusion 应带 pool_k: {router._fusion_calls}"


def test_agentic_scope_none_equals_unscoped_baseline():
    """AC2 回归: agentic scope=None 与不带 scope 调用一致（不传 pool_k）。"""
    rounds = [
        [{"node_id": "ep_0", "content": "apple revenue 10", "score": 0.5},
         {"node_id": "ep_25", "content": "apple revenue 99", "score": 0.9}],
    ]
    router = _make_agentic_scope_router(rounds, agentic_max_steps=1)
    base = router.retrieve("apple revenue", level=RetrievalLevel.FUSION)
    router2 = _make_agentic_scope_router(rounds, agentic_max_steps=1)
    none_scoped = router2.retrieve("apple revenue", level=RetrievalLevel.FUSION, scope=None)
    assert none_scoped == base, "agentic scope=None 应与不带 scope 一致"
    assert all(c["pool_k"] is None for c in router2._fusion_calls), \
        f"scope=None 不应传 pool_k: {router2._fusion_calls}"
