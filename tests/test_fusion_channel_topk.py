"""P0 达摩院收敛 — FUSION 通道深度参数化 + 跨通道近重复去重回归测试
===================================================================
覆盖验收点：
  ① FUSION 每通道候选深度独立参数化（fusion_vector/bm25/entity_topk）——
     rank 21+（超过级联 top_k_vector）证据经公共入口 retrieve(FUSION) 进入
     融合可见集；级联 L2 VECTOR 路径仍走 top_k_vector 原值（零回归）
  ② 扩池后跨通道近重复去重（fusion_dedup_*）：近重复项不把有效证据挤出 top-40
  ③ 配置 fail-fast：非法深度/阈值在配置期拒绝（禁静默失败）
  ④ 默认配置即触发修复后路径（fusion_vector_topk=100 等）

运行: python -m pytest tests/test_fusion_channel_topk.py -v
"""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel


# ─── 假件（镜像 test_core_boost_multisource 风格）───────────────

class _FakeFaiss:
    """固定 (distances, indices)；search_k 历史供断言通道深度。"""

    def __init__(self, n_items: int):
        # 距离随 index 递增：rank i 的 score = 1/(1+dist) 递减，顺序确定
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
    """每条内容彼此高区分（bigram Jaccard << 阈值），避免被近重复去重折叠。"""
    import random as _r

    vocab = ("apple banana cherry dragon elephant forest grape hotel island jungle "
             "kitchen lemon mountain notebook ocean planet quartz rocket silver "
             "tiger umbrella valley window yellow zebra anchor bridge candle "
             "diamond engine factory garden hammer igloo jacket kettle ladder "
             "mirror needle olive pillow quilt ribbon scissors tulip uranium "
             "vessel walrus xylophone yarn zephyr almond bronze copper desert "
             "emerald fountain glacier harbor ivory jasmine kangaroo lantern "
             "marble nickel opal pepper rocket sphinx temple umbrella vortex "
             "waffle xerox yucca zircon autumn beacon canvas daffodil eclipse "
             "falcon galaxy horizon icicle journey kiwi lagoon meadow nectar "
             "orchid prairie quiver sunrise twilight underpass valley whisper").split()
    out = []
    for i in range(n):
        rng = _r.Random(1000 + i)
        out.append(" ".join(rng.choice(vocab) for _ in range(18)) + f" msg{i}")
    return out


def _make_router(n_items: int = 60, contents: list[str] | None = None,
                 **cfg_kwargs) -> tuple[QueryRouter, _FakeFaiss]:
    """零依赖 QueryRouter：faiss 深度可控、关 rerank/增强通道，结果确定。"""
    contents = contents or _contents(n_items)
    episodes = [{"id": f"ep_{i}", "content": contents[i], "tau_initial": 1.0,
                 "archived": False, "fact_track": "active"} for i in range(n_items)]
    faiss = _FakeFaiss(n_items)
    cfg = QueryRouterConfig(rerank_enabled=False, **cfg_kwargs)
    qr = QueryRouter(
        graphlite_store=_FakeStore(episodes),
        faiss_index=faiss,
        tfidf_index=None,
        faiss_id_map={i: f"ep_{i}" for i in range(n_items)},
        episode_cache={},
        config=cfg,
    )
    # 关掉无关通道：bm25 索引不构建（不可用即空）；增强通道透传
    qr._build_bm25_index = lambda: None
    stack = ExitStack()
    for name in (
        "_community_expansion", "_mesa_synthesis", "_visual_recall",
        "_property_temporal_retrieve", "_entity_expansion", "_attribute_expansion",
        "_scope_retrieve", "_schema_recall", "_fact_retrieve",
    ):
        stack.enter_context(patch.object(qr, name, side_effect=lambda results, *a, **k: results))
    return qr, faiss


# ─── ① FUSION 通道深度参数化 ──────────────────────────────────

def test_fusion_vector_depth_beyond_top_k_vector_recovers_rank30():
    """① rank 21-40 证据（top_k_vector=20 截断流失段）经公共入口 FUSION 召回。"""
    qr, faiss = _make_router(
        n_items=60, fusion_vector_topk=60, top_k_vector=20,
    )
    out = qr.retrieve("hiking plan", query_embedding=np.zeros((1, 512)),
                      level=RetrievalLevel.FUSION)
    node_ids = [r["node_id"] for r in out]
    assert "ep_29" in node_ids, "vector rank 30 证据须进入 FUSION 可见集"
    assert node_ids.index("ep_29") < 40, "rank 30 证据须落在 top-40 内"
    assert len(node_ids) == 60, "扩池后 60 个可见候选全部进入融合结果"


def test_cascade_vector_path_keeps_top_k_vector_zero_regression():
    """级联 L2 VECTOR 路径仍按 top_k_vector 截断（零回归，深度解耦）。"""
    qr, faiss = _make_router(
        n_items=60, fusion_vector_topk=60, top_k_vector=20,
    )
    out = qr.retrieve("hiking plan", query_embedding=np.zeros((1, 512)),
                      level=RetrievalLevel.VECTOR)
    assert faiss.search_k_history == [20], faiss.search_k_history
    assert [r["node_id"] for r in out] == [f"ep_{i}" for i in range(20)]


def test_fusion_uses_dedicated_channel_depth_default_100():
    """默认配置即触发修复后路径：FUSION 向量通道 k=100，级联 L2 k=top_k_vector=20。"""
    qr, faiss = _make_router(n_items=100)  # QueryRouterConfig() 默认
    assert qr.config.fusion_vector_topk == 100
    qr.retrieve("hiking plan", query_embedding=np.zeros((1, 512)),
                level=RetrievalLevel.FUSION)
    qr.retrieve("hiking plan", query_embedding=np.zeros((1, 512)),
                level=RetrievalLevel.VECTOR)
    assert faiss.search_k_history == [100, 20], faiss.search_k_history


def test_fusion_per_channel_depth_dispatch():
    """每通道深度独立传参：vector/bm25/entity 分别读 fusion_*_topk。"""
    qr, faiss = _make_router(n_items=10)
    qr.config.fusion_vector_topk = 33
    qr.config.fusion_bm25_topk = 17
    qr.config.fusion_entity_topk = 9
    calls = {}
    with patch.object(qr, "_vector_retrieve",
                      side_effect=lambda q, emb, k=None: calls.update(vk=k) or []) \
            as m_v, \
         patch.object(qr, "_bm25_search",
                      side_effect=lambda q, k=20: calls.update(bk=k) or []) as m_b, \
         patch.object(qr, "_entity_match",
                      side_effect=lambda q, k=20: calls.update(ek=k) or []) as m_e:
        qr.retrieve("hiking plan", query_embedding=np.zeros((1, 512)),
                    level=RetrievalLevel.FUSION)
    assert calls["vk"] == 33 and calls["bk"] == 17 and calls["ek"] == 9, calls
    # 级联路径不改动：_vector_retrieve 不带 k → 内部回落 top_k_vector
    assert m_v.call_count == 1 and m_b.call_count == 1 and m_e.call_count == 1


# ─── ② 扩池后跨通道近重复去重 ────────────────────────────────

def _near_dup_texts() -> list[str]:
    base = ("melanie went to the lgbtq support group meeting last tuesday "
            "evening and stayed late to talk")
    # 尾缀变体：字符 bigram 几乎全同（Jaccard ~0.99 ≥ 0.9），构造稳定近重复对
    return [base, base + "!"]


def test_near_dup_dedup_via_public_fusion():
    """② 近重复文本经公共入口 FUSION 只保留融合分最高代表项（默认开）。"""
    texts = _near_dup_texts() + _contents(1)[:1]  # 近重复对 + 无关项
    # 6 条：ep_0/ep_1 近重复（高分在前），其余唯一
    pool = texts + ["unique message " + str(i) for i in range(4)]
    qr, _ = _make_router(n_items=len(pool), contents=pool)
    out = qr.retrieve("melanie", query_embedding=np.zeros((1, 512)),
                      level=RetrievalLevel.FUSION)
    contents = [r["content"] for r in out]
    # 近重复对只保留一条（更高分的 ep_0 代表）
    kept_pair = [c for c in contents if "lgbtq support group meeting" in c]
    assert len(kept_pair) == 1, kept_pair
    assert kept_pair[0] == pool[0]


def test_near_dup_dedup_flag_off_keeps_both():
    """② fusion_dedup_enabled=False → 近重复项原样保留（开关语义）。"""
    pool = _near_dup_texts() + ["unique message x", "unique message y"]
    qr, _ = _make_router(n_items=len(pool), contents=pool,
                         fusion_dedup_enabled=False)
    out = qr.retrieve("melanie", query_embedding=np.zeros((1, 512)),
                      level=RetrievalLevel.FUSION)
    contents = [r["content"] for r in out]
    kept_pair = [c for c in contents if "lgbtq support group meeting" in c]
    assert len(kept_pair) == 2, kept_pair


def test_dedup_unit_threshold_and_short_text_guard():
    """② _dedup_near_duplicates 单元：短文本不参与（防误杀）、阈值以下保留。"""
    qr, _ = _make_router(n_items=2)
    pool = [
        {"node_id": "a", "content": _near_dup_texts()[0], "score": 0.9},
        {"node_id": "b", "content": _near_dup_texts()[1], "score": 0.8},
    ]
    qr.config.fusion_dedup_min_len = 40
    assert len(qr._dedup_near_duplicates([dict(r) for r in pool])) == 1

    short = [
        {"node_id": "s1", "content": "yes sure ok great", "score": 0.9},
        {"node_id": "s2", "content": "yes sure ok great!", "score": 0.8},
    ]
    assert len(qr._dedup_near_duplicates(short)) == 2  # 短文本跳过

    distinct = [
        {"node_id": "d1", "content": "the team planned a hiking trip last saturday "
         "and checked the weather forecast", "score": 0.9},
        {"node_id": "d2", "content": "melanie painted a beautiful sunrise landscape "
         "with oil colors on canvas", "score": 0.8},
    ]
    assert len(qr._dedup_near_duplicates(distinct)) == 2  # 阈值以下保留


def test_dedup_degraded_returns_original_on_error():
    """② 去重异常 → 原列表返回（零回归，不静默丢结果）。"""
    qr, _ = _make_router(n_items=2)
    pool = [{"node_id": "a", "content": "some content here", "score": 0.9}]
    with patch.object(qr, "config") as bad_cfg:
        bad_cfg.fusion_dedup_threshold = None
        bad_cfg.fusion_dedup_min_len = None
        assert qr._dedup_near_duplicates(pool) == pool


# ─── ③ 配置 fail-fast（禁静默失败） ───────────────────────────

@pytest.mark.parametrize("bad", [
    {"fusion_vector_topk": 0},
    {"fusion_bm25_topk": -3},
    {"fusion_entity_topk": 0},
    {"fusion_dedup_threshold": 1.5},
    {"fusion_dedup_threshold": 0.0},
    {"fusion_dedup_min_len": 0},
])
def test_invalid_config_fails_fast(bad):
    """③ 非法通道深度/去重阈值配置期 ValueError（禁静默降级为空/失效）。"""
    with pytest.raises(ValueError):
        QueryRouterConfig(**bad)


def test_default_config_paths_active():
    """④ 默认 QueryRouterConfig 即修复后路径（harness 用默认配置触发）。"""
    cfg = QueryRouterConfig()
    assert (cfg.fusion_vector_topk, cfg.fusion_bm25_topk, cfg.fusion_entity_topk) \
        == (100, 100, 100)
    assert cfg.fusion_dedup_enabled is True
    assert 0.0 < cfg.fusion_dedup_threshold < 1.0
    # 级联原值不动
    assert cfg.top_k_vector == 20 and cfg.top_k_keyword == 20 and cfg.top_k_l1 == 5
