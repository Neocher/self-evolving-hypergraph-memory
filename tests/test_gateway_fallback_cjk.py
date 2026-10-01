"""v6.22.7 — GatewayAPI Cypher 兜底 CJK 分词 + 判别式打分测试。

P2 CJK 兜底修复：gateway_api.py 兜底段（query.split() 整句 CONTAINS + score: 0.5
硬编码）对中文查询近乎失效。本文件补 GatewayAPI 层 `_fallback_tokens` /
`_fallback_score` 纯函数单测 + 兜底 harness 探针（CJK 打分排序 + 英文零回归）。
"""

import pytest

from gateway.gateway_api import _fallback_tokens, _fallback_score


# ── 纯函数：_fallback_tokens ──────────────────────────────────────────────


def test_tokens_pure_cjk_2gram():
    """纯中文整句 → char 2-gram（与 BM25 char_wb 口径一致）。"""
    assert _fallback_tokens("心跳线程日志") == ["心跳", "跳线", "线程", "程日", "日志"]


def test_tokens_single_cjk_char():
    """段长 1 的 CJK 段保留单字（单字 CONTAINS 仍有意义）。"""
    assert _fallback_tokens("条") == ["条"]


def test_tokens_mixed_cjk_ascii():
    """中英混合：CJK 段 2-gram、非 CJK 段按词切、顺序保持。"""
    assert _fallback_tokens("心跳 hello world 日志") == ["心跳", "hello", "world", "日志"]


def test_tokens_punctuation_stripped():
    """非 CJK 词去首尾非字母数字。"""
    assert _fallback_tokens("hello!!") == ["hello"]
    assert _fallback_tokens("hello, world!") == ["hello", "world"]


def test_tokens_ascii_unchanged():
    """纯 ASCII/数字词行为不变（lower + 长度 ≥2）。"""
    assert _fallback_tokens("hello world 123") == ["hello", "world", "123"]
    assert _fallback_tokens("Hello World") == ["hello", "world"]


def test_tokens_dedup_preserves_order():
    """去重保序。"""
    assert _fallback_tokens("心跳 心跳 hello hello") == ["心跳", "hello"]


def test_tokens_capped_at_eight():
    """上限 8 个 token 截断保序（控查询计划，失效条件 1）。"""
    tokens = _fallback_tokens("一二三四五六七八九十")
    assert len(tokens) == 8
    assert tokens == ["一二", "二三", "三四", "四五", "五六", "六七", "七八", "八九"]


def test_tokens_empty_and_short():
    """空查询 / 纯空白 / 单字符非 CJK 词 → 空列表。"""
    assert _fallback_tokens("") == []
    assert _fallback_tokens("   ") == []
    assert _fallback_tokens("a") == []
    assert _fallback_tokens("1") == []


# ── 纯函数：_fallback_score ──────────────────────────────────────────────


def test_score_monotonic_in_hits():
    """hits 多 → score 高（严格单调）。"""
    tokens = ["心跳", "线程", "日志"]
    one = _fallback_score(tokens, "只有心跳这个词")
    three = _fallback_score(tokens, "心跳和线程和日志")
    assert three > one
    assert one == pytest.approx(0.05 + 0.45 * (1 / 3))
    assert three == pytest.approx(0.5)


def test_score_upper_bound_half():
    """全命中 score == 0.5；任何情况下永不越过 0.5。"""
    assert _fallback_score(["心跳"], "心跳") == pytest.approx(0.5)
    assert _fallback_score(["心跳", "线程", "日志"], "心跳线程日志相关") == pytest.approx(0.5)
    assert _fallback_score(["心跳", "线程", "日志"], "完全无关的内容") <= 0.5


def test_score_lower_bound_positive():
    """空 tokens 防御返回 0.05；有命中时 score > 0.05。"""
    assert _fallback_score([], "anything") == pytest.approx(0.05)
    assert _fallback_score(["心跳", "线程"], "心跳") > 0.05


def test_score_zero_hit_floor():
    """【审核建议 1】非空 tokens 但 0 hit → score 恰为噪声地板 0.05（闭下界），仍 > 0。

    规格判据"下界 >0"满足；score 区间实为 [0.05, 0.5]（新增能力"落在 (0.05,0.5]"为
    宽松措辞——0-hit 边界由 b64 遗留的"库侧 CONTAINS 命中但解码后无 token 子串"
    场景触达，给地板分而非丢弃）。
    """
    assert _fallback_score(["心跳", "线程", "日志"], "完全无关的内容") == pytest.approx(0.05)
    assert _fallback_score(["心跳", "线程"], "无关") > 0


def test_tokens_english_old_path_compat():
    """纯英文查询 token 序列与旧 split() 路径等价（英文零回归）。"""
    query = "hello world  test  "
    old = [w.strip().lower() for w in query.split() if len(w.strip()) > 1]
    assert _fallback_tokens(query) == old


# ── harness 探针：兜底打分排序 + 英文零回归 ────────────────────────────────


class _EmptyRouter:
    """FUSION 主通道返回空 → 触发 Cypher 兜底。"""

    def retrieve(self, query, include_archived=False, session_ts=None,
                 level=None, rerank=None):
        return []


class _FakeStore:
    """记录 query_cypher 的 cypher/params，并返回 canned 行。"""

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def query_cypher(self, cypher, params=None):
        self.calls.append({"cypher": cypher, "params": params})
        return self._rows


class _Svcs:
    def __init__(self, router, store):
        self.query_router = router
        self.graph_store = store


def _make_api(store):
    from gateway.gateway_api import GatewayAPI

    api = GatewayAPI.__new__(GatewayAPI)
    api._svc = _Svcs(_EmptyRouter(), store)
    api._logger = type("L", (), {
        "warning": lambda self, *a, **k: None,
        "exception": lambda self, *a, **k: None,
    })()
    return api


@pytest.mark.asyncio
async def test_retrieve_cjk_fallback_scored_sorted():
    """【探针门】CJK 兜底行 score ∈ (0.05, 0.5]、按命中降序、tie 用 created_at DESC、
    level=graphlite_fallback、degraded=True；含 created_at='Null' 防御解析。

    查询 "心跳线程日志" → tokens 5 个；乱序 canned 行：
      A(5 命中, ts=100) → score 0.5
      B(1 命中, ts=300)
      C(1 命中, ts=200)
      D(1 命中, created_at='Null' → 0.0)
    期望排序 A, B, C, D（score 降序；同分按 created_at DESC：300 > 200 > 0）。
    """
    store = _FakeStore([
        {"node_id": "B", "content": "线程 阻塞", "created_at": 300.0},
        {"node_id": "A", "content": "心跳线程日志 相关记录", "created_at": 100.0},
        {"node_id": "D", "content": "线程", "created_at": "Null"},
        {"node_id": "C", "content": "心跳", "created_at": 200.0},
    ])
    api = _make_api(store)
    resp = await api.retrieve("心跳线程日志")

    assert resp.degraded is True
    assert [r.node_id for r in resp.results] == ["A", "B", "C", "D"]
    assert resp.results[0].score == pytest.approx(0.5)
    for r in resp.results:
        assert 0.05 < r.score <= 0.5
        assert r.retrieval_level == "graphlite_fallback"
    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True), "兜底结果须按命中降序"


@pytest.mark.asyncio
async def test_retrieve_english_fallback_spy_compat():
    """【英文零回归】纯英文查询兜底 params 为英文 token（旧 split() 路径超集）。"""
    store = _FakeStore([
        {"node_id": "E", "content": "hello world", "created_at": 1.0},
    ])
    api = _make_api(store)
    await api.retrieve("hello world")
    assert store.calls, "英文查询应走 Cypher 兜底"
    params = store.calls[-1]["params"]
    assert params == {"w0": "hello", "w1": "world"}


@pytest.mark.asyncio
async def test_retrieve_fallback_created_at_real_store(overgraph_store):
    """【审核建议 2】真实 OverGraphStore：兜底 RETURN e.created_at 投影返回数值，
    打分 + created_at tie-break 排序在真实翻译链路生效（关闭覆盖空洞）。

    写 3 节点：e_high(5 命中, ts=100) / e_low(1 命中, ts=300) / e_mid(1 命中, ts=200)，
    期望兜底序 e_high → e_low → e_mid（score 降序，同分 created_at DESC）。
    """
    from gateway.gateway_api import GatewayAPI

    store = overgraph_store
    store.create_episode({"id": "e_low", "content": "线程 阻塞", "created_at": 300.0})
    store.create_episode({"id": "e_high", "content": "心跳线程日志 相关记录", "created_at": 100.0})
    store.create_episode({"id": "e_mid", "content": "心跳", "created_at": 200.0})

    class _Svcs:
        query_router = _EmptyRouter()
        graph_store = store

    api = GatewayAPI.__new__(GatewayAPI)
    api._svc = _Svcs()
    api._logger = type("L", (), {
        "warning": lambda self, *a, **k: None,
        "exception": lambda self, *a, **k: None,
    })()

    resp = await api.retrieve("心跳线程日志")
    ids = [r.node_id for r in resp.results]
    assert ids == ["e_high", "e_low", "e_mid"], ids
    assert resp.results[0].score == pytest.approx(0.5)
    for r in resp.results:
        assert 0.05 <= r.score <= 0.5
        assert r.retrieval_level == "graphlite_fallback"
