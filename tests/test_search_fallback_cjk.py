"""v6.22.8 — REST 路径 search.py Cypher 兜底 CJK 分词 + 判别式打分测试。

P2b 兜底修复：把 v6.22.7 落在 gateway_api.py 的 CJK 兜底修复同步到生产主路径
api/routes/search.py 的 Cypher 兜底块（`shm-memory-mcp.py → POST /memories/retrieve`
走这条）。修复前该副本仍是 query.split() 整句 CONTAINS（中文命中≈0）+ score 0.5
硬编码无判别力。本文件对 search.py handler 直调做兜底 harness 探针：CJK 打分降序、
英文 params 兼容、quarantine+archived 双 WHERE 保留、0-hit 地板 0.05、
created_at tie-break 防御解析。
"""

import pytest

from api.models import RetrieveRequest
from api.routes.search import retrieve as _retrieve
from api.routes._deps import Services, clear_result_cache


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


def _make_svc(store):
    svc = Services()
    svc.query_router = _EmptyRouter()
    svc.graph_store = store
    return svc


@pytest.fixture(autouse=True)
def _clear_cache():
    """清模块级 _result_cache，防各测试间缓存串扰。"""
    clear_result_cache()
    yield
    clear_result_cache()


@pytest.mark.asyncio
async def test_cjk_fallback_scored_sorted():
    """【探针门】CJK 兜底行 score ∈ [0.05, 0.5]、按命中降序、tie 用 created_at DESC、
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
    svc = _make_svc(store)
    resp = await _retrieve(RetrieveRequest(query="心跳线程日志"), deps=svc)

    assert resp.degraded is True
    assert [r.node_id for r in resp.results] == ["A", "B", "C", "D"]
    assert resp.results[0].score == pytest.approx(0.5)
    for r in resp.results:
        assert 0.05 <= r.score <= 0.5
        assert r.retrieval_level == "graphlite_fallback"
    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True), "兜底结果须按命中降序"


@pytest.mark.asyncio
async def test_score_upper_bound_half():
    """全命中 score == 0.5，不越过 0.5（与主通道"降级结果低于主通道"语义兼容）。"""
    store = _FakeStore([
        {"node_id": "full", "content": "心跳线程日志", "created_at": 1.0},
    ])
    svc = _make_svc(store)
    resp = await _retrieve(RetrieveRequest(query="心跳线程日志"), deps=svc)
    assert len(resp.results) == 1
    assert resp.results[0].score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_english_fallback_spy_compat():
    """【英文零回归】纯英文查询兜底 params 为英文 token（旧 split() 路径超集）。"""
    store = _FakeStore([
        {"node_id": "E", "content": "hello world", "created_at": 1.0},
    ])
    svc = _make_svc(store)
    await _retrieve(RetrieveRequest(query="hello world"), deps=svc)
    assert store.calls, "英文查询应走 Cypher 兜底"
    params = store.calls[-1]["params"]
    assert params == {"w0": "hello", "w1": "world"}


@pytest.mark.asyncio
async def test_cypher_contains_quarantine_and_archived():
    """【双 WHERE 保留】兜底 cypher 必须同时含 quarantine 与 archived 两个条件
    （search.py 路径业务差异，禁止对齐 gateway 副本的仅 archived）。"""
    store = _FakeStore([
        {"node_id": "Q", "content": "心跳", "created_at": 1.0},
    ])
    svc = _make_svc(store)
    await _retrieve(RetrieveRequest(query="心跳"), deps=svc)
    assert store.calls, "应走 Cypher 兜底"
    cypher = store.calls[-1]["cypher"]
    assert "e.quarantine IS NULL OR e.quarantine = false" in cypher
    assert "e.archived IS NULL OR e.archived = false" in cypher


@pytest.mark.asyncio
async def test_cypher_contains_created_at():
    """兜底 cypher RETURN 必须含 e.created_at（tie-break 排序依赖）。"""
    store = _FakeStore([
        {"node_id": "T", "content": "心跳", "created_at": 1.0},
    ])
    svc = _make_svc(store)
    await _retrieve(RetrieveRequest(query="心跳"), deps=svc)
    assert store.calls, "应走 Cypher 兜底"
    cypher = store.calls[-1]["cypher"]
    assert "e.created_at AS created_at" in cypher


@pytest.mark.asyncio
async def test_zero_hit_row_score_floor():
    """0-hit 行 score == 0.05（噪声地板），不被丢弃。"""
    store = _FakeStore([
        {"node_id": "z", "content": "完全无关的内容", "created_at": 1.0},
    ])
    svc = _make_svc(store)
    resp = await _retrieve(RetrieveRequest(query="心跳线程"), deps=svc)
    assert len(resp.results) == 1
    assert resp.results[0].score == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_list_tuple_row_created_at():
    """list/tuple 行形态取 row[2] 作 created_at（tie-break 用）。"""
    store = _FakeStore([
        ("id_old", "心跳 甲", 100.0),
        ("id_new", "心跳 乙", 200.0),
    ])
    svc = _make_svc(store)
    resp = await _retrieve(RetrieveRequest(query="心跳"), deps=svc)
    # 单 token 全命中 → 同分 0.5（content 不同防去重），tie 用 created_at DESC：200 > 100
    assert [r.node_id for r in resp.results] == ["id_new", "id_old"]


@pytest.mark.asyncio
async def test_dict_row_missing_created_at_defaults_zero():
    """dict 行缺 created_at 键 → 默认 0.0，不崩，仍参与排序。"""
    store = _FakeStore([
        {"node_id": "d1", "content": "心跳 甲"},
        {"node_id": "d2", "content": "心跳 乙", "created_at": 5.0},
    ])
    svc = _make_svc(store)
    resp = await _retrieve(RetrieveRequest(query="心跳"), deps=svc)
    assert [r.node_id for r in resp.results] == ["d2", "d1"], "d2 ts=5.0 > d1 ts=0.0"


@pytest.mark.asyncio
async def test_no_tokens_query_skips_cypher():
    """无 token 查询（如单字符非 CJK）→ 不触发 query_cypher，结果为空。"""
    store = _FakeStore([
        {"node_id": "x", "content": "whatever", "created_at": 1.0},
    ])
    svc = _make_svc(store)
    resp = await _retrieve(RetrieveRequest(query="a"), deps=svc)
    assert store.calls == [], "无 token 查询不得触发 Cypher 兜底"
    assert resp.results == []
