"""
M4/M5 检索路由测试（手术刀 R2a）
==============================
- M4: CJK 查询跳过实体通道 + L4 GraphLite 兜底
  （GraphLite CONTAINS 对中文无子串保持性，通道恒空——纯省一次全表扫描；
   一次性 warning 标志不刷屏）
- M5: EpisodeCache（OrderedDict LRU + TTL）—— TTL 过期 / LRU 逐出 / 容量封顶，
      以及 flush_faiss_buffer 作为唯一写入方填充

运行: python -m pytest tests/test_query_router_cjk_cache.py -v
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np

from graph.graphlite_store import EpisodeCache
from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel, RetrievalStrategy


class CountingStore:
    """计数 query_cypher 调用的假 GraphLiteStore。"""

    def __init__(self):
        self.calls = 0

    def query_cypher(self, *args, **kwargs):
        self.calls += 1
        return []


def _make_router(store) -> QueryRouter:
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router.graphlite_store = store
    router._cjk_warned = False
    router._episode_cache = {}
    router._time_keywords = ["最近", "刚刚", "刚才", "之前说的", "上一条", "昨天", "今天", "几分钟前", "上一次", "recent", "just now", "earlier", "last", "previous", "yesterday"]
    return router


class TestM4CjkSkipsEntityChannel:
    """【M4】CJK 查询跳过 entity 通道（CONTAINS 对中文恒空）。"""

    def test_cjk_query_zero_gql_calls(self):
        """中文查询：_entity_match 不触发任何 GraphLite 查询。"""
        store = CountingStore()
        router = _make_router(store)
        with patch.object(router, "_entity_match", wraps=router._entity_match) as em:
            with patch.object(router, "_bm25_search", return_value=[]):
                with patch.object(router, "_vector_retrieve", return_value=[]):
                    with patch.object(router, "_fuse_results", return_value=[]):
                        router._fusion_retrieve("记忆系统测试")
        assert store.calls == 0, (
            f"CJK 查询不应触发实体通道 GQL，实际 {store.calls} 次"
        )
        em.assert_not_called()

    def test_ascii_query_still_walks_entity_channel(self):
        """英文查询：实体通道照常执行（仅 CJK 跳过）。"""
        store = CountingStore()
        router = _make_router(store)
        with patch.object(router, "_bm25_search", return_value=[]):
            with patch.object(router, "_vector_retrieve", return_value=[]):
                router._fusion_retrieve("memory system test")
        assert store.calls == 1, (
            f"ASCII 查询应走实体通道（1 次 GQL），实际 {store.calls} 次"
        )

    def test_cjk_warning_once_not_flood(self):
        """一次性 warning 标志：多查询只打一次 warning（不刷屏）。"""
        store = CountingStore()
        router = _make_router(store)
        warnings = []

        import logging

        from observability.logger import configure_logging
        configure_logging("DEBUG")
        handler = logging.Handler()
        handler.emit = lambda r: warnings.append(r)  # type: ignore[method-assign]
        logger = logging.getLogger("retrieval.query_router")
        old_level = logger.level
        old_handlers = list(logger.handlers)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            with patch.object(router, "_bm25_search", return_value=[]):
                with patch.object(router, "_vector_retrieve", return_value=[]):
                    for _ in range(3):
                        router._fusion_retrieve("记忆系统测试")
        finally:
            logger.setLevel(old_level)
            for h in old_handlers:
                logger.addHandler(h)
            logger.removeHandler(handler)

        cjk_msgs = [
            r.getMessage() for r in warnings if "CJK query detected" in r.getMessage()
        ]
        assert len(cjk_msgs) == 1, (
            f"一次性标志应只打 1 次 warning，实际 {len(cjk_msgs)} 次"
        )

    def test_graphlite_text_fallback_cjk_returns_empty(self):
        """_graphlite_text_fallback 对 CJK 查询直接返回 [] 且不查询。"""
        store = CountingStore()
        router = _make_router(store)
        assert router._graphlite_text_fallback("记忆系统测试", "L3 empty") == []
        assert store.calls == 0, (
            f"CJK 查询不应触发 L4 兜底 GQL，实际 {store.calls} 次"
        )

    def test_graphlite_text_fallback_ascii_still_queries(self):
        """L4 兜底对英文查询照常查询（仅 CJK 跳过）。"""
        store = CountingStore()
        router = _make_router(store)
        assert router._graphlite_text_fallback("memory system") == []
        assert store.calls == 1, (
            f"ASCII 查询应走 L4 兜底（1 次 GQL），实际 {store.calls} 次"
        )

    def test_cjk_detect_strategy_still_hybrid(self):
        """detect_strategy 对中文仍返回 HYBRID（M4 只跳实体通道，不改策略）。"""
        router = _make_router(CountingStore())
        assert router.detect_strategy("记忆系统") == RetrievalStrategy.HYBRID


class TestM5EpisodeCache:
    """【M5】EpisodeCache LRU + TTL 行为。"""

    def test_lru_eviction_caps_size(self):
        """容量封顶：超出 maxsize 逐出最旧未访问项。"""
        c = EpisodeCache(maxsize=3, ttl=600)
        c["a"] = {"id": "a"}
        c["b"] = {"id": "b"}
        c["c"] = {"id": "c"}
        c["d"] = {"id": "d"}
        assert "a" not in c, "LRU 应逐出最旧项 a"
        assert list(c._data.keys()) == ["b", "c", "d"]
        assert len(c._data) == 3

    def test_lru_touch_refreshes_order(self):
        """访问刷新 LRU 顺序：被 touch 的项不会被先逐出。"""
        c = EpisodeCache(maxsize=3, ttl=600)
        c["a"] = {"id": "a"}
        c["b"] = {"id": "b"}
        c["c"] = {"id": "c"}
        assert c["a"] == {"id": "a"}  # touch a
        c["d"] = {"id": "d"}
        assert "b" not in c, "未 touch 的 b 应被逐出"
        assert list(c._data.keys()) == ["c", "a", "d"]

    def test_ttl_expiry_invisible(self):
        """TTL 过期：__contains__/get/__getitem__ 均视为不存在。"""
        c = EpisodeCache(maxsize=10, ttl=600)
        c["old"] = {"id": "old"}
        c["fresh"] = {"id": "fresh"}
        # 伪造过期时间戳
        c._data["old"] = (0.0, {"id": "old"})
        assert "old" not in c, "过期项不应可见"
        assert c.get("old") is None
        try:
            c["old"]
            raise AssertionError("过期项 __getitem__ 应抛 KeyError")
        except KeyError:
            pass
        assert c["fresh"] == {"id": "fresh"}

    def test_ttl_uses_epoch_not_wallclock_delta(self):
        """TTL 以写入时间戳为准：写入后 600s 内有效，超时失效。"""
        c = EpisodeCache(maxsize=10, ttl=1.0)
        c["x"] = {"id": "x"}
        assert "x" in c
        time.sleep(1.1)
        assert "x" not in c, "TTL 1s 的项应在 1.1s 后过期"

    def test_query_router_uses_shared_cache(self):
        """QueryRouter 与 flush_faiss_buffer 共享同一 EpisodeCache 引用。"""
        from api.routes._deps import Services, flush_faiss_buffer
        import types

        cache = EpisodeCache(maxsize=16, ttl=600)
        svc = Services()
        svc._episode_cache = cache
        svc._faiss_buffer_lock = __import__("threading").Lock()
        svc.faiss_index = types.SimpleNamespace(add_with_ids=lambda v, i: None)
        svc.faiss_id_map = {}
        svc._faiss_buffer = [(1, [0.1, 0.2], "ep-1"), (2, [0.3, 0.4], "ep-2")]
        assert flush_faiss_buffer(svc) == 2
        assert cache.get("ep-1") == {"id": "ep-1"}, "flush 应填充 episode cache"
        assert cache.get("ep-2") == {"id": "ep-2"}
        assert "ep-1" in cache

        router = QueryRouter.__new__(QueryRouter)
        router.config = QueryRouterConfig()
        router._episode_cache = cache
        router.graphlite_store = MagicMock()
        router._cjk_warned = False
        # L1 超图检索 cache 命中：不再回查 GraphLite
        router.faiss_index = MagicMock()
        router.faiss_index.search.return_value = (
            np.array([[-0.9]]), np.array([[1]]),
        )
        router.faiss_id_map = {1: "ep-1"}
        router.graphlite_store.get_episodes_batch = MagicMock(return_value=[])
        results = router._hypergraph_retrieve("test", query_embedding=np.zeros(8, dtype=np.float32))
        assert results, "cache 命中应返回结果（不回查 GraphLite）"
        router.graphlite_store.get_episodes_batch.assert_not_called()
