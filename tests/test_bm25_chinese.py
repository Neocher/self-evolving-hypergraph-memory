"""
BM25 中文检索测试
=================
验证 TfidfVectorizer 使用字符级 n-gram (analyzer="char_wb", ngram_range=(2,4))
后，中文语义词能够被正确召回，同时英文检索无回归。

通过 __new__ 构造 QueryRouter（跳过真实引擎依赖），
用 fake GraphLiteStore 提供语料，直接驱动 _build_bm25_index / _bm25_search。
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel


class FakeGraphLiteStore:
    """仅提供 query_cypher 的假 GraphLiteStore。"""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def query_cypher(self, *args, **kwargs) -> list[dict]:
        return self.rows


def _make_router(corpus: list[str]) -> QueryRouter:
    """用 __new__ 构造 QueryRouter，绕过真实引擎依赖。"""
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router.graphlite_store = FakeGraphLiteStore(
        [
            {"node_id": f"n{i}", "content": content, "tau_value": 1.0}
            for i, content in enumerate(corpus)
        ]
    )
    router._bm25_doc_ids = []
    router._bm25_doc_contents = []
    router._bm25_doc_tau = []
    router._bm25_built = False
    router._build_bm25_index()
    return router


def test_chinese_query_recalls_doc() -> None:
    """中文语料索引后，查询"记忆"应能召回含"记忆"的文档 (score > 0)。"""
    corpus = ["记忆系统测试", "超图神经网络", "梦境聚类"]
    router = _make_router(corpus)
    assert router._bm25_ready

    results = router._bm25_search("记忆")
    assert results, "中文查询未召回任何文档"
    assert any("记忆" in r["content"] for r in results), "未召回含'记忆'的文档"
    assert all(r["score"] > 0 for r in results)


def test_english_no_regression() -> None:
    """英文语料仍能正常匹配。"""
    corpus = ["machine learning framework", "hypergraph memory", "dream consolidation"]
    router = _make_router(corpus)
    assert router._bm25_ready

    results = router._bm25_search("framework")
    assert results, "英文查询未召回任何文档"
    assert any("framework" in r["content"] for r in results)


def test_empty_corpus_no_crash() -> None:
    """空语料构建不崩溃（返回不抛异常）。"""
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router.graphlite_store = FakeGraphLiteStore([])
    router._bm25_doc_ids = []
    router._bm25_doc_contents = []
    router._bm25_doc_tau = []
    router._bm25_built = False
    router._bm25_ready = False

    # 不应抛异常
    router._build_bm25_index()
    assert not router._bm25_ready

    # 索引未就绪时搜索返回空列表而非崩溃
    assert router._bm25_search("记忆") == []


def test_query_no_match_returns_empty() -> None:
    """无关查询返回空列表。"""
    corpus = ["记忆系统测试", "超图神经网络"]
    router = _make_router(corpus)
    assert router._bm25_ready

    results = router._bm25_search("量子纠缠计算")
    assert results == []


def test_retrieve_chinese_mapped_word_recalls() -> None:
    """完整入口回归：retrieve() 归一化会把"记忆"→"memory"，
    BM25 通道必须收到未归一化的原始中文才能命中中文语料。

    mock 掉向量/实体通道，隔离出 BM25 通道；监听其收到的查询，
    验证是原始中文而非归一化英文（修复前收到 "memory" 会召回为空）。
    """
    corpus = ["记忆系统测试", "超图神经网络", "梦境聚类"]
    router = _make_router(corpus)
    # __new__ 构造缺少 __init__ 属性，补上 retrieve() 完整路径所需
    router._zh_en_tech_map = {"记忆": "memory"}
    router._time_keywords = []

    received: list[str] = []

    def spy_bm25(query: str, k: int = 20) -> list[dict]:
        received.append(query)
        return QueryRouter._bm25_search(router, query, k)

    with (
        patch.object(router, "_vector_retrieve", return_value=[]),
        patch.object(router, "_entity_match", return_value=[]),
        patch.object(router, "_bm25_search", side_effect=spy_bm25),
    ):
        results = router.retrieve("记忆", level=RetrievalLevel.FUSION)

    assert received, "BM25 通道未被调用"
    assert received[0] == "记忆", (
        f"BM25 通道收到归一化查询 {received[0]!r}，应为原始中文 '记忆'"
    )
    assert results, "retrieve() 完整入口下中文查询未召回任何文档"
    assert any("记忆" in r["content"] for r in results)
    assert all(r["score"] > 0 for r in results)


def test_bm25_multi_doc_term_scores_land_on_own_rows() -> None:
    """P1 回归：多文档共享 term 时 BM25 分须落在各自行（CSR 列切片行索引修复）。

    修复前 col.indices 是列号（恒 0）→ 多文档贡献全部累加到 docs[0]，
    其余文档 score=0 被跳过 → 只召回第 0 篇。修复后各文档独立召回。
    """
    corpus = ["K8s 集群网络 flannel", "K8s 集群网络 calico", "买菜清单"]
    router = _make_router(corpus)
    results = router._bm25_search("K8s 网络")
    docs = [r["content"] for r in results]
    assert any("flannel" in c for c in docs), "含 flannel 的文档应被召回"
    assert any("calico" in c for c in docs), "含 calico 的文档应被召回"
    assert len(results) >= 2, f"多文档共享 term 应各自召回: {docs}"
    assert all(r["score"] > 0 for r in results)


class FailingThenWorkingStore(FakeGraphLiteStore):
    """query_cypher 先抛异常、后恢复正常（模拟 GraphLite 短暂故障恢复）。"""

    def __init__(self, rows: list[dict]):
        super().__init__(rows)
        self.fail = True

    def query_cypher(self, *args, **kwargs) -> list[dict]:
        if self.fail:
            raise RuntimeError("GraphLite down")
        return self.rows


def _make_bare_router(store) -> QueryRouter:
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router.graphlite_store = store
    router._bm25_doc_ids = []
    router._bm25_doc_contents = []
    router._bm25_doc_tau = []
    router._bm25_built = False
    router._bm25_ready = False
    router._bm25_last_attempt = 0.0
    return router


def test_bm25_build_failure_retryable() -> None:
    """【M1】构建失败不置位 _bm25_built → 保留重试机会（修复前永久降级）。

    修复前 _bm25_built=True 在函数入口提前置位：query_cypher 异常后
    _bm25_built=True / _bm25_ready=False → 永不重建。
    """
    store = FailingThenWorkingStore(
        [{"node_id": "n1", "content": "记忆系统测试", "tau_value": 1.0}]
    )
    router = _make_bare_router(store)

    # 第一次构建失败：不得置位 _bm25_built
    router._build_bm25_index()
    assert router._bm25_built is False, "构建失败不应置位 _bm25_built"
    assert router._bm25_ready is False

    # 故障恢复后重试成功：_bm25_built/_bm25_ready 置位，检索可用
    store.fail = False
    router._build_bm25_index()
    assert router._bm25_built is True
    assert router._bm25_ready is True
    assert router._bm25_search("记忆"), "恢复后 BM25 应能正常检索"


def test_bm25_lazy_build_retries_after_failure() -> None:
    """【M1】懒构建路径（_bm25_search 内）失败后保留重试机会。

    修复前 _bm25_search 懒路径先置位 _bm25_built 再构建 → 失败即永久降级。
    """
    store = FailingThenWorkingStore(
        [{"node_id": "n1", "content": "记忆系统测试", "tau_value": 1.0}]
    )
    router = _make_bare_router(store)

    # 懒构建失败：返回空但不置位 _bm25_built
    assert router._bm25_search("记忆") == []
    assert router._bm25_built is False, "懒构建失败不应置位 _bm25_built"

    # 故障恢复（冷却窗口重置为 0）→ 懒构建重试成功
    store.fail = False
    router._bm25_last_attempt = 0.0  # 模拟冷却窗口已过
    results = router._bm25_search("记忆")
    assert router._bm25_ready is True
    assert results, "懒构建重试后应能正常检索"


class GrowingStore(FakeGraphLiteStore):
    """语料从空到有（模拟空库部署后新数据累积写入）。"""

    def __init__(self):
        self.rows = []

    def query_cypher(self, *args, **kwargs) -> list[dict]:
        return self.rows


def test_empty_corpus_not_terminal_rebuilds_after_data() -> None:
    """【B-复审】空语料构建不置位 _bm25_built（非终态）→ 新数据写入后可重建。

    修复前（M1-a）空语料置 _bm25_built=True 为终态 + 启动 prewarm_bm25()
    空库预热置位 → 空库部署 BM25 进程内永久失效，即使语料已累积也不再重建。
    """
    store = GrowingStore()
    router = _make_bare_router(store)

    # 空库构建：不得置位 _bm25_built（保留冷却重试机会）
    router._build_bm25_index()
    assert router._bm25_built is False, "空语料不应置位 _bm25_built（非终态）"
    assert router._bm25_ready is False

    # 语料累积后重建成功：_bm25_built/_bm25_ready 置位，检索可用
    store.rows = [{"node_id": "n1", "content": "记忆系统测试", "tau_value": 1.0}]
    router._build_bm25_index()
    assert router._bm25_built is True, "新数据写入后应能重建 BM25"
    assert router._bm25_ready is True
    assert router._bm25_search("记忆"), "重建后应能正常检索"


def test_prewarm_empty_corpus_does_not_set_terminal() -> None:
    """【B-复审】prewarm_bm25 空库预热不置位 _bm25_built（进程内不永久失效）。

    修复前空库预热置 _bm25_built=True → 首个查询/后续写入永不触发重建。
    """
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router.graphlite_store = FakeGraphLiteStore([])
    router._bm25_doc_ids = []
    router._bm25_doc_contents = []
    router._bm25_doc_tau = []
    router._bm25_built = False
    router._bm25_ready = False
    router._bm25_last_attempt = 0.0

    asyncio.run(router.prewarm_bm25())
    assert router._bm25_built is False, "prewarm 空库不应置位 _bm25_built"
    assert router._bm25_ready is False


def test_bm25_retry_cooldown_gates_rebuild() -> None:
    """【遗留项】冷却窗口（bm25_retry_cooldown）生效：冷却期内不重建，过期后重建。

    验证 _bm25_search 懒构建路径受 _bm25_last_attempt 冷却门控：
    - 构建失败后 _bm25_last_attempt 置位 → 冷却期内重复检索不触发全量重建
    - 冷却窗口过期（或 last_attempt 重置为 0）→ 触发重建
    """
    calls = {"build": 0}
    store = GrowingStore()
    router = _make_bare_router(store)
    router.config = QueryRouterConfig()
    router.config.bm25_retry_cooldown = 60.0  # 显式冷却窗口 60s

    orig_core = router._build_bm25_index_core

    def counting_core():
        calls["build"] += 1
        orig_core()

    router._build_bm25_index_core = counting_core  # type: ignore[method-assign]

    # 空库首次构建（失败，不置位）
    router._bm25_search("记忆")
    assert calls["build"] == 1
    assert router._bm25_built is False

    # 冷却期内（_bm25_last_attempt 刚置位）再次检索 → 不重建
    router._bm25_search("记忆")
    assert calls["build"] == 1, "冷却期内不应重复触发重建"

    # 冷却窗口过期 → 重建
    router._bm25_last_attempt = 0.0
    router._bm25_search("记忆")
    assert calls["build"] == 2, "冷却窗口过期后应触发重建"


def test_bm25_empty_corpus_log_noise_reduced() -> None:
    """【遗留项】空语料日志降噪：进程内首次 warning，重复触发只打 debug。

    修复前空库每次 _build_bm25_index_core 都 logger.warning
    （冷却窗口内每 30s 刷一次），现改为首次 warning + 后续 debug。
    """
    import logging

    # structlog 默认 PrintLogger 不走 stdlib logging，先 configure 路由到 stdlib
    from observability.logger import configure_logging
    configure_logging("DEBUG")

    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r)  # type: ignore[method-assign]

    logger = logging.getLogger("retrieval.query_router")
    old_level = logger.level
    old_handlers = list(logger.handlers)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        store = GrowingStore()
        router = _make_bare_router(store)
        # 空库构建两次：第一次 warning，第二次 debug
        router._build_bm25_index()
        router._build_bm25_index()

        empty_msgs = [r.getMessage() for r in records if "empty corpus" in r.getMessage()]
        warnings = [m for m in empty_msgs if "corpus" in m]
        # 只有首次是 WARNING 级别；后续为 DEBUG
        warn_count = sum(
            1 for r in records
            if "empty corpus" in r.getMessage() and r.levelno == logging.WARNING
        )
        debug_count = sum(
            1 for r in records
            if "empty corpus" in r.getMessage() and r.levelno == logging.DEBUG
        )
        assert warn_count == 1, f"空语料 warning 应只有 1 次（进程内首次），实际 {warn_count}"
        assert debug_count >= 1, "后续空语料应打 debug 降噪"
    finally:
        logger.setLevel(old_level)
        for h in old_handlers:
            logger.addHandler(h)
        logger.removeHandler(handler)

