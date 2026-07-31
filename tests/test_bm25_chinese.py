"""
BM25 中文检索测试
=================
验证 TfidfVectorizer 使用字符级 n-gram (analyzer="char_wb", ngram_range=(2,4))
后，中文语义词能够被正确召回，同时英文检索无回归。

通过 __new__ 构造 QueryRouter（跳过真实引擎依赖），
用 fake KuzuStore 提供语料，直接驱动 _build_bm25_index / _bm25_search。
"""
from __future__ import annotations

from unittest.mock import patch

from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel


class FakeKuzuStore:
    """仅提供 query_cypher 的假 KuzuStore。"""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def query_cypher(self, *args, **kwargs) -> list[dict]:
        return self.rows


def _make_router(corpus: list[str]) -> QueryRouter:
    """用 __new__ 构造 QueryRouter，绕过真实引擎依赖。"""
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router.kuzu_store = FakeKuzuStore(
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
    router.kuzu_store = FakeKuzuStore([])
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
