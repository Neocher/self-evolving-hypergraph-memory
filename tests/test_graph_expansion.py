"""v5.26.0 图扩散检索测试 — 超边共现 + Hebbian P0 修复"""
import sys
import threading
from unittest.mock import MagicMock, patch


def _make_router(store=None, config=None):
    """构造 QueryRouter 实例，mock sklearn 导入避免环境依赖。"""
    with patch.dict(sys.modules, {"sklearn": MagicMock(), "sklearn.feature_extraction": MagicMock(), "sklearn.feature_extraction.text": MagicMock()}):
        with patch("retrieval.query_router.TfidfVectorizer", MagicMock()):
            from retrieval.query_router import QueryRouter, QueryRouterConfig
    cfg = config or QueryRouterConfig()
    return QueryRouter(
        graphlite_store=store,
        faiss_index=MagicMock(),
        tfidf_index=MagicMock(),
        config=cfg,
    )


class TestGraphExpansion:
    """图扩散检索测试（mock store 层）"""

    def test_expansion_recalls_neighbors(self):
        """扩散节点应进入结果（level=graph_expansion, _source=graph）。"""
        store = MagicMock()
        store.get_hypergraph_neighbors = MagicMock(return_value={
            "seed_a": [
                {"id": "neighbor_1", "content": "expanded content 1", "co_occurrence": 3},
                {"id": "neighbor_2", "content": "expanded content 2", "co_occurrence": 1},
            ]
        })
        router = _make_router(store=store)

        seeds = ["seed_a"]
        existing = {"seed_a"}
        tail = 0.5
        result = router._graph_expansion(seeds, existing, tail)

        assert len(result) == 2
        assert all(r["level"] == "graph_expansion" for r in result)
        assert all(r["_source"] == "graph" for r in result)
        ids = {r["node_id"] for r in result}
        assert "neighbor_1" in ids
        assert "neighbor_2" in ids

    def test_expansion_score_below_tail(self):
        """扩散分数 < 向量尾分 且 > 0（可插入结果尾部）。"""
        store = MagicMock()
        store.get_hypergraph_neighbors = MagicMock(return_value={
            "s1": [
                {"id": "n1", "content": "c1", "co_occurrence": 0},
                {"id": "n2", "content": "c2", "co_occurrence": 5},
            ]
        })
        router = _make_router(store=store)

        tail = 0.5
        result = router._graph_expansion(["s1"], set(), tail)

        for r in result:
            assert 0 < r["score"] < tail, (
                f"score={r['score']} should be between 0 and {tail}"
            )

    def test_expansion_failure_falls_back(self):
        """store 抛异常 → 返回 []（静默回退，纯向量结果不受影响）。"""
        store = MagicMock()
        store.get_hypergraph_neighbors = MagicMock(side_effect=RuntimeError("db down"))
        router = _make_router(store=store)

        result = router._graph_expansion(["s1"], set(), 0.5)
        assert result == []

    def test_expansion_filters_empty_content(self):
        """无 content 的扩散节点应被剔除。"""
        store = MagicMock()
        store.get_hypergraph_neighbors = MagicMock(return_value={
            "s1": [
                {"id": "n1", "content": "valid content", "co_occurrence": 2},
                {"id": "n2", "content": "", "co_occurrence": 5},
                {"id": "n3", "content": "also valid", "co_occurrence": 1},
            ]
        })
        router = _make_router(store=store)

        result = router._graph_expansion(["s1"], set(), 0.5)
        ids = {r["node_id"] for r in result}
        assert "n1" in ids
        assert "n3" in ids
        assert "n2" not in ids  # empty content filtered

    def test_expansion_skips_existing_ids(self):
        """已在向量结果中的 id 不应出现在扩散结果中。"""
        store = MagicMock()
        store.get_hypergraph_neighbors = MagicMock(return_value={
            "s1": [
                {"id": "existing_node", "content": "already in results", "co_occurrence": 3},
                {"id": "new_node", "content": "not in results", "co_occurrence": 2},
            ]
        })
        router = _make_router(store=store)

        existing = {"existing_node", "s1"}
        result = router._graph_expansion(["s1"], existing, 0.5)
        ids = {r["node_id"] for r in result}
        assert "existing_node" not in ids
        assert "new_node" in ids

    def test_expansion_cross_seed_truncation(self):
        """多种子汇总后按 score 降序截断到 graph_expansion_max 条。"""
        store = MagicMock()
        # 3 种子每人 5 邻居 → 15 条汇总，截断到 graph_expansion_max=5
        neighbors = {}
        for s in ["s1", "s2", "s3"]:
            neighbors[s] = [
                {"id": f"{s}_nb{i}", "content": f"content_{s}_{i}", "co_occurrence": i}
                for i in range(5)
            ]
        store.get_hypergraph_neighbors = MagicMock(return_value=neighbors)

        router = _make_router(store=store)
        router.config.graph_expansion_max = 5

        result = router._graph_expansion(["s1", "s2", "s3"], set(), 0.5)

        assert len(result) == 5, f"expected 5, got {len(result)}"
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True), f"scores not descending: {scores}"


class TestHypergraphRetrieve:
    """走公共入口 _hypergraph_retrieve 的集成测试（mock store 层）"""

    def test_hypergraph_retrieve_with_expansion(self):
        """mock get_hypergraph_neighbors 返回邻居 → 结果含 level=graph_expansion 节点。"""
        import numpy as np

        store = MagicMock()
        store.get_hypergraph_neighbors = MagicMock(return_value={
            "seed_0": [{"id": "nb_0", "content": "neighbor 0", "co_occurrence": 2}],
            "seed_1": [{"id": "nb_1", "content": "neighbor 1", "co_occurrence": 1}],
        })
        store.get_episodes_batch = MagicMock(return_value=[])

        router = _make_router(store=store)

        query_emb = np.zeros((1, 512), dtype=np.float32)
        distances = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
        indices = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        router.faiss_index.search.return_value = (distances, indices)
        router.faiss_id_map = {i: f"seed_{i}" for i in range(5)}
        for i in range(5):
            router._episode_cache[f"seed_{i}"] = {"content": f"episode {i}"}

        results = router._hypergraph_retrieve("test query", query_emb)

        levels = {r.get("level") for r in results}
        assert "graph_expansion" in levels, f"levels: {levels}"
        sources = {r.get("_source") for r in results}
        assert "graph" in sources, f"sources: {sources}"

    def test_hypergraph_retrieve_expansion_failure(self):
        """mock get_hypergraph_neighbors 抛异常 → 纯向量结果，无 graph_expansion，不抛异常。"""
        import numpy as np

        store = MagicMock()
        store.get_hypergraph_neighbors = MagicMock(side_effect=RuntimeError("db down"))
        store.get_episodes_batch = MagicMock(return_value=[])

        router = _make_router(store=store)

        query_emb = np.zeros((1, 512), dtype=np.float32)
        distances = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
        indices = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        router.faiss_index.search.return_value = (distances, indices)
        router.faiss_id_map = {i: f"seed_{i}" for i in range(5)}
        for i in range(5):
            router._episode_cache[f"seed_{i}"] = {"content": f"episode {i}"}

        results = router._hypergraph_retrieve("test query", query_emb)

        graph_nodes = [r for r in results if r.get("level") == "graph_expansion"]
        assert len(graph_nodes) == 0, f"unexpected graph nodes: {graph_nodes}"
        assert len(results) > 0
        assert all(r["level"] == "l1_faiss" for r in results)


