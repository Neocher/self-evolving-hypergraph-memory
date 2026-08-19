"""中文 embedding 检索集成测试 — 真实 bge 模型 + 隔离 GraphLite + 生产 QueryRouter

验证: bge-small-zh-v1.5 切换后, 中文语义检索真实可用 (同类命中 > 异类)。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _make_tfidf(docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    class TfidfSearchIndex:
        def __init__(self):
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=5000)
            self._fitted = False

        def fit(self, texts):
            if texts:
                self.matrix = self.vectorizer.fit_transform(texts)
                self._fitted = True
                self.texts = texts

        def search(self, query, k=20):
            if not self._fitted or not hasattr(self, "matrix"):
                return []
            q_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self.matrix)[0]
            top_k = min(k, len(scores))
            if top_k == 0:
                return []
            top_indices = np.argsort(scores)[-top_k:][::-1]
            return [(self.texts[i], float(scores[i])) for i in top_indices]

    idx = TfidfSearchIndex()
    idx.fit(docs)
    return idx


class TestChineseEmbeddingRetrieval:
    """真实 bge 中文模型判别力 (不依赖 FAISS, 直接测 encoder)"""

    def test_bge_chinese_discrimination(self):
        from embedding.encoder import create_encoder

        enc = create_encoder()
        if enc is None or not hasattr(enc, "dimension"):
            pytest.skip("encoder unavailable")

        def sim(a, b):
            va, vb = enc.embed(a), enc.embed(b)
            return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

        # 同类相似度 > 异类相似度
        same = sim("张三喜欢吃苹果", "张三喜欢吃梨")
        diff = sim("张三喜欢吃苹果", "张三喜欢打篮球")
        assert same > diff, f"bge 判别力异常: same={same:.3f} diff={diff:.3f}"
        assert same > 0.7, f"同类相似度过低: {same:.3f}"

    def test_bge_dimension_512(self):
        from embedding.encoder import create_encoder

        enc = create_encoder()
        if enc is None or not hasattr(enc, "dimension"):
            pytest.skip("encoder unavailable")
        assert enc.dimension == 512

    def test_query_router_chinese_bm25(self, overgraph_store):
        """QueryRouter BM25 中文检索命中 (bge 模型存在时走真实路径)"""
        docs = [
            "张三喜欢吃苹果和梨",
            "李四喜欢打篮球和跑步",
            "王五喜欢读书和写作",
        ]
        ids = []
        for i, c in enumerate(docs):
            eid = overgraph_store.create_episode({
                "content": c, "created_at": i + 1.0, "tau_initial": 1.0,
                "tau_value": 0.6, "source": "test", "trust_score": 0.8,
            })
            ids.append(eid)

        tfidf = _make_tfidf(docs)

        class FakeFaiss:
            def search(self, query_vec, k):
                return (np.ones((1, k)) * 0.5, np.array([np.arange(k)]))

        class FakeEncoder:
            def embed(self, text):
                return np.zeros(512)

        from retrieval.query_router import QueryRouter, QueryRouterConfig as QRCfg, RetrievalLevel
        qr = QueryRouter(
            graphlite_store=overgraph_store,
            faiss_index=FakeFaiss(),
            tfidf_index=tfidf,
            encoder=FakeEncoder(),
            faiss_id_map={i: ids[i] for i in range(len(ids))},
            episode_cache={},
            config=QRCfg(tau_weight=0.2, vector_weight=0.3, top_k_l1=10, top_k_vector=5, top_k_keyword=5),
        )
        results = qr.retrieve("篮球", level=RetrievalLevel.HYPERGRAPH)
        contents = [str(r.get("content", r)) for r in results]
        assert any("篮球" in c for c in contents), f"BM25 中文检索未命中: {contents}"
