"""端到端: 真实 bge + 真实 FAISS + 真实 GraphLite 中文向量检索闭环

验证: bge-small-zh-v1.5 切换后, 中文语义检索真实命中 (非噪声)。
标记 slow: 需要加载 bge 模型 (~5s)。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import pytest

pytestmark = pytest.mark.slow

DOCS = [
    "张三喜欢吃苹果和梨",
    "李四喜欢打篮球和跑步",
    "王五喜欢读书和写作",
    "公司发布了新产品公告",
    "团队完成了季度总结报告",
]


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


@pytest.fixture
def _enc():
    from embedding.encoder import create_encoder

    enc = create_encoder()
    enc.load()
    return enc


class TestRealChineseVectorRetrieval:
    """真实 bge + FAISS + GraphLite 全链路中文检索"""

    def test_real_chain_chinese_hits(self, _enc, graphlite_store):
        from retrieval.vector_store import FaissStore
        from retrieval.query_router import (
            QueryRouter, QueryRouterConfig as QRCfg, RetrievalLevel,
        )

        ids = []
        for i, c in enumerate(DOCS):
            eid = graphlite_store.create_episode({
                "content": c, "created_at": i + 1.0, "tau_initial": 1.0,
                "tau_value": 0.6, "source": "e2e", "trust_score": 0.8,
            })
            ids.append(eid)

        vs = FaissStore(dimension=_enc.dimension)
        faiss_nums = np.arange(len(DOCS), dtype=np.int64)
        vs.add(np.asarray(_enc.embed_batch(DOCS), dtype=np.float32), faiss_nums)
        faiss_id_map = {int(faiss_nums[i]): ids[i] for i in range(len(ids))}

        qr = QueryRouter(
            kuzu_store=graphlite_store,
            faiss_index=vs,
            tfidf_index=_make_tfidf(DOCS),
            encoder=_enc,
            faiss_id_map=faiss_id_map,
            episode_cache={},
            config=QRCfg(tau_weight=0.2, vector_weight=0.3, top_k_l1=10, top_k_vector=5, top_k_keyword=5),
        )

        for query, expected in [
            ("篮球", "李四喜欢打篮球"),
            ("读书", "王五喜欢读书"),
            ("苹果", "张三喜欢吃苹果"),
        ]:
            results = qr.retrieve(query, level=RetrievalLevel.HYPERGRAPH)
            contents = [str(r.get("content", r)) for r in results]
            assert any(expected in c for c in contents), (
                f'查"{query}" 未命中 "{expected}": {contents[:2]}'
            )

    def test_real_chain_faiss_dim_matches_encoder(self, _enc):
        from retrieval.vector_store import FaissStore

        vs = FaissStore(dimension=_enc.dimension)
        assert vs.dimension == _enc.dimension == 512
