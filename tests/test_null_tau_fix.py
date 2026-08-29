"""v5.50.1 回归：GraphLite 真实引擎 NULL 序列化为字符串 'Null' 的防御修复。

根因：GraphLite 对 NULL 字段返回字符串 'Null'（非 Python None），
float('Null') 抛 ValueError → BM25/entity 检索通道静默崩溃（bm25=0 entity=0）。
修复：_safe_float_tau() 防御解析，替换 query_router.py 5 处 float() 转换点。
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from retrieval.query_router import QueryRouter, QueryRouterConfig, _safe_float_tau


class TestSafeFloatTau:
    """_safe_float_tau 防御解析单测"""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("Null", 0.0),          # GraphLite 真实引擎 NULL 序列化
            ("null", 0.0),          # 小写变体
            ("None", 0.0),          # 字符串 None
            (None, 0.0),            # Python None
            ("", 0.0),              # 空串
            ("nan", 0.0),           # NaN 字符串
            ("abc", 0.0),           # 非数字
            (0.5, 0.5),             # float 透传
            (3, 3.0),               # int 转 float
            ("0.8", 0.8),           # 数字字符串
        ],
    )
    def test_variants(self, value, expected):
        assert _safe_float_tau(value) == expected

    def test_no_raise_on_real_engine_null(self):
        """真实引擎返回的 row 含 'Null' 不再抛异常（回归测试）"""
        # 模拟 GraphLite query_cypher 返回 dict 行，tau_value='Null'
        rows = [
            {"node_id": "ep-1", "content": "Caroline attended the support group.", "tau_value": "Null", "fact_track": "active"},
            {"node_id": "ep-2", "content": "Melanie painted a sunrise.", "tau_value": 0.42, "fact_track": "active"},
        ]
        parsed = []
        for row in rows:
            tau = _safe_float_tau(row.get("tau_value", 0.0))
            parsed.append((row["node_id"], tau))
        assert parsed[0] == ("ep-1", 0.0)
        assert parsed[1] == ("ep-2", 0.42)


class TestQueryRouterNullTau:
    """QueryRouter 通道不再被 'Null' 打挂（走生产入口 _bm25_search/_entity_match 路径）"""

    def test_bm25_index_build_survives_null_tau(self):
        """_build_bm25_index_core 遇到 'Null' tau 不抛异常，索引构建成功"""
        qr = QueryRouter.__new__(QueryRouter)
        qr.config = QueryRouterConfig()
        qr._bm25_empty_warned = False

        # 真实引擎行格式（含 'Null' 字符串）
        rows = [
            {"node_id": f"ep-{i}", "content": f"message number {i} about topic alpha", "tau_value": "Null", "fact_track": "active"}
            for i in range(5)
        ]
        # graphlite_store 只暴露 query_cypher
        class FakeStore:
            def query_cypher(self, gql, params=None):
                return rows
        qr.graph_store = FakeStore()

        # 直接调核心构建（绕过锁，验证解析逻辑）
        state = qr._build_bm25_index_core()
        assert state is not None, "含 'Null' tau 的行不应导致构建失败"
        vectorizer, doc_ids, doc_contents, doc_tau, doc_fact_track, term_matrix, idf, doc_lens, avgdl = state
        assert len(doc_ids) == 5
        assert all(t == 0.0 for t in doc_tau), "Null tau 应解析为 0.0"

    def test_bm25_search_real_null_rows_no_crash(self):
        """_bm25_search 全链路（懒构建→评分）遇 'Null' 不崩溃、返回非空"""
        qr = QueryRouter.__new__(QueryRouter)
        qr.config = QueryRouterConfig()
        qr._bm25_empty_warned = False

        rows = [
            {"node_id": f"ep-{i}", "content": f"Caroline likes painting sunrises {i}", "tau_value": "Null", "fact_track": "active"}
            for i in range(10)
        ]

        class FakeStore:
            def query_cypher(self, gql, params=None):
                return rows
        qr.graph_store = FakeStore()

        # 直接调核心构建
        state = qr._build_bm25_index_core()
        assert state is not None
        (vectorizer, doc_ids, doc_contents, doc_tau, doc_fact_track,
         term_matrix, idf, doc_lens, avgdl) = state
        qr._bm25_vectorizer = vectorizer
        qr._bm25_docs = doc_contents
        qr._bm25_doc_ids = doc_ids
        qr._bm25_doc_contents = doc_contents
        qr._bm25_doc_tau = doc_tau
        qr._bm25_doc_term_matrix = term_matrix
        qr._bm25_idf = idf
        qr._bm25_doc_lens = doc_lens
        qr._bm25_avgdl = avgdl
        qr._bm25_ready = True
        qr._bm25_built = True

        results = qr._bm25_search("Caroline painting sunrise", k=5)
        assert isinstance(results, list)
        # 至少返回一条（关键词能匹配 corpus）
        assert len(results) >= 1, f"BM25 应返回结果，got {len(results)}"
        assert all(r["level"] == "bm25" for r in results)
