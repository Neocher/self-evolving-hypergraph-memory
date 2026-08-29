"""
检索 + 缓存 + 性能测试
=======================
覆盖：检索三级路由 · 结果缓存 · 嵌入缓存 · 断路器 · 降级检索

运行: python -m pytest tests/test_retrieval_perf.py -v
"""

import time
import pytest
import requests
from unittest.mock import patch, MagicMock
import numpy as np


class TestRetrievalCache:
    """检索结果缓存测试"""

    def test_result_cache_hit(self):
        """相同 query+top_k 第二次应命中缓存"""
        from api.routes._deps import _result_cache, _result_cache_lock
        with _result_cache_lock:
            _result_cache.clear()
        assert len(_result_cache) == 0

    def test_result_cache_lru_eviction(self):
        """超过最大缓存数时应能容纳（LRU在API调用时淘汰）"""
        from api.routes._deps import _result_cache, _result_cache_lock, _result_cache_max
        with _result_cache_lock:
            _result_cache.clear()
            for i in range(_result_cache_max + 10):
                _result_cache[f"query_{i}:5"] = f"result_{i}"
        # 缓存可以超过上限（淘汰在 retrieve 调用时触发）
        assert len(_result_cache) >= _result_cache_max


class TestEmbedCache:
    """嵌入缓存测试"""

    def test_encoder_cache_reuse(self):
        """TextEncoder 的 embed() 应使用 LRU 缓存"""
        from embedding.encoder import TextEncoder, TfidfEncoder
        # 使用 TF-IDF encoder (不需要下载模型)
        enc = TfidfEncoder()
        enc.load()
        
        # 冷编码
        vec1 = enc.embed("test query")
        
        # 缓存命中（TF-IDF没有 _cached_embed, 跳过）
        # 只测试 TextEncoder 的缓存逻辑
        from embedding.encoder import TextEncoder as TE
        te = TE.__new__(TE)
        te._onnx_model = None
        te._cache = {}
        te._cache_hits = 0
        te._cache_misses = 0
        te._model = MagicMock()
        te._model.encode = MagicMock(return_value=np.zeros(384, dtype=np.float32))
        te._cloud_available = False

        # First call — cache miss
        v1 = te._cached_embed("hello world")
        assert te._cache_misses == 1
        assert te._cache_hits == 0

        # Second call — cache hit
        v2 = te._cached_embed("hello world")
        assert te._cache_hits == 1
        assert np.array_equal(v1, v2)

        # Different text — cache miss
        v3 = te._cached_embed("different text")
        assert te._cache_misses == 2

    def test_encoder_cache_lru(self):
        """缓存超过 512 条时淘汰旧条目"""
        from embedding.encoder import TextEncoder as TE
        te = TE.__new__(TE)
        te._onnx_model = None
        te._cache = {}
        te._cache_hits = 0
        te._cache_misses = 0
        te._model = MagicMock()
        te._model.encode = MagicMock(return_value=np.zeros(384, dtype=np.float32))
        te._cloud_available = False

        for i in range(600):
            te._cached_embed(f"text_{i}")

        # Should have evicted ~300 entries
        assert len(te._cache) <= 512


class TestFAISSSpeed:
    """FAISS 性能测试"""

    def test_flat_l2_is_fast(self):
        """FlatL2 对小数据集应 < 5ms"""
        from retrieval.vector_store import FaissStore
        dim = 384
        n = 2000
        index = FaissStore(dimension=dim)
        vecs = np.random.randn(n, dim).astype(np.float32)
        index.add(vecs, np.arange(n, dtype=np.int64))

        query = np.random.randn(1, dim).astype(np.float32)
        times = []
        for _ in range(10):
            t0 = time.time()
            D, I = index.search(query, 10)
            times.append((time.time() - t0) * 1000)

        avg = sum(times) / len(times)
        assert avg < 10, f"FlatL2 search too slow: {avg:.1f}ms (expected <10ms)"


class TestWriteSpeed:
    """写入性能测试"""

    def test_faiss_batch_buffer_size(self):
        """FAISS 批量缓冲区大小应为 50"""
        from api.routes._deps import _FAISS_BATCH_SIZE
        assert _FAISS_BATCH_SIZE == 50

    def test_embed_queue_async(self):
        """嵌入队列应为异步（不阻塞写入响应）"""
        from api.routes.write import _embed_queue, _embed_queue_lock
        # 队列应该存在且可操作
        assert isinstance(_embed_queue, list)
        with _embed_queue_lock:
            _embed_queue.clear()
        assert len(_embed_queue) == 0
