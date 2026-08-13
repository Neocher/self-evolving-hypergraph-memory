"""
FAISS 函数测试
=============
测试 routes.py 的 flush_faiss_buffer 和 incremental_faiss_update。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

# 模拟 Services 对象
@dataclass
class MockServices:
    faiss_index: Any = None
    faiss_id_map: dict = field(default_factory=dict)
    _faiss_buffer: list = field(default_factory=list)
    _faiss_buffer_lock: Any = None


class TestFlushFaissBuffer:
    """flush_faiss_buffer 功能测试。"""

    @pytest.fixture
    def deps(self, mock_faiss_index):
        import threading
        s = MockServices(
            faiss_index=mock_faiss_index,
            _faiss_buffer_lock=threading.Lock(),
        )
        return s

    def test_empty_buffer_returns_zero(self, deps):
        """空缓冲区应返回 0。"""
        from api.routes import flush_faiss_buffer
        assert flush_faiss_buffer(deps) == 0

    def test_flush_single_item(self, deps):
        """单条缓冲应写入索引。"""
        from api.routes import flush_faiss_buffer
        emb = np.random.randn(384).astype(np.float32)
        deps._faiss_buffer.append((42, emb, "ep-test-1"))

        count = flush_faiss_buffer(deps)
        assert count == 1
        assert deps.faiss_index.ntotal == 1

    def test_flush_batch_items(self, deps):
        """多条缓冲应批量写入。"""
        from api.routes import flush_faiss_buffer
        for i in range(5):
            emb = np.random.randn(384).astype(np.float32)
            deps._faiss_buffer.append((i, emb, f"ep-batch-{i}"))

        count = flush_faiss_buffer(deps)
        assert count == 5
        assert deps.faiss_index.ntotal == 5

    def test_flush_clears_buffer(self, deps):
        """flush 后缓冲区应清空。"""
        from api.routes import flush_faiss_buffer
        deps._faiss_buffer.append((1, np.random.randn(384).astype(np.float32), "ep-clear"))
        flush_faiss_buffer(deps)
        assert len(deps._faiss_buffer) == 0

    def test_flush_updates_id_map(self, deps):
        """flush 应同步更新 faiss_id_map。"""
        from api.routes import flush_faiss_buffer
        emb = np.random.randn(384).astype(np.float32)
        deps._faiss_buffer.append((42, emb, "ep-123"))

        flush_faiss_buffer(deps)
        assert deps.faiss_id_map.get(42) == "ep-123"

    def test_no_faiss_index_returns_zero(self, deps):
        """无 FAISS 索引时应返回 0。"""
        from api.routes import flush_faiss_buffer
        deps.faiss_index = None
        deps._faiss_buffer.append((1, np.random.randn(384).astype(np.float32)))
        assert flush_faiss_buffer(deps) == 0

    def test_flush_no_id_map(self, deps):
        """无 faiss_id_map 时 flush 不应报错。"""
        from api.routes import flush_faiss_buffer
        deps.faiss_id_map = None
        deps._faiss_buffer.append((1, np.random.randn(384).astype(np.float32)))
        count = flush_faiss_buffer(deps)
        assert count == 0  # 无 faiss_id_map → 无可写缓存项 (M5 语义), 不崩即达标


class TestIncrementalFaissUpdate:
    """incremental_faiss_update 功能测试。"""

    @pytest.fixture
    def deps(self, mock_faiss_index):
        import threading
        s = MockServices(
            faiss_index=mock_faiss_index,
            _faiss_buffer_lock=threading.Lock(),
        )
        # 预先插入一些向量
        for i in range(10):
            fid = uuid.uuid5(uuid.NAMESPACE_OID, f"ep-{i}").int & ((1 << 63) - 1)
            emb = np.random.randn(384).astype(np.float32)
            s.faiss_index.add_with_ids(emb.reshape(1, -1), np.array([fid], dtype=np.int64))
            s.faiss_id_map[fid] = f"ep-{i}"
        return s

    def test_remove_existing_nodes(self, deps):
        """删除已存在的节点应从索引移除。"""
        from api.routes import incremental_faiss_update
        before = deps.faiss_index.ntotal
        count = incremental_faiss_update(deps, ["ep-1", "ep-2"])
        assert count == 2
        assert deps.faiss_index.ntotal == before - 2

    def test_remove_nonexistent_nodes(self, deps):
        """删除不存在的节点应返回 0。"""
        from api.routes import incremental_faiss_update
        count = incremental_faiss_update(deps, ["does-not-exist"])
        assert count == 0

    def test_remove_updates_id_map(self, deps):
        """删除后 faiss_id_map 应同步。"""
        from api.routes import incremental_faiss_update
        before_count = len(deps.faiss_id_map)
        incremental_faiss_update(deps, ["ep-0"])
        assert len(deps.faiss_id_map) == before_count - 1
        # 验证正确的 ID 被移除
        fid = uuid.uuid5(uuid.NAMESPACE_OID, "ep-0").int & ((1 << 63) - 1)
        assert fid not in deps.faiss_id_map

    def test_empty_removed_list(self, deps):
        """空删除列表应返回 0。"""
        from api.routes import incremental_faiss_update
        assert incremental_faiss_update(deps, []) == 0

    def test_no_faiss_index(self, deps):
        """无 FAISS 索引应返回 0。"""
        from api.routes import incremental_faiss_update
        deps.faiss_index = None
        assert incremental_faiss_update(deps, ["ep-0"]) == 0

    def test_remove_multiple_batch(self, deps):
        """批量删除多个节点。"""
        from api.routes import incremental_faiss_update
        before = deps.faiss_index.ntotal
        count = incremental_faiss_update(deps, [f"ep-{i}" for i in range(5)])
        assert count == 5
        assert deps.faiss_index.ntotal == before - 5

    def test_search_after_removal(self, deps):
        """删除后搜索不应返回已删节点。"""
        from api.routes import incremental_faiss_update
        incremental_faiss_update(deps, ["ep-0", "ep-1"])

        query = np.random.randn(1, 384).astype(np.float32)
        _, indices = deps.faiss_index.search(query, 10)
        found_ids = set(int(idx) for idx in indices[0] if int(idx) >= 0)

        fid0 = uuid.uuid5(uuid.NAMESPACE_OID, "ep-0").int & ((1 << 63) - 1)
        fid1 = uuid.uuid5(uuid.NAMESPACE_OID, "ep-1").int & ((1 << 63) - 1)
        assert fid0 not in found_ids
        assert fid1 not in found_ids
