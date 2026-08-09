"""
SHM v4.0 测试基础设施
====================
共享 Fixtures: 临时 GraphLite 数据库、Mock FAISS、Mock Encoder。
"""
from __future__ import annotations

import os
import sys

# 确保能找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import uuid
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """提供临时数据库路径，测试后清理。"""
    tmpdir = tempfile.mkdtemp(prefix="shm_test_")
    db_path = Path(tmpdir) / "test_graphlite"
    yield db_path
    # 清理
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_graphlite_store(temp_db_path: Path):
    """图存储 mock（旧 RyuStore 已删除，引擎为 GraphLite）。

    真实 GraphLiteStore 集成测试请直接构造 GraphLiteStore 实例；
    此 fixture 仅用于不依赖真实图引擎的单元测试。
    """
    store = MagicMock()
    store.query_cypher.return_value = []
    store.get_all_nodes.return_value = {}
    store.get_all_connections.return_value = {}
    return store


@pytest.fixture
def graphlite_store(temp_db_path: Path, request):
    """真实 GraphLiteStore 临时库（本体矛盾检测等集成测试用）。

    支持 indirect parametrize 注入 cb_config：
        @pytest.mark.parametrize('graphlite_store', [cb_cfg], indirect=True)
    """
    from graph.graphlite_store import GraphLiteStore

    config = type("cfg", (), {"database_path": str(temp_db_path), "max_threads": 4})()
    cb_config = getattr(request, 'param', None)
    store = GraphLiteStore(config=config, cb_config=cb_config)
    store.connect()
    yield store
    store.close()


@pytest.fixture
def mock_faiss_index() -> Any:
    """模拟 FAISS 索引（用 numpy 模拟 search/add_with_ids/remove_ids）。"""
    import types

    class MockFaissIndex:
        def __init__(self):
            self.vectors: dict[int, np.ndarray] = {}
            self.ntotal: int = 0

        def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            for vec, fid in zip(vectors, ids):
                self.vectors[int(fid)] = vec.astype(np.float32)
            self.ntotal = len(self.vectors)

        def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            if not self.vectors:
                return (np.array([[float("inf")]]), np.array([[-1]]))
            if query.ndim == 1:
                query = query.reshape(1, -1)
            ids_arr = np.array(list(self.vectors.keys()), dtype=np.int64)
            vecs_arr = np.array(list(self.vectors.values()), dtype=np.float32)
            diffs = vecs_arr - query
            distances = np.linalg.norm(diffs, axis=1)
            top_k = min(k, len(distances))
            sorted_idx = np.argsort(distances)[:top_k]
            return (
                distances[sorted_idx].reshape(1, -1),
                ids_arr[sorted_idx].reshape(1, -1),
            )

        def remove_ids(self, id_selector: np.ndarray) -> int:
            remove_set = set(int(x) for x in id_selector)
            removed = 0
            for fid in list(self.vectors.keys()):
                if fid in remove_set:
                    del self.vectors[fid]
                    removed += 1
            self.ntotal = len(self.vectors)
            return removed

    return MockFaissIndex()


@pytest.fixture
def mock_encoder() -> Any:
    """模拟 TextEncoder，返回固定维度的随机向量。"""
    class MockEncoder:
        def __init__(self):
            self.dim = 384

        def embed(self, text: str) -> np.ndarray:
            # 用文本 hash 产生确定性向量
            rng = np.random.RandomState(hash(text) % (2**31))
            return rng.randn(self.dim).astype(np.float32)

        def embed_batch(self, texts: list[str]) -> np.ndarray:
            return np.array([self.embed(t) for t in texts])

    return MockEncoder()
