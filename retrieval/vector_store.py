"""
向量存储可插拔层
================
提供统一的 BaseVectorStore 抽象基类，使向量检索后端可替换。
当前封装 FAISS 为 FaissStore，后续可接入 Milvus / Qdrant / Pinecone / Chroma 等。

环境变量:
    SHM_VECTOR_STORE: 向量存储引擎类型 (默认: "faiss")
    SHM_FAISS_DIMENSION: 向量维度 (默认: 384)
"""

from __future__ import annotations

import abc
import os
from typing import Optional

import numpy as np


class BaseVectorStore(abc.ABC):
    """向量存储抽象基类。所有向量后端必须实现此接口。"""

    @abc.abstractmethod
    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> int:
        """批量添加向量。

        Args:
            embeddings: shape (N, dim) 的 float32 数组
            ids: shape (N,) 的 int64 数组

        Returns:
            实际添加的向量数
        """
        ...

    @abc.abstractmethod
    def remove(self, ids: np.ndarray) -> int:
        """按 ID 删除向量。

        Args:
            ids: shape (N,) 的 int64 数组

        Returns:
            实际删除的向量数
        """
        ...

    @abc.abstractmethod
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """搜索最近邻。

        Args:
            query: shape (1, dim) 的 float32 查询向量
            k: 返回 top-k 结果

        Returns:
            (distances, indices) 两个 shape (1, k) 的数组
            distances[i] 是第 i 个结果与 query 的距离
            indices[i] 是对应的向量 ID（-1 表示无结果）
        """
        ...

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """向量维度。"""
        ...

    @property
    @abc.abstractmethod
    def count(self) -> int:
        """当前存储的向量总数。"""
        ...


class FaissStore(BaseVectorStore):
    """封装 FAISS 的向量存储实现。

    保持与原始 faiss.IndexIDMap 的兼容性：
    - .index 属性暴露底层 FAISS 索引，供需要直接操作的代码使用
    - .id_map 属性管理 faiss_id → node_id 的映射
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "FlatL2",
        nlist: int = 100,
    ):
        import faiss
        import threading

        self._dim = dimension
        self._index_type = index_type
        self._nlist = nlist
        self._lock = threading.Lock()

        base_index = faiss.IndexFlatL2(dimension)
        self._index = faiss.IndexIDMap(base_index)
        self._id_map: dict[int, str] = {}

    # ── 兼容旧代码的直接访问 ──

    @property
    def index(self):  # → faiss.Index
        """暴露底层 FAISS 索引，供需要直接操作 FAISS 的代码使用。"""
        return self._index

    @property
    def id_map(self) -> dict[int, str]:
        """faiss_id → node_id 映射。"""
        return self._id_map

    @id_map.setter
    def id_map(self, value: dict[int, str]) -> None:
        self._id_map = value

    @property
    def index_type(self) -> str:
        return self._index_type

    @index_type.setter
    def index_type(self, value: str) -> None:
        self._index_type = value

    @property
    def nlist(self) -> int:
        return self._nlist

    @nlist.setter
    def nlist(self, value: int) -> None:
        self._nlist = value

    # ── BaseVectorStore 接口 ──

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return self._index.ntotal

    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> int:
        with self._lock:
            self._index.add_with_ids(embeddings, ids)
        return len(ids)

    def remove(self, ids: np.ndarray) -> int:
        with self._lock:
            return self._index.remove_ids(ids)

    def search(
        self, query: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._index.search(query, k)


class VectorStoreFactory:
    """向量存储工厂——根据配置创建对应的 BaseVectorStore 实例。"""

    _STORE_REGISTRY: dict[str, type[BaseVectorStore]] = {
        "faiss": FaissStore,
    }

    @classmethod
    def register(cls, name: str, store_cls: type[BaseVectorStore]) -> None:
        """注册自定义向量存储实现。"""
        cls._STORE_REGISTRY[name.lower()] = store_cls

    @classmethod
    def create(
        cls,
        dimension: int = 384,
        index_type: str = "FlatL2",
        nlist: int = 100,
        engine: Optional[str] = None,
    ) -> BaseVectorStore:
        """创建向量存储实例。

        Args:
            dimension: 向量维度
            index_type: 索引类型 (FAISS 专用)
            nlist: IVF 聚类数 (FAISS 专用)
            engine: 引擎类型。None 表示从 SHM_VECTOR_STORE 环境变量读取

        Returns:
            BaseVectorStore 实例

        Raises:
            ValueError: 未知的引擎类型
        """
        if engine is None:
            engine = os.environ.get("SHM_VECTOR_STORE", "faiss").lower()

        store_cls = cls._STORE_REGISTRY.get(engine)
        if store_cls is None:
            raise ValueError(
                f"Unknown vector store engine: {engine!r}. "
                f"Available: {list(cls._STORE_REGISTRY.keys())}"
            )

        # faiss 系列需要 dimension/index_type/nlist 参数
        if engine == "faiss":
            return store_cls(
                dimension=dimension,
                index_type=index_type,
                nlist=nlist,
            )

        return store_cls()
