"""
向量存储可插拔层
================
提供统一的 BaseVectorStore 抽象基类，使向量检索后端可替换。
当前封装 FAISS 为 FaissStore，后续可接入 Milvus / Qdrant / Pinecone / Chroma 等。

环境变量:
    SHM_VECTOR_STORE: 向量存储引擎类型 (默认: "faiss")
    SHM_FAISS_DIMENSION: 向量维度 (默认: 512)
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
    """纯 numpy FlatL2 向量存储（替代 faiss.IndexFlatL2 + IndexIDMap）。

    faiss 语义等价：精确暴力 L2（平方距离，与 IndexFlatL2 一致）、IndexIDMap
    用户 id 反查、search 返回 (distances, indices) 双 (1,k) 数组（不足 k 补
    inf/-1）。视觉召回通道（384d CLIP 投影空间）唯一索引实现——原为真 FAISS，
    迁移至 numpy 后行为逐位一致（FlatL2 本就是精确暴力搜索，无 IVF/量化）。

    保持与原始 faiss.IndexIDMap 的兼容性：
    - .index 属性暴露自身（faiss.Index 鸭子类型：search/ntotal/add_with_ids/remove_ids）
    - .id_map 属性管理 faiss_id → node_id 的映射
    """

    def __init__(
        self,
        dimension: int = 512,
        index_type: str = "FlatL2",
        nlist: int = 100,
    ):
        import threading

        self._dim = int(dimension)
        self._index_type = index_type
        self._nlist = nlist
        self._lock = threading.Lock()
        # 向量矩阵 (n, dim) + 用户 id 数组 (n,)：append-only（IndexIDMap 语义，
        # search 返回用户 id 而非行号）
        self._vectors = np.empty((0, self._dim), dtype=np.float32)
        self._ids = np.empty((0,), dtype=np.int64)
        self._id_map: dict[int, str] = {}

    # ── 兼容旧代码的直接访问 ──

    @property
    def index(self):
        """暴露自身（faiss.Index 鸭子类型：search/ntotal/add_with_ids/remove_ids）。"""
        return self

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
        return int(self._vectors.shape[0])

    @property
    def ntotal(self) -> int:
        """faiss.Index.ntotal 等价物（当前向量数）。"""
        return self.count

    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> int:
        with self._lock:
            emb = np.asarray(embeddings, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            ids_arr = np.asarray(ids, dtype=np.int64).reshape(-1)
            n = min(emb.shape[0], ids_arr.shape[0])
            if n == 0:
                return 0
            self._vectors = np.concatenate(
                [self._vectors, emb[:n].astype(np.float32)], axis=0
            )
            self._ids = np.concatenate([self._ids, ids_arr[:n]], axis=0)
        return n

    def add_with_ids(self, embeddings: np.ndarray, ids: np.ndarray) -> int:
        """faiss.IndexIDMap.add_with_ids 等价物（内部 id → 用户 id 映射）。"""
        return self.add(embeddings, ids)

    def remove(self, ids: np.ndarray) -> int:
        return self.remove_ids(ids)

    def remove_ids(self, ids: np.ndarray) -> int:
        with self._lock:
            remove = set(np.asarray(ids, dtype=np.int64).reshape(-1).tolist())
            if not remove:
                return 0
            keep = ~np.isin(
                self._ids, np.fromiter(remove, dtype=np.int64, count=len(remove))
            )
            removed = int((~keep).sum())
            self._vectors = self._vectors[keep]
            self._ids = self._ids[keep]
            for fid in remove:
                self._id_map.pop(fid, None)
        return removed

    def search(
        self, query: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            q = np.asarray(query, dtype=np.float32).reshape(1, -1)
            k = int(k)
            if k <= 0:
                return (np.empty((1, 0), dtype=np.float32),
                        np.empty((1, 0), dtype=np.int64))
            n = self._vectors.shape[0]
            if n == 0:
                return (np.full((1, k), float("inf"), dtype=np.float32),
                        np.full((1, k), -1, dtype=np.int64))
            # 平方 L2 距离（faiss.IndexFlatL2 语义）
            diff = self._vectors - q  # (n, dim)
            dists = np.einsum("ij,ij->i", diff, diff)  # (n,)
            if n > k:
                idx = np.argpartition(dists, k - 1)[:k]
                idx = idx[np.argsort(dists[idx])]
            else:
                idx = np.argsort(dists)
            top_d = dists[idx].astype(np.float32)
            top_i = self._ids[idx].astype(np.int64)
            if n < k:
                top_d = np.concatenate(
                    [top_d, np.full((k - n,), float("inf"), dtype=np.float32)]
                )
                top_i = np.concatenate(
                    [top_i, np.full((k - n,), -1, dtype=np.int64)]
                )
            return (top_d.reshape(1, -1), top_i.reshape(1, -1))


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
        dimension: int = 512,
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
