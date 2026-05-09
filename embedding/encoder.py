"""
文本嵌入编码器
=============
使用 sentence-transformers 生成 384 维文本嵌入向量。

默认模型: all-MiniLM-L6-v2
- 384 维输出，平衡效率与精度
- 本地部署，无 API 依赖

FAISS 索引过期策略：
- 跟踪索引中的节点 ID 集合
- 梦境阶段检查哪些节点已被修剪，从索引中标记删除
- 每 10 个梦境周期重建一次索引
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class TextEncoder:
    """
    文本嵌入编码器。

    封装 sentence-transformers，提供文本到向量的转换，
    集成 FAISS 索引过期管理。
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._indexed_node_ids: Set[str] = set()
        self._dream_cycle_count: int = 0
        self._needs_rebuild: bool = False

    def load(self) -> None:
        """加载 sentence-transformers 模型（首次调用时自动加载）。"""
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, text: str) -> np.ndarray:
        """将单条文本编码为 embedding 向量，返回 shape (384,) 的 float32 数组。"""
        if self._model is None:
            self.load()
        return self._model.encode(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码文本，返回 shape (len(texts), 384) 的 float32 矩阵。"""
        if self._model is None:
            self.load()
        return self._model.encode(texts)

    @property
    def dimension(self) -> int:
        """返回向量维度。"""
        return 384

    # ─── FAISS 索引过期管理 ───────────────────────────

    def track_indexed_node(self, node_id: str) -> None:
        """记录已索引的节点 ID。"""
        self._indexed_node_ids.add(node_id)

    def remove_pruned_nodes(self, pruned_node_ids: List[str]) -> None:
        """从索引跟踪中移除已修剪的节点。"""
        for nid in pruned_node_ids:
            self._indexed_node_ids.discard(nid)

    def should_rebuild_index(self) -> bool:
        """检查是否应重建 FAISS 索引（每 10 个梦境周期）。"""
        return self._dream_cycle_count >= 10

    def on_dream_cycle_complete(self) -> None:
        """梦境周期完成回调：增加计数，达到阈值时触发重建。"""
        self._dream_cycle_count += 1
        if self.should_rebuild_index():
            logger.info(
                "Rebuilding FAISS index after %d dream cycles", self._dream_cycle_count
            )
            self._dream_cycle_count = 0
            self._needs_rebuild = True

    @property
    def needs_rebuild(self) -> bool:
        """是否需要重建 FAISS 索引（由检索层读取并执行实际重建）。"""
        return self._needs_rebuild

    @needs_rebuild.setter
    def needs_rebuild(self, value: bool) -> None:
        self._needs_rebuild = value

    @property
    def indexed_count(self) -> int:
        """当前索引跟踪的节点数。"""
        return len(self._indexed_node_ids)
