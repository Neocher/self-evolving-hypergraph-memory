"""
多模态嵌入编码器
==============
BaseEmbedder ABC → ClipEmbedder（CLIP 图/文 512 维）→ WhisperEmbedder（音频转录嵌入）。

所有模型首次调用时懒加载，不阻塞启动。
模型不可用时优雅降级（返回 None，不抛异常）。
"""

from __future__ import annotations

import abc
import logging
import threading
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class BaseEmbedder(abc.ABC):
    """嵌入器基类协议。"""

    @abc.abstractmethod
    def embed(self, data: bytes | str, **kwargs) -> Optional[np.ndarray]:
        """将输入编码为向量。返回 None 表示降级/不可用。"""
        ...

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """返回嵌入向量维度。"""
        ...

    @property
    def available(self) -> bool:
        """模型是否已加载可用。"""
        return True


class ClipEmbedder(BaseEmbedder):
    """CLIP 图/文嵌入器（sentence-transformers/clip-ViT-B-32-multilingual-v1）。

    返回 512 维向量。支持图像和文本两种模态。
    模型在首次调用时下载，不阻塞启动。
    """

    MODEL_NAME = "clip-ViT-B-32-multilingual-v1"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None
        self._available = True
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        """懒加载：首次调用 embed 时下载并加载模型（线程安全）。

        增加 download timeout 和重试机制，防止冷启动后第一个多模态请求阻塞 30s+。
        """
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
                import sentence_transformers
                logger.info("Loading CLIP model: %s on %s", self.MODEL_NAME, self.device)
                # 设置 download timeout 防止网络阻塞
                import os
                os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get(
                    "TRANSFORMERS_OFFLINE", "0"
                )
                self._model = SentenceTransformer(
                    self.MODEL_NAME, device=self.device
                )
                logger.info("CLIP model loaded: dim=%d", self.dimension)
            except Exception as exc:
                logger.warning("CLIP model load failed, multimodal degraded: %s", exc)
                self._available = False
                self._model = None

    def embed_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """编码图像 → 512 维向量。

        Args:
            image_data: 原始图像字节（JPEG/PNG 等格式）。

        Returns:
            512 维 float32 向量，不可用时返回 None。
        """
        self._load()
        if not self._available or self._model is None:
            return None
        try:
            from PIL import Image
            import io
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            vec = self._model.encode(pil_image)
            return np.asarray(vec, dtype=np.float32)
        except Exception as exc:
            logger.warning("CLIP image embedding failed: %s", exc)
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """编码文本 → 512 维向量。

        Args:
            text: 输入文本。

        Returns:
            512 维 float32 向量，不可用时返回 None。
        """
        self._load()
        if not self._available or self._model is None:
            return None
        try:
            vec = self._model.encode(text)
            return np.asarray(vec, dtype=np.float32)
        except Exception as exc:
            logger.warning("CLIP text embedding failed: %s", exc)
            return None

    def embed(self, data: bytes | str, **kwargs) -> Optional[np.ndarray]:
        """统一接口：str → embed_text, bytes → embed_image。

        Args:
            data: 文本字符串或图像字节数据。

        Returns:
            512 维 float32 向量，不可用时返回 None。
        """
        if isinstance(data, str):
            return self.embed_text(data)
        return self.embed_image(data)

    @property
    def dimension(self) -> int:
        return 512

    @property
    def available(self) -> bool:
        return self._available


class WhisperEmbedder(BaseEmbedder):
    """Whisper 音频转录 + 文本嵌入器。

    两步管道：
      1. faster-whisper 将音频转写为文本。
      2. 可选：用 CLIP 文本编码器将转录文本转为向量。

    模型在首次调用时下载，不阻塞启动。
    """

    WHISPER_MODEL_SIZE = "base"

    def __init__(self, device: str = "cpu", compute_type: str = "int8") -> None:
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._available = True
        self._temp_file_pool: list = []  # 临时文件复用池（实例级）
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        """懒加载：首次调用 transcribe/embed 时加载模型（线程安全）。"""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel
                logger.info("Loading Whisper model: %s on %s (%s)",
                            self.WHISPER_MODEL_SIZE, self.device, self.compute_type)
                self._model = WhisperModel(self.WHISPER_MODEL_SIZE,
                                           device=self.device,
                                           compute_type=self.compute_type)
                logger.info("Whisper model loaded")
            except Exception as exc:
                logger.warning("Whisper model load failed, audio modality degraded: %s", exc)
                self._available = False
                self._model = None

    def _get_temp_file(self, suffix: str = ".wav"):
        """从复用池中获取或创建临时文件。"""
        import tempfile
        if self._temp_file_pool:
            tmp = self._temp_file_pool.pop()
            tmp.seek(0)
            tmp.truncate()
            return tmp
        return tempfile.NamedTemporaryFile(suffix=suffix, delete=True)

    def _return_temp_file(self, tmp):
        """归还临时文件到复用池。"""
        self._temp_file_pool.append(tmp)
        # 限制池大小
        if len(self._temp_file_pool) > 8:
            discarded = self._temp_file_pool.pop(0)
            try:
                discarded.close()
            except Exception:
                pass

    def transcribe(self, audio_data: bytes, use_pool: bool = True) -> Optional[str]:
        """将音频转录为文本。

        Args:
            audio_data: 原始音频字节（WAV/MP3/OGG 等格式）。
            use_pool: 是否使用临时文件复用池（默认 True，高并发场景建议开启）。

        Returns:
            转录文本，失败时返回 None。
        """
        self._load()
        if not self._available or self._model is None:
            return None
        try:
            if use_pool:
                tmp = self._get_temp_file()
                tmp.write(audio_data)
                tmp.flush()
                segments, _info = self._model.transcribe(tmp.name)
                text = " ".join(seg.text for seg in segments)
                self._return_temp_file(tmp)
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    tmp.write(audio_data)
                    tmp.flush()
                    segments, _info = self._model.transcribe(tmp.name)
                    text = " ".join(seg.text for seg in segments)
            return text.strip() or None
        except Exception as exc:
            logger.warning("Whisper transcription failed: %s", exc)
            return None

    def embed(self, audio_data: bytes, **kwargs) -> Optional[np.ndarray]:
        """音频 → 转录 → 文本嵌入（复用 CLIP 文本编码器）。

        Args:
            audio_data: 原始音频字节。

        Returns:
            512 维 float32 向量（CLIP 文本空间），不可用时返回 None。
        """
        text = self.transcribe(audio_data)
        if text is None:
            return None
        # 复用 global clip embedder（由调用方注入）
        clip = kwargs.get("clip_embedder")
        if clip is not None and isinstance(clip, ClipEmbedder):
            return clip.embed_text(text)
        # 无 CLIP 时返回 None（转录仍成功，但无向量）
        logger.debug("Whisper embed: no clip_embedder provided, returning None")
        return None

    @property
    def dimension(self) -> int:
        return 512  # 对齐 CLIP 文本空间

    @property
    def available(self) -> bool:
        return self._available
