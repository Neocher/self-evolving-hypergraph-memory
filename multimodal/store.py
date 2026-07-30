"""
媒体文件存储
==========
MediaStore — 本地文件存储 + 可选 S3 适配器。

文件路径嵌入到 Episode metadata 中，供检索时回查。
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MediaStore:
    """媒体文件存储。

    默认存储在 data/media/ 目录，按日期分片。
    可选 S3 兼容适配（通过 put/sign 回调）。

    用法:
        store = MediaStore()
        path = store.save(b"..." , ".jpg")  # → "data/media/20260728/abc123.jpg"
    """

    def __init__(
        self,
        base_dir: str | Path = "./data/media",
        s3_put: Optional[callable] = None,
        s3_sign: Optional[callable] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._s3_put = s3_put      # async callable(key, data) → str URL
        self._s3_sign = s3_sign    # async callable(key) → str signed URL

    # ─── 本地存储 ──────────────────────────────────────────

    def save(self, data: bytes, suffix: str = ".bin", prefix: str = "") -> str:
        """将数据保存到本地文件。

        Args:
            data: 文件字节。
            suffix: 文件扩展名（如 .jpg, .wav）。
            prefix: 可选文件名前缀。

        Returns:
            存储的相对路径（如 data/media/20260728/abc123.jpg）。
        """
        date_dir = time.strftime("%Y%m%d")
        target_dir = self.base_dir / date_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        content_hash = hashlib.blake2b(data, digest_size=16).hexdigest()
        filename = f"{prefix}{content_hash}{suffix}" if prefix else f"{content_hash}{suffix}"
        rel_path = f"{date_dir}/{filename}"
        full_path = self.base_dir / rel_path

        if not full_path.exists():
            full_path.write_bytes(data)
            logger.debug("Media saved: %s (%d bytes)", full_path, len(data))

        # 返回相对路径（供 metadata 存储）
        return rel_path

    def save_image(self, data: bytes) -> str:
        """便捷方法：保存图像文件。"""
        return self.save(data, suffix=".jpg", prefix="img_")

    def save_audio(self, data: bytes) -> str:
        """便捷方法：保存音频文件。"""
        return self.save(data, suffix=".wav", prefix="aud_")

    def save_video(self, data: bytes) -> str:
        """便捷方法：保存视频文件。"""
        return self.save(data, suffix=".mp4", prefix="vid_")

    def get_local_path(self, rel_path: str) -> Optional[Path]:
        """将存储的相对路径解析为本地绝对路径。

        Args:
            rel_path: save() 返回的相对路径。

        Returns:
            本地 Path 对象，文件不存在时返回 None。
        """
        full = self.base_dir / rel_path
        if full.exists() and full.is_file():
            return full.resolve()
        return None

    # ─── S3 适配 ───────────────────────────────────────────

    async def save_s3(self, data: bytes, key: str) -> Optional[str]:
        """通过 S3 适配器上传文件。

        Args:
            data: 文件字节。
            key: S3 对象键。

        Returns:
            公开 URL 或 signed URL，无适配器时返回 None。
        """
        if self._s3_put is None:
            logger.debug("S3 adapter not configured, skip upload")
            return None
        try:
            url = await self._s3_put(key, data)
            logger.debug("S3 upload: %s → %s", key, url)
            return url
        except Exception as exc:
            logger.warning("S3 upload failed: %s", exc)
            return None

    async def sign_url(self, key: str) -> Optional[str]:
        """获取 S3 签名 URL（临时访问凭证）。

        Args:
            key: S3 对象键。

        Returns:
            签名 URL，不可用时返回 None。
        """
        if self._s3_sign is None:
            return None
        try:
            return await self._s3_sign(key)
        except Exception as exc:
            logger.warning("S3 sign URL failed: %s", exc)
            return None

    # ─── 健康 ──────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """检查本地存储是否可用。"""
        return self.base_dir.exists()

    @property
    def total_size_bytes(self) -> int:
        """估算本地媒体文件总大小（不递归子目录）。"""
        total = 0
        for f in self.base_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
