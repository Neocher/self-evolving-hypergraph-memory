"""
多模态记忆系统
=============
CLIP 图/文嵌入 + Whisper 音频转录 + 媒体文件存储。

子模块:
    embedders   — BaseEmbedder ABC, ClipEmbedder, WhisperEmbedder
    store       — MediaStore（本地文件 + 可选 S3）
"""

# 惰性导入（避免 CLI 等轻量入口触发 torch/sentence-transformers 等重依赖）
def __getattr__(name):
    if name == "ClipEmbedder":
        from multimodal.embedders import ClipEmbedder as _cls
        return _cls
    if name == "WhisperEmbedder":
        from multimodal.embedders import WhisperEmbedder as _cls
        return _cls
    if name == "MediaStore":
        from multimodal.store import MediaStore as _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ClipEmbedder", "WhisperEmbedder", "MediaStore"]
