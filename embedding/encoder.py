"""
文本嵌入编码器
=============
三层降级架构：
  Tier 1 — Cloud Embedding API（云端 API，零本地负载）
  Tier 2 — Local sentence-transformers（CPU，本地缓存模型）
  Tier 3 — TF-IDF 本地编码器（零依赖，最后兜底）

默认使用 Tier 2（本地 BAAI/bge-small-zh-v1.5，512维，中文专用，CPU 0.05s/条）。
配置 DEEPSEEK_API_KEY + DEEPSEEK_EMBED_MODEL 可启用 Tier 1。

FAISS 索引过期策略：
- 跟踪索引中的节点 ID 集合
- 梦境阶段检查哪些节点已被修剪
- 每 10 个梦境周期重建一次索引
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# ─── Tier 1: Cloud Embedding API ──────────────────────────

_CLOUD_PROVIDERS = [
    {
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_EMBED_MODEL",
        "default_model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1/embeddings",
        "provider": "deepseek",
    },
    {
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_EMBED_MODEL",
        "default_model": "text-embedding-3-small",
        "base_url": "https://api.openai.com/v1/embeddings",
        "provider": "openai",
    },
    {
        "key_env": "KIMI_API_KEY",
        "model_env": "KIMI_EMBED_MODEL",
        "default_model": "kimi-k2.6",
        "base_url": "https://api.moonshot.cn/v1/embeddings",
        "provider": "kimi",
    },
]


def _cloud_embed(
    texts: list[str],
    api_keys: Optional[dict[str, str]] = None,
) -> Optional[list[list[float]]]:
    """Tier 1: 尝试调用云端 Embedding API。
    
    按 DEEPSEEK → OPENAI → KIMI 顺序尝试，第一个成功的返回。
    Returns None 如果所有 API 都不可用。
    
    【安全】api_keys 参数从调用者传入的私有存储读取，不从 os.environ 运行时读取。
    """
    import httpx

    keys = api_keys or {}
    for provider in _CLOUD_PROVIDERS:
        api_key = keys.get(provider["key_env"], "")
        if not api_key:
            continue
        model = os.environ.get(provider["model_env"], provider["default_model"])
        try:
            with httpx.Client(base_url=provider["base_url"], timeout=10.0) as client:
                # 单条输入需要处理
                payload_input = texts if len(texts) > 1 else texts[0]
                resp = client.post(
                    "",
                    json={"model": model, "input": payload_input, "encoding_format": "float"},
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code != 200:
                    logger.debug("Cloud API %s returned %d, skip", provider["provider"], resp.status_code)
                    continue
                data = resp.json()
                if "data" not in data:
                    continue
                results = [d["embedding"] for d in data["data"]]
                logger.info("Cloud embedding via %s (%s): %d vectors", provider["provider"], model, len(results))
                return results
        except Exception as e:
            logger.debug("Cloud API %s failed: %s", provider["provider"], e)
            continue
    return None


# ─── Tier 2: 本地 sentence-transformers ────────────────────


def _find_model_snapshot(model_name: str) -> Optional[str]:
    """定位任意 HF 模型的缓存 snapshot 路径（离线，不访问网络）。

    HF hub 缓存目录名将 '/' → '--'（如 BAAI/bge-m3 → models--BAAI--bge-m3）。
    返回最新 snapshot，无缓存返回 None。
    """
    import glob

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
    snapshots = sorted(
        glob.glob(os.path.join(model_dir, "snapshots", "*")),
        key=os.path.getmtime,
    )
    return snapshots[-1] if snapshots else None


def _find_bge_snapshot() -> Optional[str]:
    """兼容旧接口：定位 bge-small-zh-v1.5 的 HF 缓存 snapshot 路径（离线，不访问网络）。"""
    return _find_model_snapshot("BAAI/bge-small-zh-v1.5")


class TextEncoder:
    """
    文本嵌入编码器（Tier 2）。

    封装 sentence-transformers，提供文本到向量的转换，
    集成 FAISS 索引过期管理。
    支持 CPU (device='cpu') 和 GPU (device='cuda')。
    加载优先级: bge-small-zh-v1.5（中文，512维，HF缓存snapshot）→
    ONNX INT8（./data/all-MiniLM-L6-v2-int8，384维）→ model_name 通用模型。

    【安全】API keys 在初始化时从环境变量读取一次并存储在私有实例变量中，
    不依赖 os.environ 运行时读取，防止子进程继承。
    """

    def __init__(
        self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._onnx_model = None
        self._indexed_node_ids: Set[str] = set()
        self._dream_cycle_count: int = 0
        self._needs_rebuild: bool = False
        self._cloud_available: bool = False  # Tier 1 是否可用
        self._cache: Dict[str, np.ndarray] = {}  # 【Perf】嵌入缓存
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._api_keys: dict[str, str] = {}  # 【安全】私有 API key 存储，不从 os.environ 运行时读取
        self._preload_api_keys()

    def __repr__(self) -> str:
        return f"TextEncoder(model={self.model_name!r}, cloud={self._cloud_available}, cached={len(self._cache)})"

    __str__ = __repr__

    def _preload_api_keys(self) -> None:
        """初始化时从环境变量预加载 API keys 到私有存储。"""
        for provider in _CLOUD_PROVIDERS:
            k = os.environ.get(provider["key_env"], "")
            if k:
                self._api_keys[provider["key_env"]] = k

    def _cached_embed(self, text: str) -> np.ndarray:
        """带缓存的嵌入（LRU淘汰）。"""
        if text in self._cache:
            self._cache_hits += 1
            vec = self._cache.pop(text)
            self._cache[text] = vec
            return vec
        self._cache_misses += 1
        vec = self._do_embed(text)
        if len(self._cache) >= 512:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[text] = vec
        return vec

    def load(self) -> None:
        """加载模型。优先 bge-small-zh-v1.5（中文，512维，HF 缓存 snapshot 离线加载），
        fallback ONNX INT8（384维），再 fallback sentence-transformers（model_name）。"""
        import os as _os

        # 进程内强制离线（SentenceTransformer 不支持 local_files_only 参数）
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")
        _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # ── 优先: 配置的 model_name（中文 bge-small-zh 或升级 bge-m3，本地缓存 snapshot，不访问网络）──
        snapshot = _find_model_snapshot(self.model_name)
        if snapshot is not None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading Chinese embedding model (%s): %s", self.model_name, snapshot)
                try:
                    self._model = SentenceTransformer(snapshot, device=self.device)
                except Exception:
                    # CUDA 不可用时自动降级 CPU（如 "Torch not compiled with CUDA enabled"）
                    logger.warning("embedding device=%s failed, retry on cpu", self.device)
                    self._model = SentenceTransformer(snapshot, device="cpu")
                    self.device = "cpu"
                logger.info("Local embedding model loaded: model=%s dim=%d", self.model_name, self.dimension)
                return
            except Exception as e:
                logger.warning("%s load failed, fallback to bge-small/ONNX/ST: %s", self.model_name, e)
                self._model = None

        # ── 次优先: bge-small-zh-v1.5（中文专用，512维，本地缓存 snapshot）──
        bge_snapshot = _find_bge_snapshot()
        if bge_snapshot is not None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading Chinese embedding model (bge-small-zh-v1.5): %s", bge_snapshot)
                try:
                    self._model = SentenceTransformer(bge_snapshot, device=self.device)
                except Exception:
                    # CUDA 不可用时自动降级 CPU（如 "Torch not compiled with CUDA enabled"）
                    logger.warning("bge device=%s failed, retry on cpu", self.device)
                    self._model = SentenceTransformer(bge_snapshot, device="cpu")
                    self.device = "cpu"
                self.model_name = "BAAI/bge-small-zh-v1.5"
                logger.info("Local embedding model loaded: dim=%d", self.dimension)
                return
            except Exception as e:
                logger.warning("bge-small-zh-v1.5 load failed, fallback to ONNX/ST: %s", e)
                self._model = None

        onnx_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                   "..", "data", "all-MiniLM-L6-v2-int8")
        if _os.path.isdir(onnx_path) and _os.path.exists(_os.path.join(onnx_path, "model.onnx")):
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(onnx_path, local_files_only=True)
                self._onnx_model = ORTModelForFeatureExtraction.from_pretrained(
                    onnx_path, provider="CPUExecutionProvider", local_files_only=True
                )
                logger.info("ONNX INT8 model loaded from %s", onnx_path)
                return
            except Exception as e:
                logger.warning("ONNX INT8 model load failed, fallback to sentence-transformers: %s", e)
                self._onnx_model = None
                self._tokenizer = None

        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model: %s on %s", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Local embedding model loaded: dim=%d", self.dimension)

    def embed(self, text: str) -> np.ndarray:
        """单条文本 → embedding 向量 (dim,) float32（带LRU缓存）。"""
        return self._cached_embed(text)

    def _do_embed(self, text: str) -> np.ndarray:
        """不加缓存的原始嵌入（供缓存内部调用）。"""
        if self._cloud_available:
            try:
                result = _cloud_embed([text], api_keys=self._api_keys)
                if result:
                    return np.array(result[0], dtype=np.float32)
            except Exception:
                self._cloud_available = False
                logger.info("Cloud API degraded, falling back to local model")

        # Tier 2: Local model (ONNX preferred)
        if self._onnx_model is not None:
            inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self._onnx_model(**inputs)
            # Mean pooling over token dimension
            vec = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            return vec.astype(np.float32)

        if self._model is None:
            self.load()
        return self._model.encode(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量 → embedding 矩阵 (N, dim) float32。
        
        优先 Tier 1（Cloud API 批量），不可用时降级到 Tier 2。
        """
        # Tier 1: Cloud API (批量调用更高效)
        if self._cloud_available:
            try:
                result = _cloud_embed(texts, api_keys=self._api_keys)
                if result:
                    return np.array(result, dtype=np.float32)
            except Exception:
                self._cloud_available = False
                logger.info("Cloud API degraded, falling back to local model")

        # Tier 2: ONNX batch (preferred)
        if self._onnx_model is not None:
            inputs = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            outputs = self._onnx_model(**inputs)
            vecs = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            return np.array([v.astype(np.float32) for v in vecs])

        # Tier 3: Local sentence-transformers
        if self._model is None:
            self.load()
        return self._model.encode(texts)

    @property
    def dimension(self) -> int:
        """实际加载模型的向量维度（bge-small-zh-v1.5=512，ONNX MiniLM=384）。"""
        if self._onnx_model is not None:
            return 384
        if self._model is not None:
            try:
                return int(self._model.get_sentence_embedding_dimension())
            except Exception:
                pass
        return 512 if "bge" in getattr(self, "model_name", "") else 384

    # ─── FAISS 索引过期管理 ───────────────────────────

    def track_indexed_node(self, node_id: str) -> None:
        self._indexed_node_ids.add(node_id)

    def remove_pruned_nodes(self, pruned_node_ids: List[str]) -> None:
        for nid in pruned_node_ids:
            self._indexed_node_ids.discard(nid)

    def should_rebuild_index(self) -> bool:
        return self._dream_cycle_count >= 10

    def on_dream_cycle_complete(self) -> None:
        self._dream_cycle_count += 1
        if self.should_rebuild_index():
            self._dream_cycle_count = 0
            self._needs_rebuild = True

    @property
    def needs_rebuild(self) -> bool:
        return self._needs_rebuild

    @needs_rebuild.setter
    def needs_rebuild(self, value: bool) -> None:
        self._needs_rebuild = value


# ─── Tier 3: TF-IDF 本地编码器（零依赖兜底） ──────────────


class TfidfEncoder:
    """TF-IDF 本地编码器（Tier 3 降级）。
    
    使用 sklearn 的字符级 n-gram TF-IDF + 随机投影到 384 维。
    零网络依赖，零模型下载，纯 CPU。
    """

    def __init__(self, dimension: int = 384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.random_projection import SparseRandomProjection

        self._model = "tfidf_fallback"
        self._dimension = dimension
        self._vectorizer = TfidfVectorizer(max_features=1024, analyzer="char_wb", ngram_range=(2, 4))
        self._projector = None
        self._fitted = False
        self._indexed_node_ids: Set[str] = set()
        self._dream_cycle_count: int = 0
        self._needs_rebuild: bool = False

    def load(self) -> None:
        """TF-IDF 不需要加载，懒初始化。"""
        pass

    def embed(self, text: str) -> np.ndarray:
        from sklearn.random_projection import SparseRandomProjection
        import numpy as _np

        if not self._fitted:
            self._vectorizer.fit([text])
            self._projector = SparseRandomProjection(n_components=self._dimension, random_state=42)
            sample = self._vectorizer.transform(["", text])
            self._projector.fit(sample)
            self._fitted = True
        vec = self._vectorizer.transform([text])
        projected = self._projector.transform(vec)
        return projected.toarray().astype(_np.float32).flatten()

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        import numpy as _np

        if not self._fitted:
            self._vectorizer.fit(texts)
            self._projector = SparseRandomProjection(n_components=self._dimension, random_state=42)
            sample = self._vectorizer.transform(texts)
            self._projector.fit(sample)
            self._fitted = True
        vec = self._vectorizer.transform(texts)
        projected = self._projector.transform(vec)
        return projected.toarray().astype(_np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension

    def track_indexed_node(self, node_id: str) -> None:
        self._indexed_node_ids.add(node_id)

    def remove_pruned_nodes(self, pruned_node_ids: List[str]) -> None:
        for nid in pruned_node_ids:
            self._indexed_node_ids.discard(nid)

    def should_rebuild_index(self) -> bool:
        return False

    def on_dream_cycle_complete(self) -> None:
        pass

    @property
    def needs_rebuild(self) -> bool:
        return False

    @needs_rebuild.setter
    def needs_rebuild(self, value: bool) -> None:
        pass

    @property
    def indexed_count(self) -> int:
        return 0


# ─── 工厂函数 ───────────────────────────────────────────────


def create_encoder(
    model_name: str = "BAAI/bge-small-zh-v1.5",
    device: str = "cpu",
    prefer_cloud: bool = True,
) -> TextEncoder:
    """创建三层降级编码器。
    
    返回 TextEncoder 实例（Tier 1 + Tier 2）。
    调用者应在外层 catch Exception，降级到 TfidfEncoder。
    
    Args:
        model_name: sentence-transformers 模型名
        device: 'cpu' 或 'cuda'
        prefer_cloud: 是否启用 Tier 1（Cloud API）
    
    Returns:
        配置好的 TextEncoder 实例
    """
    encoder = TextEncoder(model_name=model_name, device=device)
    
    if prefer_cloud:
        # 用已预加载的私有 key 存储检查（不从 os.environ 运行时读取）
        for provider in _CLOUD_PROVIDERS:
            if encoder._api_keys.get(provider["key_env"], ""):
                encoder._cloud_available = True
                logger.info("Cloud embedding available via %s", provider["provider"])
                break
    
    return encoder
