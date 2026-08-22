"""
文本嵌入编码器
=============
三层降级架构：
  Tier 1 — Cloud Embedding API（云端 API，零本地负载）
  Tier 2 — Local sentence-transformers（CPU，本地缓存模型）
  Tier 3 — TF-IDF 本地编码器（零依赖，最后兜底）

默认使用 Tier 2（本地 BAAI/bge-m3，1024d 多语言，MRL 截断 512 匹配 HNSW 契约；
ONNX O2 CPU 优化版约数十 ms/条）。配置 DEEPSEEK_API_KEY + DEEPSEEK_EMBED_MODEL 可启用 Tier 1。

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

# ─── bge-m3 常量（v6.1 embedding 升级）────────────────────
_BGE_M3_MODEL = "BAAI/bge-m3"
_BGE_M3_ONNX_REPO = "EmbeddedLLM/bge-m3-onnx-o2-cpu"  # ORT O2 CPU 优化 ONNX（HF 缓存加载，不进 git）
_TRUNCATE_DIM = 512  # bge-m3 MRL 截断目标维度：匹配 overgraph.dense_vector_dimension=512（HNSW 契约）

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


# ─── 推理设备解析（auto/cpu/cuda 自适应）────────────────────

# 模型名子串 → 估算显存占用 (GB)；不含 PyTorch 运行时上下文 (~0.5GB)
_ESTIMATED_MEMORY_GB = {
    "bge-m3": 2.6,
}
_DEFAULT_ESTIMATED_MEMORY_GB = 1.5
_CUDA_CONTEXT_OVERHEAD_GB = 0.5


def _cuda_memory_ok(model_name: str) -> bool:
    """显存预检：空闲显存 > 模型估算 + 上下文开销才返回 True（防 OOM）。

    torch 缺失 / 未编译 CUDA / 驱动异常 → 一律 False（走 CPU）。
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        free_bytes, _ = torch.cuda.mem_get_info()
        estimate = _ESTIMATED_MEMORY_GB.get(
            next((k for k in _ESTIMATED_MEMORY_GB if k in model_name), ""),
            _DEFAULT_ESTIMATED_MEMORY_GB,
        )
        return free_bytes / (1024**3) > estimate + _CUDA_CONTEXT_OVERHEAD_GB
    except Exception:
        return False  # torch 缺失/未编译 CUDA/驱动异常 → 一律 CPU


def _resolve_device(requested: str, model_name: str) -> str:
    """设备解析：auto/cpu/cuda → 实际推理设备。

    - "auto"（默认）：cuda 可用且显存充足 → "cuda"，否则 "cpu"
    - "cpu"：强制 cpu（不触发 torch import，纯 CPU 环境零副作用）
    - "cuda"：强制 cuda；不可用（未编译 CUDA / 无卡）→ 降级 "cpu"
    - torch import 失败 / 异常 → "cpu"
    """
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except Exception:
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    # requested == "auto"
    if torch.cuda.is_available() and _cuda_memory_ok(model_name):
        return "cuda"
    return "cpu"


class TextEncoder:
    """文本嵌入编码器（Tier 2）。

    封装 sentence-transformers/ORT，提供文本到向量的转换，
    集成 FAISS 索引过期管理。
    """

    def __init__(
        self, model_name: str = "BAAI/bge-m3", device: str = "cpu"
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._onnx_model = None
        self._onnx_dim: Optional[int] = None  # 【v5.42】ONNX 输出维度缓存（防 384 硬编码崩 FAISS）
        self._truncate_dim: Optional[int] = None  # 【v6.1】bge-m3 MRL 截断目标维度（512=HNSW 契约）
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
        """加载模型。bge-m3（默认）走专用链路（ONNX O2 → ST snapshot →
        通用 fallback）；其他模型名走 sentence-transformers 通用加载。

        【v6.1】bge-m3 ONNX（EmbeddedLLM/bge-m3-onnx-o2-cpu）最高优先级，
        CPU ~68ms/条（单条）/~4ms/条（批量）；缺失/失败静默回退 ST 兜底。
        """
        import os as _os

        # 进程内强制离线（SentenceTransformer 不支持 local_files_only 参数）
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")
        _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # 【Device】加载前解析一次实际推理设备（auto/cpu/cuda 自适应），
        # 三处 SentenceTransformer 构造共用 resolved，避免重复 CUDA 初始化；
        # 现有异常降级兜底保留，作为运行期失败的最后防线
        resolved = _resolve_device(self.device, self.model_name)
        if resolved != self.device:
            logger.info("Embedding device resolved: requested=%s → %s", self.device, resolved)
        self.device = resolved

        # 【v6.1】模型感知加载：bge-m3（多语言 1024d + MRL 截断）走专用链路
        if "bge-m3" in self.model_name.lower():
            self._load_bge_m3()
            return

        # 非 bge-m3 模型（显式指定时）：通用 sentence-transformers 加载
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model: %s on %s", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Local embedding model loaded: dim=%d", self.dimension)

    def _load_bge_m3(self) -> None:
        """加载 bge-m3（多语言，1024d，MRL 截断 512 保持 HNSW 契约）。

        优先级: ① EmbeddedLLM/bge-m3-onnx-o2-cpu（ORT O2 CPU 优化 ONNX，
        HF 缓存 snapshot 离线加载，不进 git）→ ② BAAI/bge-m3 ST snapshot
        （PyTorch 兜底）→ ③ model_name 通用加载（离线下通常失败，由调用方兜底）。
        """
        import os as _os

        # ① ONNX（EmbeddedLLM/bge-m3-onnx-o2-cpu，外置权重 model.onnx.data）
        onnx_snap = _find_model_snapshot(_BGE_M3_ONNX_REPO)
        if onnx_snap and _os.path.exists(_os.path.join(onnx_snap, "model.onnx")):
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(onnx_snap, local_files_only=True)
                self._onnx_model = ORTModelForFeatureExtraction.from_pretrained(
                    onnx_snap, provider="CPUExecutionProvider", local_files_only=True
                )
                self._onnx_dim = self._infer_onnx_dimension()
                self._truncate_dim = _TRUNCATE_DIM
                self.model_name = _BGE_M3_MODEL
                logger.info("BGE-M3 ONNX loaded from %s (dim=%d, truncate=%d)",
                            onnx_snap, self._onnx_dim, self._truncate_dim)
                return
            except Exception as e:
                logger.warning("BGE-M3 ONNX load failed, fallback to ST: %s", e)
                self._onnx_model = None
                self._tokenizer = None
                self._onnx_dim = None

        # ② ST bge-m3 snapshot（PyTorch 兜底）
        m3_snapshot = _find_model_snapshot(_BGE_M3_MODEL)
        if m3_snapshot is not None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(m3_snapshot, device=self.device)
                self._truncate_dim = _TRUNCATE_DIM
                self.model_name = _BGE_M3_MODEL
                logger.info("BGE-M3 ST loaded: dim=%d (truncate=%d)", self.dimension, self._truncate_dim)
                return
            except Exception as e:
                logger.warning("BGE-M3 ST load failed: %s", e)
                self._model = None

        # ③ 通用 fallback（离线下通常失败，异常由调用方兜底）
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model: %s on %s", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Local embedding model loaded: dim=%d", self.dimension)

    def _maybe_truncate(self, vecs: np.ndarray) -> np.ndarray:
        """bge-m3 MRL 截断（1024d → _TRUNCATE_DIM=512）后重归一化，保持 HNSW 单位范数契约。"""
        truncate_dim = getattr(self, "_truncate_dim", None)  # 容忍 __new__ 构造（测试）
        if not truncate_dim or vecs.shape[-1] <= truncate_dim:
            return vecs
        out = vecs[..., : truncate_dim]
        norms = np.linalg.norm(out, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

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

        # 懒加载兜底：本地模型（ONNX 或 ST）未就绪时先 load
        if self._onnx_model is None and self._model is None:
            self.load()

        # Tier 2: Local model (ONNX preferred)
        if self._onnx_model is not None:
            inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self._onnx_model(**inputs)
            # bge v1.5 pooling=CLS + Normalize（实测 1_Pooling/config.json），
            # 与 ST encode 输出一致（FAISS L2 = cosine）
            vec = outputs.last_hidden_state[:, 0].squeeze().detach().numpy().astype(np.float32)
            norm = np.linalg.norm(vec)
            vec = vec / norm if norm > 0 else vec
            return self._maybe_truncate(vec.reshape(1, -1))[0]

        vec = np.asarray(self._model.encode(text), dtype=np.float32).reshape(1, -1)
        return self._maybe_truncate(vec)[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量 → embedding 矩阵 (N, dim) float32。

        优先 Tier 1（Cloud API 批量），不可用时降级到 Tier 2。

        【v5.42 Write-Throughput】
        - 缓存去重：consult/populate 现有 _cache（原文 key，LRU 512）——
          队列 flush 的重复内容不再重复编码
        - ONNX 分块 32（防 OOM：attention 矩阵峰值估算 >600MB/50 条）
        - ONNX 路径 CLS pooling + L2 归一化（与 ST encode 一致）
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

        # 【v5.42】缓存 consult：命中直接复用（LRU move_to_end），未命中收集编码
        hit_vecs: dict[int, np.ndarray] = {}
        # 【P3】批内去重：唯一原文 → 原序位置列表；同一原文批内重复出现只编码
        # 一次（去重后仍 populate 缓存，后续批次直接命中）
        unique_texts: list[str] = []
        text_positions: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            if t in self._cache:
                self._cache_hits += 1
                vec = self._cache.pop(t)
                self._cache[t] = vec
                hit_vecs[i] = vec
            else:
                if t in text_positions:
                    text_positions[t].append(i)
                else:
                    self._cache_misses += 1
                    text_positions[t] = [i]
                    unique_texts.append(t)

        if not unique_texts:
            if not texts:
                return np.empty((0, self.dimension), dtype=np.float32)
            return np.stack([hit_vecs[i] for i in range(len(texts))])

        # 懒加载兜底：本地模型（ONNX 或 ST）未就绪时先 load
        if self._onnx_model is None and self._model is None:
            self.load()

        # Tier 2: ONNX batch (preferred, chunked)
        if self._onnx_model is not None:
            chunk_vecs: list[np.ndarray] = []
            for start in range(0, len(unique_texts), 32):
                chunk = unique_texts[start : start + 32]
                inputs = self._tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
                outputs = self._onnx_model(**inputs)
                raw = outputs.last_hidden_state[:, 0].detach().numpy().astype(np.float32)
                norms = np.linalg.norm(raw, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                chunk_vecs.append(raw / norms)
            encoded = np.concatenate(chunk_vecs, axis=0)
        else:
            # Tier 3: Local sentence-transformers
            encoded = np.asarray(self._model.encode(unique_texts), dtype=np.float32)
        encoded = self._maybe_truncate(encoded)  # 【v6.1】bge-m3 MRL 截断 512

        # 组装原序矩阵 + populate 缓存（LRU 512）
        out = np.zeros((len(texts), encoded.shape[1]), dtype=np.float32)
        for j, t in enumerate(unique_texts):
            vec = encoded[j]
            for pos in text_positions[t]:
                out[pos] = vec
            if len(self._cache) >= 512:
                del self._cache[next(iter(self._cache))]
            self._cache[t] = vec
        for i, vec in hit_vecs.items():
            out[i] = vec
        return out

    def _infer_onnx_dimension(self) -> Optional[int]:
        """从 ONNX 模型输出读取维度（bge=512，非硬编码 384）。

        优先静态 shape（优化图输出 (b, s, 512) 第三维静态）；
        动态 shape 时回退一次空输入推理取实际维度。
        """
        try:
            outputs = self._onnx_model.model.get_outputs()
            for o in outputs:
                shape = list(o.shape)
                if len(shape) >= 2 and isinstance(shape[-1], int) and shape[-1] > 0:
                    return shape[-1]
        except Exception:
            pass
        try:
            inputs = self._tokenizer("", return_tensors="pt", padding=True, truncation=True)
            outputs = self._onnx_model(**inputs)
            vec = outputs.last_hidden_state[:, 0].detach().numpy()
            return int(vec.shape[-1])
        except Exception:
            return None

    @property
    def dimension(self) -> int:
        """实际加载模型的向量维度（bge-m3=1024 MRL 截断 512）。"""
        truncate_dim = getattr(self, "_truncate_dim", None)
        if truncate_dim:
            return truncate_dim
        if self._onnx_model is not None:
            if self._onnx_dim is not None:
                return self._onnx_dim
            self._onnx_dim = self._infer_onnx_dimension()
            if self._onnx_dim is not None:
                return self._onnx_dim
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
    model_name: str = "BAAI/bge-m3",
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
