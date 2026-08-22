"""
TextEncoder 加载路径测试（全部 mock，不实际加载模型）。
=========================================================
覆盖:
- load() 默认加载 bge-m3（多语言，MRL 截断 512）
- bge-m3 链路: ONNX（EmbeddedLLM O2）→ ST snapshot → model_name 通用
- 非 bge-m3 模型名走通用 sentence-transformers 加载
- dimension: bge-m3 截断后返回 512
"""
from __future__ import annotations

import sys
from unittest import mock

from embedding.encoder import TextEncoder, _resolve_device


def _fake_st_module(side_effect):
    """构造 sentence_transformers 假模块（SentenceTransformer 行为可配置）。"""
    st = mock.MagicMock()
    st.SentenceTransformer = mock.MagicMock(side_effect=side_effect)
    return st


def _fake_onnx_modules(onnx_side_effect):
    """构造 optimum.onnxruntime + transformers 假模块（ONNX 行为可配置）。"""
    opt = mock.MagicMock()
    opt.ORTModelForFeatureExtraction.from_pretrained = mock.MagicMock(
        side_effect=onnx_side_effect
    )
    tf = mock.MagicMock()
    tf.AutoTokenizer.from_pretrained = mock.MagicMock(return_value=mock.MagicMock())
    return {"optimum.onnxruntime": opt, "transformers": tf}


class TestLoadPriority:
    def test_load_prefers_bge_m3_onnx(self):
        """【v6.1】bge-m3 ONNX 快照存在 → ONNX 路径，MRL 截断 512。"""
        fake_onnx = mock.MagicMock()
        onnx_mods = _fake_onnx_modules(lambda *a, **k: fake_onnx)

        encoder = TextEncoder()
        with mock.patch(
            "embedding.encoder._find_model_snapshot", return_value="/fake/onnx_snap"
        ), mock.patch("os.path.exists", return_value=True), mock.patch.object(
            TextEncoder, "_infer_onnx_dimension", return_value=1024
        ), mock.patch.dict(
            sys.modules,
            {
                "optimum.onnxruntime": onnx_mods["optimum.onnxruntime"],
                "transformers": onnx_mods["transformers"],
            },
        ):
            encoder.load()

        assert encoder._onnx_model is fake_onnx
        assert encoder.model_name == "BAAI/bge-m3"
        assert encoder._truncate_dim == 512
        assert encoder.dimension == 512  # MRL 截断

    def test_load_prefers_bge_m3_st(self):
        """【v6.1】ONNX 快照缺失时走 ST snapshot，MRL 截断 512。"""
        fake_model = mock.MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 1024
        st = _fake_st_module([fake_model])

        def fake_snapshot(repo):
            return None if "onnx" in repo else "/fake/snapshot"

        encoder = TextEncoder()
        with mock.patch(
            "embedding.encoder._find_model_snapshot", side_effect=fake_snapshot
        ), mock.patch.dict(sys.modules, {"sentence_transformers": st}):
            encoder.load()

        assert encoder._model is fake_model
        assert encoder.model_name == "BAAI/bge-m3"
        assert encoder._truncate_dim == 512
        assert encoder.dimension == 512  # MRL 截断

    def test_fallback_chain_generic_model(self):
        """【v6.1】非 bge-m3 模型名 → 通用 sentence-transformers 加载。"""
        fake_model = mock.MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 384
        st = _fake_st_module([fake_model])

        encoder = TextEncoder(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        with mock.patch.dict(sys.modules, {"sentence_transformers": st}):
            encoder.load()

        assert encoder._model is fake_model
        assert encoder.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        assert encoder._truncate_dim is None
        assert encoder.dimension == 384


class TestDeviceResolution:
    """embedding/encoder._resolve_device 五分支测试（全 mock torch，不加载真实模型）。"""

    def _resolve(self, requested, model_name="BAAI/bge-m3",
                 cuda_ok=True, free_gb=16.0):
        with mock.patch("torch.cuda.is_available", return_value=cuda_ok), \
                mock.patch("torch.cuda.mem_get_info",
                           return_value=(int(free_gb * 1024**3), 0)):
            return _resolve_device(requested, model_name)

    def test_auto_cuda_available_uses_cuda(self):
        assert self._resolve("auto") == "cuda"

    def test_auto_no_cuda_uses_cpu(self):
        assert self._resolve("auto", cuda_ok=False) == "cpu"

    def test_auto_insufficient_memory_uses_cpu(self):
        # bge-m3 估算 2.6GB + 0.5GB 上下文 = 3.1GB，空闲 1.0GB → cpu（防 OOM）
        assert self._resolve("auto", free_gb=1.0) == "cpu"

    def test_forced_cuda_unavailable_uses_cpu(self):
        assert self._resolve("cuda", cuda_ok=False) == "cpu"

    def test_forced_cpu_uses_cpu(self):
        assert self._resolve("cpu") == "cpu"

    def test_torch_import_failure_uses_cpu(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            assert _resolve_device("auto", "BAAI/bge-m3") == "cpu"
            assert _resolve_device("cuda", "BAAI/bge-m3") == "cpu"

    def test_load_uses_resolved_device(self):
        """load() 解析一次后，SentenceTransformer 收到解析后的 device（bge-m3 默认路径）。"""
        fake_model = mock.MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 1024
        st = _fake_st_module([fake_model])

        def fake_snapshot(repo):
            return None if "onnx" in repo else "/fake/snapshot"

        encoder = TextEncoder(device="auto")
        with mock.patch(
            "embedding.encoder._find_model_snapshot", side_effect=fake_snapshot
        ), mock.patch("embedding.encoder._resolve_device", return_value="cuda"), \
                mock.patch.dict(sys.modules, {"sentence_transformers": st}):
            encoder.load()

        assert encoder._model is fake_model
        assert encoder.device == "cuda"
        # ST 路径以解析后的 device 调用
        st.SentenceTransformer.assert_called_with("/fake/snapshot", device="cuda")
