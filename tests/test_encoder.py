"""
TextEncoder bge 加载路径测试（全部 mock，不实际加载模型）。
=========================================================
覆盖 Codex 审核必改 2:
- _find_bge_snapshot: HF 缓存目录缺失返回 None，按 mtime 取最新
- load() 优先加载 bge-small-zh-v1.5 (snapshot)
- 完整 fallback 链: bge 失败 → ONNX 失败 → model_name
- dimension: bge 加载后返回 512
"""
from __future__ import annotations

import sys
from unittest import mock

from embedding.encoder import TextEncoder, _find_bge_snapshot, _resolve_device


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


class TestFindBgeSnapshot:
    def test_missing_cache_dir_returns_none(self):
        with mock.patch("glob.glob", return_value=[]):
            assert _find_bge_snapshot() is None

    def test_returns_most_recent_by_mtime(self):
        # 字典序与 mtime 序相反，验证按 mtime 取最新
        snaps = ["/snap/b", "/snap/a"]
        mtimes = {"/snap/b": 100, "/snap/a": 200}
        with mock.patch("glob.glob", return_value=snaps), mock.patch(
            "os.path.getmtime", side_effect=lambda p: mtimes[p]
        ):
            assert _find_bge_snapshot() == "/snap/a"


class TestLoadPriority:
    def test_load_prefers_bge_snapshot(self):
        fake_model = mock.MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 512
        st = _fake_st_module([fake_model])

        encoder = TextEncoder()
        with mock.patch(
            "embedding.encoder._find_bge_snapshot", return_value="/fake/snapshot"
        ), mock.patch.dict(sys.modules, {"sentence_transformers": st}):
            encoder.load()

        assert encoder._model is fake_model
        assert encoder.model_name == "BAAI/bge-small-zh-v1.5"
        assert encoder.dimension == 512

    def test_fallback_chain_bge_onnx_model_name(self):
        fake_model = mock.MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 384
        # bge 第一次 (cuda) 抛异常 → CPU 重试也抛异常 → 落到 model_name 分支
        st = _fake_st_module([RuntimeError("bge cuda fail"), RuntimeError("bge cpu fail"), fake_model])
        onnx_mods = _fake_onnx_modules(RuntimeError("onnx fail"))

        encoder = TextEncoder(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        with mock.patch(
            "embedding.encoder._find_bge_snapshot", return_value="/fake/snapshot"
        ), mock.patch("os.path.isdir", return_value=True), mock.patch(
            "os.path.exists", return_value=True
        ), mock.patch.dict(
            sys.modules,
            {
                "sentence_transformers": st,
                "optimum.onnxruntime": onnx_mods["optimum.onnxruntime"],
                "transformers": onnx_mods["transformers"],
            },
        ):
            encoder.load()

        assert encoder._model is fake_model
        assert encoder.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        assert encoder.dimension == 384


class TestDeviceResolution:
    """embedding/encoder._resolve_device 五分支测试（全 mock torch，不加载真实模型）。"""

    def _resolve(self, requested, model_name="BAAI/bge-small-zh-v1.5",
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
        # bge-small 估算 0.6GB + 0.5GB 上下文 = 1.1GB，空闲 1.0GB → cpu（防 OOM）
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
        """load() 解析一次后，SentenceTransformer 收到解析后的 device。"""
        fake_model = mock.MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 512
        st = _fake_st_module([fake_model])

        encoder = TextEncoder(device="auto")
        with mock.patch(
            "embedding.encoder._find_model_snapshot", return_value="/fake/snapshot"
        ), mock.patch("embedding.encoder._resolve_device", return_value="cuda"), \
                mock.patch.dict(sys.modules, {"sentence_transformers": st}):
            encoder.load()

        assert encoder._model is fake_model
        assert encoder.device == "cuda"
        # 三处构造均以解析后的 device 调用
        st.SentenceTransformer.assert_called_with("/fake/snapshot", device="cuda")
