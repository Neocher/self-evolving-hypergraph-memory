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

from embedding.encoder import TextEncoder, _find_bge_snapshot


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
