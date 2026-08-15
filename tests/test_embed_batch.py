"""
v5.42.0 Write-Throughput — embed_batch 批量编码测试
===================================================
覆盖:
  · 批量 vs 逐条 cosine >0.999（同 encoder 实例，防混 ONNX/PyTorch 路径）
  · 空 / 单条 / 混合长度 batch
  · 队列 flush 失败回退逐条（_process_embed_queue）
  · ONNX dimension == 512（防维度崩溃崩 FAISS）
  · embedding/onnx/ 缺失 → 静默回退 PyTorch 零回归
  · fp32 ONNX vs ST PyTorch recall@10 无损（int8 已知 recall 降幅>2% 保 fp32）
  · 缓存命中不重复编码（spy _do_embed 调用次数）

运行: cd /home/admin/shm && python -m pytest tests/test_embed_batch.py -q
"""
import asyncio
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.encoder import TextEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR = os.path.join(PROJECT_ROOT, "embedding", "onnx")
ONNX_PRESENT = os.path.isdir(ONNX_DIR) and os.path.exists(os.path.join(ONNX_DIR, "model.onnx"))

TEXTS = [
    "上海今天的天气怎么样",
    "机器学习是人工智能的一个分支",
    "中国的首都是北京",
    "我需要记住这条重要的记忆",
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def enc():
    e = TextEncoder(device="cpu")
    e.load()
    return e


# ─── 批量 vs 逐条一致性 ───────────────────────────────


def test_batch_vs_single_cosine(enc):
    enc._cache.clear()
    batch = enc.embed_batch(TEXTS)
    enc._cache.clear()
    single = np.stack([enc.embed(t) for t in TEXTS])
    assert batch.shape == single.shape == (len(TEXTS), 512)
    for i in range(len(TEXTS)):
        assert _cosine(batch[i], single[i]) > 0.999


# ─── 空 / 单条 / 混合长度 ─────────────────────────────


def test_empty_batch(enc):
    m = enc.embed_batch([])
    assert m.shape == (0, enc.dimension)


def test_single_batch(enc):
    m = enc.embed_batch(["机器学习是人工智能的一个分支"])
    assert m.shape == (1, 512)
    v = enc.embed_batch(["机器学习是人工智能的一个分支", "上海天气"])[0]
    assert v.shape == (512,)


# ─── 缓存命中不重复编码（E）───────────────────────────


def test_cache_hit_skips_reencode(enc):
    enc._cache.clear()
    base = enc._cache_misses
    enc.embed_batch(TEXTS)
    assert enc._cache_misses - base == len(TEXTS)  # 首次全部 miss
    base2 = enc._cache_misses
    enc.embed_batch(TEXTS)
    assert enc._cache_misses == base2  # 二次全命中，零新增编码


# ─── 批内去重（P3）────────────────────────────────


def test_batch_dedup_skips_duplicate_encode(enc):
    enc._cache.clear()
    base = enc._cache_misses
    m = enc.embed_batch(TEXTS + [TEXTS[0], TEXTS[1]])
    assert enc._cache_misses - base == len(TEXTS)  # 6 条原文仅 4 个唯一 → 只编码 4 次
    assert m.shape == (6, 512)
    for i in range(2):
        assert np.allclose(m[-2 + i], m[i])  # 重复原文向量与首现位置一致


# ─── ONNX dimension == 512（防 FAISS 维度崩溃）─────────


def test_onnx_dimension_512(enc):
    if enc._onnx_model is not None:
        assert enc.dimension == 512
        assert enc._onnx_dim == 512
    else:
        pytest.skip("ONNX 未加载（目录缺失时走 PyTorch）")


# ─── ONNX 缺失 → 静默回退 PyTorch 零回归 ──────────────


def test_onnx_missing_fallback():
    orig_dir = TextEncoder._ONNX_DIR
    TextEncoder._ONNX_DIR = "onnx_missing_dir_xyz"
    try:
        e = TextEncoder(device="cpu")
        e.load()
        assert e._onnx_model is None
        assert e._model is not None  # PyTorch bge snapshot
        v = e.embed("上海天气怎么样")
        assert v.shape == (512,)
        assert np.linalg.norm(v) > 0
    finally:
        TextEncoder._ONNX_DIR = orig_dir


# ─── 队列 flush 失败回退逐条（A1）──────────────────────


def test_process_embed_queue_batch_fail_fallback():
    from api.routes import Services
    from api.routes import _deps as deps_mod
    from api.routes.write import _process_embed_queue

    svc = Services()
    enc = type("FakeEncoder", (), {})()
    enc.embed_batch = lambda texts: (_ for _ in ()).throw(RuntimeError("batch boom"))
    enc.embed = lambda text: np.random.RandomState(hash(text) & 0xFFFF).rand(512).astype(np.float32)
    svc.encoder = enc
    svc.quarantine_store = None
    svc.faiss_index = type("FakeFaiss", (), {})()
    svc.faiss_index.add_with_ids = lambda vecs, ids: None
    svc._faiss_buffer_lock = threading.Lock()
    svc.hebbian_updater = None
    svc.graphlite_store = None

    items = [(f"e{i}", f"内容文本 {i} 号", float(i)) for i in range(4)]
    with deps_mod._embed_queue_lock:
        deps_mod._embed_queue[:] = items
    n = asyncio.run(_process_embed_queue(svc))
    assert n == 4  # 批量失败 → 逐条回退全部处理


# ─── fp32 ONNX vs ST PyTorch recall@10 无损 ───────────


@pytest.mark.skipif(not ONNX_PRESENT, reason="embedding/onnx/ 缺失")
def test_fp32_onnx_recall_parity_with_st():
    import faiss
    from sentence_transformers import SentenceTransformer

    snapshot = _find_bge_snapshot_for_test()
    if snapshot is None:
        pytest.skip("bge PyTorch snapshot 缺失")

    st = SentenceTransformer(snapshot, device="cpu")
    topics = {
        "机器学习": ["梯度下降优化损失函数", "卷积神经网络处理图像", "过拟合可以通过正则化缓解",
                     "模型评估使用交叉验证", "注意力机制提升翻译质量", "数据增强提升泛化能力"],
        "交通": ["早高峰地铁客流激增", "公交线路优化调整方案", "共享单车停放秩序整治",
                 "高速公路收费站拥堵", "铁路春运增加临时列车", "网约车平台夜间调度"],
        "美食": ["川菜以麻辣著称", "广式早茶点心丰富", "火锅底料配方讲究",
                 "江南菜口味偏甜", "西北面食种类繁多", "烧烤夜市人气旺"],
    }
    corpus = [s for t in topics.values() for s in t]
    queries = ["机器学习中如何防止过拟合", "地铁早高峰客流管理", "火锅底料哪家正宗"]
    cv = st.encode(corpus)
    idx_f = faiss.IndexFlatL2(512)
    idx_f.add(cv)
    gt = [set(idx_f.search(st.encode(q).reshape(1, -1), 6)[1].flatten().tolist()) for q in queries]

    e = TextEncoder(device="cpu")
    e.load()
    assert e._onnx_model is not None
    qc = e.embed_batch(corpus)
    idx_i = faiss.IndexFlatL2(512)
    idx_i.add(qc)
    qv = e.embed_batch(queries)
    for i in range(len(queries)):
        r = set(idx_i.search(qv[i].reshape(1, -1), 6)[1].flatten().tolist())
        assert len(r & gt[i]) / len(gt[i]) >= 0.98  # fp32 无损，降幅 <2%


def _find_bge_snapshot_for_test():
    from embedding.encoder import _find_bge_snapshot
    return _find_bge_snapshot()
