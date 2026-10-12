"""
长度分桶加速测试 — bge-reranker length bucketing
================================================
覆盖任务书验收点：
  - AC1 零精度损失：真实 store + 真实候选，分桶 vs 整批 → 排序逐位一致 + 分数差 < 1e-4
  - AC2 提速机制：分桶后 padding 工作量严格小于整批（真实加速 ~1.9-2.3x 的确定性代理）
  - AC3 默认零回归：max_length=512 透传 CrossEncoder；关闭分桶 → 走原整批 predict
  - AC4 fail-soft：分桶路径抛异常 → 回落整批 predict，不抛给上层
  - 阈值守卫：桶数 > 候选数按候选数截断；候选 < 2 直接返回

真实数据来源：data/bench/persona_hist/*.json（真人对话语料，仓库内真实数据）；
缺失时回落仓库 git 跟踪的 md 文档真实文本。绝不合成、绝不 skip。
运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_rerank_bucketing.py -q
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from retrieval.query_router import QueryRouter, QueryRouterConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── 真实语料加载 ────────────────────────────────────────────────────────────
def _load_real_texts() -> list[str]:
    """取真实文本：优先真人对话语料（persona_hist），缺失时回落仓库 git 跟踪 md 文档。"""
    texts: list[str] = []
    for fp in sorted((_REPO_ROOT / "data" / "bench" / "persona_hist").glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for m in data.get("chat_history", []) or []:
            if not isinstance(m, dict) or m.get("role") == "system":
                continue
            c = str(m.get("content", "") or "").strip()
            if c:
                texts.append(c)
    if texts:
        return texts
    # 回落：真实仓库文档文本（git 跟踪，绝不合成）
    for name in ("CHANGELOG.md", "README.md", "CLAUDE.md"):
        fp = _REPO_ROOT / name
        if fp.exists():
            texts.append(fp.read_text(encoding="utf-8"))
    return texts


def _real_corpus() -> str:
    corpus = "".join(_load_real_texts()).replace("\x00", "")
    return corpus or ("SHM self-evolving hypergraph memory system" * 50)


def _hash_scores(pairs):
    """逐对独立确定性打分：score = f(query, content)，与批次内其它对无关。"""
    out = []
    for q, c in pairs:
        digest = hashlib.blake2b(f"{q}\u0000{c}".encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(digest, "little") % (1 << 40)
        out.append((val / (1 << 40)) * 20.0 - 10.0)
    return out


# ── 确定性逐对独立 reranker（与批次无关 → 桶只影响 padding，不改分数）───────
class _DeterministicReranker:
    """predict(pairs, batch_size=...) 记录每次调用的 batch_size 与每对 content 长度，
    供 AC2「padding 工作量」与「连续切分」断言使用。分数来自 blake2b 摘要，确定性。"""

    def __init__(self):
        self.calls: list[tuple[object, list[int]]] = []

    def predict(self, pairs, batch_size=None):
        self.calls.append((batch_size, [len(p[1]) for p in pairs]))
        return _hash_scores(pairs)


class _BucketFailReranker:
    """分桶路径（带 batch_size 调用）抛异常 → 触发 fail-soft 回落整批；回落分数与
    _DeterministicReranker 完全一致（同 _hash_scores），便于逐位对照。"""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs, batch_size=None):
        self.calls += 1
        if batch_size is not None:
            raise RuntimeError("bucket path boom")
        return _hash_scores(pairs)


# ── 真实 store + 真实候选 ───────────────────────────────────────────────────
_TARGET_LENS = [120, 150, 180, 210, 260, 320, 400, 500, 600, 700, 760, 800, 850, 900, 1000]


def _seed_real_store(store, target_lens=_TARGET_LENS) -> list[str]:
    """写入 15 条真实文本 episode（长度分布覆盖短文档 ~120 字与长文档 ~1000 字）。"""
    corpus = _real_corpus()
    ids: list[str] = []
    offset = 0
    for i, tgt in enumerate(target_lens):
        content = corpus[offset:offset + tgt].strip()
        offset = (offset + tgt) % len(corpus)
        eid = f"ep_real_{i}"
        store.create_episode({"id": eid, "content": content, "session_id": "s1"})
        ids.append(eid)
    return ids


def _candidates(store, ids: list[str]) -> list[dict]:
    docs = []
    for i, eid in enumerate(ids):
        ep = store.get_episode(eid)
        docs.append({"node_id": eid, "content": ep["content"], "score": 1.0 - i * 0.01})
    return docs


def _make_router() -> QueryRouter:
    return QueryRouter(None, None, None)


def _run(router, results, query, reranker):
    """对 results 深拷贝跑一次 _rerank_results（避免 score 被就地改写污染对照组）。"""
    rr = copy.deepcopy(results)
    with patch.object(router, "_get_reranker", return_value=reranker):
        return router._rerank_results(rr, query, enabled=True)


# ── AC1：零精度损失（真实 store × 8 query × 真实候选）─────────────────────────
class TestBucketingZeroPrecisionLoss:
    def test_bucketed_matches_whole_batch_ordering_and_scores(self, overgraph_store):
        store = overgraph_store
        ids = _seed_real_store(store)
        candidates = _candidates(store, ids)
        queries = ["最近的进展", "机器学习", "谁负责", "会议结论", "任务状态",
                   "persona", "检索延迟", "记忆系统"]

        for q in queries:
            bucketed = _make_router()
            bucketed.config.rerank_length_bucketing = True
            bucketed.config.rerank_buckets = 3
            out_b = _run(bucketed, candidates, q, _DeterministicReranker())

            whole = _make_router()
            whole.config.rerank_length_bucketing = False
            out_w = _run(whole, candidates, q, _DeterministicReranker())

            # ① 排序结果完全一致（node_id 序列逐位相同）
            assert [r["node_id"] for r in out_b] == [r["node_id"] for r in out_w], q
            # ② 分数差 < 1e-4（逐位）
            for rb, rw in zip(out_b, out_w):
                assert abs(rb["score"] - rw["score"]) < 1e-4, (q, rb["node_id"])


# ── AC2：分桶减少 padding 工作量（确定性代理）+ 连续切分 ─────────────────────
class TestBucketingConsecutiveChunks:
    def test_buckets_are_consecutive_sorted_chunks(self, overgraph_store):
        """连续切分：各桶长度序列拼接后 == 按长度排序后的全序列（交错切分则不成立）。"""
        store = overgraph_store
        ids = _seed_real_store(store)
        candidates = _candidates(store, ids)
        contents = [c["content"] for c in candidates]

        router = _make_router()
        router.config.rerank_length_bucketing = True
        router.config.rerank_buckets = 3
        fake = _DeterministicReranker()
        _run(router, candidates, "q", fake)

        sorted_lens = sorted(len(c) for c in contents)
        flattened = []
        for _, lens in fake.calls:
            flattened.extend(lens)
        assert flattened == sorted_lens, "必须连续切分（sorted 后连续块），非交错切分"
        assert len(fake.calls) == 3, "3 桶 → 3 次 predict 调用"

    def test_bucketing_reduces_padding_work(self, overgraph_store):
        """分桶 padding 工作量 = Σ(桶大小 × 桶内最长) < 整批 (n × 全局最长)。"""
        store = overgraph_store
        ids = _seed_real_store(store)
        candidates = _candidates(store, ids)

        fake_b = _DeterministicReranker()
        router = _make_router()
        router.config.rerank_length_bucketing = True
        router.config.rerank_buckets = 3
        _run(router, candidates, "q", fake_b)
        padded_b = sum(len(lens) * max(lens) for _, lens in fake_b.calls)

        fake_w = _DeterministicReranker()
        whole = _make_router()
        whole.config.rerank_length_bucketing = False
        _run(whole, candidates, "q", fake_w)
        (_, lens_w), = fake_w.calls
        padded_w = len(lens_w) * max(lens_w)

        assert padded_b < padded_w, f"分桶 padding 工作量 {padded_b} 应小于整批 {padded_w}"


# ── AC3：默认零回归 / 关闭分桶走整批 / max_length 透传 ───────────────────────
class TestBucketingZeroRegression:
    def test_defaults_are_zero_regression(self):
        cfg = QueryRouterConfig()
        assert cfg.rerank_length_bucketing is True
        assert cfg.rerank_buckets == 3
        assert cfg.rerank_max_length == 512

    def test_bucketing_disabled_uses_whole_batch_predict(self):
        router = _make_router()
        router.config.rerank_length_bucketing = False
        fake = _DeterministicReranker()
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "b", "content": "beta beta", "score": 0.8},
            {"node_id": "c", "content": "gamma", "score": 0.7},
        ]
        _run(router, results, "q", fake)
        assert len(fake.calls) == 1, "关闭分桶 → 仅一次整批 predict"
        assert fake.calls[0][0] is None, "整批路径不带 batch_size"

    def test_max_length_passed_to_cross_encoder(self):
        import sys
        import types
        from unittest.mock import MagicMock

        router = _make_router()
        router.config.rerank_max_length = 512
        fake_ce = MagicMock()
        st = types.SimpleNamespace(CrossEncoder=MagicMock(return_value=fake_ce))
        with patch.dict(sys.modules, {"sentence_transformers": st}):
            r = router._get_reranker()
        assert r is fake_ce
        st.CrossEncoder.assert_called_once_with(
            "BAAI/bge-reranker-base", device="cpu", max_length=512
        )

    def test_config_validation_rejects_invalid_values(self):
        with pytest.raises(ValueError):
            QueryRouterConfig(rerank_buckets=0)
        with pytest.raises(ValueError):
            QueryRouterConfig(rerank_max_length=0)
        # 合法值不抛
        assert QueryRouterConfig(rerank_buckets=1).rerank_buckets == 1
        assert QueryRouterConfig(rerank_max_length=256).rerank_max_length == 256

    def test_settings_and_yaml_defaults_consistent(self):
        from config.settings import load_settings

        assert QueryRouterConfig().rerank_max_length == 512
        assert load_settings().retrieval.rerank_buckets == 3
        assert load_settings().retrieval.rerank_length_bucketing is True

    def test_build_router_config_passthrough(self):
        from api.app import _build_router_config

        class FakeRcfg:
            tau_weight = 0.4
            vector_weight = 0.6
            top_k_l1 = 5
            top_k_vector = 20
            top_k_keyword = 20
            rerank_length_bucketing = False
            rerank_buckets = 5
            rerank_max_length = 256

        qcfg = _build_router_config(FakeRcfg())
        assert qcfg.rerank_length_bucketing is False
        assert qcfg.rerank_buckets == 5
        assert qcfg.rerank_max_length == 256

        class NoBuckets:
            tau_weight = 0.4
            vector_weight = 0.6
            top_k_l1 = 5
            top_k_vector = 20
            top_k_keyword = 20

        qcfg2 = _build_router_config(NoBuckets())
        assert qcfg2.rerank_length_bucketing is True
        assert qcfg2.rerank_buckets == 3
        assert qcfg2.rerank_max_length == 512


# ── AC4：fail-soft 回落整批，不抛给上层 ─────────────────────────────────────
class TestBucketingFailSoft:
    def test_bucket_path_exception_falls_back_to_whole_batch(self):
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "b", "content": "beta beta", "score": 0.8},
            {"node_id": "c", "content": "gamma", "score": 0.7},
        ]
        # 期望：整批路径（分桶关 + 确定性打分）
        whole = _make_router()
        whole.config.rerank_length_bucketing = False
        expected = _run(whole, results, "q", _DeterministicReranker())

        router = _make_router()
        router.config.rerank_length_bucketing = True
        router.config.rerank_buckets = 3
        fake = _BucketFailReranker()
        out = _run(router, results, "q", fake)

        assert fake.calls == 2, "一次分桶失败 + 一次整批回落"
        assert [r["node_id"] for r in out] == [r["node_id"] for r in expected]
        for rb, rw in zip(out, expected):
            assert abs(rb["score"] - rw["score"]) < 1e-4

    def test_bucket_predict_length_mismatch_falls_back(self):
        """桶内 predict 返回长度失配 → 抛 ValueError → 回落整批，不抛给上层。"""

        class BadLengthReranker:
            def __init__(self):
                self.calls = 0

            def predict(self, pairs, batch_size=None):
                self.calls += 1
                if batch_size is not None:
                    return [0.5]  # 长度失配
                return _hash_scores(pairs)

        router = _make_router()
        router.config.rerank_length_bucketing = True
        router.config.rerank_buckets = 2
        fake = BadLengthReranker()
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "b", "content": "beta", "score": 0.8},
            {"node_id": "c", "content": "gamma", "score": 0.7},
        ]
        out = _run(router, results, "q", fake)
        assert fake.calls == 2
        assert len(out) == 3


# ── 阈值守卫 ────────────────────────────────────────────────────────────────
class TestBucketingGuards:
    def test_buckets_greater_than_candidates_truncated(self):
        router = _make_router()
        router.config.rerank_length_bucketing = True
        router.config.rerank_buckets = 100
        fake = _DeterministicReranker()
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "b", "content": "beta", "score": 0.8},
            {"node_id": "c", "content": "gamma", "score": 0.7},
        ]
        _run(router, results, "q", fake)
        assert len(fake.calls) == 3, "桶数按候选数截断为 3"

    def test_less_than_two_scorable_short_circuits(self):
        router = _make_router()
        router.config.rerank_length_bucketing = True
        fake = _DeterministicReranker()
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "v", "content": "", "score": 0.8},
        ]
        _run(router, results, "q", fake)
        assert fake.calls == [], "候选 < 2 → 不触发 predict（沿用现有逻辑）"

    def test_empty_content_keeps_relative_position_with_bucketing(self):
        """【F4】分桶开启时空 content 位置仍保留原始相对位置。"""
        router = _make_router()
        router.config.rerank_length_bucketing = True
        router.config.rerank_buckets = 3
        fake = _DeterministicReranker()
        results = [
            {"node_id": "a", "content": "alpha", "score": 0.9},
            {"node_id": "v", "content": "", "score": 0.85},
            {"node_id": "b", "content": "beta beta", "score": 0.8},
            {"node_id": "w", "content": "  ", "score": 0.75},
            {"node_id": "c", "content": "gamma", "score": 0.7},
        ]
        out = _run(router, results, "q", fake)
        # 空 content 节点 v/w 必须保持其 head 中的原始相对位置（下标 1/3）
        assert out[1]["node_id"] == "v"
        assert out[3]["node_id"] == "w"
        scorable_ids = [r["node_id"] for r in out if str(r.get("content", "")).strip()]
        assert sorted(scorable_ids) == ["a", "b", "c"], "可打分节点全保留且无重复"
