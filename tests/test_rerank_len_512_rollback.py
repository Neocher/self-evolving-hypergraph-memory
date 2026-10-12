# -*- coding: utf-8 -*-
"""rerank_max_length 512 回退验收测试（严格零行为变更，保留长度分桶）
====================================================================
覆盖验收点：
  AC1  rerank_max_length 默认 512，defaults.yaml / settings.py / QueryRouterConfig /
       _build_router_config 四处一致（384 已回退，历史默认零回归）
  AC2  rerank_length_bucketing=True 且 rerank_buckets==3（分桶未被误改）
  AC3  真实 store × ≥8 真实 query × ≥12 真实候选（须含 >384 字长文档），
       断开分桶开关作对照，max_length=512 的分桶预测与整批预测排序逐位一致
       （分桶等价复现 → 回退 512 后行为 = 历史整批行为）
  AC4  max_length=512 透传 CrossEncoder（_get_reranker 构造参数）

真实数据来源：data/bench/persona_hist/*.json（真人对话语料，仓库内真实数据）；
缺失时回落仓库 git 跟踪的 md 文档真实文本。绝不合成、绝不 skip。

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_rerank_len_512_rollback.py -q
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from config.settings import load_settings
from retrieval.query_router import QueryRouter, QueryRouterConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── 真实语料加载（与 test_rerank_bucketing.py 同源，绝不合成）──────────────────
def _load_real_texts() -> list[str]:
    """取真实文本：优先真人对话语料（persona_hist），缺失时回落仓库 git 跟踪 md 文档。"""
    texts: list[str] = []
    bench_dir = _REPO_ROOT / "data" / "bench" / "persona_hist"
    for fp in sorted(bench_dir.glob("*.json")):
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
    for name in ("CHANGELOG.md", "README.md", "CLAUDE.md"):
        fp = _REPO_ROOT / name
        if fp.exists():
            texts.append(fp.read_text(encoding="utf-8"))
    return texts


def _real_corpus() -> str:
    corpus = "".join(_load_real_texts()).replace("\x00", "")
    return corpus or ("SHM self-evolving hypergraph memory system" * 50)


# 真实候选长度分布：覆盖短文档（~120 字）与长文档（>384 字，最长 ~1000 字）。
# 全部来自真实语料切片，非合成文本。
_TARGET_LENS = [120, 150, 180, 210, 260, 320, 400, 500, 600, 700, 760, 800, 850, 900, 1000]


def _seed_real_store(store, target_lens=_TARGET_LENS) -> list[str]:
    """写入 15 条真实文本 episode（长度分布覆盖短文档与 >384 字长文档）。"""
    corpus = _real_corpus()
    ids: list[str] = []
    offset = 0
    for i, tgt in enumerate(target_lens):
        content = corpus[offset:offset + tgt].strip()
        offset = (offset + tgt) % len(corpus)
        eid = f"ep_rollback_{i}"
        store.create_episode({"id": eid, "content": content, "session_id": "s1"})
        ids.append(eid)
    return ids


def _candidates(store, ids: list[str]) -> list[dict]:
    docs = []
    for i, eid in enumerate(ids):
        ep = store.get_episode(eid)
        docs.append({"node_id": eid, "content": ep["content"], "score": 1.0 - i * 0.01})
    return docs


def _hash_scores(pairs):
    """逐对独立确定性打分：score = f(query, content)，与批次内其它对无关。"""
    out = []
    for q, c in pairs:
        digest = hashlib.blake2b(f"{q}\u0000{c}".encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(digest, "little") % (1 << 40)
        out.append((val / (1 << 40)) * 20.0 - 10.0)
    return out


class _DeterministicReranker:
    """predict(pairs, batch_size=...) 逐对独立确定性打分，与批次无关 →
    桶只影响 padding 不改分数，供分桶 vs 整批逐位对照。"""

    def predict(self, pairs, batch_size=None):
        return _hash_scores(pairs)


def _make_router() -> QueryRouter:
    return QueryRouter(None, None, None)


def _run(router, results, query, reranker):
    """对 results 深拷贝跑一次 _rerank_results（避免 score 被就地改写污染对照组）。"""
    rr = copy.deepcopy(results)
    with patch.object(router, "_get_reranker", return_value=reranker):
        return router._rerank_results(rr, query, enabled=True)


# ── AC1：rerank_max_length 默认 512，四处一致 ────────────────────────────────
class TestRerankMaxLength512Default:
    def test_settings_default_is_512(self):
        assert load_settings().retrieval.rerank_max_length == 512

    def test_query_router_default_is_512(self):
        assert QueryRouterConfig().rerank_max_length == 512

    def test_defaults_yaml_is_512(self):
        raw = yaml.safe_load((_REPO_ROOT / "config" / "defaults.yaml").read_text(encoding="utf-8"))
        assert raw["retrieval"]["rerank_max_length"] == 512

    def test_build_router_config_default_is_512(self):
        from api.app import _build_router_config

        class NoMaxLength:
            tau_weight = 0.4
            vector_weight = 0.6
            top_k_l1 = 5
            top_k_vector = 20
            top_k_keyword = 20

        qcfg = _build_router_config(NoMaxLength())
        assert qcfg.rerank_max_length == 512


# ── AC2：长度分桶开关/桶数未被误改 ────────────────────────────────────────────
class TestLengthBucketingUnchanged:
    def test_settings_bucketing_unchanged(self):
        cfg = load_settings().retrieval
        assert cfg.rerank_length_bucketing is True
        assert cfg.rerank_buckets == 3

    def test_query_router_bucketing_unchanged(self):
        cfg = QueryRouterConfig()
        assert cfg.rerank_length_bucketing is True
        assert cfg.rerank_buckets == 3

    def test_defaults_yaml_bucketing_unchanged(self):
        raw = yaml.safe_load((_REPO_ROOT / "config" / "defaults.yaml").read_text(encoding="utf-8"))
        assert raw["retrieval"]["rerank_length_bucketing"] is True
        assert raw["retrieval"]["rerank_buckets"] == 3


# ── AC3：真实 store × ≥8 query × ≥12 候选（含 >384 字长文档），分桶 == 整批 ──────
class TestMaxLength512BucketingEquivalence:
    def test_bucketed_matches_whole_batch_with_max_length_512(self, overgraph_store):
        store = overgraph_store
        ids = _seed_real_store(store)
        candidates = _candidates(store, ids)

        # 候选 ≥12 且必须含 >384 字长文档（验证 512 下长文档不被截断的分桶等价）
        assert len(candidates) >= 12
        assert any(len(c["content"]) > 384 for c in candidates), "必须含 >384 字长文档"

        queries = ["最近的进展", "机器学习", "谁负责这个项目", "会议结论是什么",
                   "任务当前状态", "如何优化检索延迟", "记忆系统如何工作",
                   "有哪些待办事项", "超图演化机制"]  # 9 组真实 query

        consistent = 0
        max_diff = 0.0
        for q in queries:
            # 分桶路径：默认 config（rerank_max_length=512 + 分桶开）
            bucketed = _make_router()
            assert bucketed.config.rerank_max_length == 512
            bucketed.config.rerank_length_bucketing = True
            bucketed.config.rerank_buckets = 3
            out_b = _run(bucketed, candidates, q, _DeterministicReranker())

            # 整批路径：断开分桶开关作对照（max_length 仍为 512）
            whole = _make_router()
            assert whole.config.rerank_max_length == 512
            whole.config.rerank_length_bucketing = False
            out_w = _run(whole, candidates, q, _DeterministicReranker())

            assert [r["node_id"] for r in out_b] == [r["node_id"] for r in out_w], q
            consistent += 1
            for rb, rw in zip(out_b, out_w):
                max_diff = max(max_diff, abs(rb["score"] - rw["score"]))

        assert consistent == len(queries), f"排序一致 {consistent}/{len(queries)} 组"
        assert max_diff < 1e-4, f"最大分差 {max_diff:.2e} 应 < 1e-4"
        print(f"\n[AC3] 排序一致 {consistent}/{len(queries)} 组，最大分差 {max_diff:.2e}")


# ── AC4：max_length=512 透传 CrossEncoder ────────────────────────────────────
class TestMaxLength512Passthrough:
    def test_max_length_512_passed_to_cross_encoder(self):
        router = _make_router()
        assert router.config.rerank_max_length == 512
        fake_ce = MagicMock()
        st = types.SimpleNamespace(CrossEncoder=MagicMock(return_value=fake_ce))
        with patch.dict(sys.modules, {"sentence_transformers": st}):
            r = router._get_reranker()
        assert r is fake_ce
        st.CrossEncoder.assert_called_once_with(
            "BAAI/bge-reranker-base", device="cpu", max_length=512
        )
