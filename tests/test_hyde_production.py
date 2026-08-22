"""
P3b 测试 — HyDE 假设文档增强检索生产管道集成
=============================================
覆盖设计任务书（design_p3b_hyde.md）验收点：
  - 关闭路径等价：hyde=None + config 默认关 → 结果与现状一致（单路、无 _hyde 标记）
  - hyde=True + generate_hypothesis 返回固定 hypo → dual 双路（_fusion_retrieve 2 次）
  - LLM 失败（generate_hypothesis → None）→ 单路等价（_fusion_retrieve 1 次）
  - replace 模式 → 单路但 query_embedding 为 hypo 向量
  - generate_hypothesis 失败降级单元（永久跳过 / 冷却 / 缓存 / 401 / 5xx）
  - 【P3b R1】关闭路径完整链路快照对比（不 mock _finish 内增强通道）/ 中文查询
    prompt 语言一致性约束 / _level_from_strategy 配置感知（P0-1）/ _retrieve_timeout
    （P0-2 超时预算）

集成用例全部走公共入口 retrieve(level=FUSION)，禁直调检索内部方法（防假绿）。
运行: python -m pytest tests/test_hyde_production.py -v
"""
from __future__ import annotations

import json
import time
import urllib.error
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.routes.search import _level_from_strategy, _retrieve_timeout
from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel


def _make_router(**cfg_kwargs) -> QueryRouter:
    """零依赖 QueryRouter；统一关 rerank 保证结果确定性（HyDE 与 rerank 正交）。"""
    config = QueryRouterConfig(rerank_enabled=False, **cfg_kwargs)
    return QueryRouter(None, None, None, config=config)


def _passthrough(results, *args, **kwargs):
    return results


def _enhance_passthrough_stack(router) -> ExitStack:
    """patch _finish 内部增强通道为透传（只测 HyDE 路径，不测 community/mesa 等）。"""
    stack = ExitStack()
    for name in (
        "_community_expansion",
        "_mesa_synthesis",
        "_visual_recall",
        "_property_temporal_retrieve",
    ):
        stack.enter_context(patch.object(router, name, side_effect=_passthrough))
    return stack


class TestHydeClosedPath:
    """hyde=None + config 默认关 → 现状单路逐字节等价（无 _hyde 标记、不触 LLM）。

    【P3b R1】完整链路等价强化：不 mock _finish 内增强通道（community/mesa/视觉/
    属性真实跑零依赖 router），参考快照对比——不只 call_count。
    """

    def _run(self, router, docs, **retrieve_kwargs):
        with patch.object(router, "_fusion_retrieve",
                          return_value=[dict(d) for d in docs]) as fr:
            with patch.object(router, "_encode_query") as eq:
                with patch("retrieval.query_router.generate_hypothesis") as gh:
                    out = router.retrieve("memory", level=RetrievalLevel.FUSION,
                                          **retrieve_kwargs)
        return out, fr, eq, gh

    def test_disabled_default_is_single_path(self):
        router = _make_router()  # hyde_enabled=False
        docs = [
            {"node_id": "a", "content": "alpha memory", "score": 0.9},
            {"node_id": "b", "content": "beta memory", "score": 0.7},
        ]
        out, fr, eq, gh = self._run(router, docs)

        gh.assert_not_called()
        eq.assert_not_called()
        assert fr.call_count == 1
        q, emb, raw_q = fr.call_args.args
        assert q == "memory" and raw_q == "memory"
        assert emb is None  # query_embedding 原样透传（HyDE 未介入）
        assert [r["node_id"] for r in out] == ["a", "b"]
        assert all("_hyde" not in r for r in out), "关闭路径不得有 _hyde 标记"

    def test_explicit_false_overrides_config_on(self):
        router = _make_router(hyde_enabled=True)
        docs = [{"node_id": "a", "content": "alpha", "score": 0.9}]
        out, fr, eq, gh = self._run(router, docs, hyde=False)

        gh.assert_not_called()
        assert fr.call_count == 1
        assert [r["node_id"] for r in out] == ["a"]

    def test_full_chain_snapshot_identical_across_closed_paths(self):
        """【P3b R1】参考快照对比：config 关 / hyde=True 且 LLM 失败 / 显式 hyde=False
        三条关闭路径完整链路输出逐字节一致（_finish 真实运行，不 mock 增强通道）。"""
        router = _make_router()  # hyde_enabled=False
        docs = [
            {"node_id": "a", "content": "alpha memory", "score": 0.9},
            {"node_id": "b", "content": "beta memory", "score": 0.7},
        ]

        baseline, fr_b, _, gh_b = self._run(router, docs)  # 参考快照：config 默认关
        explicit_off, fr_e, _, gh_e = self._run(router, docs, hyde=False)

        # hyde=True + LLM 失败（generate_hypothesis → None）→ 静默降级单路
        with patch.object(router, "_fusion_retrieve",
                          return_value=[dict(d) for d in docs]) as fr_d:
            with patch.object(router, "_encode_query") as eq_d:
                with patch("retrieval.query_router.generate_hypothesis",
                           return_value=None) as gh_deg:
                    degraded = router.retrieve("memory", level=RetrievalLevel.FUSION,
                                               hyde=True)

        assert gh_b.call_count == 0 and gh_e.call_count == 0
        assert gh_deg.call_count == 1  # hyde=True 触发一次 LLM 尝试（失败降级）
        assert eq_d.call_count == 0
        assert fr_b.call_count == fr_e.call_count == fr_d.call_count == 1
        # 完整链路（真实 _finish：去重/排序/增强通道）输出逐字节一致
        assert degraded == baseline
        assert explicit_off == baseline
        assert all("_hyde" not in r for r in degraded), "降级路径不得有 _hyde 标记"


class TestHydeDualMode:
    """hyde=True（或 config 开）→ 原始 + 假设双路融合合并去重。"""

    def test_dual_mode_merges_both_retrievals(self):
        router = _make_router(hyde_enabled=True)  # hyde_mode="dual" 默认
        base_docs = [{"node_id": "a", "content": "alpha memory", "score": 0.9}]
        extra_docs = [{"node_id": "b", "content": "beta memory", "score": 0.8}]
        hypo_emb = np.zeros((1, 512), dtype=np.float32)
        seen: list = []

        def fake_fusion(query, query_embedding, raw_query=None, now_ts=None):
            seen.append((query, query_embedding, raw_query))
            return [dict(d) for d in (base_docs if len(seen) == 1 else extra_docs)]

        with _enhance_passthrough_stack(router):
            with patch.object(router, "_fusion_retrieve", side_effect=fake_fusion) as fr:
                with patch.object(router, "_encode_query", return_value=hypo_emb) as eq:
                    with patch("retrieval.query_router.generate_hypothesis",
                               return_value="Hypothetical passage") as gh:
                        out = router.retrieve("memory", level=RetrievalLevel.FUSION)

        gh.assert_called_once_with("memory", timeout=1.5)  # raw_query + config 超时（P0-2 2.0→1.5）
        eq.assert_called_once_with("Hypothetical passage")
        assert fr.call_count == 2, "dual 模式应双路检索"
        assert seen[0][0] == "memory" and seen[0][1] is None  # 原始 query 路
        assert seen[1][0] == "Hypothetical passage" and seen[1][1] is hypo_emb  # 假设路
        assert {r["node_id"] for r in out} == {"a", "b"}, "双路结果合并去重"

    def test_config_off_hyde_true_override_enables(self):
        router = _make_router()  # 默认关
        docs = [{"node_id": "a", "content": "alpha", "score": 0.9}]
        hypo_emb = np.zeros((1, 512), dtype=np.float32)
        with _enhance_passthrough_stack(router):
            with patch.object(router, "_fusion_retrieve",
                              return_value=[dict(d) for d in docs]) as fr:
                with patch.object(router, "_encode_query", return_value=hypo_emb):
                    with patch("retrieval.query_router.generate_hypothesis",
                               return_value="H") as gh:
                        out = router.retrieve("memory", level=RetrievalLevel.FUSION, hyde=True)

        gh.assert_called_once()
        assert fr.call_count == 2
        assert [r["node_id"] for r in out] == ["a"]


class TestHydeLLMFailureFallback:
    """LLM 失败/编码失败 → 静默降级现状单路（_fusion_retrieve 1 次）。"""

    def test_generate_none_falls_back_to_single_path(self):
        router = _make_router(hyde_enabled=True)
        docs = [{"node_id": "a", "content": "alpha", "score": 0.9}]
        with _enhance_passthrough_stack(router):
            with patch.object(router, "_fusion_retrieve",
                              return_value=[dict(d) for d in docs]) as fr:
                with patch.object(router, "_encode_query") as eq:
                    with patch("retrieval.query_router.generate_hypothesis",
                               return_value=None) as gh:
                        out = router.retrieve("memory", level=RetrievalLevel.FUSION)

        gh.assert_called_once()
        eq.assert_not_called()
        assert fr.call_count == 1
        assert fr.call_args.args[1] is None  # 原始 query_embedding
        assert [r["node_id"] for r in out] == ["a"]

    def test_encode_none_falls_back_to_single_path(self):
        router = _make_router(hyde_enabled=True)
        docs = [{"node_id": "a", "content": "alpha", "score": 0.9}]
        with _enhance_passthrough_stack(router):
            with patch.object(router, "_fusion_retrieve",
                              return_value=[dict(d) for d in docs]) as fr:
                with patch.object(router, "_encode_query", return_value=None):
                    with patch("retrieval.query_router.generate_hypothesis",
                               return_value="H"):
                        out = router.retrieve("memory", level=RetrievalLevel.FUSION)

        assert fr.call_count == 1
        assert fr.call_args.args[1] is None
        assert [r["node_id"] for r in out] == ["a"]


class TestHydeReplaceMode:
    """replace 模式 → 单路检索但 query_embedding 替换为 hypo 向量。"""

    def test_replace_uses_hypo_embedding_single_call(self):
        router = _make_router(hyde_enabled=True, hyde_mode="replace")
        docs = [{"node_id": "a", "content": "alpha", "score": 0.9}]
        hypo_emb = np.zeros((1, 512), dtype=np.float32)
        with _enhance_passthrough_stack(router):
            with patch.object(router, "_fusion_retrieve",
                              return_value=[dict(d) for d in docs]) as fr:
                with patch.object(router, "_encode_query", return_value=hypo_emb) as eq:
                    with patch("retrieval.query_router.generate_hypothesis",
                               return_value="Hypothetical passage") as gh:
                        out = router.retrieve("memory", level=RetrievalLevel.FUSION)

        gh.assert_called_once()
        eq.assert_called_once()
        assert fr.call_count == 1
        assert fr.call_args.args[1] is hypo_emb, "replace 模式 query_embedding 应为 hypo 向量"
        assert [r["node_id"] for r in out] == ["a"]


# ─── generate_hypothesis 失败降级单元（防每查询重试）──────────────────────

def _make_ok_opener():
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"hypo answer text"}}]}'

    opener = MagicMock()
    opener.open.return_value = FakeResp()
    return opener


class TestHydeGenerateFailureDegradation:
    """确定性失败永久跳过 / 瞬时失败冷却 / 缓存命中 / 401 / 5xx / 成功解析缓存。"""

    @pytest.fixture(autouse=True)
    def _reset_hyde_state(self, monkeypatch):
        from retrieval import hyde
        hyde._PERM_FAILED = False
        hyde._last_fail_ts = 0.0
        hyde._cache.clear()
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    def test_missing_key_marks_permanent_skip(self):
        from retrieval import hyde
        with patch.object(hyde.urllib.request, "build_opener") as bo:
            assert hyde.generate_hypothesis("q") is None
            assert hyde._PERM_FAILED is True
            assert hyde.generate_hypothesis("q2") is None  # 永久跳过不再触网
        bo.assert_not_called()

    def test_cooldown_window_skips_network(self, monkeypatch):
        from retrieval import hyde
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        hyde._last_fail_ts = time.time()  # 模拟刚发生瞬时失败
        with patch.object(hyde.urllib.request, "build_opener") as bo:
            assert hyde.generate_hypothesis("q") is None
        bo.assert_not_called()

    def test_cache_hit_skips_network(self, monkeypatch):
        from retrieval import hyde
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        hyde._cache["q"] = (time.time(), "cached hypo")
        with patch.object(hyde.urllib.request, "build_opener") as bo:
            assert hyde.generate_hypothesis("q") == "cached hypo"
        bo.assert_not_called()

    def test_http_401_marks_permanent(self, monkeypatch):
        from retrieval import hyde
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        opener = MagicMock()
        opener.open.side_effect = urllib.error.HTTPError(
            hyde._API_URL, 401, "Unauthorized", {}, None)
        with patch.object(hyde.urllib.request, "build_opener", return_value=opener):
            assert hyde.generate_hypothesis("q") is None
        assert hyde._PERM_FAILED is True
        assert hyde._last_fail_ts == 0.0  # 永久失败不落冷却窗口

    def test_http_5xx_sets_cooldown(self, monkeypatch):
        from retrieval import hyde
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        opener = MagicMock()
        opener.open.side_effect = urllib.error.HTTPError(
            hyde._API_URL, 500, "Internal Server Error", {}, None)
        with patch.object(hyde.urllib.request, "build_opener", return_value=opener):
            assert hyde.generate_hypothesis("q") is None
        assert hyde._PERM_FAILED is False
        assert time.time() - hyde._last_fail_ts < 5.0
        with patch.object(hyde.urllib.request, "build_opener") as bo:
            assert hyde.generate_hypothesis("q") is None  # 冷却窗口内不再触网
        bo.assert_not_called()

    def test_success_parses_caches_and_matches_eval_prompt(self, monkeypatch):
        from retrieval import hyde
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        opener = _make_ok_opener()
        with patch.object(hyde.urllib.request, "build_opener", return_value=opener):
            assert hyde.generate_hypothesis("q") == "hypo answer text"

        req = opener.open.call_args.args[0]
        body = json.loads(req.data)
        assert body["model"] == "deepseek-chat"
        assert body["max_tokens"] == 150
        assert body["temperature"] == 0.3
        prompt = body["messages"][0]["content"]
        assert "Based on the question below, write a short factual paragraph" in prompt
        assert "Question: q" in prompt
        assert "Hypothetical passage:" in prompt
        assert req.full_url == hyde._API_URL
        assert req.get_header("Authorization") == "Bearer sk-test"

        with patch.object(hyde.urllib.request, "build_opener") as bo2:
            assert hyde.generate_hypothesis("q") == "hypo answer text"  # 缓存命中
        bo2.assert_not_called()

    def test_model_override_env(self, monkeypatch):
        from retrieval import hyde
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat-v3")
        opener = _make_ok_opener()
        with patch.object(hyde.urllib.request, "build_opener", return_value=opener):
            hyde.generate_hypothesis("q")
        body = json.loads(opener.open.call_args.args[0].data)
        assert body["model"] == "deepseek-chat-v3"

    def test_chinese_query_prompt_has_language_constraint(self, monkeypatch):
        """【P3b R1 P2】中文查询 → prompt 含语言一致性约束（语种匹配提升检索）。"""
        from retrieval import hyde
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        opener = _make_ok_opener()
        with patch.object(hyde.urllib.request, "build_opener", return_value=opener):
            hyde.generate_hypothesis("记忆是什么")
        body = json.loads(opener.open.call_args.args[0].data)
        prompt = body["messages"][0]["content"]
        assert "Write in the same language as the question." in prompt
        # 模板结构保留（评测脚本等效性不变）
        assert "Question: 记忆是什么" in prompt
        assert "Hypothetical passage:" in prompt


# ─── P3b R1：配置感知 level 映射（P0-1）+ 超时预算 helper（P0-2）────────────

class TestLevelFromStrategyConfigAware:
    """_level_from_strategy(strategy, config)：hybrid 恒 FUSION；auto/空/None 且
    rerank/hyDE 任一开启 → FUSION；其余回落 HYPERGRAPH（零回归）。"""

    def test_hybrid_always_fusion(self):
        assert _level_from_strategy("hybrid") == RetrievalLevel.FUSION
        assert _level_from_strategy("hybrid", QueryRouterConfig()) == RetrievalLevel.FUSION

    def test_auto_with_rerank_enabled_is_fusion(self):
        # rerank_enabled=True（默认）→ 生产 auto 请求进入 FUSION
        cfg = QueryRouterConfig(rerank_enabled=True)
        assert _level_from_strategy("auto", cfg) == RetrievalLevel.FUSION

    def test_auto_with_hyde_enabled_is_fusion(self):
        cfg = QueryRouterConfig(rerank_enabled=False, hyde_enabled=True)
        assert _level_from_strategy("auto", cfg) == RetrievalLevel.FUSION

    def test_auto_without_rerank_hyde_is_hypergraph(self):
        cfg = QueryRouterConfig(rerank_enabled=False, hyde_enabled=False)
        assert _level_from_strategy("auto", cfg) == RetrievalLevel.HYPERGRAPH

    def test_empty_and_none_with_config_are_hypergraph_without_flags(self):
        cfg = QueryRouterConfig(rerank_enabled=False, hyde_enabled=False)
        assert _level_from_strategy("", cfg) == RetrievalLevel.HYPERGRAPH
        assert _level_from_strategy(None, cfg) == RetrievalLevel.HYPERGRAPH

    def test_none_config_keeps_old_behavior(self):
        # config 缺省（None，旧调用方/无 config 的 mock router）→ auto 回落 HYPERGRAPH
        assert _level_from_strategy("auto", None) == RetrievalLevel.HYPERGRAPH
        assert _level_from_strategy("", None) == RetrievalLevel.HYPERGRAPH
        assert _level_from_strategy(None, None) == RetrievalLevel.HYPERGRAPH

    def test_tau_first_and_unknown_stay_hypergraph(self):
        cfg = QueryRouterConfig()  # rerank_enabled=True 也不影响显式策略
        assert _level_from_strategy("tau_first", cfg) == RetrievalLevel.HYPERGRAPH
        assert _level_from_strategy("vector_first", cfg) == RetrievalLevel.HYPERGRAPH
        assert _level_from_strategy("garbage", cfg) == RetrievalLevel.HYPERGRAPH


class TestRetrieveTimeout:
    """_retrieve_timeout(level, config)：FUSION+HyDE → 5.0；否则 3.0（零回归）。"""

    def test_fusion_with_hyde_is_5s(self):
        cfg = QueryRouterConfig(rerank_enabled=False, hyde_enabled=True)
        assert _retrieve_timeout(RetrievalLevel.FUSION, cfg) == 5.0

    def test_fusion_without_hyde_is_3s(self):
        cfg = QueryRouterConfig(rerank_enabled=False, hyde_enabled=False)
        assert _retrieve_timeout(RetrievalLevel.FUSION, cfg) == 3.0

    def test_hypergraph_is_3s_even_with_hyde(self):
        cfg = QueryRouterConfig(hyde_enabled=True)
        assert _retrieve_timeout(RetrievalLevel.HYPERGRAPH, cfg) == 3.0

    def test_none_config_is_3s(self):
        assert _retrieve_timeout(RetrievalLevel.FUSION, None) == 3.0


class TestQueryRouterConfigHydeModeValidation:
    """【P3b R1 P2】QueryRouterConfig.__post_init__ 校验 hyde_mode ∈ {dual, replace}。"""

    def test_invalid_hyde_mode_raises(self):
        with pytest.raises(ValueError):
            QueryRouterConfig(hyde_mode="triple")

    def test_valid_modes_ok(self):
        QueryRouterConfig(hyde_mode="dual")
        QueryRouterConfig(hyde_mode="replace")
