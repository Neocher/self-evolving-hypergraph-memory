"""
v6.29 检索超时修复第二轮验收测试
================================
覆盖任务书 AC1..AC6：
  AC1 配置生效：rerank_input_k 默认 15（defaults.yaml / settings.py / QueryRouterConfig 一致）
  AC2 rerank 提速机制：k=15 只送 15 对候选进 reranker；rerank=False 与 k 无关逐字节一致
  AC3 预算化生效（调用计数而非计时）：慢函数第 1 次调用后该步熔断打开，后续 <50ms 跳过并返回原入参
  AC4 半开恢复：熔断窗口过期 + 恢复正常耗时 → 该步重新生效（再次调用）
  AC5 预算关闭零回归：budget_s<=0 时调用次数与无包装一致
  AC6 端到端：真实 store + 生产同构配置 FUSION ≥6 次，中位 <4.0s 且 100% <6.0s

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_retrieval_step_budget.py -q
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel


def _make_router(config=None, store=None):
    """构造零依赖 QueryRouter（__new__ 绕过 __init__，不触发真实模型/索引加载）。"""
    router = QueryRouter.__new__(QueryRouter)
    router.config = config or QueryRouterConfig()
    router._zh_en_tech_map = {}
    router._time_keywords = set()
    router.graph_store = store
    router.encoder = None
    router._hypergraph_retrieve = MagicMock(return_value=[])
    return router


class _FakeReranker:
    """记录 pairs 的假 CrossEncoder：可选每对耗时，模拟 CPU 推理。"""

    def __init__(self, scores=None, per_pair_s: float = 0.0):
        self._scores = scores
        self.per_pair_s = per_pair_s
        self.pairs = None
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        self.pairs = list(pairs)
        if self.per_pair_s > 0:
            time.sleep(self.per_pair_s * len(pairs))
        if callable(self._scores):
            return self._scores(pairs)
        return list(self._scores) if self._scores else [0.5] * len(pairs)


class TestRerankInputKConfig:
    """AC1 — rerank_input_k 默认 15（三处一致）。"""

    def test_query_router_config_default_15(self):
        assert QueryRouterConfig().rerank_input_k == 15

    def test_settings_default_15(self):
        from config.settings import load_settings
        assert load_settings().retrieval.rerank_input_k == 15

    def test_yaml_default_15_and_step_budget(self):
        yaml_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert raw["retrieval"]["rerank_input_k"] == 15
        assert raw["retrieval"]["step_budget_s"] == 1.0
        assert raw["retrieval"]["step_breaker_s"] == 30.0


class TestRerankInputKMechanism:
    """AC2 — k=15 只送 15 对候选；rerank=False 与 k 无关逐字节一致。"""

    def _results(self, n=40):
        return [
            {"node_id": f"n{i}", "content": f"content{i}", "score": 1.0 - i * 0.01}
            for i in range(n)
        ]

    def test_k15_limits_reranker_pairs(self):
        router = _make_router()
        router.config.rerank_input_k = 15
        fake = _FakeReranker(scores=[0.5] * 40)
        with patch.object(router, "_get_reranker", return_value=fake):
            router._rerank_results(self._results(), "q", enabled=True)
        assert len(fake.pairs) == 15

    def test_k40_reranks_more_pairs_than_k15(self):
        # 证明「少送候选」的提速机制：k=40 送 40 对，k=15 送 15 对
        pairs = {}
        for k in (15, 40):
            router = _make_router()
            router.config.rerank_input_k = k
            fake = _FakeReranker(scores=[0.5] * 40)
            with patch.object(router, "_get_reranker", return_value=fake):
                router._rerank_results(self._results(), "q", enabled=True)
            pairs[k] = len(fake.pairs)
        assert pairs[15] == 15
        assert pairs[40] == 40

    def test_rerank_disabled_identical_regardless_of_k(self):
        results = self._results()
        r15 = _make_router()
        r15.config.rerank_input_k = 15
        r40 = _make_router()
        r40.config.rerank_input_k = 40
        out15 = r15._rerank_results(results, "q", enabled=False)
        out40 = r40._rerank_results(results, "q", enabled=False)
        assert out15 is results and out40 is results
        assert out15 == out40


class TestRunWithBudget:
    """AC3/AC4/AC5 — _run_with_budget 预算熔断（调用计数 + 跳过耗时断言）。"""

    def test_breaker_skips_after_slow_call(self):
        router = _make_router()
        router.config.step_budget_s = 0.1
        router.config.step_breaker_s = 60.0
        calls = {"n": 0}

        def fn(results):
            calls["n"] += 1
            time.sleep(0.5)
            return results + [{"node_id": "extra"}]

        results = [{"node_id": "seed"}]
        t0 = time.monotonic()
        out1 = router._run_with_budget("test_step", fn, results)
        first_elapsed = time.monotonic() - t0
        assert first_elapsed >= 0.4, f"第 1 次应走慢调用: {first_elapsed:.3f}s"
        assert calls["n"] == 1
        assert out1 == results + [{"node_id": "extra"}]

        # 熔断打开：后续 <50ms 直接跳过，且不再调用 fn（调用计数区分「跳过」与「真调用」）
        t1 = time.monotonic()
        out2 = router._run_with_budget("test_step", fn, results)
        second_elapsed = time.monotonic() - t1
        assert second_elapsed < 0.05, f"熔断未跳过，耗时 {second_elapsed:.3f}s"
        assert calls["n"] == 1, "熔断窗口内应跳过（不再调用 fn）"
        assert out2 == results, "跳过应返回原入参（fail-soft）"

    def test_half_open_recovers(self):
        router = _make_router()
        router.config.step_budget_s = 0.1
        router.config.step_breaker_s = 0.5
        state = {"slow": True}
        calls = {"n": 0}

        def fn(results):
            calls["n"] += 1
            if state["slow"]:
                time.sleep(0.3)
            return results + [{"node_id": "x"}]

        results = [{"node_id": "seed"}]
        router._run_with_budget("s", fn, results)
        assert calls["n"] == 1
        # 窗口内跳过
        router._run_with_budget("s", fn, results)
        assert calls["n"] == 1

        # 熔断窗口过期 + 恢复正常 → 半开重试，重新调用
        time.sleep(0.6)
        state["slow"] = False
        out = router._run_with_budget("s", fn, results)
        assert calls["n"] == 2, "半开未重试（call_count 仍为 1）"
        assert out == results + [{"node_id": "x"}]

    def test_budget_disabled_calls_every_time(self):
        # budget_s<=0 → 每次直接调用（与无包装逐字节一致）
        router = _make_router()
        router.config.step_budget_s = 0.0
        calls = {"n": 0}

        def fn(results):
            calls["n"] += 1
            return results + [{"node_id": "x"}]

        for _ in range(3):
            out = router._run_with_budget("s", fn, [])
            assert out == [{"node_id": "x"}], "关闭预算时输出应与直接调用 fn 一致"
        assert calls["n"] == 3

    def test_per_step_override(self):
        # step_budget per-step 覆盖：统一关闭，个别步开启
        router = _make_router()
        router.config.step_budget_s = 0.0
        router.config.step_breaker_s = 30.0
        router.config.step_budget = {"only_this": {"budget_s": 0.1, "breaker_s": 60.0}}
        calls = {"n": 0}

        def fn(results):
            calls["n"] += 1
            time.sleep(0.3)
            return results

        router._run_with_budget("only_this", fn, [])  # 超预算 → 打开熔断
        assert calls["n"] == 1
        router._run_with_budget("only_this", fn, [])  # 窗口内跳过
        assert calls["n"] == 1

        # 另一未覆盖步：统一关闭 → 每次调用
        for _ in range(2):
            router._run_with_budget("other", fn, [])
        assert calls["n"] == 3


class TestBudgetWiredIntoFinish:
    """AC5（retrieve 级）— budget_s<=0 时各步调用次数与改动前完全一致。"""

    def _passthrough(self, results, *args, **kwargs):
        return results

    def test_steps_called_once_when_budget_disabled(self):
        router = _make_router()
        router.config.step_budget_s = 0.0
        router.config.rerank_enabled = False
        router._fusion_retrieve = MagicMock(return_value=[
            {"node_id": "n0", "content": "alpha beta", "score": 0.9,
             "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss"},
            {"node_id": "n1", "content": "gamma delta", "score": 0.8,
             "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss"},
        ])

        counters = {}
        for name in (
            "_community_expansion",
            "_mesa_synthesis",
            "_visual_recall",
            "_property_temporal_retrieve",
            "_entity_expansion",
            "_attribute_expansion",
            "_scope_retrieve",
            "_schema_recall",
            "_fact_retrieve",
        ):
            counters[name] = 0
            setattr(router, name, MagicMock(side_effect=self._passthrough))
        # 用 wrap 计数：包装后各步仍会被调用恰好一次（budget 关闭 → 直接调用）
        for name in (
            "_property_temporal_retrieve",
            "_entity_expansion",
            "_attribute_expansion",
            "_scope_retrieve",
        ):
            orig = getattr(router, name)

            def _counted(results, *args, _orig=orig, _name=name, **kwargs):
                counters[_name] += 1
                return _orig(results, *args, **kwargs)

            setattr(router, name, _counted)

        router.retrieve("alpha beta", level=RetrievalLevel.FUSION)

        assert counters["_property_temporal_retrieve"] == 1
        assert counters["_entity_expansion"] == 1
        assert counters["_attribute_expansion"] == 1
        assert counters["_scope_retrieve"] == 1


class TestEndToEndFusionLatency:
    """AC6 — 真实 store + 生产同构配置 FUSION ≥6 次：中位 <4.0s 且 100% <6.0s。"""

    def test_fusion_median_under_4s_all_under_6s(self, overgraph_store):
        from core.dream_pipeline import DreamPipeline

        # 生产同构数据：40 个 EpisodeNode + 1 个社区（成员 τ 分化）
        contents = [
            f"多会话记忆 K8s 集群 flannel calico 网络排障 话题{i}" for i in range(40)
        ]
        for i, c in enumerate(contents):
            overgraph_store.create_episode({
                "id": f"ep{i}", "content": c, "tau_initial": 1.0 - i * 0.01,
                "fact_track": "active", "archived": False,
            })
        DreamPipeline()._persist_one_community(
            overgraph_store,
            {"id": "comm_prod", "members": [f"ep{i}" for i in range(40)],
             "report": "K8s 集群网络 flannel calico 多会话排障 记忆"},
            "eval", 0,
        )

        # 生产同构配置：rerank 开启（k 默认 15）、扩展全开、预算默认宽松。
        router = QueryRouter(
            graphlite_store=overgraph_store,
            faiss_index=MagicMock(),
            tfidf_index=None,
            encoder=None,
            faiss_id_map={},
            episode_cache={},
            config=QueryRouterConfig(rerank_enabled=True),
        )
        seeds = [
            {"node_id": f"ep{i}", "content": contents[i], "score": 1.0 - i * 0.01,
             "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss"}
            for i in range(40)
        ]
        router._fusion_retrieve = MagicMock(return_value=[dict(s) for s in seeds])
        # 模拟 bge-reranker CPU 推理：每对 ~0.11s（对齐生产 ~4.5s/40 对）。
        # k=40 约 4.4s（修复前中位被拖过 4s），k=15 约 1.7s（修复后达标）。
        fake = _FakeReranker(scores=[0.5] * 40, per_pair_s=0.11)
        with patch.object(router, "_get_reranker", return_value=fake):
            # 预热一次让懒加载模型就位，再开始计时
            router.retrieve("多会话 K8s 集群 flannel 网络问题", level=RetrievalLevel.FUSION)
            times = []
            for _ in range(6):
                t0 = time.monotonic()
                router.retrieve(
                    "多会话 K8s 集群 flannel 网络问题", level=RetrievalLevel.FUSION)
                times.append(time.monotonic() - t0)

        # rerank 头部只送 15 对（k=15 生效，防「关闭 rerank」作弊）
        assert len(fake.pairs) <= 15, f"reranker 应只收到 ≤15 对，实际 {len(fake.pairs)}"
        median = sorted(times)[len(times) // 2]
        assert median < 4.0, (
            f"FUSION 中位耗时 {median:.3f}s >= 4.0s: {[round(t, 3) for t in times]}"
        )
        assert max(times) < 6.0, (
            f"FUSION 存在 >6.0s 请求（会被 6s 预算砍掉）: {[round(t, 3) for t in times]}"
        )
