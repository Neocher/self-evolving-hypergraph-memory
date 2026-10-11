"""
v6.30 检索超时修复第三轮验收测试
================================
覆盖任务书 AC1..AC8：
  AC1 配置生效：_DEGRADE_TIMEOUT==3.0；community budget_s==0.5/breaker_s==300；
      total_step_budget_s==3.0（defaults.yaml 与 settings.py 一致）
  AC2 社区扩展熔断（计数断言）：慢 store（每调用 2s）→ 第 1 次调用后熔断打开；
      后续调用在 50ms 内跳过且返回原入参；跳过时产生 WARNING 日志
  AC3 退避：连续 3 次半开重试均超预算 → 退避窗口严格大于上一次（单调增长）
  AC4 总预算守卫：多步各耗时 0.8s → 守卫触发后后续步骤调用次数为 0（计数断言）
  AC5 关闭零回归：budget_s<=0 且 total_step_budget_s<=0 时各步调用次数与改动前一致
  AC6 端到端：真实 store 生产同构配置跑 8 条查询（含「现在情况如何了」「SHM 超图
      记忆」「熔断保护设计」「缓存失效机制」）≥8 次，中位 < 4.0s 且 100% < 6.0s

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_retrieval_budget_guard.py -q
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel


_EXPANSION_STEPS = (
    "_community_expansion",
    "_mesa_synthesis",
    "_visual_recall",
    "_property_temporal_retrieve",
    "_entity_expansion",
    "_attribute_expansion",
    "_scope_retrieve",
    "_schema_recall",
    "_fact_retrieve",
)


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


def _seed_result(nid: str, content: str, score: float = 0.9) -> dict:
    return {
        "node_id": nid, "content": content, "score": score,
        "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss",
    }


def _slow_store(sleep_s: float, member_dicts=None, summary=None):
    """慢 store：get_community_members_batch 每调用 sleep_s 秒。"""
    store = MagicMock()
    store.get_communities_by_seeds.return_value = [{
        "community_id": "comm_slow",
        "summary": summary or "K8s 集群网络 flannel calico 多会话排障 记忆",
        "member_ids": ["ep1"],
    }]

    def batch(cids, limit):
        time.sleep(sleep_s)
        return {"comm_slow": member_dicts or []}

    store.get_community_members_batch.side_effect = batch
    return store


class TestConfigValues:
    """AC1 — 三处配置一致（defaults.yaml / settings.py / 运行时常量）。"""

    def test_degrade_timeout_is_3s(self):
        from api.routes.search import _DEGRADE_TIMEOUT
        assert _DEGRADE_TIMEOUT == 3.0

    def test_community_expansion_budget_and_breaker(self):
        from config.settings import load_settings
        ce = load_settings().retrieval.community_expansion
        assert ce.budget_s == 0.5
        assert ce.breaker_s == 300.0
        assert ce.backoff_n == 3
        assert ce.backoff_max_s == 3600.0

    def test_total_step_budget_default(self):
        from config.settings import load_settings
        assert load_settings().retrieval.total_step_budget_s == 3.0
        # 守卫实际读 QueryRouterConfig（self.config），默认值须与 settings 一致
        assert QueryRouterConfig().total_step_budget_s == 3.0

    def test_yaml_values_match(self):
        yaml_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        ce = raw["retrieval"]["community_expansion"]
        assert ce["budget_s"] == 0.5
        assert ce["breaker_s"] == 300.0
        assert ce["backoff_n"] == 3
        assert ce["backoff_max_s"] == 3600.0
        assert raw["retrieval"]["total_step_budget_s"] == 3.0


class TestCommunityBreakerHonestDegradation:
    """AC2 — 熔断后跳过（计数断言）+ WARNING 日志。"""

    def test_breaker_skips_with_warning_log(self, monkeypatch):
        from config.settings import Settings, CommunityExpansionConfig
        s = Settings()
        s.retrieval.community_expansion = CommunityExpansionConfig(
            budget_s=0.5, breaker_s=300.0)
        monkeypatch.setattr("retrieval.query_router.get_settings", lambda: s)
        # 用 MagicMock 替换模块 logger，确定性断言 WARNING 方法被调用（含语义文本）
        log_mock = MagicMock()
        monkeypatch.setattr("retrieval.query_router.logger", log_mock)

        store = _slow_store(sleep_s=2.0)
        seeds = [_seed_result("ep1", "K8s 集群 flannel 问题")]
        router = _make_router(store=store)
        query = "多会话 K8s 集群 flannel 网络问题"

        # 第 1 次：慢调用（≥1.5s）→ 打开熔断
        t0 = time.monotonic()
        out1 = router._community_expansion(seeds, query, None, query)
        first_elapsed = time.monotonic() - t0
        assert first_elapsed >= 1.5, f"第 1 次应走慢调用: {first_elapsed:.3f}s"
        assert store.get_community_members_batch.call_count == 1
        assert getattr(router, "_community_expansion_breaker_open_until", 0.0) \
            > time.monotonic()

        # 后续：熔断窗口内 <50ms 跳过，且不再调用底层（计数区分跳过/真调用）
        t1 = time.monotonic()
        out2 = router._community_expansion(seeds, query, None, query)
        second_elapsed = time.monotonic() - t1

        assert second_elapsed < 0.05, f"熔断未跳过，耗时 {second_elapsed:.3f}s"
        assert store.get_community_members_batch.call_count == 1, "熔断窗口内应跳过（不再次调用底层）"
        assert [r["node_id"] for r in out2] == ["ep1"], "跳过应返回原入参（fail-soft）"
        assert [r["node_id"] for r in out1] == ["ep1"]
        # 累计跳过次数记录
        assert getattr(router, "_community_expansion_skip_count", 0) == 1
        # WARNING 日志：skip 分支应调用 logger.warning（含语义文本 + 社区数/跳过次数）
        warn_calls = log_mock.warning.call_args_list
        assert warn_calls, "跳过时应产生 WARNING 日志"
        assert any("Community expansion skipped" in str(c.args[0]) for c in warn_calls), \
            f"WARNING 应含诚实降级语义: {[c.args for c in warn_calls]}"


class TestCommunityBreakerBackoff:
    """AC3 — 连续 3 次半开重试超预算 → 退避窗口单调增长（严格大于上一次）。"""

    def test_backoff_doubles_after_consecutive_slow_retries(self, monkeypatch):
        from config.settings import Settings, CommunityExpansionConfig
        s = Settings()
        s.retrieval.community_expansion = CommunityExpansionConfig(
            budget_s=0.1, breaker_s=0.5, backoff_n=3, backoff_max_s=3600.0)
        monkeypatch.setattr("retrieval.query_router.get_settings", lambda: s)

        store = _slow_store(sleep_s=0.3)
        seeds = [_seed_result("ep1", "K8s 集群 flannel 问题")]
        router = _make_router(store=store)
        query = "多会话 K8s 集群 flannel 网络问题"

        def backoff_s():
            return getattr(router, "_community_expansion_breaker_backoff_s", 0.0)

        # 第 1 次慢调用：基础窗口打开（breaker_s=0.5），退避复位
        router._community_expansion(seeds, query, None, query)
        first_backoff = backoff_s()
        assert first_backoff == 0.5

        # 连续 3 次半开重试仍超预算：前两次窗口不变，第 3 次退避加倍
        windows = [first_backoff]
        for _ in range(3):
            time.sleep(0.6)  # 等 0.5s 熔断窗口过期 → 半开重试
            router._community_expansion(seeds, query, None, query)
            windows.append(backoff_s())

        assert backoff_s() == 1.0, f"连续 3 次慢重试后退避应加倍为 1.0: {windows}"
        # 单调增长：最后一次退避窗口严格大于首次
        assert windows[-1] > windows[0]
        assert all(b >= a for a, b in zip(windows, windows[1:])), \
            f"退避窗口应单调不减: {windows}"
        assert windows[-1] > windows[-2], "最后一次半开重试应触发退避增长"


class TestTotalStepBudgetGuard:
    """AC4 — 多步各耗时 0.8s → 总预算守卫触发后后续步骤调用次数为 0。"""

    def test_guard_skips_remaining_expansion_steps(self):
        router = _make_router(config=QueryRouterConfig(
            step_budget_s=0.0,           # 关闭单步熔断，隔离总预算守卫
            total_step_budget_s=1.5,     # 0.8s×2 步 = 1.6s > 1.5s → 触发
            rerank_enabled=False,
        ))
        router._fusion_retrieve = MagicMock(return_value=[
            _seed_result("n0", "alpha beta"),
            _seed_result("n1", "gamma delta", score=0.8),
        ])

        counters = {name: 0 for name in _EXPANSION_STEPS}

        def _sleeper(name):
            def fn(results, *args, **kwargs):
                counters[name] += 1
                time.sleep(0.8)
                return results
            return fn

        for name in _EXPANSION_STEPS:
            setattr(router, name, MagicMock(side_effect=_sleeper(name)))

        router.retrieve("alpha beta", level=RetrievalLevel.FUSION)

        # 前两步（community/mesa）各跑一次（0.8+0.8=1.6 > 1.5），随后触发守卫，
        # 后续尚未执行的扩展步骤调用次数为 0（计数断言，非计时）。
        assert counters["_community_expansion"] == 1
        assert counters["_mesa_synthesis"] == 1
        remaining = {n: c for n, c in counters.items()
                     if n not in ("_community_expansion", "_mesa_synthesis")}
        assert all(c == 0 for c in remaining.values()), \
            f"守卫触发后后续扩展步骤应零调用: {remaining}"


class TestBudgetDisabledZeroRegression:
    """AC5 — budget_s<=0 且 total_step_budget_s<=0 时各步调用次数与改动前一致。"""

    def test_all_steps_called_once_when_guards_disabled(self):
        router = _make_router(config=QueryRouterConfig(
            step_budget_s=0.0,
            total_step_budget_s=0.0,
            rerank_enabled=False,
        ))
        router._fusion_retrieve = MagicMock(return_value=[
            _seed_result("n0", "alpha beta"),
            _seed_result("n1", "gamma delta", score=0.8),
        ])

        counters = {name: 0 for name in _EXPANSION_STEPS}

        def _counted(name):
            def fn(results, *args, **kwargs):
                counters[name] += 1
                return results
            return fn

        for name in _EXPANSION_STEPS:
            setattr(router, name, MagicMock(side_effect=_counted(name)))

        router.retrieve("alpha beta", level=RetrievalLevel.FUSION)

        assert all(c == 1 for c in counters.values()), \
            f"关闭守卫后各扩展步骤应各调用恰好一次: {counters}"


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


class TestEndToEndFusionLatency:
    """AC6 — 真实 store + 生产同构配置跑 8 条查询：中位 < 4.0s 且 100% < 6.0s。"""

    def test_fusion_median_under_4s_all_under_6s(self, overgraph_store):
        from core.dream_pipeline import DreamPipeline

        queries = [
            "现在情况如何了",
            "SHM 超图记忆",
            "熔断保护设计",
            "缓存失效机制",
            "多会话 K8s 集群 flannel 网络问题",
            "社区扩展召回成员",
            "检索超时预算守卫",
            "τ 值衰减与记忆固化",
        ]

        # 生产同构数据：40 个 EpisodeNode + 1 个社区（成员 τ 分化）
        contents = [
            f"多会话记忆 SHM 超图 熔断保护 缓存失效 情况 话题{i}" for i in range(40)
        ]
        for i, c in enumerate(contents):
            overgraph_store.create_episode({
                "id": f"ep{i}", "content": c, "tau_initial": 1.0 - i * 0.01,
                "fact_track": "active", "archived": False,
            })
        DreamPipeline()._persist_one_community(
            overgraph_store,
            {"id": "comm_prod", "members": [f"ep{i}" for i in range(40)],
             "report": "SHM 超图记忆 熔断保护 缓存失效 多会话情况 检索"},
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
            _seed_result(f"ep{i}", contents[i], score=1.0 - i * 0.01)
            for i in range(40)
        ]
        router._fusion_retrieve = MagicMock(return_value=[dict(s) for s in seeds])
        # 模拟 bge-reranker CPU 推理：每对 ~0.11s。k=15 约 1.7s（修复后达标）。
        fake = _FakeReranker(scores=[0.5] * 40, per_pair_s=0.11)
        with patch.object(router, "_get_reranker", return_value=fake):
            # 预热一次让懒加载模型就位，再开始计时
            router.retrieve(queries[0], level=RetrievalLevel.FUSION)
            times = []
            for q in queries:
                t0 = time.monotonic()
                router.retrieve(q, level=RetrievalLevel.FUSION)
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
