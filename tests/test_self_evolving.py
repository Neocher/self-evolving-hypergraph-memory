"""测试检索自演化模块（v5.42 真实化改造）

覆盖：生效旋钮映射（top_k_*，不再调 weight_fusion_*/tau_weight/vector_weight——
后两者仅被零调用的 hybrid_score 消费，死旋钮）、
周期强制 + 即时硬失败触发（质量阈值仅统计）、_sync_params 仅演化后调用、
持久化 roundtrip、梦境探针召回喂入、AC 端到端（3 次 degraded → top_k_l1 演化）。
"""
import asyncio
import json
import os
import tempfile
import threading
import time

import numpy as np
import pytest
import faiss

from retrieval.self_evolving import (
    EvolvableParams, FailureLogger, RetrievalSnapshot,
    DiagnosisEngine, EvolutionGuard, SelfEvolvingRetrieval,
)
from retrieval.query_router import QueryRouter, QueryRouterConfig
from core.dream_pipeline import DreamPipeline


def _cfg(**over):
    """构造带全字段的 QueryRouterConfig 替身（可覆盖个别字段）。"""
    class Config:
        weight_fusion_vector = 0.35
        weight_fusion_bm25 = 0.40
        weight_fusion_entity = 0.25
        tau_weight = 0.4
        vector_weight = 0.6
        top_k_l1 = 5
        top_k_fusion = 30
        top_k_keyword = 20
        top_k_vector = 20
        bm25_k1 = 1.5
        bm25_b = 0.75

    c = Config()
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _tmp_persist_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    return path


class MockRouter:
    """默认返回低质量但非降级结果（score 0.7，1 条）"""

    def __init__(self, config=None, results=None):
        self.config = config or _cfg()
        self._results = (results if results is not None
                         else [{"content": "result", "score": 0.7, "node_id": "r1"}])

    def retrieve(self, q, include_archived=False, session_ts=None):
        return self._results

    def state(self):
        return {}


def make_se(router=None, **kw):
    """构造 SE；默认持久化到临时文件（避免污染仓库 data/）。"""
    kw.setdefault("persist_path", _tmp_persist_path())
    return SelfEvolvingRetrieval(router or MockRouter(), **kw)


def _real_router(n_docs: int = 10, top_k_l1: int = 2, **cfg_over):
    """真实 FAISS + mock GraphLite 的 QueryRouter（生产默认 HYPERGRAPH 路径）。

    Returns:
        (QueryRouter, DegradableFaiss) — index.down=True 时检索降级为空（硬失败信号）。
    """
    dim = 8
    rng = np.random.RandomState(42)
    base = faiss.IndexFlatL2(dim)
    base.add(rng.randn(n_docs, dim).astype("float32"))

    class DegradableFaiss:
        """真实 FAISS 索引封装：down=True 时返回全 -1（模拟检索降级）"""

        def search(self, emb, k):
            if self.down:
                return (np.full((1, k), 999.0, dtype="float32"),
                        np.full((1, k), -1, dtype="int64"))
            return base.search(emb, k)

    index = DegradableFaiss()
    index.down = False
    episodes = {
        str(i): {"id": str(i), "content": f"doc_{i}",
                 "tau_initial": 0.5, "fact_track": "active", "archived": False}
        for i in range(n_docs)
    }

    class FakeEncoder:
        def embed(self, text):
            return np.zeros(dim, dtype="float32")

    class FakeStore:
        def get_episodes_batch(self, ids):
            # 返回副本：真实 GraphLite 每次回查返回新 dict（查询路径会 pop/改写）
            return [dict(episodes[i]) for i in ids if i in episodes]

    qr = QueryRouter(
        graphlite_store=FakeStore(),
        faiss_index=index,
        tfidf_index=None,
        encoder=FakeEncoder(),
        faiss_id_map={i: str(i) for i in range(n_docs)},
        episode_cache={},
        config=QueryRouterConfig(top_k_l1=top_k_l1, **cfg_over),
    )
    return qr, index


class TestEvolvableParams:
    def test_validate_ok(self):
        assert EvolvableParams().validate() == []

    def test_validate_out_of_range(self):
        p = EvolvableParams()
        p.weight_fusion_vector = 1.5
        p.tau_weight = 2.0
        p.top_k_vector = 0
        errs = p.validate()
        assert any("weight_fusion_vector" in e for e in errs)
        assert any("tau_weight" in e for e in errs)
        assert any("top_k_vector" in e for e in errs)

    def test_snapshot_roundtrip(self):
        snap = EvolvableParams().snapshot()
        assert snap["weight_fusion_vector"] == 0.35
        assert snap["tau_weight"] == 0.4
        assert snap["bm25_k1"] == 1.5

    def test_from_config_reads_user_values(self):
        """根因#3：初始值从 config 读，不硬编码覆盖用户配置"""
        p = EvolvableParams.from_config(_cfg(tau_weight=0.7, top_k_vector=44))
        assert p.tau_weight == 0.7
        assert p.top_k_vector == 44

    def test_from_config_missing_field_default(self):
        """cfg 缺字段（测试 mock）→ 默认值兜底不抛错"""
        class PartialCfg:
            tau_weight = 0.5
        p = EvolvableParams.from_config(PartialCfg())
        assert p.tau_weight == 0.5
        assert p.top_k_vector == 20  # 缺失字段用 dataclass 默认


class TestFailureLogger:
    def _snap(self, query, num, score, distinct, degraded):
        return RetrievalSnapshot(
            timestamp=time.time(), query=query, params_before={},
            num_results=num, top_scores=[score], avg_score=score,
            top_distinct=distinct, latency_ms=100, degraded=degraded,
        )

    def test_log_appends_and_state(self):
        logger = FailureLogger(quality_threshold=0.5)
        logger.log(self._snap("poor", 0, 0.0, 0, True))       # quality 0 → poor
        logger.log(self._snap("good", 10, 0.9, 8, False))     # quality 高 → good
        st = logger.state()
        assert st["total"] == 2
        assert st["poor"] == 1 and st["good"] == 1

    def test_recent_poor_filters(self):
        logger = FailureLogger(quality_threshold=0.5)
        logger.log(self._snap("poor", 0, 0.0, 0, True))
        logger.log(self._snap("good", 10, 0.9, 8, False))
        poor = logger.recent_poor(5)
        assert len(poor) == 1 and poor[0].query == "poor"

    def test_log_returns_none(self):
        """log 不再返回触发信号（触发改周期/硬失败）"""
        logger = FailureLogger()
        assert logger.log(self._snap("q", 0, 0.0, 0, True)) is None


class TestDiagnosisEngine:
    def _snap(self, num, score, distinct, latency=100, degraded=False):
        return RetrievalSnapshot(
            timestamp=time.time(), query="t", params_before={},
            num_results=num, top_scores=[score], avg_score=score,
            top_distinct=distinct, latency_ms=latency, degraded=degraded,
        )

    def test_no_snapshots(self):
        result = DiagnosisEngine().diagnose([], EvolvableParams())
        assert result.suggested == {}

    def test_few_results_raises_topk(self):
        """规则 1：结果少 → 扩大候选集（含生产默认路径旋钮 top_k_l1）"""
        engine = DiagnosisEngine()
        p = EvolvableParams()
        snaps = [self._snap(1, 0.5, 1)]
        result = engine.diagnose(snaps, p)
        assert result.suggested.get("top_k_l1", 5) == 6  # 固定 +1（乘性会 int 截断）
        assert result.suggested.get("top_k_vector", 20) > 20
        assert result.suggested.get("top_k_keyword", 20) > 20
        assert "weight_fusion_vector" not in result.suggested

    def test_low_score_expands_vector_candidates(self):
        """规则 2：分数低/噪声 → 扩 top_k_vector（原调 vector_weight 死旋钮）"""
        engine = DiagnosisEngine()
        p = EvolvableParams()
        snaps = [self._snap(10, 0.2, 5)]  # 数量足、多样足，仅分数低
        result = engine.diagnose(snaps, p)
        assert result.suggested.get("top_k_vector", 20) > 20
        assert "vector_weight" not in result.suggested, "死旋钮不再演化"

    def test_single_content_raises_keyword_narrows_vector(self):
        """规则 3：内容单一 → 提 top_k_keyword + 收窄 top_k_vector"""
        engine = DiagnosisEngine()
        p = EvolvableParams()
        snaps = [self._snap(10, 0.7, 1)]  # 数量足、分数高，仅多样低
        result = engine.diagnose(snaps, p)
        assert result.suggested.get("top_k_keyword", 20) > 20
        assert result.suggested.get("top_k_vector", 20) < 20
        assert "vector_weight" not in result.suggested, "死旋钮不再演化"

    def test_high_latency_lowers_topk(self):
        """规则 4：延迟高 → 降 top_k_vector/top_k_keyword（不动 top_k_l1）"""
        engine = DiagnosisEngine()
        p = EvolvableParams()
        snaps = [self._snap(5, 0.65, 4, latency=600)]
        result = engine.diagnose(snaps, p)
        assert "top_k_l1" not in result.suggested, "延迟规则不再降 top_k_l1"
        assert result.suggested.get("top_k_vector", 20) < 20
        assert result.suggested.get("top_k_keyword", 20) < 20

    def test_clamp_lower_bound_preserves_shrink_direction(self):
        """P2：top_k_vector=3 + 延迟规则 → 应降为 2，而非被 clamp 下界 5 抬到 5"""
        engine = DiagnosisEngine()
        p = EvolvableParams(top_k_vector=3, top_k_keyword=20)
        snaps = [self._snap(5, 0.65, 4, latency=600)]
        result = engine.diagnose(snaps, p)
        assert result.suggested.get("top_k_vector") == 2, \
            f"3 应降为 2 而非升 5: {result.suggested.get('top_k_vector')}"

    def test_clamp_upper_bound_preserves_expand_direction(self):
        """P2：top_k_vector=120 + 扩规则 → 应升 121，而非被 clamp 上界 100 压回 100"""
        engine = DiagnosisEngine()
        p = EvolvableParams(top_k_vector=120, top_k_keyword=20)
        snaps = [self._snap(1, 0.5, 1)]  # 结果太少 → 扩候选集
        result = engine.diagnose(snaps, p)
        assert result.suggested.get("top_k_vector") == 121, \
            f"120 应升 121 而非降 100: {result.suggested.get('top_k_vector')}"

    def test_rule3_reason_branches_when_few_results(self):
        """P3-2：avg_n<3 时规则 3 只扩关键词不收窄向量——reason 不得写「收窄向量」"""
        engine = DiagnosisEngine()
        p = EvolvableParams()
        snaps = [self._snap(1, 0.7, 1)]  # 结果太少 + 单一 → 规则 1 扩、规则 3 只提关键词
        result = engine.diagnose(snaps, p)
        assert "提关键词+收窄向量" not in result.root_cause, \
            f"avg_n<3 时 reason 不应写「提关键词+收窄向量」: {result.root_cause}"
        assert "不收窄向量" in result.root_cause, \
            f"avg_n<3 时 reason 应注明不收窄向量: {result.root_cause}"
        assert result.suggested.get("top_k_keyword", 20) > 20
        assert result.suggested.get("top_k_vector", 20) > 20

    def test_degraded_expands_l1(self):
        """规则 5：降级 → 扩 top_k_l1（原调 tau_weight 死旋钮）"""
        engine = DiagnosisEngine()
        p = EvolvableParams()
        snaps = [self._snap(0, 0.0, 0, degraded=True)]
        result = engine.diagnose(snaps, p)
        assert result.suggested.get("top_k_l1", 5) == 6
        assert "tau_weight" not in result.suggested, "死旋钮不再演化"
        assert result.confidence >= 0.5

    def test_degraded_plus_slow_expansion_priority(self):
        """P1-2 回归：0 结果 + 延迟 600ms → 结果太少优先扩大——
        规则 1 的扩大不被规则 4 的延迟降覆盖（同旋钮不再互覆）"""
        engine = DiagnosisEngine()
        p = EvolvableParams()
        snaps = [self._snap(0, 0.0, 0, latency=600, degraded=True)]
        result = engine.diagnose(snaps, p)
        assert result.suggested["top_k_l1"] == 6, \
            "结果太少应扩大 top_k_l1，而非被延迟降为 4"
        assert result.suggested["top_k_vector"] > 20, "候选集应扩大而非被降"
        assert result.suggested["top_k_keyword"] > 20

    def test_rule_clash_delta_merge(self):
        """P1-1：同旋钮多规则增量合并——num=10, score=0.2, distinct=5,
        latency=600：规则 2 扩向量（+0.25*d）与规则 4 降候选（-0.2*d）
        合并为净 delta 统一应用，而非规则 4 后写覆写规则 2。

        修复前规则 4 无条件覆写 suggested["top_k_vector"] → 诊断理由写
        「扩向量候选+0.125」但实际值被降到 18（理由与结果相反）；修复后
        top_k_vector = int(20*(1+0.125-0.1)) = 20（净合并），keyword 仅被
        规则 4 降 → 18。"""
        engine = DiagnosisEngine()
        p = EvolvableParams()
        d = engine._damping
        snaps = [self._snap(10, 0.2, 5, latency=600)]
        result = engine.diagnose(snaps, p)
        assert result.suggested["top_k_vector"] == int(
            20 * (1.0 + 0.25 * d - 0.2 * d)), \
            "规则2(+0.125) 与规则4(-0.1) 的 delta 应合并应用"
        assert result.suggested["top_k_vector"] != int(20 * (1.0 - 0.2 * d)), \
            "不得是规则 4 单独覆写的结果（覆写=18，合并=20）"
        assert result.suggested["top_k_keyword"] == int(20 * (1.0 - 0.2 * d))

    def test_never_targets_dead_or_fusion_weights(self):
        """规则不再调 weight_fusion_*（fusion 消费但非失败直接旋钮）
        与 tau_weight/vector_weight（hybrid_score 零调用，死旋钮）"""
        engine = DiagnosisEngine()
        p = EvolvableParams()
        snaps = [self._snap(0, 0.0, 0, degraded=True)]
        result = engine.diagnose(snaps, p)
        for k in ("weight_fusion_vector", "weight_fusion_bm25",
                  "weight_fusion_entity", "tau_weight", "vector_weight"):
            assert k not in result.suggested


class TestEvolutionGuard:
    def test_apply_valid(self):
        guard = EvolutionGuard()
        ok, msg = guard.apply({"weight_fusion_vector": 0.45})
        assert ok
        assert guard.current().weight_fusion_vector == 0.45
        assert "#1" in msg

    def test_apply_unknown_param(self):
        guard = EvolutionGuard()
        ok, _ = guard.apply({"nonexistent": 0.5})
        assert ok is False

    def test_apply_out_of_range(self):
        guard = EvolutionGuard()
        ok, _ = guard.apply({"weight_fusion_vector": 99.0})
        assert ok is False

    def test_apply_no_change_skipped(self):
        """空转保护：建议值 == 当前值 → 不生成新版本"""
        guard = EvolutionGuard()
        ok, msg = guard.apply({"tau_weight": 0.4})  # 默认就是 0.4
        assert ok is False
        assert guard._version == 0
        assert "无变化" in msg

    def test_initial_params_honored(self):
        guard = EvolutionGuard(initial_params=EvolvableParams(tau_weight=0.7))
        assert guard.current().tau_weight == 0.7

    def test_revert_on_regression(self):
        guard = EvolutionGuard(revert_threshold=0.1)
        guard.report_quality(0.8)
        guard.check_revert()
        guard.report_quality(0.8)
        guard.check_revert()
        guard.apply({"weight_fusion_vector": 0.5})
        for _ in range(5):
            guard.report_quality(0.3)
        reverted, reason = guard.check_revert()
        assert reverted is True, f"应回滚但未回滚: {reason}"
        assert "下降" in (reason or "")

    def test_no_revert_on_improvement(self):
        guard = EvolutionGuard(revert_threshold=0.1)
        guard.report_quality(0.3)
        guard.check_revert()
        guard.apply({"weight_fusion_vector": 0.5})
        for _ in range(5):
            guard.report_quality(0.8)
        reverted, _ = guard.check_revert()
        assert reverted is False

    def test_stagnation_explore_goes_through_apply(self):
        """P2：停滞探索走 apply()——版本号递增 + history 记录（非裸 setattr）

        修复前 _explore_if_stagnant 直接 setattr(self.params, ...)，
        不产生版本/history，持久化丢失探索改动，且探索改动无回滚保护。"""
        import random as _random

        _random.seed(42)
        guard = EvolutionGuard(explore_after=1)
        guard.report_quality(0.8)  # v0 基线
        guard.check_revert()
        guard.apply({"weight_fusion_vector": 0.45})  # v1, 建立 pending
        guard.report_quality(0.8)  # v1.avg_quality_after 生效（探索前置条件）
        v1 = guard._version
        assert guard._explore_if_stagnant() is True
        assert guard._version == v1 + 1, "探索必须经 apply 递增版本号"
        assert len(guard.history) == 3, "v0 + v1 + 探索版本都应入 history"


class TestSelfEvolvingRetrieval:
    def test_basic_retrieve(self):
        se = make_se()
        result = se.retrieve("hello")
        assert isinstance(result, list)
        assert len(result) > 0
        assert se.state()["total_calls"] == 1

    def test_retrieve_passthrough_include_archived(self):
        """v5.32.0 回归：include_archived 必须透传给底层 QueryRouter"""
        received = {}

        class SpyRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                received["include_archived"] = include_archived
                return [{"content": f"result_{q}", "score": 0.7, "node_id": "r1"}]

        se = make_se(SpyRouter())
        se.retrieve("hello", include_archived=True)
        assert received.get("include_archived") is True
        se.retrieve("world")
        assert received.get("include_archived") is False

    def test_retrieve_exception_returns_list(self):
        """P1-1 回归：底层抛异常 → []（list），且不计入调用统计"""
        class RaisingRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                raise RuntimeError("simulated retrieval failure")

        se = make_se(RaisingRouter())
        result = se.retrieve("hello")
        assert isinstance(result, list) and result == []
        assert se.state()["total_calls"] == 0

    def test_initial_params_from_config(self):
        """根因#3：guard 初始值来自 config，而非硬编码默认"""
        se = make_se(MockRouter(config=_cfg(tau_weight=0.7, top_k_vector=44)))
        assert se.guard.current().tau_weight == 0.7
        assert se.guard.current().top_k_vector == 44

    def test_config_not_overwritten_every_retrieve(self):
        """根因#3：_sync_params 仅演化 apply 后调用，普通 retrieve 不再覆盖配置"""
        se = make_se(MockRouter(config=_cfg(tau_weight=0.7)))
        for i in range(3):
            se.retrieve(f"q{i}")
        assert se._qr.config.tau_weight == 0.7
        assert se.guard.current().tau_weight == 0.7
        assert se.guard._version == 0  # 健康（无退化）不演化

    def test_params_synced_after_apply(self):
        """演化应用后参数同步到 QueryRouter"""
        se = make_se()
        se.guard.apply({"vector_weight": 0.8})
        se._sync_params()
        assert se._qr.config.vector_weight == 0.8

    def test_ac_evolution_changes_retrieval_behavior(self):
        """AC（P1 修复，替代假绿内存断言）：真实 QueryRouter 端到端——
        degraded 硬失败 → 演化（top_k_l1 上调）→ 真实检索结果条数随演化增长。

        旧版只断言 guard.current()/cfg 内存值，MockRouter.retrieve 不消费
        config——配置动了行为没动。本测试断言检索行为/结果真实变化。
        """
        qr, index = _real_router(n_docs=10, top_k_l1=2)
        se = SelfEvolvingRetrieval(qr, persist_path=_tmp_persist_path(),
                                   probe_every=10**9)
        # 基线：k=2 → 2 条
        assert len(se.retrieve("q0")) == 2
        # FAISS 降级 → 0 条 → degraded 硬失败 → 即时演化并同步到真实 config
        index.down = True
        assert se.retrieve("q1") == []
        assert qr.config.top_k_l1 > 2, "演化参数必须同步到生产 QueryRouter config"
        # 恢复 → 真实检索行为随演化变化（条数 = 新 top_k_l1）
        index.down = False
        n = len(se.retrieve("q2"))
        assert n == qr.config.top_k_l1 and n > 2, f"检索结果应随演化变化: {n}"

    def test_p0_top_k_l1_drives_real_retrieval_behavior(self):
        """P0 修复：top_k_l1 是生产默认路径的真实旋钮——
        config 变化 → 真实 FAISS 检索结果条数变化（行为断言，非内存值）"""
        qr, _ = _real_router(n_docs=10, top_k_l1=2)
        assert len(qr.retrieve("q")) == 2
        qr.config.top_k_l1 = 6
        assert len(qr.retrieve("q")) == 6, "top_k_l1 变化必须真实改变检索条数"

    def test_ac_evolved_then_normal_retrieve(self):
        """AC 端到端：degraded 后检索恢复返回正常结果"""
        class FlakyRouter(MockRouter):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def retrieve(self, q, include_archived=False, session_ts=None):
                self.calls += 1
                if self.calls <= 3:
                    return []
                return [{"content": f"ok_{q}", "score": 0.8, "node_id": "r1"}]

        se = make_se(FlakyRouter())
        for i in range(3):
            se.retrieve(f"q{i}")
        r = se.retrieve("final")
        assert isinstance(r, list) and len(r) > 0
        assert se.state()["total_calls"] == 4

    def test_periodic_forced_evaluation(self):
        """周期强制触发：probe_every 次检索后即使非降级也评估"""
        class LowQualityRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                return [{"content": f"r_{q}", "score": 0.2, "node_id": "r1"}]

        se = make_se(LowQualityRouter(), probe_every=2)
        v0 = se.guard._version
        se.retrieve("a")  # calls=1, 1%2≠0 → 不触发
        assert se.guard._version == v0
        se.retrieve("b")  # calls=2, 2%2==0 → 周期评估 → 低质量诊断演化
        assert se.guard._version > v0, "周期评估应触发演化"

    def test_periodic_healthy_no_change(self):
        """周期评估但检索健康 → 无演化"""
        class GoodRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                return [{"content": f"result_{i}_{q}", "score": 0.9 - i * 0.05,
                         "node_id": f"n{i}"} for i in range(8)]

        se = make_se(GoodRouter(), probe_every=2)
        se.retrieve("a")
        se.retrieve("b")
        assert se.guard._version == 0

    def test_time_interval_forces_evaluation(self):
        """P2：低流量兜底——超过 probe_interval_s 未评估 → 强制评估演化

        修复前 probe_every=100 对低流量过长（100 次检索才周期评估），
        时间兜底保证慢流量下也定期触发。"""
        class LowQualityRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                return [{"content": f"r_{q}", "score": 0.2, "node_id": "r1"}]

        se = make_se(LowQualityRouter(), probe_every=10**9, probe_interval_s=0.02)
        v0 = se.guard._version
        se.retrieve("a")
        assert se.guard._version == v0, "未到时间间隔不强制评估"
        time.sleep(0.08)
        se.retrieve("b")
        assert se.guard._version > v0, "超时间隔应强制评估并演化"

    def test_periodic_diagnosis_excludes_probe_source(self):
        """P2：周期/时间兜底路径诊断只合并 retrieve 源——probe 合成快照
        不污染生产周期诊断（修复前 recent_poor(10) 不带 source 混入 probe）"""
        class LowQualityRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                return [{"content": f"r_{q}", "score": 0.2, "node_id": "r1"}]

        se = make_se(LowQualityRouter(), probe_every=10**9, probe_interval_s=0.02)
        se.retrieve("a")  # 生产低质量快照（quality≈0.16 → recent_poor 命中）
        # 注入 probe 合成低质量快照（不经 report_probe，避免立即触发演化）
        se.logger.log(RetrievalSnapshot(
            timestamp=time.time(), query="<health-probe>", params_before={},
            num_results=0, top_scores=[], avg_score=0.1, top_distinct=0,
            latency_ms=0.0, degraded=True, source="probe"))
        # 兜住 diagnose 输入，验证周期路径只喂 retrieve 源快照
        captured: dict = {}
        orig_diagnose = se.diagnoser.diagnose

        def spy(snaps, params):
            captured["snaps"] = list(snaps)
            return orig_diagnose(snaps, params)

        se.diagnoser.diagnose = spy
        time.sleep(0.05)
        se.retrieve("b")  # 超时间隔 → 周期路径 _evolve
        assert captured.get("snaps"), "周期路径应触发 diagnose"
        assert captured["snaps"], "周期诊断应取到 retrieve 源低质量快照"
        assert all(s.source == "retrieve" for s in captured["snaps"]), \
            "周期诊断不得混入 probe 合成快照"

    def test_latency_hard_fail_triggers_evolution(self):
        """即时硬失败：延迟超阈值 → 触发演化"""
        class SlowRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                time.sleep(0.55)  # > latency_threshold_ms=500
                return [{"content": f"result_{q}", "score": 0.7, "node_id": "r1"}]

        se = make_se(SlowRouter(), latency_threshold_ms=500.0)
        v0 = se.guard._version
        se.retrieve("a")
        assert se.guard._version > v0, "延迟超时硬失败应触发演化"

    def test_latency_hard_fail_high_quality_triggers_evolution(self):
        """P1 修复：延迟硬失败不被质量过滤吞掉——quality≥0.4 的慢检索也必须诊断

        快照 quality≈0.85（8 条高分多样结果 + 550ms）会被 recent_poor 过滤；
        修复前 _evolve 取不到快照 → 延迟规则永不生效。"""
        class SlowHighQualityRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                time.sleep(0.55)
                return [{"content": f"r{i}_{q}", "score": 0.9 - i * 0.02,
                         "node_id": f"n{i}"} for i in range(8)]

        se = make_se(SlowHighQualityRouter(), latency_threshold_ms=500.0)
        v0 = se.guard._version
        se.retrieve("a")
        assert se.guard._version > v0, "高质量慢检索也应触发延迟规则演化"
        assert se.guard.current().top_k_vector < 20, "延迟规则应降候选集"
        assert se.guard.current().top_k_l1 == 5, "延迟规则不降 top_k_l1（P1-2）"

    def test_evolve_aggregate_preserves_trigger_signal(self):
        """P1-2：_evolve 聚合路径——零结果慢触发 + 5 条 num=4 慢历史，
        触发快照加权后 avg_n 不被历史稀释到 ≥3：规则 1 扩大候选集生效，
        不被规则 4（延迟降候选）压制。

        修复前触发快照只占 1/6：avg_n=(0+5*4)/6=3.3 → 规则 1 跳过、规则 4
        把 top_k_vector 降到 20（结果太少反而降候选，与「结果太少→扩大」相反）。"""
        se = make_se(MockRouter())
        for i in range(5):
            se.logger.log(RetrievalSnapshot(
                timestamp=time.time(), query=f"hist_{i}", params_before={},
                num_results=4, top_scores=[0.25], avg_score=0.25,
                top_distinct=3, latency_ms=600, degraded=False))
        trigger = RetrievalSnapshot(
            timestamp=time.time(), query="trigger", params_before={},
            num_results=0, top_scores=[], avg_score=0.0,
            top_distinct=0, latency_ms=600, degraded=True)
        se._evolve(trigger_snapshot=trigger)
        assert se.guard.current().top_k_l1 > 5, "零结果触发应扩大 L1 候选集"
        assert se.guard.current().top_k_vector > 20, \
            "零结果触发应扩向量候选——触发信号不被 num=4 历史稀释"
        assert se.guard.current().top_k_keyword > 20

    def test_retrieve_hard_fail_mixed_history_full_path(self):
        """P2：degraded+slow 回归走生产入口 retrieve 全链路——预置混合历史
        后，硬失败触发（0 结果 + 延迟超阈值）→ _evolve 聚合 → 最终建议
        仍扩大候选集（断言 guard 实际生效值 + cfg 同步，非单快照 diagnose）。"""
        class EmptySlowRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                time.sleep(0.55)
                return []

        se = make_se(EmptySlowRouter(), latency_threshold_ms=500.0)
        for i in range(5):
            se.logger.log(RetrievalSnapshot(
                timestamp=time.time(), query=f"hist_{i}", params_before={},
                num_results=4, top_scores=[0.25], avg_score=0.25,
                top_distinct=3, latency_ms=600, degraded=False))
        se.retrieve("q")
        assert se.guard._version > 0, "硬失败应触发演化"
        assert se.guard.current().top_k_l1 > 5
        assert se.guard.current().top_k_vector > 20, \
            "生产入口聚合后仍应扩大向量候选（触发信号不被历史稀释）"
        assert se.guard.current().top_k_keyword > 20
        assert se._qr.config.top_k_l1 > 5, "演化应同步到生产 config"


class TestPersistence:
    def test_persistence_roundtrip(self, tmp_path):
        """持久化 roundtrip：save → 新实例 restore → 参数/版本/history/cfg 一致"""
        path = str(tmp_path / "retrieval_evolved.json")
        se = make_se(persist_path=path)
        ok, _ = se.guard.apply({"tau_weight": 0.65, "top_k_vector": 33})
        assert ok
        assert se.save_state(path) is True

        loaded = se.load_state(path)
        assert loaded["version"] == 1
        assert loaded["params"]["tau_weight"] == pytest.approx(0.65)
        assert loaded["params"]["top_k_vector"] == 33
        assert "history" in loaded and "applied_at" in loaded

        se2 = make_se(persist_path=path)
        assert se2.restore_state(path) is True
        assert se2.guard.current().snapshot() == se.guard.current().snapshot()
        assert se2.guard._version == 1
        assert len(se2.guard.history) == 1
        assert se2._qr.config.tau_weight == pytest.approx(0.65)  # 恢复后同步 cfg

    def test_restore_missing_or_corrupt_no_crash(self, tmp_path):
        """缺失/损坏持久化 → restore 返回 False，保持默认值"""
        se = make_se(persist_path=str(tmp_path / "none.json"))
        assert se.restore_state(str(tmp_path / "none.json")) is False
        assert se.guard.current().tau_weight == pytest.approx(0.4)

        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{not-json", encoding="utf-8")
        assert se.restore_state(str(corrupt)) is False

    def test_restore_invalid_params_rejected(self, tmp_path):
        """P2：restore_state 对读回参数执行 validate()——非法参数拒绝恢复

        修复前 restore 直接 EvolvableParams(**params)，越界值（tau_weight=5.0）
        原样写回生产 cfg；修复后 validate() 拦截，保持默认参数。"""
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"version": 3, "params": {"tau_weight": 5.0}}),
                       encoding="utf-8")
        se = make_se(persist_path=str(bad))
        assert se.restore_state(str(bad)) is False
        assert se.guard.current().tau_weight == pytest.approx(0.4)
        assert se.guard._version == 0
        assert se._qr.config.tau_weight == pytest.approx(0.4), "非法参数不得同步到 cfg"


class TestProbe:
    def _nodes(self, n=5):
        return [
            {"id": f"n{i}", "content": f"梦境核心记忆主题 {i} 详细内容片段 " + "y" * 50,
             "tau_initial": 0.9 - i * 0.1}
            for i in range(n)
        ]

    def test_probe_recall_low_triggers_evolution(self):
        """梦境探针：低召回 → report_probe → 演化触发（不污染 total_calls）"""
        class EmptyRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                return []

        se = make_se(EmptyRouter())
        pipe = DreamPipeline(retrieval_guard=se)
        recall = asyncio.run(pipe.retrieval_health_probe(self._nodes()))
        assert recall == 0.0
        assert se.state()["total_calls"] == 0, "探针直调内层，不能自增 total_calls"
        assert se.guard.current().top_k_l1 > 5, "低召回应触发扩大 L1 候选集演化"

    def test_probe_recall_high_no_trigger(self):
        """梦境探针：高召回 → 不触发演化"""
        nodes = self._nodes()

        class RecallRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                for n in nodes:
                    if (n["content"] or "").startswith(q):
                        return [{"node_id": n["id"], "content": n["content"],
                                 "score": 0.9}]
                return []

        se = make_se(RecallRouter())
        pipe = DreamPipeline(retrieval_guard=se)
        recall = asyncio.run(pipe.retrieval_health_probe(nodes))
        assert recall == 1.0
        assert se.guard.current().top_k_l1 == 5, "高召回不触发演化"
        assert se.state()["total_calls"] == 0

    def test_probe_recall_040_triggers_evolution(self):
        """P1 修复：report_probe 低召回但合成快照 quality()≥0.4 时必须触发演化

        recall=0.4 → num_results=12/avg_score=0.4/distinct=5 → quality()≈0.76，
        recent_poor（<0.4 门槛）会过滤掉它；修复前探针信号被质量过滤吞掉。
        修复后触发快照显式传给 diagnose → 降级规则（top_k_l1 上调）生效。"""
        se = make_se(MockRouter())
        se.report_probe(recall=0.4, sample_size=30)
        assert se.guard._version > 0, "recall=0.4 探针应触发演化"
        assert se.guard.current().top_k_l1 > 5, "降级规则应扩大 L1 候选集"

    def test_probe_snapshot_source_separate(self):
        """P2：probe 合成快照带 source=probe，与生产快照分离（诊断互不污染）"""
        se = make_se(MockRouter())
        se.report_probe(recall=0.2, sample_size=30)  # probe 快照
        se.retrieve("x")                              # 生产快照
        snaps = list(se.logger.snapshots)
        assert [s.source for s in snaps] == ["probe", "retrieve"]
        assert snaps[0].query == "<health-probe>"
        # probe 触发诊断只合并同源快照：生产诊断不受合成快照污染
        se.report_probe(recall=0.1, sample_size=30)
        assert all(s.source == "probe" for s in se.logger.snapshots
                   if s.query == "<health-probe>")

    def test_probe_no_guard_or_empty_nodes_safe(self):
        """无 guard / 空节点 → 返回 0.0 不抛错"""
        pipe = DreamPipeline()
        assert asyncio.run(pipe.retrieval_health_probe([])) == 0.0
        assert asyncio.run(pipe.retrieval_health_probe(self._nodes())) == 0.0


class TestSelfEvolvingRetrievalConcurrency:
    """【H1】多线程并发 retrieve 回归：无锁检索 + 短锁段更新共享状态"""

    def test_concurrent_retrieve_no_race(self):
        class SlowRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                time.sleep(0.002)
                return [{"content": f"result_{q}", "score": 0.7, "node_id": "r1"}]

        se = make_se(SlowRouter())
        errors: list[Exception] = []
        N_THREADS, N_CALLS = 10, 20

        def worker():
            try:
                for i in range(N_CALLS):
                    r = se.retrieve(f"w{i}")
                    assert isinstance(r, list)
                    assert r, "retrieve 应返回非空结果"
            except Exception as e:  # pragma: no cover - 失败时记录
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"并发 retrieve 抛异常: {errors}"
        assert se.state()["total_calls"] == N_THREADS * N_CALLS

    def test_concurrent_retrieve_low_quality_evolves_safely(self):
        """低质量并发检索触发演化路径（guard.apply/回滚/探索）也不抛异常"""
        class BadRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                time.sleep(0.001)
                return []

        se = make_se(BadRouter(), probe_every=2)
        errors: list[Exception] = []

        def worker():
            try:
                for i in range(10):
                    se.retrieve(f"q{i}")
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"并发演化路径抛异常: {errors}"
        assert se.state()["total_calls"] == 100

    def test_concurrent_retrieve_not_serialized(self):
        """【H1-a】锁粒度收窄：并发 retrieve 不再被方法级大锁串行化"""
        import time as _time

        class SlowRouter(MockRouter):
            def retrieve(self, q, include_archived=False, session_ts=None):
                _time.sleep(0.1)  # 模拟 100ms 检索
                return [{"content": f"result_{q}", "score": 0.7, "node_id": "r1"}]

        se = make_se(SlowRouter(), probe_every=10**9)
        N_THREADS = 10

        def worker():
            se.retrieve("q")

        start = _time.monotonic()
        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = _time.monotonic() - start

        assert elapsed < 0.6, f"并发检索被串行化: {elapsed:.2f}s (期望 < 0.6s)"
        assert se.state()["total_calls"] == N_THREADS
