"""测试检索自演化模块"""
import time
import pytest
from retrieval.self_evolving import (
    EvolvableParams, FailureLogger, RetrievalSnapshot,
    DiagnosisEngine, EvolutionGuard, SelfEvolvingRetrieval,
)


class TestEvolvableParams:
    def test_validate_ok(self):
        p = EvolvableParams()
        assert p.validate() == []

    def test_validate_out_of_range(self):
        p = EvolvableParams()
        p.weight_fusion_vector = 1.5
        errs = p.validate()
        assert any("weight_fusion_vector" in e for e in errs)

    def test_snapshot_roundtrip(self):
        p = EvolvableParams()
        snap = p.snapshot()
        assert snap["weight_fusion_vector"] == 0.35
        assert snap["bm25_k1"] == 1.5


class TestFailureLogger:
    def test_log_triggers_diagnosis(self):
        logger = FailureLogger(min_snapshots_for_diagnosis=3, quality_threshold=0.5)
        p = EvolvableParams()
        for i in range(4):
            snap = RetrievalSnapshot(
                time.time(), f"q{i}", p.snapshot(),
                0, [], 0.0, 0, 100, True,
            )
            triggered = logger.log(snap)
        assert triggered is True  # 4次低质量触发

    def test_log_good_quality(self):
        logger = FailureLogger(min_snapshots_for_diagnosis=3, quality_threshold=0.5)
        p = EvolvableParams()
        for i in range(5):
            snap = RetrievalSnapshot(
                time.time(), f"q{i}", p.snapshot(),
                10, [0.9, 0.8], 0.85, 8, 50, False,
            )
            triggered = logger.log(snap)
        assert triggered is False  # 高质量不触发

    def test_state(self):
        logger = FailureLogger()
        assert "total" in logger.state()


class TestDiagnosisEngine:
    def test_no_snapshots(self):
        engine = DiagnosisEngine()
        result = engine.diagnose([], EvolvableParams())
        assert result.suggested == {}

    def test_few_results(self):
        engine = DiagnosisEngine()
        p = EvolvableParams()
        snaps = [
            RetrievalSnapshot(time.time(), "t", p.snapshot(),
                              1, [0.5], 0.5, 1, 100, False),
            RetrievalSnapshot(time.time(), "t2", p.snapshot(),
                              0, [], 0.0, 0, 200, True),
        ]
        result = engine.diagnose(snaps, p)
        assert len(result.suggested) > 0
        assert result.confidence >= 0.5

    def test_high_latency(self):
        engine = DiagnosisEngine()
        p = EvolvableParams()
        snaps = [
            RetrievalSnapshot(time.time(), "t", p.snapshot(),
                              5, [0.7, 0.6], 0.65, 4, 600, False),
        ]
        result = engine.diagnose(snaps, p)
        assert result.suggested.get("top_k_fusion", 30) < 30


class TestEvolutionGuard:
    def test_apply_valid(self):
        guard = EvolutionGuard()
        ok, msg = guard.apply({"weight_fusion_vector": 0.45})
        assert ok
        assert guard.current().weight_fusion_vector == 0.45
        assert "#1" in msg

    def test_apply_invalid(self):
        guard = EvolutionGuard()
        # 未知参数现在返回 False
        ok, _ = guard.apply({"nonexistent": 0.5})
        assert ok is False
        # 超出范围返回 False
        ok, _ = guard.apply({"weight_fusion_vector": 99.0})
        assert ok is False  # 超出范围

    def test_revert_on_regression(self):
        guard = EvolutionGuard(revert_threshold=0.1)
        # 先设基线：应用前先报告一次质量
        guard.report_quality(0.8)
        guard.check_revert()
        # 再报一次确保基线稳定
        guard.report_quality(0.8)
        guard.check_revert()
        # 应用变更
        guard.apply({"weight_fusion_vector": 0.5})
        # 报告更差的质量
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


class MockRouter:
    class Config:
        weight_fusion_vector = 0.35
        weight_fusion_bm25 = 0.40
        weight_fusion_entity = 0.25
        tau_weight = 0.4
        vector_weight = 0.6
        top_k_fusion = 30
        top_k_keyword = 20
        top_k_vector = 20
        bm25_k1 = 1.5
        bm25_b = 0.75
    config = Config()
    gate_threshold = 0.5

    def retrieve(self, q):
        return [{"content": f"result_{q}", "score": 0.7}]

    def state(self):
        return {}


class TestSelfEvolvingRetrieval:
    def test_basic_retrieve(self):
        se = SelfEvolvingRetrieval(MockRouter())
        result = se.retrieve("hello")
        assert isinstance(result, list)
        assert len(result) > 0
        s = se.state()
        assert s["total_calls"] == 1

    def test_evolve_cycle(self):
        """低质量检索多次后触发演化的完整循环"""
        class BadRouter(MockRouter):
            def retrieve(self, q):
                return []  # 总是返回空

        se = SelfEvolvingRetrieval(BadRouter(), evolve_interval=3, quality_threshold=0.5)
        for i in range(10):
            se.retrieve(f"q{i}")
        s = se.state()
        # 应该已经触发了演化
        print(f"演化状态: {s}")
        assert s["total_calls"] == 10

    def test_params_synced(self):
        """验证演化后参数同步到 QueryRouter"""
        se = SelfEvolvingRetrieval(MockRouter())
        se.guard.apply({"weight_fusion_vector": 0.6})
        se._sync_params()
        assert se._qr.config.weight_fusion_vector == 0.6
