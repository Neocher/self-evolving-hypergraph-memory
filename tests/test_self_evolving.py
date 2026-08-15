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

    def retrieve(self, q, include_archived=False):
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
            def retrieve(self, q, include_archived=False):
                return []  # 总是返回空

        se = SelfEvolvingRetrieval(BadRouter(), evolve_interval=3, quality_threshold=0.5)
        for i in range(10):
            se.retrieve(f"q{i}")
        s = se.state()
        # 应该已经触发了演化
        print(f"演化状态: {s}")
        assert s["total_calls"] == 10

    def test_retrieve_exception_returns_list(self):
        """P1-1 回归测试：底层检索抛异常时 retrieve() 必须返回 []（list），
        不能返回 dict —— 下游 search.py/gateway_api.py 按 list 遍历，
        dict 会触发 AttributeError → 500（曾导致线上故障）。"""
        class RaisingRouter(MockRouter):
            def retrieve(self, q, include_archived=False):
                raise RuntimeError("simulated retrieval failure")

        se = SelfEvolvingRetrieval(RaisingRouter())
        result = se.retrieve("hello")
        assert isinstance(result, list)
        assert result == []
        # 失败不应计入成功调用统计
        assert se.state()["total_calls"] == 0

    def test_params_synced(self):
        """验证演化后参数同步到 QueryRouter"""
        se = SelfEvolvingRetrieval(MockRouter())
        se.guard.apply({"weight_fusion_vector": 0.6})
        se._sync_params()
        assert se._qr.config.weight_fusion_vector == 0.6


class TestSelfEvolvingRetrievalConcurrency:
    """【H1】多线程并发 retrieve 回归测试

    to_thread 改造后多请求在独立线程并发执行：若 retrieve() 不加锁，
    _sync_params 的 setattr / _total_calls += 1 / deque、list 迭代中并发修改
    会引发 RuntimeError（deque/list mutated during iteration）或计数丢失。
    """

    def test_concurrent_retrieve_no_race(self):
        import threading

        class SlowRouter(MockRouter):
            def retrieve(self, q, include_archived=False):
                time.sleep(0.002)  # 拉大并发交错窗口
                return [{"content": f"result_{q}", "score": 0.7}]

        se = SelfEvolvingRetrieval(SlowRouter())
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
        # 计数正确：无丢失更新
        assert se.state()["total_calls"] == N_THREADS * N_CALLS

    def test_concurrent_retrieve_low_quality_evolves_safely(self):
        """低质量并发检索触发演化路径（guard.apply/回滚/探索）也不抛异常"""
        import threading

        class BadRouter(MockRouter):
            def retrieve(self, q, include_archived=False):
                time.sleep(0.001)
                return []  # 空结果 → 低质量 → 触发诊断/演化

        se = SelfEvolvingRetrieval(BadRouter(), evolve_interval=2, quality_threshold=0.5)
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
        """【H1-a】锁粒度收窄：并发 retrieve 不再被方法级大锁串行化。

        修复前方法级大锁包裹整个 _qr.retrieve（含 FAISS+GraphLite+encoder，
        100-500ms）：10 并发 × 100ms 检索 → 串行耗时 ~1s，且超时 zombie 线程
        持锁会楔死后续请求。修复后仅锁共享状态变更段，检索读操作可并行。
        """
        import threading
        import time as _time

        class SlowRouter(MockRouter):
            def retrieve(self, q, include_archived=False):
                _time.sleep(0.1)  # 模拟 100ms 检索
                return [{"content": f"result_{q}", "score": 0.7}]

        se = SelfEvolvingRetrieval(SlowRouter())
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

        # 串行化时 ~1.0s；并行应远低于此（0.6s 阈值留足 CI 余量）
        assert elapsed < 0.6, f"并发检索被串行化: {elapsed:.2f}s (期望 < 0.6s)"
        # 锁内更新计数正确（无丢失更新）
        assert se.state()["total_calls"] == N_THREADS
