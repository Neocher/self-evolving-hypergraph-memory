# -*- coding: utf-8 -*-
"""reranker 并发闸门验收测试（进程级 Semaphore + 有界等待降级）
=================================================================
覆盖验收点：
  AC1  rerank_max_concurrency 默认 1、rerank_gate_wait_ms 默认 300，
       defaults.yaml / settings.py / QueryRouterConfig / _build_router_config 四处一致
  AC2  进程级 Semaphore 有界等待：拿不到闸门 → WARNING + 原列表返回（优雅降级，
       不调用 predict、不抛异常、不无限等待）
  AC3  finally 释放：持有者释放后后续请求可正常重排（无泄漏 / 无死锁）
  AC4  rerank_max_concurrency=0 → 完全禁用闸门（predict 照常执行，与改动前一致）
  AC5  并发 4/8 条全新查询全部 <6s 且各返回 20 条；闸门限制同时 predict 数为 1
  AC6  单条零回归：默认配置（并发度 1）下单条重排与禁用闸门逐位一致

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_rerank_concurrency_gate.py -q
"""
from __future__ import annotations

import copy
import hashlib
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import retrieval.query_router as qr
from config.settings import load_settings
from retrieval.query_router import QueryRouter, QueryRouterConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _reset_gate():
    """每个用例前后重置进程级闸门单例，防跨用例污染（容量变化 / 手动占用泄漏）。"""
    qr._RERANK_GATE = None
    qr._RERANK_GATE_VALUE = -1
    yield
    qr._RERANK_GATE = None
    qr._RERANK_GATE_VALUE = -1


def _make_router(**cfg_overrides) -> QueryRouter:
    """构造零依赖 QueryRouter 并覆盖 config 字段（绕过 __post_init__ 的默认值）。"""
    router = QueryRouter(None, None, None)
    for key, value in cfg_overrides.items():
        setattr(router.config, key, value)
    return router


def _candidates(n: int = 20) -> list[dict]:
    return [
        {"node_id": f"n{i}", "content": f"content {i}", "score": 1.0 - i * 0.01}
        for i in range(n)
    ]


def _hash_scores(pairs):
    """逐对独立确定性打分：score = f(query, content)，与批次内其它对无关。"""
    out = []
    for q, c in pairs:
        digest = hashlib.blake2b(f"{q}\u0000{c}".encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(digest, "little") % (1 << 40)
        out.append((val / (1 << 40)) * 20.0 - 10.0)
    return out


class _SlowReranker:
    """带延迟的确定性 reranker，记录并发度供闸门限流与并发竞态断言。"""

    def __init__(self, delay: float = 0.03):
        self.delay = delay
        self.calls = 0
        self._active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def predict(self, pairs, batch_size=None):
        with self._lock:
            self.calls += 1
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(self.delay)
            return _hash_scores(pairs)
        finally:
            with self._lock:
                self._active -= 1


# ── AC1：默认值四处一致 ────────────────────────────────────────────────────
class TestGateDefaults:
    def test_settings_defaults(self):
        cfg = load_settings().retrieval
        assert cfg.rerank_max_concurrency == 1
        assert cfg.rerank_gate_wait_ms == 300

    def test_query_router_defaults(self):
        cfg = QueryRouterConfig()
        assert cfg.rerank_max_concurrency == 1
        assert cfg.rerank_gate_wait_ms == 300

    def test_defaults_yaml(self):
        raw = yaml.safe_load((_REPO_ROOT / "config" / "defaults.yaml").read_text(encoding="utf-8"))
        assert raw["retrieval"]["rerank_max_concurrency"] == 1
        assert raw["retrieval"]["rerank_gate_wait_ms"] == 300

    def test_build_router_config_defaults(self):
        from api.app import _build_router_config

        class _NoGate:
            tau_weight = 0.4
            vector_weight = 0.6
            top_k_l1 = 5
            top_k_vector = 20
            top_k_keyword = 20

        qcfg = _build_router_config(_NoGate())
        assert qcfg.rerank_max_concurrency == 1
        assert qcfg.rerank_gate_wait_ms == 300

    def test_build_router_config_passthrough(self):
        from api.app import _build_router_config

        class _GateCfg:
            tau_weight = 0.4
            vector_weight = 0.6
            top_k_l1 = 5
            top_k_vector = 20
            top_k_keyword = 20
            rerank_max_concurrency = 3
            rerank_gate_wait_ms = 500

        qcfg = _build_router_config(_GateCfg())
        assert qcfg.rerank_max_concurrency == 3
        assert qcfg.rerank_gate_wait_ms == 500


# ── AC2：有界等待降级（拿不到闸门 → WARNING + 原列表返回）───────────────────
class TestBoundedWaitDegradation:
    def test_busy_gate_skips_rerank_with_warning(self):
        router = _make_router(rerank_max_concurrency=1, rerank_gate_wait_ms=150)
        fake = _SlowReranker(delay=0.05)
        router._reranker = fake
        gate = router._get_rerank_gate()
        assert gate is not None
        assert gate.acquire(timeout=0), "预占闸门应成功"
        try:
            results = _candidates(20)
            with patch.object(qr, "logger") as mock_logger:
                start = time.monotonic()
                out = router._rerank_results(copy.deepcopy(results), "q", enabled=True)
                elapsed = time.monotonic() - start
            assert fake.calls == 0, "拿不到闸门不应调用 predict"
            assert [r["node_id"] for r in out] == [r["node_id"] for r in results], \
                "应原序返回完整融合结果（20 条）"
            assert elapsed >= 0.10, f"应有界等待约 150ms，实际 {elapsed:.3f}s（非无限等待也非立即返回）"
            assert mock_logger.warning.called, "拿不到闸门应记 WARNING"
            msg = " ".join(str(a) for a in mock_logger.warning.call_args.args)
            assert "skipping rerank" in msg
        finally:
            gate.release()


# ── AC3：finally 释放（无泄漏 / 无死锁）────────────────────────────────────
class TestGateRelease:
    def test_releases_after_rerank(self):
        router = _make_router(rerank_max_concurrency=1, rerank_gate_wait_ms=500,
                              rerank_length_bucketing=False)
        fake = _SlowReranker(delay=0.01)
        router._reranker = fake
        gate = router._get_rerank_gate()
        # 连续两次调用都应成功：第一次 finally 释放后，第二次才能拿到闸门
        out1 = router._rerank_results(copy.deepcopy(_candidates(20)), "q1", enabled=True)
        out2 = router._rerank_results(copy.deepcopy(_candidates(20)), "q2", enabled=True)
        assert len(out1) == 20 and len(out2) == 20
        assert fake.calls == 2, "两次都应完成 predict（闸门被正确释放）"
        # 闸门当前应空闲（可无阻塞获取）
        assert gate.acquire(timeout=0), "finally 释放后闸门应空闲"
        gate.release()

    def test_releases_after_degraded_skip(self):
        router = _make_router(rerank_max_concurrency=1, rerank_gate_wait_ms=100,
                              rerank_length_bucketing=False)
        fake = _SlowReranker(delay=0.01)
        router._reranker = fake
        gate = router._get_rerank_gate()
        assert gate.acquire(timeout=0)
        try:
            router._rerank_results(copy.deepcopy(_candidates(20)), "q", enabled=True)
        finally:
            gate.release()
        # 释放后，后续请求恢复正常重排
        out = router._rerank_results(copy.deepcopy(_candidates(20)), "q2", enabled=True)
        assert len(out) == 20
        assert fake.calls == 1, "持有者释放后应恢复重排（降级那一次未调用 predict）"


# ── AC4：并发度 0 → 完全禁用闸门 ───────────────────────────────────────────
class TestGateDisabled:
    def test_zero_concurrency_disables_gate(self):
        router = _make_router(rerank_max_concurrency=0, rerank_gate_wait_ms=2000,
                              rerank_length_bucketing=False)
        assert router._get_rerank_gate() is None
        fake = _SlowReranker(delay=0.01)
        router._reranker = fake
        out = router._rerank_results(copy.deepcopy(_candidates(20)), "q", enabled=True)
        assert len(out) == 20
        assert fake.calls == 1, "禁用闸门时 predict 应照常执行"


# ── AC5：并发 4/8 条全新查询 <6s 且各返回 20 条 ────────────────────────────
class TestConcurrentQueries:
    def test_concurrent_4_and_8_queries(self):
        for n in (4, 8):
            fake = _SlowReranker(delay=0.03)
            routers = [_make_router(rerank_max_concurrency=1, rerank_gate_wait_ms=2000,
                                    rerank_length_bucketing=False)
                       for _ in range(n)]
            for r in routers:
                r._reranker = fake
            template = _candidates(20)
            queries = [f"query {i}" for i in range(n)]  # 全新查询（各不相同的 query）
            outputs: list = [None] * n
            errors: list = []

            def worker(idx: int) -> None:
                try:
                    outputs[idx] = routers[idx]._rerank_results(
                        copy.deepcopy(template), queries[idx], enabled=True
                    )
                except Exception as exc:  # pragma: no cover - 失败即测试失败
                    errors.append(exc)

            start = time.monotonic()
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.monotonic() - start

            assert not errors, f"并发 {n} 条出现异常: {errors}"
            assert all(len(o) == 20 for o in outputs), f"并发 {n} 条应各返回 20 条"
            assert elapsed < 6.0, f"并发 {n} 条应 <6s，实际 {elapsed:.2f}s"
            assert fake.max_active == 1, f"闸门应限制同时 predict 为 1，实际 {fake.max_active}"
            assert fake.calls == n, f"每条查询都应完成 predict（不降级），实际 {fake.calls}/{n}"


# ── AC6：单条零回归（默认并发度 1 vs 禁用闸门逐位一致）─────────────────────
class TestSingleQueryZeroRegression:
    def test_default_gate_equals_disabled(self):
        results = _candidates(20)

        gated = _make_router(rerank_max_concurrency=1, rerank_gate_wait_ms=2000,
                             rerank_length_bucketing=False)
        fake1 = _SlowReranker(delay=0.0)
        gated._reranker = fake1
        out_gated = gated._rerank_results(copy.deepcopy(results), "q", enabled=True)

        ungated = _make_router(rerank_max_concurrency=0, rerank_gate_wait_ms=2000,
                               rerank_length_bucketing=False)
        fake2 = _SlowReranker(delay=0.0)
        ungated._reranker = fake2
        out_ungated = ungated._rerank_results(copy.deepcopy(results), "q", enabled=True)

        assert [r["node_id"] for r in out_gated] == [r["node_id"] for r in out_ungated]
        for a, b in zip(out_gated, out_ungated):
            assert abs(a["score"] - b["score"]) < 1e-9
