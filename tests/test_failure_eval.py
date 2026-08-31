"""测试 Phase 2 失败驱动闭环：失败快照 → held-out 评估集 → 验证门。

覆盖：
  - 失败查询集提取（quality 阈值过滤 + degraded 硬失败 + 去重 + probe 排除）
  - 重放分数生成（N seeds、RetrievalSnapshot.quality() 口径、cfg 恢复）
  - 配置变更：明显提升 ACCEPT / 无差异 REJECT / 不足 12 item 回退启发式
  - 与 Phase 1 held_out_paired_gate 的真实集成（不 mock gate）

全部走公共入口（FailureLogger.log / FailedQueryEval / EvolutionGuard.apply），
不直调被 mock 的内部方法。
"""
import tempfile

import pytest

from retrieval.self_evolving import (
    EvolvableParams, FailureLogger, RetrievalSnapshot, EvolutionGuard,
)
from retrieval.failure_eval import FailedQueryEval, query_id
from core.validation_gate import held_out_paired_gate


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
        mesa_boost = 0.4

    c = Config()
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _tmp_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    import os
    os.close(fd)
    os.unlink(path)
    return path


def _snap(query: str, num_results: int, avg_score: float, top_distinct: int,
          degraded: bool = False, source: str = "retrieve") -> RetrievalSnapshot:
    return RetrievalSnapshot(
        timestamp=1.0,
        query=query,
        params_before={},
        num_results=num_results,
        top_scores=[avg_score] * min(5, max(1, num_results)),
        avg_score=avg_score,
        top_distinct=top_distinct,
        latency_ms=10.0,
        degraded=degraded,
        source=source,
    )


def _failed_logger(n: int, source: str = "retrieve") -> FailureLogger:
    """n 个互不相同的失败查询（quality < 0.4）：num=2/score=0.2/distinct=2 → 0.24。"""
    fl = FailureLogger()
    for i in range(n):
        fl.log(_snap(f"failed query {i}", num_results=2, avg_score=0.2,
                     top_distinct=2, source=source))
    return fl


class FixedRouter:
    """结果与参数无关的 mock QueryRouter（用于无差异 REJECT / 口径断言）。"""

    def __init__(self, config=None, results=None):
        self.config = config or _cfg()
        self._results = results or [{"content": "r0"}, {"content": "r1"}]

    def retrieve(self, q, include_archived=False, session_ts=None, level=None, rerank=None):
        return self._results


class ParamSensitiveRouter:
    """结果随 top_k_l1 变化的 mock QueryRouter（明显提升 → ACCEPT）。

    top_k_l1 < 10 → 2 条 score 0.3（quality=0.4*0.2+0.4*0.3+0.2*0.4=0.28）
    top_k_l1 ≥ 10 → 10 条 score 0.8（quality=0.4*1+0.4*0.8+0.2*1=0.92）
    """

    def __init__(self, config=None):
        self.config = config or _cfg()

    def retrieve(self, q, include_archived=False, session_ts=None, level=None, rerank=None):
        n = self.config.top_k_l1
        if n >= 10:
            return [{"content": f"c{i}", "score": 0.8} for i in range(10)]
        return [{"content": f"c{i}", "score": 0.3} for i in range(min(2, n))]


# ═══════════════ 1. 失败查询集提取 ═══════════════

class TestExtractFailedQueries:
    def test_quality_threshold_filter(self, tmp_path):
        fl = FailureLogger()
        fl.log(_snap("poor query", num_results=2, avg_score=0.2, top_distinct=2))     # 0.24 < 0.4 进
        fl.log(_snap("good query", num_results=10, avg_score=0.9, top_distinct=5))   # 0.96 ≥ 0.4 不进
        ev = FailedQueryEval(fl, FixedRouter(), persist_path=str(tmp_path / "fq.json"))
        items = ev.extract_failed_queries()
        assert query_id("poor query") in items
        assert query_id("good query") not in items

    def test_degraded_included_even_high_quality(self, tmp_path):
        # degraded 硬失败信号不被质量门槛吞掉（quality 0.96 也进）
        fl = FailureLogger()
        fl.log(_snap("degraded-but-rich", num_results=10, avg_score=0.9,
                     top_distinct=5, degraded=True))
        ev = FailedQueryEval(fl, FixedRouter(), persist_path=str(tmp_path / "fq.json"))
        items = ev.extract_failed_queries()
        assert query_id("degraded-but-rich") in items
        assert items[query_id("degraded-but-rich")]["quality"] == 0.96

    def test_dedupe_by_query(self, tmp_path):
        fl = FailureLogger()
        fl.log(_snap("dup query", num_results=2, avg_score=0.2, top_distinct=2))
        fl.log(_snap("dup query", num_results=1, avg_score=0.1, top_distinct=1))  # 同 query 再失败
        ev = FailedQueryEval(fl, FixedRouter(), persist_path=str(tmp_path / "fq.json"))
        items = ev.extract_failed_queries()
        assert len(items) == 1

    def test_probe_snapshots_excluded(self, tmp_path):
        fl = FailureLogger()
        fl.log(_snap("real failure", num_results=2, avg_score=0.2, top_distinct=2))
        fl.log(_snap("<health-probe>", num_results=0, avg_score=0.0, top_distinct=0,
                     degraded=True, source="probe"))
        fl.log(_snap("probe real query", num_results=2, avg_score=0.2, top_distinct=2,
                     source="probe"))
        ev = FailedQueryEval(fl, FixedRouter(), persist_path=str(tmp_path / "fq.json"))
        items = ev.extract_failed_queries()
        assert query_id("real failure") in items
        assert query_id("<health-probe>") not in items
        assert query_id("probe real query") not in items


# ═══════════════ 2. 重放分数生成 ═══════════════

class TestReplayScores:
    def test_per_seed_scores_and_quality_metric(self, tmp_path):
        # 2 条无 score 字段结果 → score 默认 0.5 → quality = 0.4*0.2+0.4*0.5+0.2*0.4 = 0.36
        fl = _failed_logger(3)
        router = FixedRouter(results=[{"content": "a"}, {"content": "b"}])
        ev = FailedQueryEval(fl, router, n_seeds=3, persist_path=str(tmp_path / "fq.json"))
        scores = ev.replay(EvolvableParams())
        assert len(scores) == 3
        for qid, per_seed in scores.items():
            assert per_seed == [0.36, 0.36, 0.36]  # N=3 次重放、口径=quality()

    def test_config_restored_after_replay(self, tmp_path):
        fl = _failed_logger(3)
        router = FixedRouter(config=_cfg(top_k_l1=7))
        ev = FailedQueryEval(fl, router, persist_path=str(tmp_path / "fq.json"))
        scores = ev.build_heldout_scores(EvolvableParams(top_k_l1=7),
                                         EvolvableParams(top_k_l1=20))
        # 重放后 cfg 恢复原值（cand top_k_l1=20 被临时写入后还原）
        assert router.config.top_k_l1 == 7
        assert set(scores["base"]) == set(scores["cand"])

    def test_build_heldout_scores_structure(self, tmp_path):
        fl = _failed_logger(4)
        ev = FailedQueryEval(fl, FixedRouter(), persist_path=str(tmp_path / "fq.json"))
        scores = ev.build_heldout_scores(EvolvableParams(), EvolvableParams(top_k_l1=20))
        assert set(scores.keys()) == {"base", "cand"}
        qids = set(scores["base"])
        assert qids == set(scores["cand"]) and len(qids) == 4

    def test_failure_queries_persisted(self, tmp_path):
        fl = _failed_logger(2)
        path = str(tmp_path / "failure_queries.json")
        ev = FailedQueryEval(fl, FixedRouter(), persist_path=path)
        ev.build_heldout_scores(EvolvableParams(), EvolvableParams())
        import json
        import os
        assert os.path.exists(path)
        data = json.load(open(path, encoding="utf-8"))
        assert len(data) == 2
        assert data[query_id("failed query 0")]["query"] == "failed query 0"


# ═══════════════ 3. EvolutionGuard 统计门判定 ═══════════════

class TestEvolutionGuardGate:
    def test_accept_keeps_new_params(self, tmp_path):
        """明显提升（top_k_l1 5→20，quality 0.28→0.92）→ ACCEPT 保留新配置。"""
        fl = _failed_logger(15)  # ≥12 item
        router = ParamSensitiveRouter()
        ev = FailedQueryEval(fl, router, persist_path=str(tmp_path / "fq.json"))
        guard = EvolutionGuard(initial_params=EvolvableParams(top_k_l1=5),
                               heldout_builder=ev.build_heldout_scores)
        ok, msg = guard.apply({"top_k_l1": 20})
        assert ok is True
        assert guard.current().top_k_l1 == 20  # 新配置保留
        assert guard.history[-1].reverted is False
        assert "配对检验 REJECT" not in msg

    def test_reject_rolls_back(self, tmp_path):
        """无差异（base/cand 结果相同）→ REJECT 回滚旧参数。"""
        fl = _failed_logger(15)
        router = FixedRouter(results=[{"content": "r0"}, {"content": "r1"}])
        ev = FailedQueryEval(fl, router, persist_path=str(tmp_path / "fq.json"))
        guard = EvolutionGuard(initial_params=EvolvableParams(top_k_l1=5),
                               heldout_builder=ev.build_heldout_scores)
        ok, msg = guard.apply({"top_k_l1": 20})
        assert ok is False
        assert "配对检验 REJECT" in msg
        assert guard.current().top_k_l1 == 5   # 回滚到旧参数
        assert guard.history[-1].reverted is True
        assert guard._pending is None

    def test_below_12_items_falls_back_to_heuristic(self, tmp_path, monkeypatch):
        """失败查询集不足 12 → 不触发统计门，回退在线启发式（原路径）。"""
        fl = _failed_logger(5)  # <12 item
        router = ParamSensitiveRouter()
        ev = FailedQueryEval(fl, router, persist_path=str(tmp_path / "fq.json"))
        guard = EvolutionGuard(initial_params=EvolvableParams(top_k_l1=5),
                               heldout_builder=ev.build_heldout_scores)

        # 统计门若被调用必然 import core.validation_gate.held_out_paired_gate ——
        # 打补丁使其抛错即可证明未走统计门
        def _boom(*a, **k):
            raise AssertionError("held_out_paired_gate 不应被调用（item < 12）")
        monkeypatch.setattr("core.validation_gate.held_out_paired_gate", _boom)

        ok, msg = guard.apply({"top_k_l1": 20})
        assert ok is True                      # 启发式路径：应用成功不回滚
        assert guard.current().top_k_l1 == 20
        assert guard.history[-1].reverted is False

    def test_no_builder_keeps_original_behavior(self):
        """无 builder（默认）→ 行为与旧版一致：直接应用不触发评估。"""
        guard = EvolutionGuard(initial_params=EvolvableParams(top_k_l1=5))
        ok, msg = guard.apply({"top_k_l1": 20})
        assert ok is True
        assert guard.current().top_k_l1 == 20
        assert guard.history[-1].reverted is False

    def test_guard_reject_persists_history_audit(self, tmp_path):
        """REJECT 回滚在 history 保留审计（reverted 标志 + 旧参数），格式兼容。"""
        fl = _failed_logger(15)
        router = FixedRouter()
        ev = FailedQueryEval(fl, router, persist_path=str(tmp_path / "fq.json"))
        guard = EvolutionGuard(initial_params=EvolvableParams(top_k_l1=5),
                               heldout_builder=ev.build_heldout_scores)
        guard.apply({"top_k_l1": 20})
        entry = guard.history[-1]
        assert entry.reverted is True
        assert entry.version == 1
        # ConfigVersion 序列化字段（save_state 用）保持原样
        assert {"version", "params", "applied_at", "avg_quality_after",
                "reverted"} <= set(entry.__dataclass_fields__)


# ═══════════════ 4. 与 Phase 1 held_out_paired_gate 真实集成 ═══════════════

class TestGateIntegration:
    def test_real_held_out_paired_gate_accept(self, tmp_path):
        """build_heldout_scores 输出直接喂真实 gate：明显提升 → ACCEPT。"""
        fl = _failed_logger(15)
        router = ParamSensitiveRouter()
        ev = FailedQueryEval(fl, router, persist_path=str(tmp_path / "fq.json"))
        scores = ev.build_heldout_scores(EvolvableParams(top_k_l1=5),
                                         EvolvableParams(top_k_l1=20))
        verdict = held_out_paired_gate(scores["base"], scores["cand"])
        assert verdict.accept is True
        assert verdict.net > 0.5          # 0.28 → 0.92 每 item 大幅提升
        assert verdict.n_regressed == 0   # reg_cap=0：零回归
        assert verdict.n_improved == 15
        assert verdict.ci[0] > 0          # CI 在改善侧排除 0

    def test_real_held_out_paired_gate_reject(self, tmp_path):
        """无差异 → 真实 gate REJECT（CI 含 0）。"""
        fl = _failed_logger(15)
        router = FixedRouter()
        ev = FailedQueryEval(fl, router, persist_path=str(tmp_path / "fq.json"))
        scores = ev.build_heldout_scores(EvolvableParams(), EvolvableParams(top_k_l1=20))
        verdict = held_out_paired_gate(scores["base"], scores["cand"])
        assert verdict.accept is False
        assert verdict.net == 0.0
