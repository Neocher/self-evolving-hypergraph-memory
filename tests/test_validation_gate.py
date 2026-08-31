"""Phase 1 Recuris 验证门控测试
================================
覆盖：
· held_out_paired_gate 单元：明显提升 ACCEPT / 无差异 REJECT / 回归 REJECT
  (reg_cap=0) / 回归数 ≤ cap ACCEPT / 空交集 REJECT / material 容忍密集噪声 /
  bootstrap 重采样 items 而非 trials（trials 数不影响区间）
· evolve_once 集成：无评估集（get_heldout_scores → None）落盘不变（向后兼容）；
  评估集 REJECT → gate_rejected 且不落盘。
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.ontology_evolution import evolve_once, load_extended
from core.validation_gate import held_out_paired_gate


def run(coro):
    return asyncio.run(coro)


def _llm(response: str) -> MagicMock:
    client = MagicMock()
    client.api_key = "test-key"
    client.chat = AsyncMock(return_value=response)
    return client


def _summaries() -> list[dict]:
    return [
        {
            "topics": ["quantum entanglement", "nonlocality"],
            "report": "Community about quantum entanglement experiments and nonlocality.",
        },
    ]


# ─── held_out_paired_gate 单元测试 ────────────────────────────


def test_clear_improvement_accept():
    """明显提升（base 全 0.4 → cand 全 0.7）→ ACCEPT。"""
    base = {f"item{i}": [0.4, 0.4, 0.4] for i in range(6)}
    cand = {f"item{i}": [0.7, 0.7, 0.7] for i in range(6)}
    v = held_out_paired_gate(base, cand)
    assert v.accept is True
    assert v.reason == "net improvement, CI excludes 0"
    assert v.n_regressed == 0
    assert v.n_improved == 6
    assert v.net == pytest.approx(0.3, abs=1e-3)
    lo, hi = v.ci
    assert lo > 0


def test_no_difference_reject():
    """base=cand（无差异）→ REJECT，CI 包含 0。"""
    scores = {f"item{i}": [0.5, 0.6, 0.4] for i in range(8)}
    v = held_out_paired_gate(scores, dict(scores))
    assert v.accept is False
    assert "CI includes 0" in v.reason
    assert v.net == 0.0


def test_regression_reject_reg_cap_zero():
    """2 个 item 回归（其余提升）且 reg_cap=0 → REJECT（回归数超 cap）。"""
    base = {f"up{i}": [0.4, 0.4, 0.4] for i in range(4)}
    cand = {f"up{i}": [0.7, 0.7, 0.7] for i in range(4)}
    base.update({f"dn{i}": [0.5, 0.5, 0.5] for i in range(2)})
    cand.update({f"dn{i}": [0.49, 0.49, 0.49] for i in range(2)})
    v = held_out_paired_gate(base, cand)
    assert v.accept is False
    assert "regressed" in v.reason
    assert "exceeds the cap" in v.reason
    assert v.n_regressed == 2


def test_regression_within_cap_accept():
    """同上但 reg_cap=2（容忍 2 个回归）→ ACCEPT。"""
    base = {f"up{i}": [0.4, 0.4, 0.4] for i in range(4)}
    cand = {f"up{i}": [0.7, 0.7, 0.7] for i in range(4)}
    base.update({f"dn{i}": [0.5, 0.5, 0.5] for i in range(2)})
    cand.update({f"dn{i}": [0.49, 0.49, 0.49] for i in range(2)})
    v = held_out_paired_gate(base, cand, reg_cap=2)
    assert v.accept is True
    assert v.reason == "net improvement, CI excludes 0"
    assert v.n_regressed == 2


def test_empty_intersection_reject():
    """base/cand 无共有 item（空交集）→ REJECT "no comparable items"。"""
    base = {"item_a": [0.5, 0.5]}
    cand = {"item_b": [0.5, 0.5]}
    v = held_out_paired_gate(base, cand)
    assert v.accept is False
    assert v.reason == "no comparable items"
    assert v.net == 0.0
    assert v.n_improved == 0 and v.n_regressed == 0


def test_material_tolerates_dense_reward_noise():
    """material 容忍密集奖励噪声：同一数据 material=0 拒绝（噪声计入回归），
    material=0.05 时不把 ±0.01 噪声计入回归 → ACCEPT。"""
    base = {f"up{i}": [0.5, 0.5, 0.5] for i in range(4)}
    cand = {f"up{i}": [0.8, 0.8, 0.8] for i in range(4)}
    base.update({f"n{i}": [0.5, 0.5, 0.5] for i in range(2)})
    cand.update({f"n{i}": [0.49, 0.49, 0.49] for i in range(2)})
    strict = held_out_paired_gate(base, cand)  # reg_cap=0, material=0
    assert strict.accept is False
    assert "regressed" in strict.reason
    tolerant = held_out_paired_gate(base, cand, material=0.05)
    assert tolerant.accept is True
    assert tolerant.n_regressed == 0


def test_bootstrap_resamples_items_not_trials():
    """bootstrap 重采样 items 而非 trials：单 item 内 trials 数翻倍不改变区间
    （同一 item 的 trials 不独立，重采样 trials 会报出过窄区间）。"""
    v1 = held_out_paired_gate({"i": [0.4]}, {"i": [0.7]})
    v2 = held_out_paired_gate({"i": [0.4] * 50}, {"i": [0.7] * 50})
    assert v1.ci == v2.ci  # diffs 结构相同 → 区间完全一致
    assert v1.accept is True and v2.accept is True


# ─── evolve_once 集成测试 ─────────────────────────────────────


def test_evolve_once_backward_compat_no_heldout(tmp_path, monkeypatch):
    """无评估集（get_heldout_scores → None）→ 行为与现状完全一致：LLM 提议 →
    语法守卫 → 直接落盘。"""
    monkeypatch.setattr(
        "core.ontology_evolution.get_heldout_scores",
        AsyncMock(return_value=None),
    )
    p = tmp_path / "ontology_extended.json"
    client = _llm(json.dumps({
        "action": "new_type",
        "type": "quantum_result",
        "description": "量子实验发现",
        "conflict_keys": ["quantum", "entanglement", "nonlocality"],
    }))
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "new_type"
    assert result["type"] == "quantum_result"
    extended = load_extended(str(p))
    assert "quantum_result" in extended


def test_evolve_once_gate_rejected_no_persist(tmp_path, monkeypatch):
    """评估集 REJECT（base=cand 无差异）→ evolve_once 返回 gate_rejected 且不落盘。"""
    heldout = {
        "base": {f"item{i}": [0.5, 0.5] for i in range(6)},
        "cand": {f"item{i}": [0.5, 0.5] for i in range(6)},
    }
    monkeypatch.setattr(
        "core.ontology_evolution.get_heldout_scores",
        AsyncMock(return_value=heldout),
    )
    p = tmp_path / "ontology_extended.json"
    client = _llm(json.dumps({
        "action": "new_type",
        "type": "quantum_result",
        "description": "量子实验发现",
        "conflict_keys": ["quantum", "entanglement", "nonlocality"],
    }))
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "skip"
    assert result["reason"].startswith("gate_rejected:")
    assert "CI includes 0" in result["reason"]
    assert not p.exists()  # 未落盘


def test_evolve_once_gate_accepted_persists(tmp_path, monkeypatch):
    """评估集 ACCEPT（明显提升）→ 验证门放行，正常落盘。"""
    heldout = {
        "base": {f"item{i}": [0.4, 0.4] for i in range(6)},
        "cand": {f"item{i}": [0.7, 0.7] for i in range(6)},
    }
    monkeypatch.setattr(
        "core.ontology_evolution.get_heldout_scores",
        AsyncMock(return_value=heldout),
    )
    p = tmp_path / "ontology_extended.json"
    client = _llm(json.dumps({
        "action": "new_type",
        "type": "quantum_result",
        "description": "量子实验发现",
        "conflict_keys": ["quantum", "entanglement", "nonlocality"],
    }))
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "new_type"
    assert result["type"] == "quantum_result"
    extended = load_extended(str(p))
    assert "quantum_result" in extended
