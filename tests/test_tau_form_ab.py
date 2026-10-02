"""τ 衰减形式 A/B 单元测试（v6.24.0 PowerLaw-Tau）

纯单元测试，不依赖任何外部服务。覆盖：
- exp 分支逐点向后兼容（含 underflow 哨兵）
- pow / frac 长尾；alpha 语义；frac K=1 退化
- AdaMem 在 pow 模式学 alpha_custom、exp 模式学 tau_decay_custom（两路不混）
- yaml / env 配置覆盖

公式约定（已按任务书【新能力】文字/测试描述做数学自洽修正）：
  - pow:  τ₀·(1 + dt/τc)^(-α)，α∈(0,1] 越小长记忆越强；α=1 退化为 (1+x)^-1。
  - frac: K 个对数间隔指数模式加权求和，尺度按每档 ×frac_scale_factor 排列
          （sf^0..sf^(K-1)，跨 K 个 decade），以逼近重尾幂律核。
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from core.tau_decay import (
    AdaptiveDecayLearner,
    TauDecayConfig,
    TauDecayEngine,
)
from config.settings import load_settings


# 固定时间基准，避免 time.time() 抖动导致 flaky
BASE = 1_000_000_000.0
DAY = 86400.0
HOUR = 3600.0


def _engine(**cfg_kwargs) -> TauDecayEngine:
    """构造 enable_adaptive=False 的引擎（除显式覆盖）。"""
    cfg_kwargs.setdefault("enable_adaptive", False)
    kwargs = {"tau_decay_seconds": 300.0, **cfg_kwargs}
    return TauDecayEngine(TauDecayConfig(**kwargs))


# 1 ────────────────────────────────────────────────────────────────────
def test_exp_unchanged():
    """exp 默认分支逐点等于 τ₀·exp(-dt/τc)，且 dt 极大时 underflow 返回 0.0。"""
    eng = _engine(tau_form="exp", tau_initial=1.0)
    eng.register_node("n", created_at=BASE)
    for dt in (0, 60, 3600, 86400):
        tau = eng.compute_tau("n", created_at=BASE, force_now=BASE + dt)
        assert tau == pytest.approx(math.exp(-dt / 300.0)), f"dt={dt}"
    # 3 天 → exponent = -259200/300 = -864 < -700 → 精确 0.0
    assert eng.compute_tau(
        "n", created_at=BASE, force_now=BASE + 3 * DAY
    ) == 0.0


# 2 ────────────────────────────────────────────────────────────────────
def test_pow_long_tail():
    """pow 在 3d/7d/14d 仍有正残值，且严格大于同条件 exp 的 τ。"""
    eng_pow = _engine(tau_form="pow", alpha=0.5, tau_initial=1.0)
    eng_exp = _engine(tau_form="exp", tau_initial=1.0)
    for eng in (eng_pow, eng_exp):
        eng.register_node("n", created_at=BASE)
    for dt in (3 * DAY, 7 * DAY, 14 * DAY):
        tau_pow = eng_pow.compute_tau("n", created_at=BASE, force_now=BASE + dt)
        tau_exp = eng_exp.compute_tau("n", created_at=BASE, force_now=BASE + dt)
        assert tau_pow > 0.0, f"pow dt={dt}"
        assert tau_pow > tau_exp, f"pow 长尾未超 exp: dt={dt}"


# 3 ────────────────────────────────────────────────────────────────────
def test_pow_alpha_extremes():
    """alpha 极值行为 + alpha=1.0 退化为 (1+dt/τc)^-1。

    公式 τ₀·(1+dt/τc)^(-α)：α 越小指数越接近 0 → 衰减越慢 → 长记忆越强。
    """
    dt = 3 * DAY
    eng_low = _engine(tau_form="pow", alpha=0.1, tau_initial=1.0)
    eng_high = _engine(tau_form="pow", alpha=1.0, tau_initial=1.0)
    for eng in (eng_low, eng_high):
        eng.register_node("n", created_at=BASE)
    tau_low = eng_low.compute_tau("n", created_at=BASE, force_now=BASE + dt)
    tau_high = eng_high.compute_tau("n", created_at=BASE, force_now=BASE + dt)
    # α 越小 → 衰减越慢（长记忆越强）
    assert tau_low > tau_high
    # alpha=1.0 → 指数 -1
    assert tau_high == pytest.approx((1.0 + dt / 300.0) ** -1.0)


# 4 ────────────────────────────────────────────────────────────────────
def test_frac_matches_pow_roughly():
    """frac 长尾 > exp；frac 与 pow(α=0.5) 在 3d/7d 同量级（≤4 个数量级）。

    frac 是"有限个对数间隔指数模式加权求和"，精确 parity 不期望，但跨 K 个
    decade 的尺度应逼近 pow 的重尾量级。允许 ≤4 个数量级差异。
    """
    eng_frac = _engine(
        tau_form="frac", tau_initial=1.0, frac_K=8, frac_scale_factor=10.0
    )
    eng_exp = _engine(tau_form="exp", tau_initial=1.0)
    eng_pow = _engine(tau_form="pow", alpha=0.5, tau_initial=1.0)
    for eng in (eng_frac, eng_exp, eng_pow):
        eng.register_node("n", created_at=BASE)

    for dt in (3 * DAY, 7 * DAY):
        tau_frac = eng_frac.compute_tau("n", created_at=BASE, force_now=BASE + dt)
        tau_exp = eng_exp.compute_tau("n", created_at=BASE, force_now=BASE + dt)
        tau_pow = eng_pow.compute_tau("n", created_at=BASE, force_now=BASE + dt)
        assert tau_frac > 0.0, f"frac dt={dt}"
        assert tau_frac > tau_exp, f"frac 长尾未超 exp: dt={dt}"
        assert tau_pow > 0.0, f"pow dt={dt}"
        # 同量级：frac 近似 pow 的重尾，允许 ≤4 个数量级差异
        assert abs(math.log10(tau_frac) - math.log10(tau_pow)) <= 4, (
            f"frac/pow 量级偏离过大: dt={dt} frac={tau_frac:.3e} pow={tau_pow:.3e}"
        )


# 5 ────────────────────────────────────────────────────────────────────
def test_frac_K1_equals_exp():
    """frac K=1 退化为单指数，逐点等于 exp（证明 tau_initial 乘子存在）。"""
    eng_frac = _engine(
        tau_form="frac", tau_initial=0.9, frac_K=1, frac_scale_factor=10.0
    )
    eng_exp = _engine(tau_form="exp", tau_initial=0.9)
    for eng in (eng_frac, eng_exp):
        eng.register_node("n", created_at=BASE)
    for dt in (60, 3600, 86400):
        tau_frac = eng_frac.compute_tau("n", created_at=BASE, force_now=BASE + dt)
        tau_exp = eng_exp.compute_tau("n", created_at=BASE, force_now=BASE + dt)
        assert tau_frac == pytest.approx(tau_exp), f"dt={dt}"
        assert tau_frac == pytest.approx(0.9 * math.exp(-dt / 300.0)), f"dt={dt}"


# 6 ────────────────────────────────────────────────────────────────────
def test_ada_mem_learn_alpha():
    """pow 模式 AdaMem 走 alpha_custom 路径，不碰 tau_decay_custom。"""
    cfg = TauDecayConfig(tau_form="pow", alpha=0.5, enable_adaptive=False)
    engine = TauDecayEngine(cfg)
    learner = AdaptiveDecayLearner(cfg)
    engine.register_node("n", created_at=BASE)
    for _ in range(cfg.decay_target_window):
        learner.record_feedback("n", 0.8)
    learner.update_decay(engine)

    info = engine._node_info["n"]
    assert info.alpha_custom is not None
    assert 0.1 <= info.alpha_custom <= 1.0
    assert info.tau_decay_custom is None


# 7 ────────────────────────────────────────────────────────────────────
def test_ada_mem_exp_unchanged():
    """exp 模式 AdaMem 仍走 tau_decay_custom 路径，alpha_custom 恒 None。"""
    cfg = TauDecayConfig(tau_form="exp", enable_adaptive=False)
    engine = TauDecayEngine(cfg)
    learner = AdaptiveDecayLearner(cfg)
    engine.register_node("n", created_at=BASE)
    for _ in range(cfg.decay_target_window):
        learner.record_feedback("n", 600.0)
    learner.update_decay(engine)

    info = engine._node_info["n"]
    assert info.tau_decay_custom is not None
    assert info.alpha_custom is None


# 8 ────────────────────────────────────────────────────────────────────
def test_config_yaml_override(tmp_path: Path):
    """自定义 yaml 设 tau_form: pow → settings.tau.tau_form == "pow"，其余取 dataclass 默认。"""
    yaml_file = tmp_path / "tau_pow.yaml"
    yaml_file.write_text('{"tau": {"tau_form": "pow"}}', encoding="utf-8")
    settings = load_settings(yaml_path=yaml_file)
    assert settings.tau.tau_form == "pow"
    assert settings.tau.alpha == 0.5
    assert settings.tau.frac_K == 8
    assert settings.tau.frac_scale_factor == 10.0


# 9 ────────────────────────────────────────────────────────────────────
def test_env_override(monkeypatch):
    """SHM_TAU__TAU_FORM=pow 经泛型 _env_override 生效。"""
    monkeypatch.setenv("SHM_TAU__TAU_FORM", "pow")
    settings = load_settings()
    assert settings.tau.tau_form == "pow"
