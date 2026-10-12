# -*- coding: utf-8 -*-
"""【Part B + Part C】rerank_max_length 512→384 + 启动后台任务错峰 验收测试。

覆盖任务书验收点：
  AC1(B)  rerank_max_length 默认 384，defaults.yaml / settings.py / QueryRouterConfig 一致
  AC2(B)  精度不掉：真实 store/真实候选 + 真实 query，384 vs 512 top-5 重合 ≥95%
          （真实 bge-reranker-base 交叉编码器实测，绝不合成短文本、绝不空集）
  AC3(B)  提速：同一批真实数据 384 比 512 快 ≥1.15x（真实推理计时，贴出实测数字）
  AC5(C)  可回退：startup_background_stagger_s = 0 → 后台任务立即执行（无错峰延迟）
  AC7(C)  错峰机制：4 个启动后台任务首次执行前各加 stagger*slot 初始延迟，fail-soft

真实数据来源：data/bench/persona_hist/*.json（真人对话语料，仓库内真实数据）；
缺失时回落仓库 git 跟踪的 md 文档真实文本。绝不合成、绝不 skip。

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_startup_stagger_and_rerank_len.py -q
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import time
import types
from pathlib import Path

import pytest
import yaml

import api.app as app_module
from config.settings import APIConfig, load_settings
from retrieval.query_router import QueryRouterConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── 真实语料加载（与 test_rerank_bucketing.py 同源，绝不合成）──────────────────
def _real_messages() -> list[str]:
    """取真人对话语料（persona_hist）；缺失时回落仓库 git 跟踪的 md 文档真实文本。"""
    texts: list[str] = []
    bench_dir = _REPO_ROOT / "data" / "bench" / "persona_hist"
    for fp in sorted(bench_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for m in data.get("chat_history", []) or []:
            if not isinstance(m, dict) or m.get("role") == "system":
                continue
            c = str(m.get("content", "") or "").strip()
            if c:
                texts.append(c)
    if texts:
        return texts
    for name in ("CHANGELOG.md", "README.md", "CLAUDE.md"):
        fp = _REPO_ROOT / name
        if fp.exists():
            texts.append(fp.read_text(encoding="utf-8"))
    return texts


# 真实候选：多数为短文档（≤384 token，真实检索候选的自然长度分布），少量长文档
# （>384 token）用于验证 384 vs 512 的截断差异。全部来自真实语料，非合成。
def _precision_candidates() -> list[str]:
    msgs = _real_messages()
    short = [c for c in msgs if 200 < len(c) < 1200][:12]
    long = [c for c in msgs if 1500 <= len(c) <= 3500][:3]
    return short + long


_QUERIES = ["最近的进展", "机器学习", "谁负责这个项目", "会议结论是什么",
            "任务当前状态", "如何优化检索延迟", "记忆系统如何工作", "有哪些待办事项"]


@functools.lru_cache(maxsize=None)
def _cross_encoder(max_length: int):
    """懒加载真实 bge-reranker-base 交叉编码器（离线加载，进程内缓存按 max_length 分键）。"""
    import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-base", device="cpu", max_length=int(max_length))


def _top5(model, query: str, candidates: list[str]) -> list[int]:
    scores = model.predict([(query, c) for c in candidates])
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [i for i in order[:5]]


# ── AC1(B)：默认值 384，defaults / settings / QueryRouterConfig 三处一致 ──────
class TestRerankMaxLengthDefault:
    def test_settings_default_is_384(self):
        assert load_settings().retrieval.rerank_max_length == 384

    def test_query_router_default_is_384(self):
        assert QueryRouterConfig().rerank_max_length == 384

    def test_defaults_yaml_is_384(self):
        raw = yaml.safe_load((_REPO_ROOT / "config" / "defaults.yaml").read_text(encoding="utf-8"))
        assert raw["retrieval"]["rerank_max_length"] == 384

    def test_build_router_config_default_is_384(self):
        class NoMaxLength:
            tau_weight = 0.4
            vector_weight = 0.6
            top_k_l1 = 5
            top_k_vector = 20
            top_k_keyword = 20

        qcfg = app_module._build_router_config(NoMaxLength())
        assert qcfg.rerank_max_length == 384


# ── AC2(B) + AC3(B)：真实数据 × 真实 reranker，384 vs 512 精度 + 提速 ─────────
class TestRerank384Vs512:
    def test_384_vs_512_preserves_top5_and_is_faster(self):
        candidates = _precision_candidates()
        assert len(candidates) == 15, "必须真实候选（12 短 + 3 长）"
        m384 = _cross_encoder(384)
        m512 = _cross_encoder(512)

        # 短文档（≤384 token）在两种 max_length 下分数必须逐位一致（截断机制校验）
        short_scores_384 = m384.predict([(q, candidates[0]) for q in _QUERIES])
        short_scores_512 = m512.predict([(q, candidates[0]) for q in _QUERIES])
        assert list(short_scores_384) == list(short_scores_512), (
            "短文档不被截断 → 384 与 512 分数逐位一致"
        )

        # 预热（首次 predict 含懒初始化，不计入计时）
        m384.predict([(_QUERIES[0], candidates[0])])
        m512.predict([(_QUERIES[0], candidates[0])])

        overlaps: list[float] = []
        t384_total = 0.0
        t512_total = 0.0
        for q in _QUERIES:
            t0 = time.monotonic()
            top384 = _top5(m384, q, candidates)
            t384_total += time.monotonic() - t0

            t1 = time.monotonic()
            top512 = _top5(m512, q, candidates)
            t512_total += time.monotonic() - t1

            overlaps.append(len(set(top384) & set(top512)) / 5.0)

        mean_overlap = sum(overlaps) / len(overlaps)
        speedup = t512_total / t384_total
        print(f"\n[AC2] top-5 重合率 per-query: {[round(o, 2) for o in overlaps]} "
              f"→ 均值 {mean_overlap:.3%}")
        print(f"[AC3] 384 总耗时 {t384_total:.3f}s vs 512 总耗时 {t512_total:.3f}s "
              f"→ 提速 {speedup:.3f}x")

        # AC2：top-5 重合 ≥95%（预期 97.5%）
        assert mean_overlap >= 0.95, f"top-5 重合率 {mean_overlap:.3%} < 95%"
        # AC3：提速 ≥1.15x（预期 1.27x）
        assert speedup >= 1.15, f"384 vs 512 提速 {speedup:.3f}x < 1.15x"


# ── AC5(C) + 错峰机制：配置默认 / 可回退 / slot 递增 / fail-soft ──────────────
def _cfg(stagger: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(api=types.SimpleNamespace(startup_background_stagger_s=stagger))


class TestStartupStaggerConfig:
    def test_default_is_0(self):
        """【v6.35 回退】默认 0.0（关闭错峰）；10.0 已被 Hermes py-spy 证伪。"""
        assert APIConfig().startup_background_stagger_s == 0.0
        assert load_settings().api.startup_background_stagger_s == 0.0

    def test_defaults_yaml_is_0(self):
        raw = yaml.safe_load((_REPO_ROOT / "config" / "defaults.yaml").read_text(encoding="utf-8"))
        assert raw["api"]["startup_background_stagger_s"] == 0.0


class TestStartupStaggerDelay:
    def test_slot_scaling(self, monkeypatch):
        monkeypatch.setattr(app_module, "get_settings", lambda: _cfg(10.0))
        assert app_module._startup_stagger_delay(1) == 10.0
        assert app_module._startup_stagger_delay(2) == 20.0
        assert app_module._startup_stagger_delay(4) == 40.0

    def test_zero_disables_reverts_to_immediate(self, monkeypatch):
        """AC5：stagger=0 → 行为与改动前一致（立即执行，无错峰）。"""
        monkeypatch.setattr(app_module, "get_settings", lambda: _cfg(0.0))
        assert app_module._startup_stagger_delay(1) == 0.0
        assert app_module._startup_stagger_delay(3) == 0.0

    def test_fail_soft_on_config_read_error(self, monkeypatch):
        def boom():
            raise RuntimeError("cfg read boom")

        monkeypatch.setattr(app_module, "get_settings", boom)
        assert app_module._startup_stagger_delay(1) == 0.0


class TestStaggeredStart:
    def test_sleeps_once_with_scaled_delay(self, monkeypatch):
        slept: list[float] = []

        async def fake_sleep(sec: float) -> None:
            slept.append(sec)

        monkeypatch.setattr(app_module, "get_settings", lambda: _cfg(10.0))
        monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
        asyncio.run(app_module._staggered_start(3))
        assert slept == [30.0], "错峰延迟 = stagger * slot，且只睡一次"

    def test_no_sleep_when_disabled(self, monkeypatch):
        slept: list[float] = []

        async def fake_sleep(sec: float) -> None:
            slept.append(sec)

        monkeypatch.setattr(app_module, "get_settings", lambda: _cfg(0.0))
        monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
        asyncio.run(app_module._staggered_start(1))
        assert slept == [], "stagger=0 不得 sleep（可回退）"

    def test_propagates_cancellation(self, monkeypatch):
        """CancelledError 必须向上传播（shutdown 取消任务时能正常退出），不得被吞。"""

        async def fake_sleep(sec: float) -> None:
            raise asyncio.CancelledError()

        monkeypatch.setattr(app_module, "get_settings", lambda: _cfg(10.0))
        monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)

        caught: list[bool] = []

        async def runner() -> None:
            try:
                await app_module._staggered_start(1)
            except asyncio.CancelledError:
                caught.append(True)

        asyncio.run(runner())
        assert caught == [True]

    def test_fail_soft_on_sleep_error(self, monkeypatch):
        """非取消异常（如 sleep 出错）静默回退立即执行，不影响后台任务本体。"""

        async def fake_sleep(sec: float) -> None:
            raise RuntimeError("sleep boom")

        monkeypatch.setattr(app_module, "get_settings", lambda: _cfg(10.0))
        monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
        asyncio.run(app_module._staggered_start(1))  # 不得抛出


class TestBackgroundTasksStaggered:
    def test_four_startup_tasks_await_stagger(self):
        """4 个启动后台任务各自首次执行前 await _staggered_start(...)。"""
        src = inspect.getsource(app_module.lifespan.__wrapped__)
        for name in ("_startup_user_profile", "_dream_poll_loop",
                     "_embed_poll_loop", "_hyperedge_sweep"):
            assert f"async def {name}()" in src, f"{name} 必须在 lifespan 内定义"
        assert src.count("await _staggered_start(") == 4, (
            "4 个后台任务必须各自在首次执行前 await 错峰延迟"
        )
