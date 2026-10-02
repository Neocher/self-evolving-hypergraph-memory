#!/usr/bin/env python3
"""τ 衰减形式离线仿真（v6.24.0 PowerLaw-Tau，A/B 第一道门）

三种衰减形式（exp / pow α=0.5 / frac K=8 sf=10）在多个时间跨度上的 τ 对比：
- 优先从现有 OverGraph DB 只读抽样至多 1000 条 EpisodeNode.created_at；
- 连接/查询失败或无数据 → 回退 1000 条合成 created_at（最近 30 天，确定性种子）；
- 绝不写库。

断言（3d/7d/14d）：mean pow τ > mean exp τ 且 mean frac τ > mean exp τ，
失败以非零码退出。

独立运行：
    python scripts/tau_form_sim.py
"""
from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path

# —— 确保仓库根在 sys.path（脚本可独立运行）——
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.tau_decay import TauDecayConfig, TauDecayEngine  # noqa: E402

DB_PATH = str(_REPO_ROOT / "data" / "shm_overgraph_db")
DAY = 86400.0
HOUR = 3600.0
HORIZONS = [
    ("1h", 1 * HOUR),
    ("1d", 1 * DAY),
    ("3d", 3 * DAY),
    ("7d", 7 * DAY),
    ("14d", 14 * DAY),
]
SAMPLE_LIMIT = 1000


def _sample_created_at() -> tuple[list[float], str]:
    """返回 (created_at 列表, 来源标签)。失败回退合成数据。"""
    fallback_label = "synthetic(30d, seed=42)"
    try:
        from graph.overgraph_store import OverGraphStore

        cfg = type(
            "cfg",
            (),
            {
                "database_path": DB_PATH,
                "dense_vector_dimension": 512,
                "dense_vector_metric": "cosine",
            },
        )()
        store = OverGraphStore(config=cfg)
        try:
            store.connect()
            rows = store.query_cypher(
                "MATCH (e:EpisodeNode) "
                "RETURN e.created_at AS created_at LIMIT 1000"
            )
        finally:
            store.close()

        values: list[float] = []
        for row in rows or []:
            raw = row.get("created_at") if isinstance(row, dict) else None
            if raw is None:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        if values:
            return values[:SAMPLE_LIMIT], f"overgraph_db({len(values[:SAMPLE_LIMIT])})"
        print("[sim] OverGraph query returned no usable created_at → fallback")
    except Exception as e:  # noqa: BLE001 — 采样失败一律回退
        print(f"[sim] OverGraph sampling failed ({e!r}) → fallback")

    # 合成回退：最近 30 天，确定性
    rng = random.Random(42)
    now = time.time()
    synth = [now - rng.uniform(0.0, 30 * DAY) for _ in range(SAMPLE_LIMIT)]
    return synth, fallback_label


def _build_engines() -> dict[str, TauDecayEngine]:
    return {
        "exp": TauDecayEngine(
            TauDecayConfig(
                tau_form="exp", tau_initial=1.0, tau_decay_seconds=300.0,
                enable_adaptive=False,
            )
        ),
        "pow(a=0.5)": TauDecayEngine(
            TauDecayConfig(
                tau_form="pow", alpha=0.5, tau_initial=1.0,
                tau_decay_seconds=300.0, enable_adaptive=False,
            )
        ),
        "frac(K=8,sf=10)": TauDecayEngine(
            TauDecayConfig(
                tau_form="frac", frac_K=8, frac_scale_factor=10.0,
                tau_initial=1.0, tau_decay_seconds=300.0, enable_adaptive=False,
            )
        ),
    }


def run() -> int:
    samples, source = _sample_created_at()
    print(f"[sim] created_at source: {source} (n={len(samples)})")
    now = time.time()
    engines = _build_engines()

    # results[form_label][horizon_label] = (mean, median)
    results: dict[str, dict[str, tuple[float, float]]] = {}
    for form_label, engine in engines.items():
        per_horizon: dict[str, tuple[float, float]] = {}
        for h_label, h_seconds in HORIZONS:
            taus: list[float] = []
            for i, _ in enumerate(samples):
                taus.append(
                    engine.compute_tau(
                        f"m{i}",
                        created_at=now - h_seconds,
                        force_now=now,
                    )
                )
            per_horizon[h_label] = (
                statistics.fmean(taus),
                statistics.median(taus),
            )
        results[form_label] = per_horizon

    # —— 表格输出 ——
    col_w = max([len(h) for h, _ in HORIZONS] + [12]) + 1
    print()
    header = f"{'form':<16}" + "".join(f"{h:>{col_w}}" for h, _ in HORIZONS)
    print(header)
    print("-" * len(header))
    for form_label in engines:
        cells = []
        for h_label, _ in HORIZONS:
            mean, _med = results[form_label][h_label]
            cells.append(f"{mean:>{col_w}.3e}")
        print(f"{form_label:<16}" + "".join(cells))
    print()
    print("median:")
    print(header)
    print("-" * len(header))
    for form_label in engines:
        cells = []
        for h_label, _ in HORIZONS:
            _mean, med = results[form_label][h_label]
            cells.append(f"{med:>{col_w}.3e}")
        print(f"{form_label:<16}" + "".join(cells))

    # —— 断言：3d/7d/14d 长尾生效 ——
    print()
    ok = True
    for h_label, _ in HORIZONS:
        if h_label not in ("3d", "7d", "14d"):
            continue
        mean_exp = results["exp"][h_label][0]
        mean_pow = results["pow(a=0.5)"][h_label][0]
        mean_frac = results["frac(K=8,sf=10)"][h_label][0]
        cond = (mean_pow > mean_exp) and (mean_frac > mean_exp)
        ok = ok and cond
        print(
            f"[assert] {h_label}: mean pow={mean_pow:.3e} > exp={mean_exp:.3e}"
            f"  AND frac={mean_frac:.3e} > exp={mean_exp:.3e}  "
            f"=> {'PASS' if cond else 'FAIL'}"
        )

    if not ok:
        print("[sim] LONG-TAIL ASSERTIONS FAILED", file=sys.stderr)
        return 1
    print("[sim] all long-tail assertions PASS")
    return 0


if __name__ == "__main__":
    sys.exit(run())
