"""Recuris 验证门控（Phase 1 移植）— held_out_paired_gate
========================================================
移植自 Gen-Verse/Recuris（arXiv:2608.24876，Apache-2.0）：
  https://github.com/Gen-Verse/Recuris/blob/main/src/recuris/metaagent/gates.py

核心思想「模型提议，算术裁决」：候选记忆/schema 变更必须在配对 held-out 集上
通过统计检验（bootstrap CI 排除 0 + 回归 item 数 ≤ reg_cap）才被接纳。

本文件是纯函数层：无 I/O、无 LLM 调用，仅标准库 random + statistics。
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass


@dataclass
class Verdict:
    accept: bool
    net: float
    ci: tuple
    n_improved: int
    n_regressed: int
    reason: str


def held_out_paired_gate(base, cand, alpha=0.05, reg_cap=0, eps=1e-9,
                         n_boot=3000, seed=0, material=0.0) -> Verdict:
    """配对 held-out 显著性检验 — 演化接纳门（Recuris held_out_paired_gate 移植）。

    ``base`` / ``cand`` 映射 item id → 该 item 的 per-seed 分数列表。
    二进制基准给 0/1，密集奖励基准给 [0,1] 浮点；估计量对两者连续，一份实现通用。

    逐 item 取种子均值，再取候选与基线之差 ``d_i``。bootstrap **重采样 items**
    （不是 trials）：同一 item 内的 trials 不独立，重采样 trials 会报出比证据
    支撑的更窄区间。

    ACCEPT iff 区间在改善侧排除 0 且回归 item 数 ≤ ``reg_cap``。

    ``material``：单 item 差值需超过的幅度才计入改善/回归。默认 0.0 落到
    ``eps``，二进制调用方不受影响；密集奖励下抬高它，否则 0.97→0.95 这类噪声
    会计入 ``n_dn``，而 ``reg_cap`` 是硬拒绝条件，把噪声计入会让门变成掷硬币。
    """
    items = sorted(set(base) & set(cand))
    diffs = [statistics.mean(cand[i]) - statistics.mean(base[i]) for i in items]
    if not diffs:
        return Verdict(False, 0.0, (0.0, 0.0), 0, 0, "no comparable items")
    net = statistics.mean(diffs)
    floor = max(eps, material)
    n_up = sum(1 for x in diffs if x > floor)
    n_dn = sum(1 for x in diffs if x < -floor)
    rng = random.Random(seed)
    boots = sorted(statistics.mean([diffs[rng.randrange(len(diffs))] for _ in diffs])
                   for _ in range(n_boot))
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    accept = (lo > 0) and (n_dn <= reg_cap)
    if accept:
        reason = "net improvement, CI excludes 0"
    elif lo <= 0:
        reason = "CI includes 0: not significant"
    else:
        reason = f"{n_dn} regressed items exceeds the cap of {reg_cap}"
    return Verdict(accept, round(net, 4), (round(lo, 4), round(hi, 4)), n_up, n_dn, reason)
