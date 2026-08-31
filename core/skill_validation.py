"""技能候选验证模块 (Skill Candidate Validation) — Phase 3 记忆→技能闭环
========================================================================
把 Phase 2 失败驱动闭环与 Phase 1 Recuris 配对检验接到技能生成上：
「模型提议，算术裁决」——技能候选也必须通过配对 held-out 检验才被接纳。

  - base: 失败查询在**无技能注入**下的重放分数（RetrievalSnapshot.quality() 口径）
  - cand: 同一批查询在**技能知识注入**下（SKILL.md 正文进入检索上下文）的重放分数
  - held_out_paired_gate 判定：ACCEPT → 技能写入技能库；REJECT → 丢弃（不污染）

技能注入方式可插拔：
  - 默认策略 = **查询增强适配器**（_SkillAugmentedRouter）：把 SKILL.md 正文
    拼进查询文本后再调内层 QueryRouter.retrieve —— 技能知识作为增强检索上下文
    进入重放，不改写 QueryRouter 本体（纯只读调用）。
  - 换用其他策略：给 validate_skill_candidate 传自定义 ``injector``，
    形如 ``injector(router, skill_md) -> router_like``，返回对象需暴露
    ``retrieve(query, **kwargs)`` 与 ``config``（如把技能向量注入重放参数、
    把技能正文写入检索索引的增强字段等）。

纯逻辑 + 只读检索调用：不修改 QueryRouter（不写 config、不留残留状态）。
"""
from __future__ import annotations

import json
import logging
import os
import time

from retrieval.failure_eval import query_id
from retrieval.self_evolving import RetrievalSnapshot

logger = logging.getLogger(__name__)

# 技能验证所需的最少失败查询数（与 EvolutionGuard 统计门门槛一致）
_MIN_FAILURE_ITEMS = 12

# 仓库 data/failure_queries.json 默认路径（Phase 2 FailedQueryEval 持久化路径）
DEFAULT_FAILURE_QUERIES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "failure_queries.json",
)


# ═══════════════ 失败查询集提取 ═══════════════

def extract_failure_queries_from_file(path: str) -> list[dict]:
    """读取 data/failure_queries.json → 失败查询 meta 列表。

    兼容两种格式：
      - dict-of-dicts（Phase 2 persist_failed_queries 持久化格式）：
        {qid: {"query", "num_results", "avg_score", "quality", "source",
               "first_failed_at"}}
      - list：[{"query": ...}, ...]
    缺失 / 损坏 / 空 → 返回 []（warning 日志，不抛异常，调用方保持兼容）。
    """
    if not path or not os.path.isfile(path):
        logger.warning("Skill-Validation: failure queries file not found: %s", path)
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("Skill-Validation: cannot parse %s: %s", path, e)
        return []
    if isinstance(data, dict):
        items = [v for v in data.values() if isinstance(v, dict)]
    elif isinstance(data, list):
        items = [v for v in data if isinstance(v, dict)]
    else:
        items = []
    return items


# ═══════════════ 技能注入（可插拔，默认=查询增强适配器） ═══════════════

class _SkillAugmentedRouter:
    """查询增强注入适配器：把 SKILL.md 正文拼进查询文本再调内层 QueryRouter。

    - ``retrieve(query, *args, **kwargs)`` → 内层 ``retrieve(augmented, ...)``
    - ``config`` 属性代理到内层 router（与 Phase 2 重放代码的 cfg 访问兼容）
    """

    def __init__(self, router, skill_md: str):
        self._inner = router
        self._skill_md = skill_md

    @property
    def config(self):
        return getattr(self._inner, "config", None)

    def retrieve(self, query, *args, **kwargs):
        augmented = f"{query}\n\n[技能知识]\n{self._skill_md}"
        return self._inner.retrieve(augmented, *args, **kwargs)


def _default_injector(router, skill_md: str):
    """默认注入策略：查询增强适配器（技能正文进入检索的增强上下文）。"""
    return _SkillAugmentedRouter(router, skill_md)


# ═══════════════ 重放打分 ═══════════════

def _to_items(failure_queries: list[dict]) -> dict:
    """失败查询 meta 列表 → {qid: meta}（qid=query_id(query)，无 query 条目跳过）。"""
    items: dict = {}
    for meta in failure_queries or []:
        q = (meta or {}).get("query", "")
        if not q:
            continue
        items[query_id(q)] = meta
    return items


def _snapshot_from_result(query: str, results) -> RetrievalSnapshot:
    """把一次检索结果转成临时快照，复用 RetrievalSnapshot.quality() 口径。"""
    raw = results if isinstance(results, list) else results.get("results", [])
    scores = [r.get("score", 0.5) for r in raw[:10]] if raw else [0.0]
    contents = [r.get("content", "")[:100] for r in raw[:10]]
    return RetrievalSnapshot(
        timestamp=time.time(),
        query=query,
        params_before={},
        num_results=len(raw),
        top_scores=scores[:5],
        avg_score=sum(scores) / max(1, len(scores)),
        top_distinct=len(set(contents)),
        latency_ms=0.0,
        degraded=len(raw) == 0,
    )


def _replay_scores(items: dict, router, n_seeds: int) -> dict:
    """在给定 router 下重放失败查询集 → {qid: [per-seed quality 分数]}。

    只读检索调用（不改 config）；单次重放异常记 0 分（degraded 语义），不中断整批。
    """
    out: dict = {}
    for qid, meta in items.items():
        query = meta["query"] if isinstance(meta, dict) else str(meta)
        scores = []
        for _ in range(max(1, int(n_seeds))):
            try:
                res = router.retrieve(query)
            except Exception:
                res = []  # 重放失败 → degraded 语义（quality 0），不中断
            snap = _snapshot_from_result(query, res)
            scores.append(round(snap.quality(), 4))
        out[qid] = scores
    return out


# ═══════════════ 验证入口 ═══════════════

def validate_skill_candidate(skill_md: str, failure_queries: list[dict],
                             query_router, n_seeds: int = 3,
                             injector=None, **gate_kwargs):
    """技能候选 A/B 配对检验 — Recuris「模型提议，算术裁决」第三阶段落地。

    base = 失败查询在无技能注入下的重放分数；cand = 同一批查询在技能知识
    注入下（默认查询增强适配器）的重放分数；held_out_paired_gate 判定：
    ACCEPT → 技能可写入技能库；REJECT → 丢弃（不污染技能库）。

    Args:
        skill_md: SKILL.md 全文（frontmatter + 正文）
        failure_queries: 失败查询 meta 列表（data/failure_queries.json 条目）
        query_router: 只读重放目标 QueryRouter（不修改其 config）
        n_seeds: 每个查询重放次数（per-seed 分数），默认 3
        injector: 技能注入策略 ``injector(router, skill_md) -> router_like``；
                  默认 None → 查询增强适配器（技能正文拼进查询文本）
        **gate_kwargs: 透传给 held_out_paired_gate（alpha/reg_cap/...）

    Returns:
        Verdict | None：失败查询集 < 12 item 时返回 None（调用方跳过验证，
        保持兼容）。否则返回 held_out_paired_gate 的 Verdict。
    """
    items = _to_items(failure_queries)
    if len(items) < _MIN_FAILURE_ITEMS:
        logger.info("Skill-Validation: %d failure items < %d, skip gate",
                    len(items), _MIN_FAILURE_ITEMS)
        return None
    if injector is None:
        injector = _default_injector
    base = _replay_scores(items, query_router, n_seeds)
    cand = _replay_scores(items, injector(query_router, skill_md), n_seeds)
    # 惰性 import：validation_gate 为已发布稳定依赖，延迟取用便于测试探针
    # （与 self_evolving 对 failure_eval 的惰性 import 同一模式）
    from core.validation_gate import held_out_paired_gate
    verdict = held_out_paired_gate(base, cand, **gate_kwargs)
    logger.info(
        "Skill-Validation: gate accept=%s net=%.4f ci=%s n_up=%d n_dn=%d (%s)",
        verdict.accept, verdict.net, verdict.ci, verdict.n_improved,
        verdict.n_regressed, verdict.reason,
    )
    return verdict
