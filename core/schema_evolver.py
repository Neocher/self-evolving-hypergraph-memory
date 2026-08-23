"""Schema Evolver — 实体属性/关系演化引擎（纯函数，无 I/O）。分区计票 + 阈值固化 + 迟滞防抖动。"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

T_EMERGE = 0.35
T_SOLIDIFY = 0.60
HYST = 0.15
CAP = 5
MIN_PARTITIONS = 2


def blake3_hex(data: str) -> str:
    """blake3 前 8 hex（Python 3.11 无内置 blake3，用 sha256 前 8 模拟——保持 8 hex 契约）。"""
    return hashlib.sha256(data.encode()).hexdigest()[:8]


class Action:
    IGNORE = "IGNORE"
    EMERGE = "EMERGE"
    SOLIDIFY = "SOLIDIFY"
    STRENGTHEN = "STRENGTHEN"
    CORRECT = "CORRECT"


@dataclass(slots=True)
class AttrStat:
    attr_name: str
    value: str
    value_blake3: str
    votes: dict[str, int] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(slots=True)
class ExtractedRelation:
    src_entity_id: str
    dst_entity_id: str
    predicate: str
    partition: str
    evidence_episode_id: str
    evidence_span: str
    local_conf: float = 0.5


def confidence(votes: Mapping[str, int], weights: Mapping[str, float] | None = None) -> float:
    """§5.1 分区计票公式: 0.6*diversity + 0.4*strength。单分区封顶 CAP=5，需 >=MIN_PARTITIONS 独立分区。"""
    weights = weights or {}
    if not votes:
        return 0.0
    sat = {p: min(n, CAP) / CAP for p, n in votes.items()}
    # 负权重/合计 ≤ 0 → 回退等权（防负 confidence 契约缺陷）
    if any(w < 0 for w in weights.values()):
        weights = {}
    wsum = sum(weights.get(p, 1.0) for p in sat) or 1.0
    strength = sum(weights.get(p, 1.0) * s for p, s in sat.items()) / wsum
    diversity = min(1.0, len(sat) / MIN_PARTITIONS)
    return round(0.6 * diversity + 0.4 * strength, 4)


def accumulate_votes(sidecar: dict, extracted: Sequence[Any]) -> dict:
    """纯函数：candidate 按 value 聚合 + blake3 证据键去重 → 分区累票 → 重算 confidence。"""
    import copy
    out = copy.deepcopy(sidecar)
    for ex in extracted:
        attr = ex.attr_name
        candidates = out.setdefault(attr, {}).setdefault("candidates", {})
        # candidate 按 value 聚合（同一 value 的所有证据合并计票）
        vkey = blake3_hex(ex.attr_value)
        cand = candidates.get(vkey)
        if cand is None:
            cand = candidates[vkey] = {
                "value": ex.attr_value, "votes": {}, "evidence": [], "conf": 0.0,
                "_seen": [],
            }
        # 证据键去重（同一证据重放不计二次票；含 partition，防同 episode 同 value
        # 不同分区规则命中只计一票）
        part = ex.partition
        ekey = blake3_hex(f"{ex.evidence_episode_id}:{attr}:{ex.attr_value}:{part}")
        if ekey in cand["_seen"]:
            continue
        cand["_seen"].append(ekey)
        cand["votes"][part] = cand["votes"].get(part, 0) + 1
        if ex.evidence_episode_id not in cand["evidence"]:
            cand["evidence"].append(ex.evidence_episode_id)
        cand["conf"] = confidence(cand["votes"])
    return out


def decide(stat: AttrStat, solidified: dict | None, *, t_solidify: float = T_SOLIDIFY, hyst: float = HYST) -> str:
    """§5.2 状态机：IGNORE/EMERGE/SOLIDIFY/STRENGTHEN/CORRECT。"""
    if stat.confidence < T_EMERGE:
        return Action.IGNORE
    if not solidified or not solidified.get("active"):
        return Action.SOLIDIFY if stat.confidence >= t_solidify else Action.EMERGE
    if solidified.get("value_blake3") == stat.value_blake3:
        return Action.STRENGTHEN if stat.confidence > solidified.get("conf", 0.0) else Action.IGNORE
    # 冲突值：需领先迟滞带才修正
    if stat.confidence >= solidified.get("conf", 0.0) + hyst:
        return Action.CORRECT
    return Action.IGNORE
