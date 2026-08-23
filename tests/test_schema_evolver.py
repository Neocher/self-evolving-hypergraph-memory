"""Test Schema Evolver — 置信度公式 + 状态机 + 幂等 + 聚合。"""
import sys
sys.path.insert(0, "/home/admin/shm")
import pytest
from core.schema_evolver import (
    confidence, accumulate_votes, decide, Action, AttrStat, T_EMERGE, T_SOLIDIFY, HYST,
)
from core.attribute_extractor import ExtractedAttribute


def _ea(attr, value, part, ep, eid="e1"):
    return ExtractedAttribute(eid, attr, value, part, ep, "s", 0.5)


def test_confidence_formula():
    # §5.1 四例
    assert abs(confidence({"p1": 1}) - 0.38) < 0.01
    assert abs(confidence({"p1": 1, "p2": 1}) - 0.68) < 0.01
    assert abs(confidence({"p1": 5}) - 0.70) < 0.01
    assert abs(confidence({"p1": 5, "p2": 5}) - 1.0) < 0.01


def test_decide_state_machine():
    assert decide(AttrStat("t", "CEO", "x", {"a": 1}, [], 0.38), None) == Action.EMERGE
    assert decide(AttrStat("t", "CEO", "x", {"a": 5, "b": 5}, [], 1.0), None) == Action.SOLIDIFY
    solid = {"value": "CEO", "value_blake3": "x", "conf": 0.7, "active": True}
    assert decide(AttrStat("t", "CEO", "x", {"a": 5, "b": 5}, [], 1.0), solid) == Action.STRENGTHEN
    assert decide(AttrStat("t", "CTO", "y", {"a": 5, "b": 5, "c": 5}, [], 0.90), solid) == Action.CORRECT
    assert decide(AttrStat("t", "CTO", "y", {"a": 2, "b": 2}, [], 0.74), solid) == Action.IGNORE
    assert decide(AttrStat("t", "CEO", "x", {"a": 1}, [], 0.30), None) == Action.IGNORE


def test_accumulate_idempotent():
    sidecar = accumulate_votes({}, [_ea("title", "CEO", "p1", "ep1")])
    sidecar2 = accumulate_votes(sidecar, [_ea("title", "CEO", "p1", "ep1")])
    cand = list(sidecar2["title"]["candidates"].values())[0]
    assert cand["votes"]["p1"] == 1
    assert abs(cand["conf"] - 0.38) < 0.01


def test_accumulate_aggregates_same_value():
    sidecar = accumulate_votes({}, [_ea("title", "CEO", "p1", "ep1")])
    sidecar2 = accumulate_votes(sidecar, [_ea("title", "CEO", "p1", "ep2")])
    cands = list(sidecar2["title"]["candidates"].values())
    assert len(cands) == 1, cands  # 同一 value 聚合为一个 candidate
    cand = cands[0]
    assert cand["votes"]["p1"] == 2
    assert len(cand["evidence"]) == 2
    assert abs(cand["conf"] - 0.46) < 0.01


def test_accumulate_multi_partition():
    sidecar = accumulate_votes({}, [_ea("title", "CEO", "p1", "ep1")])
    sidecar2 = accumulate_votes(sidecar, [_ea("title", "CEO", "p2", "ep2")])
    cand = list(sidecar2["title"]["candidates"].values())[0]
    assert cand["votes"] == {"p1": 1, "p2": 1}
    assert abs(cand["conf"] - 0.68) < 0.01
