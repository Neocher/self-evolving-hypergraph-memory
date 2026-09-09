"""R8 E1' 状态时点裁决 — core/state_projection.py G1 oracle 单测 (TDD 先行).

本体论验收句: 题问时点该 (实体,槽) 的值由什么决定?
    → 由 (值,生效时间) 集合的确定性 latest-wins 裁决决定 → 过。

G1 硬性覆盖: ① latest-wins ② 历史插入不改 latest ③ session_ts 全早 → N/A
④ 同秒 tie-break=ep_id 序 ⑤ ts==session_ts 含边界 ⑥ 否定行裁决
⑦ 真日历跨日 (datetime 入参注入, 禁 mock 裁决逻辑)。
"""
import datetime

import pytest

from core.state_projection import resolve, timeline

_E = "u_amy"
_A = "city"


def _fact(value, ts, ep_id, *, attr=_A, entity=_E, negation=False):
    row = {"entity": entity, "attr": attr, "value": value, "ts": ts, "ep_id": ep_id}
    if negation:
        row["negation"] = True
    return row


# ── ① latest-wins ─────────────────────────────────────────────────────────

def test_latest_wins_returns_newest_visible_value():
    """同 (entity,attr) 多版本: ts 最大者胜出, 返回其 value/ts/ep_id。"""
    facts = [
        _fact("NYC", 10, 1),
        _fact("SF", 20, 2),
    ]
    assert resolve(facts, _E, _A, 30) == {"value": "SF", "ts": 20, "ep_id": 2}


def test_latest_wins_only_within_attr_and_entity():
    """裁决按 (entity, attr) 隔离, 其它实体/属性的事实不干扰。"""
    facts = [
        _fact("NYC", 50, 1),
        _fact("SF", 60, 2),
        _fact("seoul", 70, 3, entity="u_bob", attr="city"),
        _fact("42", 80, 4, attr="age"),
    ]
    assert resolve(facts, _E, _A, 100) == {"value": "SF", "ts": 60, "ep_id": 2}


# ── ② 历史插入不改 latest ─────────────────────────────────────────────────

def test_backfill_old_fact_does_not_change_latest():
    """历史(更早 ts)事实后补入库, 不得翻转当前 latest 裁决结果。"""
    facts = [_fact("NYC", 10, 1), _fact("SF", 20, 2)]
    before = resolve(facts, _E, _A, 50)
    older = _fact("old_town", 1, 0)
    after = resolve(facts + [older], _E, _A, 50)
    assert after == before == {"value": "SF", "ts": 20, "ep_id": 2}


# ── ③ session_ts 全早 → N/A ───────────────────────────────────────────────

def test_session_ts_before_all_facts_returns_none():
    """session_ts 早于全部事实 ts → 无可见版本, 返回 None (N/A)。"""
    facts = [_fact("NYC", 100, 1)]
    assert resolve(facts, _E, _A, 99) is None


# ── ④ 同秒 tie-break=ep_id 序 ─────────────────────────────────────────────

def test_same_ts_tie_break_prefers_larger_ep_id():
    """ts 相同 → 按 ep_id 序裁决: 后到 episode (更大 ep_id) 胜出。"""
    facts = [
        _fact("dog", 50, 3, attr="pet"),
        _fact("cat", 50, 7, attr="pet"),
    ]
    assert resolve(facts, _E, "pet", 50) == {"value": "cat", "ts": 50, "ep_id": 7}


def test_same_ts_tie_break_in_timeline_keeps_ep_ascending():
    """timeline 同秒内按 ep_id 升序输出, 保证版本链与 episode 序一致。"""
    facts = [
        _fact("dog", 50, 7, attr="pet"),
        _fact("cat", 50, 3, attr="pet"),
        _fact("bird", 50, 5, attr="pet"),
    ]
    values = [s["value"] for s in timeline(facts, _E, "pet")]
    assert values == ["cat", "bird", "dog"]


# ── ⑤ ts==session_ts 含边界 ───────────────────────────────────────────────

def test_boundary_ts_equal_session_ts_is_visible():
    """ts == session_ts 的行属于可见版本 (含边界, latest-wins 用 <=)。"""
    facts = [_fact("NYC", 30, 1)]
    assert resolve(facts, _E, _A, 30) == {"value": "NYC", "ts": 30, "ep_id": 1}


# ── ⑥ 否定行裁决 ──────────────────────────────────────────────────────────

def test_negation_as_latest_returns_none_never_leaks_value():
    """最新可见行为否定行 (negation=True) → 返回 None, 被否值不外泄。"""
    facts = [
        _fact("dog", 10, 1, attr="pet"),
        _fact("dog", 20, 2, attr="pet", negation=True),
    ]
    assert resolve(facts, _E, "pet", 30) is None


def test_negation_can_be_superseded_by_later_positive():
    """否定行之后的更新肯定行使属性恢复取值。"""
    facts = [
        _fact("dog", 10, 1, attr="pet"),
        _fact("dog", 20, 2, attr="pet", negation=True),
        _fact("cat", 30, 3, attr="pet"),
    ]
    assert resolve(facts, _E, "pet", 40) == {"value": "cat", "ts": 30, "ep_id": 3}


def test_negation_at_session_boundary_is_visible():
    """否定行恰在 session_ts 边界 → 同样参与裁决并裁决为 N/A。"""
    facts = [
        _fact("dog", 20, 1, attr="pet"),
        _fact("dog", 30, 2, attr="pet", negation=True),
    ]
    assert resolve(facts, _E, "pet", 30) is None


# ── ⑦ 真日历跨日 (datetime 注入, 非 mock) ─────────────────────────────────

def test_cross_day_real_calendar_resolution():
    """真日历 datetime 跨日: 新日期的后写值胜出, 边界含入, 全早为 N/A。"""
    t_old = datetime.datetime(2023, 12, 31, 23, 59, 59)
    t_new = datetime.datetime(2024, 1, 1, 0, 0, 1)
    facts = [
        _fact("2023_city", t_old, 1),
        _fact("2024_city", t_new, 2),
    ]
    # session 在 2024 首秒后 → 最新为跨日后值
    assert resolve(facts, _E, _A, datetime.datetime(2024, 1, 1, 0, 0, 30)) == {
        "value": "2024_city", "ts": t_new, "ep_id": 2,
    }
    # session == 旧行 ts (含边界) → 旧值可见
    assert resolve(facts, _E, _A, t_old) == {
        "value": "2023_city", "ts": t_old, "ep_id": 1,
    }
    # session 全早于任何事实 → N/A
    assert resolve(facts, _E, _A, datetime.datetime(2023, 12, 31, 0, 0, 0)) is None


# ── timeline 全版本链 ──────────────────────────────────────────────────────

def test_timeline_returns_full_chronological_chain():
    """timeline 返回按 (ts asc, ep_id asc) 的全版本链。"""
    facts = [
        _fact("NYC", 10, 1),
        _fact("SF", 20, 2),
        _fact("LA", 30, 3),
    ]
    states = timeline(facts, _E, _A)
    assert [s["value"] for s in states] == ["NYC", "SF", "LA"]
    assert [s["ts"] for s in states] == [10, 20, 30]
    assert [s["ep_id"] for s in states] == [1, 2, 3]


def test_timeline_includes_negation_row_without_leaking_value():
    """timeline 全版本链含否定行 (value=None, negation=True), 被否值不出现。"""
    facts = [
        _fact("dog", 10, 1, attr="pet"),
        _fact("dog", 20, 2, attr="pet", negation=True),
    ]
    states = timeline(facts, _E, "pet")
    assert len(states) == 2
    assert states[1]["value"] is None
    assert states[1]["negation"] is True
    assert all("dog" != s["value"] for s in states)


def test_timeline_unknown_entity_attr_returns_empty():
    """无匹配 (entity,attr) → timeline 空链, resolve None。"""
    facts = [_fact("NYC", 10, 1)]
    assert timeline(facts, "nobody", "nothing") == []
    assert resolve(facts, "nobody", "nothing", 99) is None
