"""P3 实例去重与计数 — core/instance_dedup.py G1~G9 oracle 单测 (TDD 先行).

本体论验收句: 两次提及是同一实例吗?
    → 由「成员规范化 + 实例去重 + 按周分桶 + 同实例合并」确定性裁决 → 过。

覆盖: G1~G9 内联 oracle + dict 形态 + 生成器输入 + window_days 边界 (7/8 天)
+ 组内升序 + 空输入 + unparseable ts + 桶内去重。
"""
from core.instance_dedup import (
    distinct_count,
    distinct_members,
    members_by_week,
    merge_same_instance,
    normalize_member,
)


class F:
    """对齐规格内联 oracle 的 fact 形状: .value / .ts (可选)。"""

    def __init__(self, value, ts=None):
        self.value = value
        self.ts = ts


def _value_of(f):
    """对象属性或 dict 键取值 (对齐被测模块的 duck-typing 兼容)。"""
    return f["value"] if isinstance(f, dict) else f.value


def _group_values(groups):
    """merge_same_instance 的组列表 → 各组 value 列表 (便于断言)。"""
    return [[_value_of(f) for f in g] for g in groups]


# --- G1 / G9 规范化 ---

def test_normalize_strips_quotes_article_punct():
    assert normalize_member('"Charlotte\'s Web"') == "Charlotte's Web"
    assert normalize_member('the mansion.') == "mansion"
    assert normalize_member('“Becoming Nicole”') == "Becoming Nicole"


def test_normalize_collapses_whitespace():
    assert normalize_member("  the   mansion  ") == "mansion"


def test_normalize_article_case_insensitive():
    assert normalize_member("The Mansion") == "Mansion"
    assert normalize_member("A mansion") == "mansion"
    assert normalize_member("an Apple") == "Apple"


# --- G2 / G8 去重与计数 ---

def test_distinct_case_and_article_insensitive():
    assert distinct_members([F("mansion"), F("Mansion"), F("a mansion")]) == ["mansion"]


def test_distinct_preserves_first_seen_order():
    assert distinct_members([F("B"), F("a"), F("b")]) == ["B", "a"]


def test_distinct_count_generator():
    assert distinct_count(F(v) for v in ["a", "A", "b"]) == 2


# --- G5 空输入 ---

def test_empty_input():
    assert distinct_members([]) == []
    assert distinct_count([]) == 0
    assert members_by_week([]) == {}


# --- G3 / G6 / G7 周分桶 ---

def test_members_by_week_bucketing():
    w = members_by_week([
        F("Oregon", "2023-05-08T10:00"),
        F("Florida", "2023-05-09T10:00"),
        F("Canada", "2023-05-25T10:00"),
    ])
    assert sorted(w) == ["2023-W19", "2023-W21"]
    assert w["2023-W19"] == ["Oregon", "Florida"]


def test_members_by_week_epoch():
    assert list(members_by_week([F("x", 1683525360.0)]))[0].startswith("2023-W")


def test_members_by_week_none_unknown():
    assert members_by_week([F("x", None)]) == {"unknown": ["x"]}


def test_members_by_week_unparseable_unknown():
    assert members_by_week([F("x", "last summer")]) == {"unknown": ["x"]}


def test_members_by_week_dedup_within_bucket():
    w = members_by_week([F("mansion", "2023-05-08"), F("Mansion", "2023-05-08")])
    assert w == {"2023-W19": ["mansion"]}


# --- G4 / G7 同实例合并 ---

def test_merge_adjacent_within_window():
    assert len(merge_same_instance([F("trip", "2023-05-01"), F("trip", "2023-05-07")])) == 1
    assert len(merge_same_instance([F("trip", "2023-05-01"), F("trip", "2023-05-09")])) == 2


def test_merge_window_boundary_inclusive():
    # 恰好 7 天 → 合并 (≤); 8 天 → 分 (>)
    assert len(merge_same_instance([F("trip", "2023-05-01"), F("trip", "2023-05-08")])) == 1
    assert len(merge_same_instance([F("trip", "2023-05-01"), F("trip", "2023-05-09")])) == 2


def test_merge_none_each_own_group():
    assert len(merge_same_instance([F("x", None), F("x", None)])) == 2


def test_merge_group_sorted_by_ts():
    groups = merge_same_instance([
        F("trip", "2023-05-03"),
        F("trip", "2023-05-01"),
        F("trip", "2023-05-02"),
    ])
    assert len(groups) == 1
    assert [f.ts for f in groups[0]] == ["2023-05-01", "2023-05-02", "2023-05-03"]


def test_merge_member_case_article_insensitive():
    assert len(merge_same_instance([F("mansion", "2023-05-01"), F("a Mansion", "2023-05-02")])) == 1


def test_merge_generator_input():
    assert len(merge_same_instance(F("trip", ts) for ts in ["2023-05-01", "2023-05-09"])) == 2


# --- dict 形态 ---

def test_dict_form_facts():
    facts = [
        {"value": "mansion", "ts": "2023-05-08"},
        {"value": "Mansion", "ts": "2023-05-08"},
    ]
    assert distinct_members(facts) == ["mansion"]
    assert members_by_week(facts) == {"2023-W19": ["mansion"]}
    merged = merge_same_instance([
        {"value": "trip", "ts": "2023-05-01"},
        {"value": "trip", "ts": "2023-05-07"},
    ])
    assert len(merged) == 1
    assert _group_values(merged) == [["trip", "trip"]]
