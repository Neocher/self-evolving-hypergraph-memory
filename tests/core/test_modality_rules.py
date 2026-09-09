"""R8 E2 模态/归属一等化 — core/modality_rules.py G1 oracle 单测 (TDD 先行).

本体论验收句: 模态与主体是否被确定性标注?
    → 规则模块五值闭域标注 + speaker/recent_entity 归属规则 → 过。

G1 硬性覆盖: ① wish/aspire/would/hope → 不标 actual ② in_talks 与 wish 可分
(q0004 形状) ③ "had planned to" 完成时已执行 → actual ④ "if you want" 修辞
不标愿望 ⑤ 不该过滤的绝不过滤 (q0013/q0026/q0004 反例) ⑥ subject 双人
he/she 追踪 (speaker 输入)。
"""
import pytest

from core.modality_rules import MODALITIES, classify, extract_subject

# ── ① wish 类不标 actual ───────────────────────────────────────────────────

def test_wish_aspire_would_hope_not_actual():
    """wish/hope/would/aspire 驱动的句子 → 归 wish, 绝不标 actual。"""
    for snippet in (
        "I wish I had a guitar",
        "I hope to visit Paris this summer",
        "That would be really cool",
        "I aspire to become a doctor",
    ):
        label = classify(snippet)
        assert label in ("planned", "wish"), snippet
        assert label != "actual", snippet
    assert classify("I hope to visit Paris this summer") == "wish"
    assert classify("I aspire to become a doctor") == "wish"


# ── ② in_talks 与 wish 可分 (q0004 形状) ──────────────────────────────────

def test_in_talks_distinct_from_wish():
    """'in talks with X' → in_talks; 'X would be really cool' → wish; 二者可分。"""
    in_talks = classify("We are in talks with the potential sponsor about the event")
    wish = classify("The sponsorship would be really cool")
    assert in_talks == "in_talks"
    assert wish == "wish"
    assert in_talks != wish


# ── ③ 完成时已执行 → actual ───────────────────────────────────────────────

def test_had_planned_to_completed_becomes_actual():
    """'had planned to' (完成时, 已执行) → actual, 不误标 planned/wish。"""
    assert classify("I had planned to visit my grandmother last week") == "actual"


# ── ④ 修辞 want 不标愿望 ──────────────────────────────────────────────────

def test_rhetorical_if_you_want_not_wish():
    """'if you want' 是修辞条件, 不构成愿望 → 不标 wish (回落 actual)。"""
    assert classify("I can come over and help if you want") == "actual"
    assert classify("we can drive there if you want") != "wish"


# ── ⑤ 不该过滤的绝不过滤 (反例 oracle) ────────────────────────────────────

def test_plan_aspire_talks_gold_never_downgraded_to_actual():
    """q0013/q0026/q0004 反例: 计划/aspire/in_talks 金标不得被标 actual
    (减法过滤不得在 plan/aspire/want/potential 题面误删)。"""
    assert classify("we planned to get together three times") == "planned"
    assert classify("I aspire to be like him one day") == "wish"
    assert classify("we are in talks for the sponsorship") == "in_talks"


# ── 五值域闭合 ─────────────────────────────────────────────────────────────

def test_classify_closed_domain_five_values():
    """classify 输出恒落五值闭域, 且五值均可达。"""
    samples = {
        "I own a red car": "actual",
        "we are going to move next month": "planned",
        "I hope to get a promotion": "wish",
        "I think he lives in Berlin": "inferred",
        "I am in talks with a sponsor": "in_talks",
    }
    for text, expected in samples.items():
        got = classify(text)
        assert got in MODALITIES, text
        assert got == expected, (text, got)


# ── ⑥ subject 双人 he/she 追踪 ────────────────────────────────────────────

def test_subject_first_person_is_speaker():
    """第一人称 (I/my/we) → 归属 speaker。"""
    assert extract_subject("I bought a new car", "Alice", "Bob") == "Alice"
    assert extract_subject("my dog is brown", "Alice", "Bob") == "Alice"


def test_subject_third_person_tracks_recent_entity():
    """双人会话 he/she/they → 归属 recent_entity (非说话人)。"""
    assert extract_subject("He bought a car", "Alice", "Bob") == "Bob"
    assert extract_subject("She loves the shiny guitar", "Bob", "Alice") == "Alice"
    assert extract_subject("their new house is big", "Alice", "Bob") == "Bob"


def test_subject_named_mention_and_default():
    """具名 recent_entity 命中 → recent_entity; 无代词省略句默认归属 speaker。"""
    assert extract_subject("Bob bought the car", "Alice", "Bob") == "Bob"
    assert extract_subject("bought a car yesterday", "Alice", "Bob") == "Alice"
