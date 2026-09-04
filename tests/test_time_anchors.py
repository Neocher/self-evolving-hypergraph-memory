"""达摩院 P0-b 确定性时间层 — time_anchors 模块单元测试 (真日历 oracle + 双形态 + AC3).

设计依据: shm-round3-technical-paradigm-priorities.md §3 (P0-b; R4 已证证据在场
14B 仍不做日历算术 — cat2 oracle 61.2% < 生产 65.0%; R2c prompt 教算术 -8.3pp →
算术下沉确定性层) + M5 语料 (~40 种模式) + M6 (299 题 186 题证据含相对词)。

AC1 真日历 oracle (2023-07-23 是周日 → 'last Friday' = 21 日; 周一起始/周末/月底
语义) + 语料覆盖率 + AC3 原文逐字不变 + 双形态渲染 (自然相对短语 + 绝对区间) +
双路径呈现 (组织段 [TIME ANCHORS] / round2 平铺内联)。
"""
import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import time_anchors as ta  # noqa: E402


def _d(y, m, dd):
    return datetime.date(y, m, dd)


# ── AC1: 真日历 oracle ─────────────────────────────────────────────────────

def test_calendar_oracle_last_friday():
    """2023-07-23 是周日 → 'last Friday' 严格早于锚 = 2023-07-21 (周五)。"""
    sunday = _d(2023, 7, 23)
    assert sunday.weekday() == 6  # 真日历前置条件: 周日
    assert ta.last_weekday(sunday, 4) == _d(2023, 7, 21)
    assert ta.last_weekday(sunday, 4).weekday() == 4  # 周五


def test_monday_start_week_semantics():
    """周一起始: 2023-07-23 (周日) 所在自然周 [7/17 周一, 7/23 周日]; last week = 前一周。"""
    sunday = _d(2023, 7, 23)
    mon, sun = ta.week_interval(sunday)
    assert mon.weekday() == 0 and mon == _d(2023, 7, 17)
    assert sun == _d(2023, 7, 23)
    # last week = 上一自然周 [10..16] (消息日周日, 上上周四说 last week 不指本周)
    lw = ta._shift_span_week(sunday, -1)
    assert lw == (_d(2023, 7, 10), _d(2023, 7, 16))


def test_weekend_semantics_two_weekends_ago():
    """周末=周六+周日块; 2023-07-17 (周一) last weekend = 7/15-16, two weekends ago = 7/8-9
    (与 R2c V2 few-shot gold 'Two weekends before 17 July 2023 (July 8-9)' 一致)。"""
    monday = _d(2023, 7, 17)
    assert monday.weekday() == 0
    assert ta.last_weekend(monday) == (_d(2023, 7, 15), _d(2023, 7, 16))
    assert ta._weekends_ago_span(monday, 2) == (_d(2023, 7, 8), _d(2023, 7, 9))
    assert ta._weekends_ago_span(monday, 1) == ta.last_weekend(monday)


def test_month_end_semantics():
    """月底: 2023-01-31 last month = 2022-12 全月; next month = 2023-02 全月 (28 天); 闰年 2 月 29 天。"""
    jan31 = _d(2023, 1, 31)
    assert ta.shift_month(jan31, -1) == (_d(2022, 12, 1), _d(2022, 12, 31))
    assert ta.shift_month(jan31, +1) == (_d(2023, 2, 1), _d(2023, 2, 28))
    assert ta.shift_month(_d(2024, 1, 31), +1) == (_d(2024, 2, 1), _d(2024, 2, 29))  # 闰年


def test_week_before_date_span():
    """'the week before 9 June 2023' → 9 June 所在周 (周一始) 前一整周 = 29 May to 4 June
    (R2c V2 few-shot gold 口径)。"""
    nine_june = _d(2023, 6, 9)
    assert ta.week_interval(nine_june) == (_d(2023, 6, 5), _d(2023, 6, 11))
    # 通过完整消息渲染断言 (无需构造 re.Match)
    msg = "[date: 9 June, 2023] [X] my school event the week before 9 June 2023 was great."
    ann = ta.inline_annotation(msg)
    assert ann is not None and "29 May 2023 to 4 June 2023" in ann


def test_last_weekday_anchor_when_anchor_is_target():
    """锚本身就是目标周几 → 前推 7 天 (周五说 last Friday = 上一周五, 非当天)。"""
    friday = _d(2023, 7, 21)
    assert ta.last_weekday(friday, 4) == _d(2023, 7, 14)


# ── 双形态渲染 + 绝对区间格式 ──────────────────────────────────────────────

def test_render_absolute_single_vs_range():
    assert ta.render_absolute((_d(2023, 5, 7), _d(2023, 5, 7))) == "7 May 2023"
    assert ta.render_absolute((_d(2023, 5, 29), _d(2023, 6, 4))) == "29 May 2023 to 4 June 2023"


def test_inline_annotation_two_forms_present():
    """内联注解同时含自然相对短语 (原文) 与精确绝对区间 (形态一+形态二)。"""
    msg = "[date: 2:31 pm on 17 July, 2023] [Melanie] we went camping with my fam two weekends ago."
    ann = ta.inline_annotation(msg)
    assert ann is not None
    assert "two weekends ago" in ann            # 形态一: 自然相对短语 (原文)
    assert "8 July 2023 to 9 July 2023" in ann  # 形态二: 精确绝对区间


def test_no_anchor_no_annotation():
    """无 [date:] 锚 (块摘要/旧裸文本) → 不注解 (无法做日历算术, 不猜)。"""
    assert ta.inline_annotation("plain text without date prefix yesterday") is None
    assert ta.find_anchors("no anchor here") == []


def test_fuzzy_words_not_rendered_as_absolute():
    """模糊量词 (a few days ago / the other day / recently) 不渲染绝对区间 —
    无法给出精确锚点, 渲染会误导 reader (识别但不虚构)。"""
    for body in ("a few days ago we talked", "the other day I saw him", "recently started"):
        msg = f"[date: 10 May, 2023] [X] {body}"
        assert ta.inline_annotation(msg) is None, body
        assert ta.find_anchors(msg) == [], body


# ── AC3: 原文逐字不变 ──────────────────────────────────────────────────────

def test_message_text_verbatim_in_both_paths():
    """组织段 [TIME ANCHORS] 与平铺内联都不得改动 raw 消息原文 (AC3)。

    注解是附加行/附加段; 消息行本身逐字保留 (判卷近重复敏感, F5 + R2c 教训)。"""
    org = ("[ENTITY: X]\n- fact\n"
           "[DIRECT EVIDENCE]\n"
           "[1] [date: 1:56 pm on 8 May, 2023] [Caroline] I went to a support group yesterday and it was so powerful.\n"
           "[2] plain block summary without anchor")
    out_org = ta.append_anchor_block(org)
    assert "[1] [date: 1:56 pm on 8 May, 2023] [Caroline] I went to a support group yesterday and it was so powerful." in out_org
    assert "[TIME ANCHORS" in out_org and "[1] yesterday -> 7 May 2023" in out_org
    assert "[2] plain block summary without anchor" in out_org  # 无锚消息原样保留

    flat = ("[MEMORY BLOCK 1] summary\n"
            "[1] [date: 9:12 am on 3 March, 2023] [Alex] I finished my screenplay last night; the first one is my favorite.")
    out_flat = "\n".join(ta.inline_numbered_lines(flat.splitlines()))
    assert "[1] [date: 9:12 am on 3 March, 2023] [Alex] I finished my screenplay last night; the first one is my favorite." in out_flat
    assert "[time: last night -> 2 March 2023]" in out_flat
    assert out_flat.count("[date: 9:12 am on 3 March, 2023]") == 1  # 原文不重复/不改写


def test_find_anchors_does_not_mutate_input():
    """find_anchors 纯函数: 输入字符串逐字节不变 (AC3 前置)。"""
    msg = "[date: 10 May, 2023] [X] we met yesterday and planned last week."
    before = msg
    ta.find_anchors(msg)
    assert msg == before


# ── 覆盖率 (语料口径, M6 参考) ─────────────────────────────────────────────

_REAL_STYLE_MESSAGES = [
    "[date: 1:56 pm on 8 May, 2023] [Caroline] I went to a LGBTQ support group yesterday and it was so powerful.",
    "[date: 1:14 pm on 25 May, 2023] [Melanie] We're thinking about going camping next month.",
    "[date: 7:55 pm on 9 June, 2023] [Caroline] I wanted to tell you about my school event last week.",
    "[date: 2:31 pm on 17 July, 2023] [Melanie] we went camping with my fam two weekends ago.",
    "[date: 2:31 pm on 17 July, 2023] [Caroline] We had a great time at the LGBT pride event last month.",
    "[date: 9:12 am on 3 March, 2023] [Alex] I finally finished writing my second screenplay last night.",
    "[date: 23 August, 2023] [Caroline] Guess what I did this week? I applied to adoption.",
    "[date: 14 August, 2023] [Melanie] Last night was amazing! We celebrated my daughter's birthday.",
    "[date: 31 July, 2023] [John] Last Sunday we had our first call-out, and it was intense.",
    "[date: 10 May, 2023] [X] plain message with no relative expression here.",
]


def test_corpus_coverage_stats_on_real_style_messages():
    """真实形态语料上的覆盖率统计: 识别含相对词消息 + 可精确解析消息计数 (M6 口径)。"""
    st = ta.corpus_coverage(_REAL_STYLE_MESSAGES)
    assert st["messages"] == len(_REAL_STYLE_MESSAGES)
    # 9/10 含相对词 (最后一条无), 全部 9 条有 [date:] 锚可精确解析 (含 last sunday)
    assert st["with_relative"] == 9
    assert st["with_exact_anchor"] == 9
    assert st["occurrences"] >= 9
    assert st["exact_occurrences"] == st["occurrences"] - st["fuzzy_occurrences"]


def test_top_corpus_patterns_resolvable():
    """M5 高频词 (last week/yesterday/last friday/next month/last weekend/last month/
    last night/this week) 每条至少能解析出一个精确绝对区间 — 确定性层覆盖高频即可。"""
    cases = {
        "last week": ("8 May, 2023", "1 May 2023 to 7 May 2023"),
        "yesterday": ("8 May, 2023", "7 May 2023"),
        "last friday": ("23 July, 2023", "21 July 2023"),
        "next month": ("3 March, 2023", "1 April 2023 to 30 April 2023"),
        "last weekend": ("17 July, 2023", "15 July 2023 to 16 July 2023"),
        "last month": ("31 January, 2023", "1 December 2022 to 31 December 2022"),
        "last night": ("3 March, 2023", "2 March 2023"),
        "this week": ("23 July, 2023", "17 July 2023 to 23 July 2023"),
    }
    for phrase, (anchor, expect) in cases.items():
        msg = f"[date: {anchor}] [X] we did {phrase} stuff."
        ann = ta.inline_annotation(msg)
        assert ann is not None, phrase
        assert expect in ann, (phrase, ann)
