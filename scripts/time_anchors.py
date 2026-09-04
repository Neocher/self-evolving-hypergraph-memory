# -*- coding: utf-8 -*-
"""达摩院 P0-b 确定性时间层 — 零 LLM 相对时间词 → 绝对区间解析模块 (纯 stdlib).

设计依据: /home/user/taiji/agent_skills/damoyuan/outputs/shm-round3-technical-paradigm-
priorities.md §3 (P0-b 确定性时间层; R4 已证证据在场 14B 仍不做日历算术 — cat2 oracle
61.2% < 生产 65.0%; R2c 用 prompt 教算术失败 -8.3pp → 算术从 reader 移到确定性层,
与 semantica TemporalNormalizer 同一哲学) + M5 语料词频 (last week 108 · yesterday 67 ·
last friday 40 · next month 36 · last year 28 · last weekend 27 · last month 21 ·
last night 16 · ago 14 · the other day 13, ~40 种有界模式) + M6 (299 题中 186 题证据
含可解析相对词, 101 题 gold 为绝对日期/精确区间)。

本模块零 LLM 调用、无外部依赖 (re + datetime + calendar): 输入消息原文 + 消息日期锚
(消息 [date: ...] 前缀, 会话 date_time 已在评测 DB) → 纯日历算术 → 精确绝对区间。
两种形态并存供 reader 摘取: 自然相对短语 (原文逐字保留) + 精确绝对区间
("29 May 2023 to 4 June 2023")。

语义口径 (全部以真日历断言在 tests/test_time_anchors.py):
- 周起始: 周一 (ISO) — 'last week' = 锚所在周的前一自然周 [Mon, Sun];
- 周末: 周六+周日块 — 'last weekend' = 结束日 (周日) 严格早于锚的最近周末;
  'two weekends ago' = 再前一周末 (与 R2c V2 few-shot gold 对齐:
  锚 2023-07-17 周一 → 7/8-9 July 2023);
- 'last <weekday>' = 严格早于锚的最近目标周几 (锚本身是目标 → 前推 7 天);
  AC1 oracle: 锚 2023-07-23 (周日) → 'last Friday' = 2023-07-21;
- 月底: 自然月边界 (calendar.monthrange), 'last/next month' 取整月区间;
- 'the week before <D>' (D 为绝对日期) = D 所在周 (周一起始) 的前一整周
  (锚 2023-06-09 周五 → 29 May 2023 to 4 June 2023, R2c V2 few-shot 口径);
- 模糊量词 (a few/several/couple of days ago, the other day, recently) 不渲染
  绝对区间 — 无法给出精确锚点, 渲染会误导 reader (标注在 coverage 统计的
  '识别未精确解析' 类)。
"""
from __future__ import annotations

import calendar
import datetime
import re
from typing import Iterable, List, Optional, Tuple

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a": 1, "an": 1,
}
_FULL_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"]


# ── 日期解析 (消息 [date:] 锚 + 句内绝对日期) ───────────────────────────────

def _parse_month(token: str) -> Optional[int]:
    """月名/月名缩写 → 1-12; 无法识别 → None。"""
    return _MONTHS.get((token or "").strip().lower()[:3]) or _MONTHS.get((token or "").strip().lower())


def parse_absolute_date(text: str) -> Optional[datetime.date]:
    """句内绝对日期 → date; 支持官方形态 ('8 May, 2023' / '8th May, 2023' /
    'May 8, 2023') 与 ISO 'YYYY-MM-DD'; 无法解析 → None。不做无年份推断
    (锚年不可靠, 见 _p1_dt_to_epoch 同款保守口径)。"""
    s = (text or "").strip()
    if not s:
        return None
    try:
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
        if m:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = re.match(
            r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})$", s)
        if m:
            mon = _parse_month(m.group(2))
            if mon:
                return datetime.date(int(m.group(3)), mon, int(m.group(1)))
        m = re.match(
            r"^([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$", s)
        if m:
            mon = _parse_month(m.group(1))
            if mon:
                return datetime.date(int(m.group(3)), mon, int(m.group(2)))
    except (ValueError, OverflowError):
        return None
    return None


_DATE_IN_MSG = re.compile(
    r"(?<![\w-])"
    r"(?:(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9}),?\s+(\d{4})"
    r"|([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4}))"
    r"(?![\w-])", re.IGNORECASE)


def message_anchor_date(text: str) -> Optional[datetime.date]:
    """消息 [date: ...] 前缀 → date (日级锚; 时刻被丢弃 — 相对日词只需日历日)。

    date_time 官方形态 '1:56 pm on 8 May, 2023' / '8 May, 2023' /
    '8th May, 2023' / 'May 8, 2023' / ISO '2023-05-08'。无前缀/无法解析 → None
    (调用方跳过该消息注解, 不中断)。"""
    m = re.search(r"\[date:\s*([^\]]+)\]", (text or "")[:300])
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        mm = re.match(
            r"^(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(.+)$", raw, re.IGNORECASE)
        if mm:
            return parse_absolute_date(mm.group(4).strip())
    except Exception:
        pass
    return parse_absolute_date(raw)


def _msg_body(text: str) -> str:
    """去掉 [date: ...] 与 [speaker] 前缀 → 消息正文 (相对词扫描区)。"""
    s = re.sub(r"^\[date:[^\]]*\]\s*", "", (text or "").strip())
    s = re.sub(r"^\[[^\]]*\]\s*", "", s)
    return s


# ── 日历算术原语 (周一起始/周末/月边界) ─────────────────────────────────────

def monday_of(d: datetime.date) -> datetime.date:
    """d 所在周的周一 (ISO 周一起始)。"""
    return d - datetime.timedelta(days=d.weekday())


def week_interval(d: datetime.date) -> Tuple[datetime.date, datetime.date]:
    """d 所在自然周 [周一, 周日]。"""
    mon = monday_of(d)
    return mon, mon + datetime.timedelta(days=6)


def last_weekday(d: datetime.date, wd: int) -> datetime.date:
    """严格早于 d 的最近目标周几; d 本身是目标 → 前推 7 天。
    AC1 oracle: d=2023-07-23 (周日) → last Friday(wd=4): (6-4)%7=2 → 7/21。"""
    delta = (d.weekday() - wd) % 7
    return d - datetime.timedelta(days=delta or 7)


def next_weekday(d: datetime.date, wd: int) -> datetime.date:
    """严格晚于 d 的最近目标周几; d 本身是目标 → 后推 7 天。"""
    delta = (wd - d.weekday()) % 7
    return d + datetime.timedelta(days=delta or 7)


def _last_sunday(d: datetime.date) -> datetime.date:
    """严格早于 d 的最近周日 (last weekend 的结束日)。d 是周日 → 前推 7。"""
    delta = (d.weekday() + 1) % 7
    return d - datetime.timedelta(days=delta or 7)


def weekend_interval(d: datetime.date) -> Tuple[datetime.date, datetime.date]:
    """含 d 的周末 [周六, 周日]。"""
    sat = d - datetime.timedelta(days=(d.weekday() - 5) % 7)
    return sat, sat + datetime.timedelta(days=1)


def last_weekend(d: datetime.date) -> Tuple[datetime.date, datetime.date]:
    """结束日 (周日) 严格早于 d 的最近周末。d=7/17 周一 → 7/15-16 (V2 gold)。"""
    sun = _last_sunday(d)
    return sun - datetime.timedelta(days=1), sun


def month_interval(d: datetime.date) -> Tuple[datetime.date, datetime.date]:
    """d 所在自然月 [1 日, 月末] (calendar.monthrange 处理月底)。"""
    last = calendar.monthrange(d.year, d.month)[1]
    return datetime.date(d.year, d.month, 1), datetime.date(d.year, d.month, last)


def shift_month(d: datetime.date, delta: int) -> Tuple[datetime.date, datetime.date]:
    """d 所在月平移 delta 个自然月的整月区间 (月底正确, 如 1/31 +1 → Feb)。"""
    ym, y = (d.month - 1 + delta) % 12 + 1, d.year + (d.month - 1 + delta) // 12
    last = calendar.monthrange(y, ym)[1]
    return datetime.date(y, ym, 1), datetime.date(y, ym, last)


def shift_year(d: datetime.date, delta: int) -> Tuple[datetime.date, datetime.date]:
    """d 所在年平移 delta 个自然年的整年区间。"""
    return datetime.date(d.year + delta, 1, 1), datetime.date(d.year + delta, 12, 31)


def _shift_days(d: datetime.date, delta: int) -> Tuple[datetime.date, datetime.date]:
    x = d + datetime.timedelta(days=delta)
    return x, x


# ── 渲染 (两种形态: 原文相对短语 + 精确绝对区间) ───────────────────────────

def _fmt_day(x: datetime.date) -> str:
    return f"{x.day} {_FULL_MONTHS[x.month - 1]} {x.year}"


def render_absolute(span: Tuple[datetime.date, datetime.date]) -> str:
    """绝对区间渲染: 单日 → '7 May 2023'; 区间 → '29 May 2023 to 4 June 2023'。"""
    s, e = span
    if s == e:
        return _fmt_day(s)
    return f"{_fmt_day(s)} to {_fmt_day(e)}"


# ── 相对时间词模式表 (有序, 先长后短防误吞; ~40 种覆盖 M5 高频) ─────────────

# 每条: (识别正则, 解析函数(date, match) -> Optional[span], 短语重渲染)
# 短语重渲染 None → 保留原文逐字 (相对词原样是形态一)。

def _n_of(match: re.Match, group: str = "n") -> Optional[int]:
    g = (match.group(group) or "").lower()
    if g.isdigit():
        return int(g)
    return _NUM_WORDS.get(g)


_RE_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)
_RE_DAY_BEFORE_YEST = re.compile(r"\bthe\s+day\s+before\s+yesterday\b", re.IGNORECASE)
_RE_TODAY = re.compile(r"\btoday\b", re.IGNORECASE)
_RE_TONIGHT = re.compile(r"\btonight\b", re.IGNORECASE)
_RE_THIS_TOD = re.compile(r"\bthis\s+(morning|afternoon|evening)\b", re.IGNORECASE)
_RE_LAST_NIGHT = re.compile(r"\blast\s+night\b", re.IGNORECASE)
_RE_LAST_WEEK = re.compile(r"\blast\s+week\b", re.IGNORECASE)
_RE_THIS_WEEK = re.compile(r"\bthis\s+week\b", re.IGNORECASE)
_RE_NEXT_WEEK = re.compile(r"\bnext\s+week\b", re.IGNORECASE)
_RE_WEEK_BEFORE = re.compile(
    r"\bthe\s+week\s+before\s+"
    r"((?:\d{1,2})(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{4}"
    r"|[A-Za-z]{3,9}\s+(?:\d{1,2})(?:st|nd|rd|th)?,?\s+\d{4})\b", re.IGNORECASE)
_RE_WEEK_OF = re.compile(
    r"\bthe\s+week\s+of\s+"
    r"((?:\d{1,2})(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{4}"
    r"|[A-Za-z]{3,9}\s+(?:\d{1,2})(?:st|nd|rd|th)?,?\s+\d{4})\b", re.IGNORECASE)
_RE_LAST_WEEKEND = re.compile(r"\blast\s+weekend\b", re.IGNORECASE)
_RE_THIS_WEEKEND = re.compile(r"\bthis\s+weekend\b", re.IGNORECASE)
_RE_NEXT_WEEKEND = re.compile(r"\bnext\s+weekend\b", re.IGNORECASE)
_RE_N_WEEKENDS_AGO = re.compile(
    r"\b(?P<n>one|two|three|four|five|\d+)\s+weekends?\s+ago\b", re.IGNORECASE)
_RE_LAST_WD = re.compile(
    r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE)
_RE_NEXT_WD = re.compile(
    r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE)
_RE_LAST_MONTH = re.compile(r"\blast\s+month\b", re.IGNORECASE)
_RE_THIS_MONTH = re.compile(r"\bthis\s+month\b", re.IGNORECASE)
_RE_NEXT_MONTH = re.compile(r"\bnext\s+month\b", re.IGNORECASE)
_RE_LAST_YEAR = re.compile(r"\blast\s+year\b", re.IGNORECASE)
_RE_THIS_YEAR = re.compile(r"\bthis\s+year\b", re.IGNORECASE)
_RE_NEXT_YEAR = re.compile(r"\bnext\s+year\b", re.IGNORECASE)
_RE_N_UNIT_AGO = re.compile(
    r"\b(?P<n>one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|\d+)\s+(?P<u>day|week|month|year)s?\s+ago\b", re.IGNORECASE)
_RE_IN_N_UNIT = re.compile(
    r"\bin\s+(?P<n>a|an|one|two|three|four|five|\d+)\s+"
    r"(?P<u>day|week|month|year)s?\b", re.IGNORECASE)
# 模糊量词: 可识别但无法给出精确绝对区间 (标注统计用, 不渲染)
_RE_FUZZY_AGO = re.compile(
    r"\b(a\s+few|several|a\s+couple\s+of)\s+(day|week|month|year)s?\s+ago\b",
    re.IGNORECASE)
_RE_THE_OTHER_DAY = re.compile(r"\bthe\s+other\s+day\b", re.IGNORECASE)
_RE_RECENTLY = re.compile(r"\brecently\b", re.IGNORECASE)

_UNIT_DELTAS = {"day": "days", "week": "weeks", "month": "months", "year": "years"}


def _shift_units(d: datetime.date, n: int, unit: str, sign: int
                 ) -> Tuple[datetime.date, datetime.date]:
    """d 平移 n 个 unit (day/week/month/year) → 单日区间。"""
    if unit == "day":
        return _shift_days(d, sign * n)
    if unit == "week":
        return _shift_days(d, sign * n * 7)
    if unit == "month":
        return shift_month(d, sign * n)
    return shift_year(d, sign * n)


# 有序模式表: (名称, 正则, 解析器)。先长后短、先具体后一般 — 贪心按出现顺序,
# 重叠命中取表序靠前者 (collect_matches 保证)。
_PATTERNS = [
    ("day before yesterday", _RE_DAY_BEFORE_YEST,
     lambda d, m: _shift_days(d, -2)),
    ("yesterday", _RE_YESTERDAY, lambda d, m: _shift_days(d, -1)),
    ("today", _RE_TODAY, lambda d, m: _shift_days(d, 0)),
    ("tonight", _RE_TONIGHT, lambda d, m: _shift_days(d, 0)),
    ("this morning/afternoon/evening", _RE_THIS_TOD,
     lambda d, m: _shift_days(d, 0)),
    ("last night", _RE_LAST_NIGHT, lambda d, m: _shift_days(d, -1)),
    ("N weekends ago", _RE_N_WEEKENDS_AGO,
     lambda d, m: _weekends_ago_span(d, _n_of(m))),
    ("last weekend", _RE_LAST_WEEKEND, lambda d, m: last_weekend(d)),
    ("this weekend", _RE_THIS_WEEKEND,
     lambda d, m: (weekend_interval(d) if d.weekday() >= 5
                   else _upcoming_weekend(d))),
    ("next weekend", _RE_NEXT_WEEKEND, lambda d, m: _next_weekend_span(d)),
    ("the week before D", _RE_WEEK_BEFORE,
     lambda d, m: _week_before_span(d, m)),
    ("the week of D", _RE_WEEK_OF,
     lambda d, m: (week_interval(parse_absolute_date(m.group(1)))
                   if parse_absolute_date(m.group(1)) else None)),
    ("last week", _RE_LAST_WEEK,
     lambda d, m: _shift_span_week(d, -1)),
    ("this week", _RE_THIS_WEEK,
     lambda d, m: week_interval(d)),
    ("next week", _RE_NEXT_WEEK,
     lambda d, m: _shift_span_week(d, +1)),
    ("last <weekday>", _RE_LAST_WD,
     lambda d, m: _single(last_weekday(d, _WEEKDAYS[m.group(1).lower()]))),
    ("next <weekday>", _RE_NEXT_WD,
     lambda d, m: _single(next_weekday(d, _WEEKDAYS[m.group(1).lower()]))),
    ("last month", _RE_LAST_MONTH, lambda d, m: shift_month(d, -1)),
    ("this month", _RE_THIS_MONTH, lambda d, m: month_interval(d)),
    ("next month", _RE_NEXT_MONTH, lambda d, m: shift_month(d, +1)),
    ("last year", _RE_LAST_YEAR, lambda d, m: shift_year(d, -1)),
    ("this year", _RE_THIS_YEAR, lambda d, m: shift_year(d, 0)),
    ("next year", _RE_NEXT_YEAR, lambda d, m: shift_year(d, +1)),
    ("N unit ago", _RE_N_UNIT_AGO,
     lambda d, m: _shift_units(d, _n_of(m) or 0, m.group("u").lower(), -1)),
    ("in N unit", _RE_IN_N_UNIT,
     lambda d, m: _shift_units(d, _n_of(m) or 0, m.group("u").lower(), +1)),
    # 模糊词 (识别, 不渲染绝对区间 — 表尾兜底, 供 coverage 统计)
    ("fuzzy ago", _RE_FUZZY_AGO, lambda d, m: None),
    ("the other day", _RE_THE_OTHER_DAY, lambda d, m: None),
    ("recently", _RE_RECENTLY, lambda d, m: None),
]


def _single(x: datetime.date) -> Tuple[datetime.date, datetime.date]:
    return x, x


def _weekends_ago_span(d: datetime.date, n: Optional[int]
                       ) -> Optional[Tuple[datetime.date, datetime.date]]:
    if not n or n < 1:
        return None
    sun = _last_sunday(d) - datetime.timedelta(days=7 * (n - 1))
    return sun - datetime.timedelta(days=1), sun


def _upcoming_weekend(d: datetime.date) -> Tuple[datetime.date, datetime.date]:
    sat = d + datetime.timedelta(days=(5 - d.weekday()) % 7)
    return sat, sat + datetime.timedelta(days=1)


def _next_weekend_span(d: datetime.date) -> Tuple[datetime.date, datetime.date]:
    this_sat = d + datetime.timedelta(days=(5 - d.weekday()) % 7)
    sat = this_sat + datetime.timedelta(days=7)
    return sat, sat + datetime.timedelta(days=1)


def _shift_span_week(d: datetime.date, delta: int
                     ) -> Tuple[datetime.date, datetime.date]:
    mon = monday_of(d) + datetime.timedelta(days=7 * delta)
    return mon, mon + datetime.timedelta(days=6)


def _week_before_span(d: datetime.date, m: re.Match
                      ) -> Optional[Tuple[datetime.date, datetime.date]]:
    x = parse_absolute_date(m.group(1))
    if x is None:
        return None
    mon = monday_of(x) - datetime.timedelta(days=7)
    return mon, mon + datetime.timedelta(days=6)


# ── 对外 API ───────────────────────────────────────────────────────────────

class TimeAnchor:
    """一条解析出的时间锚注解: 原文相对短语 + 绝对区间 (双形态)。"""

    __slots__ = ("phrase", "start", "end", "pattern")

    def __init__(self, phrase: str, span: Tuple[datetime.date, datetime.date],
                 pattern: str):
        self.phrase = phrase          # 形态一: 原文相对短语 (逐字)
        self.start, self.end = span   # 精确绝对区间 (形态二)
        self.pattern = pattern

    @property
    def absolute(self) -> str:
        return render_absolute((self.start, self.end))

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<TimeAnchor {self.phrase!r} -> {self.absolute}>"


def _dedup(anchors: List[TimeAnchor]) -> List[TimeAnchor]:
    out, seen = [], set()
    for a in anchors:
        key = (a.phrase.lower(), a.start, a.end)
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def find_anchors(text: str, anchor_date: Optional[datetime.date] = None
                 ) -> List[TimeAnchor]:
    """消息正文 → 相对时间锚注解列表 (按原文出现顺序)。

    锚缺失 (无 [date:] 前缀 / 无法解析) → 返回 [] (调用方跳过该消息)。
    模糊词 (a few days ago / the other day / recently) 可识别但无法给出精确
    绝对区间 → 不返回 (保证每条注解的 absolute 都精确可摘取)。
    """
    if anchor_date is None:
        anchor_date = message_anchor_date(text)
    if anchor_date is None:
        return []
    body = _msg_body(text)
    if not body:
        return []
    out: List[TimeAnchor] = []
    for name, rx, fn in _PATTERNS:
        for m in rx.finditer(body):
            span = fn(anchor_date, m)
            if span is None:
                continue
            out.append(TimeAnchor(m.group(0).strip(), span, name))
    return _dedup(sorted(out, key=lambda a: body.lower().find(a.phrase.lower())))


def inline_annotation(text: str, anchor_date: Optional[datetime.date] = None
                      ) -> Optional[str]:
    """单条消息 → 内联注解行 '[time: <相对词> → <绝对区间>]' (多词分号连接)。

    无锚/无注解 → None (调用方原样保留消息行, 原文逐字不变)。"""
    anchors = find_anchors(text, anchor_date)
    if not anchors:
        return None
    joined = "; ".join(f"{a.phrase} -> {a.absolute}" for a in anchors)
    return f"[time: {joined}]"


def numbered_anchor_block(numbered_docs: Iterable[Tuple[int, str]],
                          max_docs: int = 30) -> str:
    """[TIME ANCHORS] 段 (组织路径 (a)): 引用 DIRECT EVIDENCE 消息编号。

    入参编号文档 [(N, content), ...] (N 与 DIRECT EVIDENCE 段 '[N]' 同号);
    只含可解析锚的消息; 无注解 → 返回 '' (不输出空段)。组织段文本只陈述事实。"""
    lines = []
    for n, content in numbered_docs:
        anchors = find_anchors(content)
        if not anchors:
            continue
        for a in anchors[:3]:  # 单条消息至多 3 条, 控 ctx 预算
            lines.append(f"[{n}] {a.phrase} -> {a.absolute}")
        if len(anchors) > 3:
            lines.append(f"[{n}] ... ({len(anchors) - 3} more)")
    if not lines:
        return ""
    return "[TIME ANCHORS (deterministic calendar resolution)]\n" + "\n".join(lines)


def split_message_line(line: str) -> Optional[Tuple[int, str]]:
    """ctx 编号消息行 '[N] <content>' → (N, content); 非编号消息行 → None。

    覆盖 DIRECT EVIDENCE 段与 round2 平铺/追加段的行形态 (两者同为
    '[<数字>] [date: ...] [speaker] text')。"""
    m = re.match(r"^\[(\d+)\]\s+(.*)$", line, re.DOTALL)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def inline_numbered_lines(lines: Iterable[str], max_docs: int = 20
                          ) -> List[str]:
    """round2 平铺路径 (b): ctx 行列表 → 注解后行列表 (原文逐字不变)。

    每条编号 raw 消息行 (形态 '[N] [date: ...] ...') 之后附加一行
    '[time: <相对词> -> <绝对区间>]' (只对前 max_docs 条, 控 ctx 预算);
    非编号行 (组织段/块摘要标题等) 原样保留。消息行本身零改动 — AC3 逐字不变。"""
    out: List[str] = []
    n = 0
    for ln in lines:
        out.append(ln)
        hit = split_message_line(ln)
        if hit is not None and n < max_docs:
            ann = inline_annotation(hit[1])
            if ann is not None:
                out.append(ann)
            n += 1
    return out


def append_anchor_block(ctx: str, max_docs: int = 30) -> str:
    """组织路径 (a): ctx 尾部追加 '[TIME ANCHORS]' 汇总段 (引用消息编号)。

    扫描 ctx 内全部编号 raw 消息行 (DIRECT EVIDENCE / ROUND2 SUPPLEMENTAL /
    平铺同形态), 只保留有精确锚的条目; 无注解 → 返回原 ctx (不输出空段)。
    注解段在 ctx 尾部, 不挤占 raw 证据行 — 失效条件 ④。组织段文本只陈述事实。"""
    items: List[Tuple[int, str]] = []
    for ln in ctx.splitlines():
        hit = split_message_line(ln)
        if hit is not None:
            items.append(hit)
    block = numbered_anchor_block(items, max_docs=max_docs)
    if not block:
        return ctx
    return ctx + "\n\n" + block


# ── 覆盖率统计 (真实语料, M6 口径) ─────────────────────────────────────────

def corpus_coverage(messages: Iterable[str]) -> dict:
    """真实消息语料上的相对词覆盖率 (供 AC1 报告/断言)。

    返回: 消息总数 / 含可识别相对词消息数 / 含可精确解析注解消息数 /
    识别词出现总次数 / 精确解析词次数 / 模糊词次数。识别 = 模式表命中
    (含模糊词); 精确解析 = 渲染出绝对区间的注解条数。"""
    total = n_rec = n_exact = n_word = n_exact_word = n_fuzzy = 0
    for text in messages:
        total += 1
        body = _msg_body(text).lower()
        if not body:
            continue
        rec = exact = False
        for name, rx, fn in _PATTERNS:
            for m in rx.finditer(text):
                if name in ("fuzzy ago", "the other day", "recently"):
                    n_fuzzy += 1
                    n_word += 1
                    rec = True
                    continue
                if message_anchor_date(text) is None:
                    continue  # 无锚消息的词不计数 (无法算日历)
                n_word += 1
                rec = True
                if fn(message_anchor_date(text), m) is not None:
                    n_exact_word += 1
                    exact = True
        if rec:
            n_rec += 1
        if exact:
            n_exact += 1
    return {"messages": total, "with_relative": n_rec, "with_exact_anchor": n_exact,
            "occurrences": n_word, "exact_occurrences": n_exact_word,
            "fuzzy_occurrences": n_fuzzy}
