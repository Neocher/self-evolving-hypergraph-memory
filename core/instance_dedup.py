"""P3 实例去重与计数基础件 — core/instance_dedup.py (纯函数).

本体论验收句: 两次提及是同一实例吗?
    → 由「成员规范化 + 实例去重 + 按周分桶 + 同实例合并」确定性裁决 → 过。

五个公开纯函数 (零 IO / 零 LLM / 零系统时钟):
    - normalize_member: 成员值规范化 (去引号/尾随标点/冠词, 折叠空白, 保留大小写原形)
    - distinct_members / distinct_count: 规范化后不区分大小写去重, 保首现序原值
    - members_by_week: 按 ISO 周分桶 (YYYY-Www), ts 不可解析 → "unknown"
    - merge_same_instance: 同规范化成员 + 相邻 ts 间隔 ≤ window_days → 同一实例归一组

facts 兼容对象属性 (.value/.ts) 与 dict 形态 (键取值), 接受任意可迭代 (含生成器)。

边界 (承继规格 §4 失效条件):
    - ts 仅接受 ISO8601 字符串或 epoch 秒 (int/float); 其余 (None/"last summer") → "unknown" / 各自成组
    - window_days=7 为启发式, 会误合并「同周两次不同行程」, 须下游 E3 instance_key 精化后依赖
    - 本件不做跨实体/跨槽位判定 (调用方负责按 entity/slot 分组后传入)
"""
from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping
from typing import Any, Optional

# 引号字符集 (ASCII + typographic + 中文引号 + 角括号)
_QUOTES = frozenset('"\'“”‘’«»「」『』„‟‹›')
# 尾随标点字符集 (半角 + 全角 + 省略号)
_TRAILING_PUNCT = frozenset(".,;:!?。，；：！？…")
# 开头冠词 (去重/合并判定统一不区分大小写)
_ARTICLES = frozenset({"the", "a", "an"})


def _get(item: Any, key: str, default: Any = None) -> Any:
    """对象属性或 dict 键取值: Mapping → 键取值, 否则 getattr (duck-typing 兼容)。"""
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _member_key(value: Any) -> str:
    """规范化 + 小写化后的成员比较键 (不区分大小写)。"""
    if value is None:
        s = ""
    else:
        s = value if isinstance(value, str) else str(value)
    return normalize_member(s).lower()


def _to_date(ts: Any) -> Optional[datetime.date]:
    """ts → datetime.date; ISO8601 字符串或 epoch 秒 (int/float); 其余 → None。"""
    if ts is None or isinstance(ts, bool):
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return None
        for candidate in (s, s.replace("Z", "+00:00"), s.replace("z", "+00:00")):
            try:
                return datetime.datetime.fromisoformat(candidate).date()
            except ValueError:
                continue
        try:
            return datetime.date.fromisoformat(s)
        except ValueError:
            return None
    return None


def _week_key(ts: Any) -> str:
    """ts → "YYYY-Www" (ISO 周, 真日历); 不可解析 → "unknown"。"""
    d = _to_date(ts)
    if d is None:
        return "unknown"
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def normalize_member(value: str) -> str:
    """成员值规范化: 去引号(含中文引号)/首尾标点/冠词, 折叠空白, 小写化前的原形保留大小写。

    规则 (确定):
      - 去 typographic/ASCII 引号与尾随标点
      - 去开头冠词 the/a/an (不区分大小写)
      - 折叠连续空白
    """
    s = value.strip()
    while s and s[0] in _QUOTES:
        s = s[1:].strip()
    while s and s[-1] in _QUOTES.union(_TRAILING_PUNCT):
        s = s[:-1].strip()
    s = " ".join(s.split())
    head, _, rest = s.partition(" ")
    if head.lower() in _ARTICLES:
        s = rest
    return s


def distinct_members(facts) -> list[str]:
    """去重(规范化后不区分大小写) → 保留首现顺序的原值列表。"""
    seen: set[str] = set()
    result: list[str] = []
    for f in facts:
        v = _get(f, "value")
        key = _member_key(v)
        if key not in seen:
            seen.add(key)
            result.append(v)
    return result


def distinct_count(facts) -> int:
    """= len(distinct_members(facts))。"""
    return len(distinct_members(facts))


def members_by_week(facts) -> dict[str, list[str]]:
    """按 ISO 周分桶 (key 形如 "2023-W19"); ts 不可解析 → key "unknown"。桶内成员同样去重、保首现序。"""
    buckets: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for f in facts:
        v = _get(f, "value")
        key = _week_key(_get(f, "ts"))
        bucket = buckets.setdefault(key, [])
        seen_set = seen.setdefault(key, set())
        mk = _member_key(v)
        if mk not in seen_set:
            seen_set.add(mk)
            bucket.append(v)
    return buckets


def merge_same_instance(facts, window_days: int = 7) -> list[list]:
    """同规范化成员且相邻 ts 间隔 ≤ window_days 天的连续出现 → 同一实例, 归为一组;
    返回组列表(组内按 ts 升序, ts 为 None 的各自成组)。"""
    groups: dict[str, list[tuple[Optional[datetime.date], Any]]] = {}
    for f in facts:
        mk = _member_key(_get(f, "value"))
        d = _to_date(_get(f, "ts"))
        groups.setdefault(mk, []).append((d, f))

    result: list[list] = []
    for items in groups.values():
        undated = [f for d, f in items if d is None]
        dated = sorted((x for x in items if x[0] is not None), key=lambda x: x[0])
        for f in undated:  # ts=None 各自成组 (原序)
            result.append([f])
        chain: list[Any] = []
        last: Optional[datetime.date] = None
        for d, f in dated:
            if last is not None and (d - last).days > window_days:
                result.append(chain)
                chain = []
            chain.append(f)
            last = d
        if chain:
            result.append(chain)
    return result
