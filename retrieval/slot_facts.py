"""R8 E4 槽位原文检索 — retrieval/slot_facts.py (纯, 无 IO / 无 LLM / 无时间依赖).

本体论验收句: (entity×slot) 成员全集如何确定性取得?
    → 集合查询 + 确定性排序/去重/词形归一, 不做 top-N → 过。

数据面注入式: SlotFact 行由调用方 (灌库/原文 loader) 构建注入, ts 一律入参注入
(不取系统时钟); 本模块零 IO 零 env 读取 (env 读取仅限 query_router.retrieve_slot)。

`SlotIndex` 确定性语义 (ts<=session_ts 含边界与 core.state_projection.resolve 一致):
    - query:  全量返回不截断; 过滤 (entity,slot) + scope 会话隔离 + session_ts 含边界;
              排序 ts 升序, 同 ts 按 ep_id 升序; 无命中 → []
    - members: 成员行去重语义 = 按 value 归一 (小写/strip), 每 value 保留首次出现行
               (排序后 ts 最小者), mentions=[全部命中 ep_id]; 供 cat1 LIST/SET 判卷形态。

`parse_session_datetime` 确定性解析会话时间锚 "H:MM am/pm on D Month, YYYY" 为
可比较的 "%Y-%m-%dT%H:%M" 字符串 (定长零填充, 字典序即时间序; 自定义基准, 不取系统时钟)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

# "5:53 pm on 24 September, 2023" → 12h 时钟 + 月名 + 日/年 (确定性, 无区域时钟)。
_DT_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\s*$",
    re.IGNORECASE,
)
_MONTHS: Dict[str, int] = {
    name: i for i, name in enumerate(
        ("january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december"),
        start=1,
    )
}


@dataclass
class SlotFact:
    """单条 (entity×slot) 成员事实行 (episode 原文的一处提及)。

    session_id: 会话归属 (跨会话同名实体按它隔离)
    ep_id:      会话内 episode 序 (可比较, 排序 tie-break)
    ts:         可比较时间戳 (调用方注入, 不取系统时钟)
    entity/slot/value: 检索与成员判定键
    msg_ref:    原文消息引用 (可选, 原文装配属后续步)
    """

    session_id: str
    ep_id: Any
    ts: Any
    entity: str
    slot: str
    value: str
    msg_ref: Optional[str] = None


@dataclass
class SlotIndex:
    """槽位确定性检索引擎: 注入事实全集的确定性过滤/排序/成员去重 (纯, 无 IO)。"""

    rows: List[SlotFact] = field(default_factory=list)

    @classmethod
    def from_facts(
        cls, facts: Sequence[Union[SlotFact, Dict[str, Any]]]
    ) -> "SlotIndex":
        """注入 (entity×slot) 成员事实行: 归并/去重完全相同的行, 每行保留。"""
        seen: set = set()
        merged: List[SlotFact] = []
        for item in facts:
            f = item if isinstance(item, SlotFact) else _coerce(item)
            ident = (f.session_id, f.ep_id, f.ts, f.entity, f.slot, f.value, f.msg_ref)
            if ident in seen:
                continue
            seen.add(ident)
            merged.append(f)
        return cls(rows=merged)

    def query(
        self,
        entity: str,
        slot: str,
        session_ts: Any = None,
        scope: Any = None,
    ) -> List[SlotFact]:
        """(entity, slot) 成员全量事实行: 全量返回不截断。

        scope 非 None → 仅 session_id==scope 行 (跨会话同名隔离);
        session_ts 非 None → 仅 ts<=session_ts (含边界, 与 state_projection resolve 一致);
        排序 ts 升序, 同 ts 按 ep_id 升序; 无命中 → []。
        """
        out = [
            f for f in self.rows
            if f.entity == entity and f.slot == slot
            and (scope is None or f.session_id == scope)
            and (session_ts is None or (f.ts is not None and f.ts <= session_ts))
        ]
        out.sort(key=lambda f: (f.ts, f.ep_id))
        return out

    def members(
        self,
        entity: str,
        slot: str,
        session_ts: Any = None,
        scope: Any = None,
    ) -> List[dict]:
        """成员行去重语义: 按 value 归一 (小写/strip) 去重, 每 value 首次行 + mentions。

        成员行排序 ts 升序 (接口契约); mentions = 该 value 全部命中 ep_id (按时间序)。
        """
        rows = self.query(entity, slot, session_ts=session_ts, scope=scope)
        first: Dict[str, SlotFact] = {}
        mentions: Dict[str, List[Any]] = {}
        ordered: List[str] = []
        for f in rows:
            key = (f.value or "").strip().lower()
            if key not in first:
                first[key] = f
                mentions[key] = []
                ordered.append(key)
            if f.ep_id not in mentions[key]:
                mentions[key].append(f.ep_id)
        return [
            {
                "session_id": first[k].session_id,
                "ep_id": first[k].ep_id,
                "ts": first[k].ts,
                "entity": first[k].entity,
                "slot": first[k].slot,
                "value": first[k].value,
                "msg_ref": first[k].msg_ref,
                "mentions": mentions[k],
            }
            for k in ordered
        ]


def _coerce(row: Dict[str, Any]) -> SlotFact:
    """dict 事实行 → SlotFact (原文 loader 侧复用; 缺失字段容错为 None)。"""
    return SlotFact(
        session_id=row["session_id"],
        ep_id=row["ep_id"],
        ts=row["ts"],
        entity=row["entity"],
        slot=row["slot"],
        value=row["value"],
        msg_ref=row.get("msg_ref"),
    )


def parse_session_datetime(text: str) -> str:
    """确定性解析会话时间锚 "H:MM am/pm on D Month, YYYY" → "%Y-%m-%dT%H:%M"。

    定长零填充字符串的字典序 == 时间序, 可与同函数输出直接比较; 不取系统时钟。
    无法解析 / 未知月名 → ValueError (显式失败, 不静默兜底)。
    """
    m = _DT_RE.match(text or "")
    if not m:
        raise ValueError(f"unparseable session datetime: {text!r}")
    h12, minute, meridiem, day, month_name, year = m.groups()
    month = _MONTHS.get(month_name.lower())
    if month is None:
        raise ValueError(f"unknown month in session datetime: {text!r}")
    hour = int(h12) % 12
    if meridiem.lower() == "pm":
        hour += 12
    return f"{int(year):04d}-{month:02d}-{int(day):02d}T{hour:02d}:{int(minute):02d}"
