"""R8 E1' 状态时点裁决 — core/state_projection.py (纯函数).

本体论验收句: 题问时点该 (实体,槽) 的值由什么决定?
    → 由 (值,生效时间) 集合的确定性 latest-wins 裁决决定 → 过。

事实行 (dict) 契约:
    {"entity": str, "attr": str, "value": Any,
     "ts": 可比较时间戳 (int/float/datetime 一律入参注入, 不取系统时钟),
     "ep_id": 可比较 episode 序,
     "negation": True (可选, 否定证据行)}

裁决语义与 query_router 既有 `_property_temporal_retrieve` / `_fact_retrieve`
同构 (共享裁决器, 无第三条平行实现):
    - 可见版本 = ts <= session_ts (含边界);
    - latest-wins: ts 最大者胜出; 同 ts tie-break 取 ep_id 更大者 (后到 episode 覆盖);
    - 否定行参与裁决: 最新可见为否定行 → N/A (None), 被否值不外泄;
    - session_ts 早于全部事实 → N/A (None)。

`timeline` 输出全版本链 (ts asc, ep_id asc); 被否值 (被其后否定行否掉的 value)
不出现于任何 State — 全链不外泄被否值。
"""
from __future__ import annotations

from typing import Any, Optional


def resolve(facts: list[dict], entity: str, attr: str,
            session_ts: Any) -> Optional[dict]:
    """裁决 (entity, attr) 在 session_ts 时点的值。

    可见版本 = ts <= session_ts; 取 ts 最大者, 同秒按 ep_id 更大者胜;
    最新可见为否定行或全早无可见 → None (N/A)。
    """
    visible = [
        f for f in facts
        if f.get("entity") == entity and f.get("attr") == attr
        and f.get("ts") is not None and f.get("ts") <= session_ts
    ]
    if not visible:
        return None
    best = max(visible, key=lambda f: (f.get("ts"), f.get("ep_id", 0)))
    if best.get("negation"):
        # 否定行只参与裁决, 不返回被否值。
        return None
    return {
        "value": best.get("value"),
        "ts": best.get("ts"),
        "ep_id": best.get("ep_id"),
    }


def timeline(facts: list[dict], entity: str, attr: str) -> list[dict]:
    """返回 (entity, attr) 的全版本链 (ts asc, ep_id asc)。

    每个事实行产出一个 State; 否定行 State 记 value=None + negation=True;
    被其后否定行 (同 value, 更晚 (ts, ep_id)) 否掉的正行 State 值掩为 None,
    保证被否值不出现于全链。
    """
    rows = [
        f for f in facts
        if f.get("entity") == entity and f.get("attr") == attr
        and f.get("ts") is not None
    ]
    rows.sort(key=lambda f: (f.get("ts"), f.get("ep_id", 0)))
    neg_rows = [f for f in rows if f.get("negation")]
    states: list[dict] = []
    for f in rows:
        pos = (f.get("ts"), f.get("ep_id", 0))
        if f.get("negation"):
            states.append({
                "value": None,
                "ts": f.get("ts"),
                "ep_id": f.get("ep_id"),
                "negation": True,
            })
            continue
        value = f.get("value")
        negated_later = any(
            n.get("value") == value
            and (n.get("ts"), n.get("ep_id", 0)) > pos
            for n in neg_rows
        )
        states.append({
            "value": None if negated_later else value,
            "ts": f.get("ts"),
            "ep_id": f.get("ep_id"),
        })
    return states
