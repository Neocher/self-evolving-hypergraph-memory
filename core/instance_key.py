"""R8 E3 实例身份 key — core/instance_key.py (纯函数).

本体论验收句: 两提及同实例?
    → 由 key 判定: sha1(entity|type|iso_week|obj|obj_desc) 确定性裁决 → 过。

key = sha1(normalize_alias(entity) | entity_type | iso_week | obj | obj_desc)[:12]:
    - entity 先经 normalize_alias 别名归一 (colleague→Rob 形状示例);
    - iso_week 接受 "YYYY-Www" 字符串或 date/datetime 本体 (内部归一到 iso_week);
    - obj / obj_desc 缺失 (None/空串) 归一到固定空位, 行为固定可复现;
    - obj_desc 含修饰语描述维: 同 obj 异修饰语 → 异 key (q0139 双吉他形状);
    - 跨年邻接周: 2023-12-31 (2023-W52) vs 2024-01-01 (2024-W01) → 异 key。
"""
from __future__ import annotations

import datetime
import hashlib
from typing import Any, Mapping, Optional

DEFAULT_ALIASES: dict[str, str] = {"colleague": "Rob"}


def normalize_alias(name: str,
                    aliases: Optional[Mapping[str, str]] = None) -> str:
    """别名归一到 canonical。

    aliases=None → 默认词典 (DEFAULT_ALIASES); 显式传入 (含空 dict) 覆盖默认,
    未命中原样返回。
    """
    table = DEFAULT_ALIASES if aliases is None else aliases
    return table.get(name, name)


def iso_week(value: Any) -> str:
    """date/datetime 本体 → ISO-8601 周串 "YYYY-Www" (真日历, 不取系统时钟)。"""
    iso_year, iso_week_no, _ = value.isocalendar()
    return f"{iso_year:04d}-W{iso_week_no:02d}"


def _to_iso_week(week: Any) -> str:
    """str "YYYY-Www" 原样返回; date/datetime 归一到 iso_week; 缺失 → 空位。"""
    if week is None:
        return ""
    if hasattr(week, "isocalendar"):
        return iso_week(week)
    return str(week)


def _part(value: Any) -> str:
    """obj / obj_desc 分段: None/空串统一归一到空位 (行为固定)。"""
    if value is None:
        return ""
    return str(value)


def instance_key(entity: str, entity_type: str, week: Any,
                 obj: Any, obj_desc: Optional[str] = None) -> str:
    """(entity, type, iso_week, obj, obj_desc) → sha1 前 12 位 hex 确定性实例 key。

    同入参必同 key; 任一分段不同 → 异 key (实例身份隔离)。
    """
    canon_entity = normalize_alias(entity)
    payload = "|".join((
        str(canon_entity),
        str(entity_type),
        _to_iso_week(week),
        _part(obj),
        _part(obj_desc),
    ))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
