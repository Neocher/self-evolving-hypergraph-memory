"""双轨事实记忆 (Dual-Track Facts) 分类器。

MemSIF 双轨启发：稳定事实（core）不应因 τ 衰减被误归档（DUM），
事件性/临时内容（active）随 τ 正常衰减。

分类结果落到 episode 的 fact_track 字段（"core" / "active"），
写时默认 "active"（保守）。仅依赖 re（标准库），无外部依赖。
"""
from __future__ import annotations

import re
from typing import Optional

# 稳定事实本体类型（映射分支）：这些本体类型天然持久，直判 core。
# 11 种本体类型中 core 候选 = 6 类；event_date/generic_fact 默认 active。
CORE_ONTOLOGY_TYPES: set[str] = {
    "person_birth",
    "person_death",
    "organization_founded",
    "relationship",
    "location_fact",
    "scientific_claim",
}

# 中文持久化语义词（关键词分支）：内容含这些词 → core（稳定偏好/身份/长期状态）。
CORE_KEYWORDS: tuple[str, ...] = (
    "喜欢", "我是", "我住", "一直", "偏好", "经常", "住在",
    "爱好", "擅长", "讨厌", "习惯", "始终", "总是", "职业",
)

# 事件语义词（可选，语义标注）：显式事件/临时安排 → active。
# 分类默认已是 active，此常量仅作语义说明，不改变分支结果。
ACTIVE_KEYWORDS: tuple[str, ...] = (
    "今天", "明天", "下午", "会议", "活动", "计划", "安排", "记得",
)

# 事件性本体类型（映射分支）：默认 active，走关键词分支兜底补 core。
_ACTIVE_ONTOLOGY_TYPES: set[str] = {"event_date", "generic_fact"}

# 预编译 core 关键词正则（单次扫描，避免逐词 in 判断）
_CORE_PATTERN = re.compile("|".join(re.escape(kw) for kw in CORE_KEYWORDS))


def classify_fact_track(
    content: str,
    entities: Optional[list] = None,
    ontology_type: Optional[str] = None,
) -> str:
    """按 (本体类型 → 关键词 → 默认 active) 判定 fact_track。

    1. ontology_type ∈ CORE_ONTOLOGY_TYPES → "core"
    2. ontology_type ∈ {event_date, generic_fact} 或 None → 关键词分支
    3. 内容含 CORE_KEYWORDS → "core"；否则默认 "active"
    """
    if ontology_type in CORE_ONTOLOGY_TYPES:
        return "core"
    if content and _CORE_PATTERN.search(content):
        return "core"
    return "active"


def is_core_track(track: str) -> bool:
    """是否为稳定事实轨。"""
    return track == "core"
