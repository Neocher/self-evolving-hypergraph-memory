# -*- coding: utf-8 -*-
"""R8 Step3b 确定性槽证据装配 — scripts/slot_assembly.py (bench 侧, 禁 LLM).

权威: data/r8/r8-step3b-spec.md (装配先例 L1212 `_r6_evidence_forms_ctx`;
词表只读 q2slot-v1.json / slot-triggers-v1.json / fixtures-r8-oracle.json)。

Step3a 抽取管线 (retrieval/slot_extract.build_slot_index, 纯规则) 已把会话原文
→ (entity×slot) SlotFact; 本模块在评测组织段 ctx 装配处追加 `[SLOT EVIDENCE]`
段: 对题面 (qa_id → lexicon 得 entity/slot) 取槽位确定性证据 (members 全量,
时间序不截断), 每行 = value + dia 引用 + ts, 不复制原文逐字 (超 120 字符截断),
供 reader 完整枚举。原文消息行逐字不变 (红线), reader prompt V1 不动 (红线)。

形态 (每 entity 一段; 无词条/零命中 → 原样返回 ctx 不追加空段):
    [SLOT EVIDENCE] Calvin | purchase_item (确定性槽证据, 时间序):
    - value: mansion  (msg D1:3, 2023-03-23T11:53)

开关在 bench (R8_SLOT env, 默认 off); 本模块零 IO 零 LLM — 词表一律由调用方
注入, ts 一律入参注入 (不取系统时钟; 数值 ts 仅做确定性 UTC 格式化显示)。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

# 行首 "Name:" 说话人前缀 (G1 ⑦): "Calvin: text" → speaker=Calvin。
_SPEAKER_PREFIX_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_.' -]{0,39}?)\s*:\s*(.*)$", re.S)
# 灌库 DB 同形前缀 "[date: ...] [speaker] " (bench msg_by_id content 前缀剥离)。
_DATE_BRACKET_RE = re.compile(r"^\[date:[^\]]*\]\s*")
_SPEAKER_BRACKET_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$", re.S)
# 行引用不复制原文逐字: value 超 120 字符截断 (G1 ⑤)。
_MAX_VALUE_CHARS = 120


def append_slot_evidence(
    ctx: str,
    question: str,
    qa_id: Optional[str],
    msgs: Sequence[Mapping[str, Any]],
    lexicon: Optional[Mapping[str, Mapping[str, Any]]],
    triggers: Optional[Mapping[str, Sequence[str]]],
) -> str:
    """ctx 尾部追加 '[SLOT EVIDENCE]' 段 (每 entity 一段, 行全量时间序)。

    qa_id 形如 "conv-50#q0001" (lexicon key); 无词条 → 原样返回 ctx (不报错)。
    msgs: [{dia, session_id, speaker, text, ts}] 会话内时间序 (调用方给本 conv)。
    经 Step3a build_slot_index 确定性抽取 → SlotIndex.query(entity, slot) 行全量;
    无命中 → 原样返回 ctx (不追加空段)。原文消息零改动 (红线)。
    """
    entry = (lexicon or {}).get(qa_id) if qa_id else None
    if not entry:
        return ctx
    entities = list(entry.get("entities") or [])
    slot = (entry.get("slot") or "").strip()
    if not entities or not slot or not msgs:
        return ctx
    build_slot_index = _lazy_build_slot_index()
    idx = build_slot_index(list(msgs), {qa_id: entry}, triggers)
    sections: List[str] = []
    for entity in entities:
        rows = idx.query(entity, slot)
        if not rows:
            continue
        lines = [f"[SLOT EVIDENCE] {entity} | {slot} (确定性槽证据, 时间序):"]
        for f in rows:
            val = _truncate_value(f.value or "")
            dia = (f.msg_ref or "").strip()
            ts = _fmt_ts(f.ts)
            lines.append(f"- value: {val}  (msg {dia}, {ts})".rstrip())
        sections.append("\n".join(lines))
    if not sections:
        return ctx
    return ctx + "\n\n" + "\n\n".join(sections)


def build_msgs_from_cache(
    msg_by_id: Mapping[str, Any],
    episode_cache: Mapping[str, Mapping[str, Any]],
    session_date_by_conv: Mapping[Any, str],
) -> List[Dict[str, Any]]:
    """bench 内存 → msgs 标准形态 (供 append_slot_evidence / Step3a 抽取).

    msg_by_id: {ep_id: 消息文本}; episode_cache: {ep_id: {session_id, created_at,
    content, ...}}; session_date_by_conv: {会话 key → 会话日期/时间串} (会话 key
    与 episode_cache['session_id'] 同坐标, bench 为 conv idx int / 会话标签)。

    每条: dia=key; text=原文去 [date:]/[speaker]/'Name:' 前缀后的正文; speaker=
    前缀解析 (bracket [Name] 优先, 其次行首 "Name:", 无则 ""); session_id=
    episode_cache['session_id']; ts=会话日期串经 parse_session_datetime 规范化
    (可解析) 否则 episode_cache['created_at'] 原值。顺序=msg_by_id 输入序。
    """
    parse_sess_dt = _lazy_parse_session_datetime()
    out: List[Dict[str, Any]] = []
    for key, content in msg_by_id.items():
        episode = episode_cache.get(key) or {}
        raw = content if content not in (None, "") else episode.get("content")
        body, speaker = _split_prefixes(str(raw or ""))
        sess = episode.get("session_id")
        ts: Any = None
        if sess is not None:
            try:
                date_raw = session_date_by_conv.get(sess)
            except TypeError:
                date_raw = None  # sess 不可哈希 (list/dict) → 回落 created_at
            if date_raw:
                try:
                    ts = parse_sess_dt(str(date_raw))
                except Exception:
                    ts = date_raw  # 非 "H:MM am/pm on D Month, YYYY" 形态 → 原值
        if ts is None:
            ts = episode.get("created_at")
        out.append({"dia": key, "session_id": sess, "speaker": speaker,
                    "text": body, "ts": ts})
    return out


# ── 前缀剥离 / 说话人解析 (G1 ⑦ ⑧) ────────────────────────────────────────

def _split_prefixes(raw: str) -> tuple:
    """原文 → (正文, speaker): 剥离 [date:]/[speaker]/行首 'Name:' 前缀。

    bench 灌库 content 为 "[date: ...] [Speaker] text" DB 同形; 纯文本形态
    兼容行首 "Calvin: text" (G1 ⑦)。无任何前缀 → (原文, "") (G1 ⑧)。
    """
    s = (raw or "").strip()
    s = _DATE_BRACKET_RE.sub("", s)
    m = _SPEAKER_BRACKET_RE.match(s)
    if m:
        return m.group(2).strip(), m.group(1).strip()
    m = _SPEAKER_PREFIX_RE.match(s)
    if m and m.group(2).strip():
        return m.group(2).strip(), m.group(1).strip()
    return s, ""


def _truncate_value(value: str) -> str:
    """值文本: 折行空白后截断 ≤120 字符 (G1 ⑤ 引用不复制原文逐字长句)。"""
    s = " ".join((value or "").split())
    if len(s) <= _MAX_VALUE_CHARS:
        return s
    return s[:_MAX_VALUE_CHARS - 1] + "…"


def _fmt_ts(ts: Any) -> str:
    """ts → 展示串 (G1 ⑤ 行含 <ts 日期串>): 字符串直通; 数值 (epoch) 确定性
    UTC 格式化 (不取系统时钟)。"""
    if ts is None:
        return ""
    if isinstance(ts, (int, float)):
        import datetime as _dt
        try:
            return _dt.datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
        except (OSError, OverflowError, ValueError):
            return str(ts)
    return str(ts)


# ── 惰性 import (bench off 模式零依赖: 仅 R8_SLOT 装配路径触发 retrieval) ──

def _lazy_build_slot_index():
    from retrieval.slot_extract import build_slot_index  # noqa: PLC0415
    return build_slot_index


def _lazy_parse_session_datetime():
    from retrieval.slot_facts import parse_session_datetime  # noqa: PLC0415
    return parse_session_datetime
