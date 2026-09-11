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


# ── P2 槽位闭包 v2: 路由 + 去噪排序 + 替换式装配 (2026-09-11) ─────────────
# 依据: studies/r8-p2-taskbook.md 【新增能力】2/3/4。默认不启用 (bench env R8_SLOT_V2)。

_SEG_HEAD_RE = re.compile(r"^\[(GLOBAL CONTEXT|DIRECT EVIDENCE|MEMORY BLOCK|SLOT EVIDENCE|"
                          r"TIME ANCHORS|FACT CLUSTERS|ENTITY:|RELATIONS)")


def _replace_entity_segment(ctx: str, replacement: str) -> str:
    """把 [ENTITY: ...] 组织段整体替换为 replacement —— 原文消息行逐字不动 (红线)。

    段边界: 从 "[ENTITY:" 行起, 到空行或下一个 section 头 (非 "- " 开头) 为止。
    无 [ENTITY:] 段 → 追加到尾部 (与 v1 行为一致)。
    """
    lines = ctx.splitlines()
    out: List[str] = []
    i, done = 0, False
    while i < len(lines):
        ln = lines[i]
        if not done and ln.startswith("[ENTITY:"):
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() == "" or _SEG_HEAD_RE.match(nxt):
                    break
                i += 1
            if replacement:
                out.append(replacement)
            done = True
            continue
        out.append(ln)
        i += 1
    if not done and replacement:
        return ctx + "\n\n" + replacement
    return "\n".join(out)


def render_slot_block_v2(
    ctx: str,
    question: str,
    qa_id: Optional[str],
    msgs: Sequence[Mapping[str, Any]],
    lexicon: Optional[Mapping[str, Mapping[str, Any]]],
    triggers: Optional[Mapping[str, Sequence[str]]] = None,
    budget_chars: int = 1800,
    max_members: int = 80,   # 实测: 60→R .355 / 无上限→.384, 取 80 折中
    route_first: bool = True,
) -> str:
    """P2 闭包 v2 装配: 仅集合/计数题注入; 槽位闭包替换 [ENTITY:] 段。

    路由: route_first=True 时先走题面现场路由 (P6-lite), 未命中再回落词表 (记录 via)。
    非集合题 / 无词条 / 零命中 → 原样返回 ctx (不注入, 消除噪声面)。
    """
    from retrieval.slot_closure import (is_set_question, render_closure_for_entry,
                                        route_question)
    if not is_set_question(question):
        return ctx
    if not msgs:
        return ctx

    entries: List[Dict[str, Any]] = []
    if route_first:
        speakers = sorted({str(m.get("speaker") or "") for m in msgs if m.get("speaker")})
        for r in route_question(question, speakers=speakers, triggers=triggers, max_slots=2):
            if r.get("entity") and r.get("slot"):
                entries.append({"entities": [r["entity"]], "slot": r["slot"],
                                "gate": "did", "via": r.get("via", "route")})
    if not entries and qa_id:
        entry = (lexicon or {}).get(qa_id)
        if entry:
            e = dict(entry)
            e["via"] = "lexicon"
            entries.append(e)
    if not entries:
        return ctx

    from retrieval.slot_closure import clean_members, extract_objects_v2, render_closure
    blocks: List[str] = []
    for entry in entries:
        slot = (entry.get("slot") or "").strip()
        trigs = list((triggers or {}).get(slot, []) or [])
        gate = entry.get("gate") or "did"
        for ent in entry.get("entities") or []:
            if not ent or not slot or not trigs:
                continue
            rows = extract_objects_v2(msgs, ent, slot, trigs, gate=gate, question=question)
            members = clean_members([(getattr(r, "value", "") or "", r) for r in rows],
                                    question=question, entity=ent)
            seg = render_closure(members, ent, slot, budget_chars=budget_chars,
                                 max_members=max_members)
            if seg:
                blocks.append(seg)
    if not blocks:
        return ctx
    return _replace_entity_segment(ctx, "\n\n".join(blocks))


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
