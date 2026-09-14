# -*- coding: utf-8 -*-
"""P2 槽位闭包 v2 — retrieval/slot_closure.py (纯规则 / 零 IO / 零 LLM / 零系统钟)。

立项依据: /home/user/shm-evolve/studies/r8-p2-taskbook.md (四段模板)
现场缺口 (2026-09-11 实测, v1 基线):
  - value 3438 条中 36.8% <15 字符、12.8% 是问句、54 条纯垃圾 ("I"/"me"/"so")
  - 成员 precision 0.032 / recall 0.399 (以题面 gold 为参照, 确定性度量)
  - 段均 5,028 字符、最大 23,689 (尾部追加第 7 段)

本模块补齐的能力:
  1. 问题类型路由 — 只有集合/计数问法才吃槽位闭包 (非集合题不注入, 消除噪声面)
  2. 值规范化与去噪 — 去前介词/尾标点/引号/图像标记; 丢问句/代词/碎句/从句
  3. 成员排序与预算 — 题面相关度 + NP 形态 + 专名优先, 受 budget 约束截断
  4. 渲染 — 行级 provenance (msg/dia + ts), 供替换式装配使用
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# 集合/计数问法 (确定性规则, 可单测)
_SET_PATTERNS = (
    re.compile(r"^\s*how many\b", re.I),
    re.compile(r"^\s*(what|which)\b.*\b(has|have|did|does|do)\b", re.I),
    re.compile(r"\b(list|name all|enumerate)\b", re.I),
    re.compile(r"\bwhat (books|events|activities|places|states|cities|countries|items|things|games|foods|sports|pets|songs|movies|shows|hobbies|skills|jobs|gifts|instruments|classes|trips|vacations)\b", re.I),
)
_COUNT_RE = re.compile(r"^\s*how many\b", re.I)

# 值级垃圾 (代词/限定词/连接词/纯标点)
_STOP_VALUES = {
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "we", "us", "our", "ours",
    "he", "him", "his", "she", "her", "hers", "they", "them", "their", "it", "its",
    "the", "a", "an", "and", "or", "but", "so", "because", "if", "then", "that", "this",
    "there", "here", "what", "which", "who", "yes", "no", "ok", "okay", "oh", "wow",
    "-", "--", "…", "...", ".", ",", "?", "!", ";", ":",
}
_LEADING_FILLER = re.compile(r"^\s*(to|in|at|from|for|about|on|with|of|into|like|very|really|just|also|then|and|but)\s+", re.I)
_PRONOUN_START = re.compile(r"^\s*(i|you|we|he|she|they|it|my|your|our|his|her|their|me|us|them)\b", re.I)
_CLAUSE_START = re.compile(r"^\s*(and|or|but|so|because|if|when|while|although|though|since|as|then|also|just|very)\b", re.I)
_IMAGE_MARKER = re.compile(r"\[img:[^\]]*\]", re.I)
_ENUM_SPLIT = re.compile(r"\s*(?:,|;|\band\b|\+)\s*", re.I)


def is_set_question(question: str) -> bool:
    """集合/计数问法判定 (确定性, 无 LLM)。"""
    q = (question or "").strip()
    if not q:
        return False
    return any(p.search(q) for p in _SET_PATTERNS)


def is_count_question(question: str) -> bool:
    return bool(_COUNT_RE.match((question or "").strip()))


_QUOTED_RE = re.compile(r"[""\u201c\u2018]([^""\u201d\u2019]{2,60})[""\u201d\u2019]")


def _quoted_span(value: str) -> Optional[str]:
    """值中的引号片段 (书名/名称常被引号包裹) → 片段; 无则 None。"""
    m = _QUOTED_RE.search(value or "")
    if not m:
        m = re.search(r"\"([^\"]{2,60})\"", value or "")
    return m.group(1).strip() if m else None


def normalize_value(value: str) -> str:
    """值规范化: 去图像标记/引号/前介词/尾标点, 折叠空白。"""
    v = _IMAGE_MARKER.sub(" ", value or "")
    qs = _quoted_span(value or "")
    if qs:
        return qs
    v = v.strip().strip("\"'“”‘’")
    v = re.sub(r"\s+", " ", v).strip()
    v = _LEADING_FILLER.sub("", v)
    v = v.strip(" \t.,;:!-—…")
    return v.strip()


def is_noise_value(value: str) -> bool:
    """垃圾值判定 (纯规则): 空/极短/代词/连接词开头/问句/从句/无信息。

    注意: 形态判定在**规范化之前**做 —— 规范化会剥掉前导介词/连词, 使
    "and then we went" 之类的从句碎片漏检 (2026-09-11 单测抓出)。
    """
    raw = (value or "").strip()
    if _PRONOUN_START.match(raw) or _CLAUSE_START.match(raw):
        return True
    v = normalize_value(value)
    if len(v) < 2:
        return True
    low = v.lower()
    if low in _STOP_VALUES:
        return True
    if v.endswith("?"):
        return True
    if _PRONOUN_START.match(v) or _CLAUSE_START.match(v):
        return True
    if not re.search(r"[A-Za-z0-9]", v):          # 纯标点/符号
        return True
    if len(v.split()) > 14:                        # 从句样长句 (非 NP)
        return True
    if low in {"so much", "so good", "a lot", "kind of", "sort of"}:
        return True
    return False


def split_enumeration(value: str) -> List[str]:
    """把 "A, B and C" 形态的枚举值拆成成员 (仅当各部分都非噪声)。"""
    v = normalize_value(value)
    if "," not in v and " and " not in v.lower() and ";" not in v:
        return [v] if v else []
    parts = [p.strip() for p in _ENUM_SPLIT.split(v) if p.strip()]
    parts = [normalize_value(p) for p in parts]
    parts = [p for p in parts if p and not is_noise_value(p)]
    return parts or ([v] if v and not is_noise_value(v) else [])


def _tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-z0-9']+", (s or "").lower()) if len(t) > 2}


# ── A) 谓词论元抽取 (v2 专用; 修 generic 题的子句碎片值) ──────────────────
# 现场缺口: v1 抽取取"触发词后 8-token 窗再切分" → 产出子句碎片
# ("group has made me feel accepted"). 本实现取**触发词支配的宾语 NP**:
# 触发词命中 → 跳过限定词/前置介词 → 取 1..N 个 token 直至停用词 (介词/连词/代词/助动词/标点)。

_NP_STOP = {
    "in", "on", "at", "for", "to", "from", "with", "about", "of", "during", "after", "before",
    "by", "into", "over", "under", "near", "around", "through", "across", "than", "as",
    "and", "or", "but", "so", "because", "then", "also", "if", "when", "while", "that",
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "should", "might", "may", "must", "not", "no",
    "i", "you", "we", "he", "she", "they", "it", "me", "us", "him", "her", "them",
    "my", "your", "our", "his", "their", "its", "this", "these", "those", "there", "here",
}
_NP_DET = {"a", "an", "the", "some", "any", "another", "each", "every", "my", "your", "his",
           "her", "their", "our", "its", "this", "that", "these", "those"}
_NP_TIME = {"last", "next", "this", "year", "years", "week", "weeks", "day", "days", "month",
            "months", "time", "times", "ago", "today", "yesterday", "tomorrow", "soon", "later",
            "again", "now", "recently", "lately", "already", "still", "ever", "never"}
_MAX_NP_TOKENS = 4


def _trigger_span(sentence: str, trigger: str):
    """句内触发词出现位 (词边界起, 允许词尾屈折: visit→visited) → (start, end) 或 None。

    2026-09-11 实测教训: 用 str.find 会命中单词内部 ("visit" 命中 "visited" 的中段),
    产出 "ed Italy"/"ing basketball" 这类碎片; 必须词边界匹配并把屈折尾一起吃进 span。
    """
    if not sentence or not trigger:
        return None
    m = re.search(r"\b" + re.escape(str(trigger)) + r"\w{0,4}\b", sentence, re.I)
    return (m.start(), m.end()) if m else None


def _np_after_trigger(sentence: str, trigger: str) -> Optional[str]:
    """句子 + 触发词 → 触发词支配的宾语 NP (None = 无可取)。"""
    s = (sentence or "").strip()
    span = _trigger_span(s, trigger)
    if span is None:
        return None
    tail = s[span[1]:]
    toks = re.findall(r"[A-Za-z0-9'’\-]+|[.,;:!?]", tail)
    out: List[str] = []
    for tok in toks:
        if re.fullmatch(r"[.,;:!?]", tok):
            break
        lw = tok.lower()
        if not out:
            if lw in _NP_DET or lw in _NP_STOP:      # 先跳过限定词/前置介词
                continue
            out.append(tok)
            continue
        if lw in _NP_STOP or lw in _NP_TIME:
            break
        out.append(tok)
        if len(out) >= _MAX_NP_TOKENS:
            break
    if not out:
        return None
    return normalize_value(" ".join(out))


def extract_objects_v2(
    msgs: Sequence[Mapping[str, Any]],
    entity: str,
    slot: str,
    triggers: Sequence[str],
    gate: str = "did",
    question: str = "",
) -> List[Dict[str, Any]]:
    """会话消息 → 宾语 NP 事实 [{value, ts, msg_ref}] (v2 抽取, 纯规则)。"""
    import types as _types
    from retrieval.slot_extract import _attributed, _modality_gated, _split_sentences
    out: List[Any] = []
    for msg in msgs:
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        speaker = msg.get("speaker") or ""
        ts = msg.get("ts")
        dia = msg.get("dia")
        for sent in _split_sentences(text):
            if not _attributed(sent, speaker, entity):
                continue
            low = sent.lower()
            for trig in triggers or []:
                if not trig or _trigger_span(sent, str(trig)) is None:
                    continue
                if _modality_gated(sent, gate):
                    continue
                np = _np_after_trigger(sent, str(trig))
                if not np or is_noise_value(np):
                    continue
                out.append(_types.SimpleNamespace(value=np, ts=ts, msg_ref=dia, trigger=str(trig)))
    return out


# ── 计数题路径 v1 (P3 第一步: 符号读出; 2026-09-11) ────────────────────────
# 现场缺口: 20 道 how many 题里 6 道连槽位候选都没有 (route 返回空), 且计数的输入
# 是"值"而非"事件实例" → 同一事件多次提及被重复计数 (gold=3 却 distinct=105)。

_COUNT_ANCHOR_STOP = _NP_STOP | _NP_TIME | {
    "many", "much", "times", "time", "did", "does", "do", "has", "have", "had",
    "year", "years", "month", "months", "week", "weeks", "day", "days",
}
_ENTITY_LIKE = re.compile(r"^[A-Z][a-z]+$")


def count_anchor_terms(question: str, speakers: Sequence[str] = ()) -> List[str]:
    """how many 题 → 内容锚词 (含单复数变体), 用于无槽位命中时的伪槽抽取。

    规则 (确定): 取题面里长度≥3 的词, 去掉停用词/说话人名/纯数字, 最多 4 个;
    "how many times …" 的 times 属停用词 → 锚词落到真正的对象名词 (如 beach)。
    """
    q = question or ""
    if not q:
        return []
    spk = {s.lower() for s in speakers if s}
    out: List[str] = []
    for w in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", q):
        lw = w.lower()
        if lw in _COUNT_ANCHOR_STOP or lw in spk:
            continue
        if _ENTITY_LIKE.match(w) and w not in ("How", "What", "Which", "When", "Where"):
            continue          # 专名 (人名/地名) 已是 entity 维度, 不做锚
        if lw not in out:
            out.append(lw)
        if len(out) >= 4:
            break
    forms: List[str] = []
    for w in out:
        forms.append(w)
        if not w.endswith("s"):
            forms.append(w + "s")
    return forms


def extract_anchor_mentions(
    msgs: Sequence[Mapping[str, Any]],
    anchors: Sequence[str],
    entity: str = "",
    gate: str = "did",
) -> List[Any]:
    """锚词提及 → 事实 (value=锚词原形, ts=消息 ts) — 供无槽位的计数题使用。

    值取锚词本身: 计数题要的是"事件出现次数", 由 count_instances 按时间窗聚类。
    """
    import types as _types
    from retrieval.slot_extract import _attributed, _modality_gated, _split_sentences
    norm_anchors = [a for a in anchors if a]
    out: List[Any] = []
    for msg in msgs:
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        speaker = msg.get("speaker") or ""
        for sent in _split_sentences(text):
            low = sent.lower()
            hit = next((a for a in norm_anchors if re.search(r"\b" + re.escape(a.lower()) + r"\b", low)), None)
            if not hit:
                continue
            if entity and not _attributed(sent, speaker, entity):
                continue
            if _modality_gated(sent, gate):
                continue
            out.append(_types.SimpleNamespace(value=hit, ts=msg.get("ts"), msg_ref=msg.get("dia"),
                                              trigger=hit))
    return out


def count_instances(facts: Sequence[Any], window_days: int = 7) -> int:
    """事实 → 实例数 (同一成员 + 时间窗内相邻提及 → 1 个实例)。"""
    from core.instance_dedup import merge_same_instance
    items = [f for f in facts if getattr(f, "value", "")]
    if not items:
        return 0
    return len(merge_same_instance(items, window_days=window_days))


def route_count_question(
    question: str,
    speakers: Sequence[str] = (),
    triggers: Optional[Mapping[str, Sequence[str]]] = None,
    max_slots: int = 2,
) -> List[Dict[str, Any]]:
    """how many 题候选: 既有槽位路由 ∪ 锚词伪槽 (保证无触发词命中时仍有候选)。"""
    cands = list(route_question(question, speakers=speakers, triggers=triggers, max_slots=max_slots))
    if cands:
        return cands
    anchors = count_anchor_terms(question, speakers=speakers)
    if not anchors:
        return []
    q = (question or "").strip()
    proper = [t for t in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", q) if t.lower() not in _STOP_TOKENS]
    spk = {s for s in speakers if s}
    ent = next((t for t in proper if t in spk), proper[0] if proper else "")
    return [{"entity": ent, "slot": "anchor:" + anchors[0], "score": 1.0,
             "via": "route:anchor", "terms": anchors}]


# ── P1 类型化值校验 (题面推断期望类型 → 只保留同类型成员) ────────────────
# 资源: data/r8/value-types.json (封闭词表, 通用; 不读 qa_id — 红线)

_VT_PATH = None
_VT_CACHE: Optional[Dict[str, Any]] = None

_TYPE_PATTERNS = (
    ("count", re.compile(r"^\s*how (many|often|much)\b", re.I)),
    ("us_state", re.compile(r"\bstates?\b|\bprovinces?\b", re.I)),
    ("country", re.compile(r"\b(countr(y|ies)|nations?)\b|\babroad\b", re.I)),
    ("place", re.compile(r"\b(cit(y|ies)|place|places|location|locations|where)\b", re.I)),
    ("work_book", re.compile(r"\b(books?|novels?|reading list|literature)\b", re.I)),
    ("work_media", re.compile(r"\b(movies?|films?|shows?|series|songs?|albums?|podcasts?)\b", re.I)),
    ("activity", re.compile(r"\b(activit(y|ies)|hobb(y|ies)|sports?|exercises?|things to do)\b", re.I)),
)


def detect_value_type(question: str) -> str:
    """题面 → 期望值类型 (确定性, 无 LLM): count/us_state/country/place/work_book/work_media/activity/generic。"""
    q = (question or "").strip()
    for name, pat in _TYPE_PATTERNS:
        if pat.search(q):
            return name
    return "generic"


def load_value_types(path: Optional[str] = None) -> Dict[str, Any]:
    """惰性加载类型资源 (data/r8/value-types.json); 缺失时返回空表 (退化为形态过滤)。"""
    global _VT_CACHE, _VT_PATH
    import json as _json
    import pathlib as _pl
    if _VT_CACHE is not None and path is None:
        return _VT_CACHE
    if path is None:
        path = str(_pl.Path(__file__).resolve().parents[1] / "data" / "r8" / "value-types.json")
    try:
        data = _json.loads(_pl.Path(path).read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if path is None or _VT_PATH in (None, path):
        _VT_CACHE, _VT_PATH = data, path
    return data


def type_match(value: str, vtype: str, resources: Mapping[str, Any]) -> bool:
    """值是否属于期望类型 (封闭词表/形态规则)。generic → 一律通过 (由形态过滤负责)。"""
    if vtype in ("", "generic"):
        return True
    v = normalize_value(value)
    if not v:
        return False
    low = v.lower()
    def _in(key, item):
        return any(item == x.lower() for x in (resources.get(key) or []))
    def _contains(key, item):
        return any(re.search(r"\b" + re.escape(x.lower()) + r"\b", item) for x in (resources.get(key) or []))
    if vtype == "count":
        return bool(re.search(r"\b\d+\b|\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", low))
    if vtype == "us_state":
        return _in("us_states", low) or _contains("us_states", low)
    if vtype == "country":
        return _in("countries", low) or _contains("countries", low)
    if vtype == "place":
        if _in("cities", low) or _in("us_states", low) or _in("countries", low):
            return True
        # 所有格 ("Charlotte's Web") 与长 NP 不得仅凭包含判为地名 —— 城市名与
        # 人名大量撞名 (Charlotte/Austin/Orlando…), 2026-09-11 单测抓出的假阳性源
        if "'s" in low or "’s" in low:
            return False
        words = low.split()
        if len(words) <= 3:
            return _contains("cities", low) or _contains("us_states", low) or _contains("countries", low)
        return False
    if vtype in ("work_book", "work_media"):
        if v[:1] in "\"'“”" or '"' in value:
            return True
        words = v.split()
        titled = len(words) >= 2 and sum(1 for w in words if w[:1].isupper()) >= max(2, len(words) - 1)
        return bool(titled or _in("media_words", words[-1].lower().strip(".,")))
    if vtype == "activity":
        if _in("activities", low):
            return True
        words = low.split()
        if len(words) <= 3 and words and re.match(r"^\w+ing$", words[0]):
            return True
        return _contains("activities", low) and len(words) <= 2
    return True


def score_member(value: str, question: str) -> int:
    """确定性打分: 题面词重合 + NP 形态 + 专名 + 长度适中。"""
    s = 0
    if _tokens(value) & _tokens(question):
        s += 2
    words = value.split()
    if 1 <= len(words) <= 6:
        s += 1
    if value[:1].isupper():
        s += 1
    if len(value) >= 40:
        s -= 1
    return s


def clean_members(
    pairs: Sequence[Tuple[str, Any]],
    question: str = "",
    entity: str = "",
    vtype: Optional[str] = None,
    resources: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """(value, fact) 对 → 规范化/去噪/去重/排序后的成员列表。

    pairs: [(value, fact)] — fact 至少有 .ts / .msg_ref (SlotFact 兼容)。
    返回: [{"value","ts","msg_ref","score"}] 按 (score desc, ts asc, value) 稳定排序。
    """
    vt = vtype if vtype is not None else detect_value_type(question)
    if resources is None and vt not in ("", "generic"):
        resources = load_value_types()
    out: Dict[str, Dict[str, Any]] = {}
    for value, fact in pairs:
        for part in split_enumeration(value or ""):
            if is_noise_value(part):
                continue
            if vt and vt != "generic" and resources is not None:
                if not type_match(part, vt, resources):
                    continue
            key = part.lower()
            if key in out:
                continue
            out[key] = {
                "value": part,
                "ts": getattr(fact, "ts", None),
                "msg_ref": getattr(fact, "msg_ref", "") or "",
                "score": score_member(part, question),
            }
    members = list(out.values())
    members.sort(key=lambda m: (-m["score"], str(m["ts"] or ""), m["value"]))
    return members


def render_closure(
    members: Sequence[Mapping[str, Any]],
    entity: str,
    slot: str,
    budget_chars: int = 1800,
    max_members: int = 25,
) -> str:
    """成员 → "[SLOT EVIDENCE]" 段 (受预算约束, 行含 provenance)。"""
    if not members:
        return ""
    head = f"[SLOT EVIDENCE] {entity} | {slot} (确定性槽证据, 时间序):"
    lines: List[str] = []
    used = len(head)
    for m in members[:max_members]:
        line = f"- value: {m['value']}  (msg {m['msg_ref']})".rstrip()
        if used + len(line) + 1 > budget_chars:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return "\n".join([head] + lines)


def render_closure_for_entry(
    rows: Sequence[Any],
    entity: str,
    slot: str,
    question: str,
    budget_chars: int = 1800,
    max_members: int = 25,
) -> str:
    """SlotFact 行 → 闭包段 (去噪 + 排序 + 预算)。"""
    pairs = [(getattr(f, "value", "") or "", f) for f in rows]
    members = clean_members(pairs, question=question, entity=entity)
    return render_closure(members, entity, slot, budget_chars=budget_chars,
                          max_members=max_members)


# ── 题面现场路由 (P6-lite, 替代 qa_id→slot 查表; 无 LLM) ──────────────────

_STOP_TOKENS = {"what", "which", "when", "where", "who", "whom", "whose", "why", "how",
                "many", "much", "does", "did", "has", "have", "had", "is", "are", "was",
                "were", "the", "a", "an", "and", "or", "of", "in", "at", "to", "for",
                "with", "on", "by", "from", "about", "that", "this", "these", "those"}


def route_question(
    question: str,
    speakers: Sequence[str] = (),
    triggers: Optional[Mapping[str, Sequence[str]]] = None,
    max_slots: int = 2,
) -> List[Dict[str, Any]]:
    """题面 → [{"entity","slot","score","via"}] (确定性; 不读 qa_id)。

    entity: 题面专名 ∩ 会话说话人 (无交集时取首个专名);
    slot: 触发词在题面出现 (子串/词边界), 按命中数与词长打分取前 max_slots。
    """
    q = (question or "").strip()
    if not q:
        return []
    q_low = q.lower()
    proper = [t for t in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", q) if t.lower() not in _STOP_TOKENS]
    spk_set = {s for s in speakers if s}
    entity = next((t for t in proper if t in spk_set), proper[0] if proper else "")
    slots: List[Dict[str, Any]] = []
    for slot, trigs in (triggers or {}).items():
        hits = [t for t in (trigs or []) if t and re.search(r"\b" + re.escape(str(t).lower()), q_low)]
        if hits:
            slots.append({"slot": slot, "score": len(hits) + max(len(h) for h in hits) / 100.0,
                          "hits": hits})
    slots.sort(key=lambda s: (-s["score"], s["slot"]))
    out = []
    for s in slots[:max_slots]:
        out.append({"entity": entity, "slot": s["slot"], "score": round(s["score"], 3),
                    "via": "route:question"})
    return out
