# -*- coding: utf-8 -*-
"""达摩院 R6-A 事实簇聚合 — 会话内确定性实体-谓词完整列表 (纯 stdlib, 零 LLM).

实证依据: 达摩院 r5 研究 (cat1 88 错: 计数/枚举漏项 46.6% + 追加超集 26.1%;
cat4 list-gold 溢出) — reader 从平铺消息里自己数, 数漏 (只 Kyle/Sara、3/4 狗名、
7→6 漏算)、重复计数同一事件 (conv-42#q0042 同一拒信数两次)。R6-A 在 ctx 装配
层 (P0-b TIME ANCHORS 同路径) 追加 [FACT CLUSTERS] 段: 确定性正则把同类事实
(same entity category + same predicate) 汇聚为带消息号引用的完整列表, 服务
计数/枚举题 (how many / how many times / list / name all / who); 事件实例按
"同实体 + 同谓词 + 同日期窗口 (周归一)" 去重, 只计一次。原文消息逐字不变
(聚合是附加段); 无 LLM、无外部依赖 (re + datetime), 与 time_anchors 同哲学
(确定性正则 + 实体名索引, 防 P1-a graphify 的重型依赖)。

形态:
    [FACT CLUSTERS]
    owns_pet: 3 pets: Toby (msg 1), Sara (msg 2), Mango (msg 3)
    rejection: 1 instance from Acme (msg 4, 5)

组织段文本只陈述事实 (无祈使)。开关 env FACT_CLUSTERS=0|1 (默认 0, 与
SESSION_SCOPE 正交 — 仅消费 DIRECT EVIDENCE 已限定消息, scope=on 时即会话内)。
"""
from __future__ import annotations

import datetime
import re
from typing import Iterable, List, Optional, Tuple

import time_anchors as ta

# ── 谓词模式表 (确定性, 有界; 值实体 = 捕获的专名) ──────────────────────────
# 每条: (predicate, category_hint, regex)。category_hint 用于计数行名词 (pets)。

_PET_KINDS = r"dog|cat|puppy|kitten|parrot|fish|hamster|rabbit|turtle|snake|bird|pet"
# "my dog Toby" / "our new puppy Rex"
_RE_PET_POSS = re.compile(
    rf"\b(?:my|our|his|her)\s+(?:new\s+)?(?P<kind>{_PET_KINDS})\s+"
    rf"(?P<name>[A-Z][a-zA-Z]+)\b")
# "a dog named Toby" / "adopted a kitten called Sara" / "got a puppy Rex"
_RE_PET_NAMED = re.compile(
    rf"\b(?P<kind>{_PET_KINDS})\s+(?:named|called|that I call)\s+"
    rf"(?P<name>[A-Z][a-zA-Z]+)\b")
# "I have a dog" / "we adopted a cat" — 无名字: 不产出值 (无法枚举名), 跳过
_RE_PET_ADOPT = re.compile(
    rf"\b(?:adopted|got|bought|welcomed|rescued)\s+(?:a|an|another|new)?\s*"
    rf"(?P<kind>{_PET_KINDS})\s+(?:named|called)\s+(?P<name>[A-Z][a-zA-Z]+)\b")
# "I have 3 pets" / "we have two dogs" — 计数直述 (作为计数证据行)
_RE_PET_COUNT = re.compile(
    rf"\b(?:I|we|they|he|she)\s+(?:have|has|had)\s+(?P<n>one|two|three|four|five|"
    rf"six|seven|eight|nine|ten|\d+)\s+(?:pets|pet)\b", re.IGNORECASE)
# "my son X" / "my daughter Y" — 孩子 (家人计数题常用)
_RE_KID_POSS = re.compile(
    r"\bmy\s+(?:son|daughter|kid|child|boy|girl)\s+(?:named|called)?\s*"
    r"(?P<name>[A-Z][a-zA-Z]+)\b")

# ── 事件实例模式 (去重对象: 同实体+同事件类型+同日期窗口只计一次) ──────────
# 实体 = 来源/对象专名 (公司/学校/申请对象), 事件类型 = rejection/offer/...。
# 实义表述 (assertion=True): 消息含动作动词, 事件日期可锚 (相对词) 或取消息日。
# 回指表述 (assertion=False): 仅 'the X rejection' 式回顾 — 不产生新实例,
# 只并入同 (type, entity) 已有实例 (conv-42 同一拒信两条消息数两次的修复路径)。
_RE_EVENT_REJECTION = re.compile(
    r"\b(?:got|received|had|sent|sent\s+in|sent\s+out|got\s+back)\s+"
    r"(?:a|an|the|another|one)?\s*(?:rejection|offer|acceptance|admission|"
    r"response|email|letter|news)\s*(?:letter|email)?\s*(?:from|about)?\s*"
    r"(?P<entity>[A-Z][a-zA-Z]+)?", re.IGNORECASE)
# "applied to X" / "interview at X" — 申请事件 (实体=机构)
_RE_EVENT_APPLY = re.compile(
    r"\b(?:applied|applying|interview(?:ed|ing)?)\s+(?:to|at|for|with)\s+"
    r"(?P<entity>[A-Z][A-Za-z]+)\b")
# "went to X" / "visited X" — 地点事件 (实体=地点, 单词大写或常见地点词)
_RE_EVENT_TRIP = re.compile(
    r"\b(?:went|visited|went\s+to|travel(?:led|ed)?\s+to)\s+"
    r"(?P<entity>[A-Z][A-Za-z]+|the\s+[A-Za-z]+)\b")
# 回指表述: 'the Acme rejection' / 'that offer from Acme' — 并入已有实例
_RE_EVENT_REFER = re.compile(
    r"\b(?:the|that|this)\s+(?:(?P<entity>[A-Z][a-zA-Z]+)\s+)?"
    r"(?:rejection|offer|letter|email|news|response)\s+from\s+"
    r"(?P<entity2>[A-Z][a-zA-Z]+)|"
    r"\b(?:the|that|this)\s+(?P<entity3>[A-Z][a-zA-Z]+)\s+"
    r"(?:rejection|offer|letter|email|news|response)\b", re.IGNORECASE)

_OWN_PREDICATES = [("owns_pet", "pet", _RE_PET_POSS),
                   ("owns_pet", "pet", _RE_PET_NAMED),
                   ("owns_pet", "pet", _RE_PET_ADOPT),
                   ("owns_child", "child", _RE_KID_POSS)]
_EVENT_PREDICATES = [("rejection", _RE_EVENT_REJECTION, True),
                     ("application", _RE_EVENT_APPLY, True),
                     ("trip", _RE_EVENT_TRIP, True),
                     ("rejection", _RE_EVENT_REFER, False)]


def _strip_prefix(content: str) -> str:
    """去掉 [date: ...] 与 [speaker] 前缀 → 正文 (值抽取区; 复用 time_anchors)。"""
    return ta._msg_body(content)  # noqa: SLF001 — 同模块族共享前缀剥离


def _speaker_of(content: str) -> Optional[str]:
    """消息 [speaker] 前缀 → 说话人名; 无 → None。"""
    m = re.search(r"\[date:[^\]]*\]\s*\[([^\]]+)\]", (content or "")[:200])
    return m.group(1).strip() if m else None


def _value_lines(content: str) -> List[Tuple[str, str, str]]:
    """单条消息 → [(predicate, category, value)] (有界谓词模式, 确定性)。

    值实体去重 (同消息同谓词同值只留一次); 计数直述 (I have N pets) 不产出值,
    由聚合层转成完整性信号 (列表行已含消息号, 计数 = len)。"""
    body = _strip_prefix(content)
    out: List[Tuple[str, str, str]] = []
    seen = set()
    for pred, cat, rx in _OWN_PREDICATES:
        for m in rx.finditer(body):
            v = (m.group("name") or "").strip()
            if v and (pred, v) not in seen:
                seen.add((pred, v))
                out.append((pred, cat, v))
    return out


def _event_instances(content: str, assertion_only: bool = False
                     ) -> List[Tuple[str, str]]:
    """单条消息 → [(event_type, entity, is_assertion)] 事件实例候选。

    实体缺省 (如 "got a rejection" 未指明来源) → 用说话人作实体, 保证跨消息
    同事件可归并 (conv-42: 同说话人同周内多次提到同一拒信 → 只计一次)。
    assertion_only=True → 只保留实义表述 (回指表述在聚合层并入已有实例)。"""
    body = _strip_prefix(content)
    speaker = _speaker_of(content) or "?"
    out: List[Tuple[str, str, bool]] = []
    for typ, rx, is_assert in _EVENT_PREDICATES:
        if assertion_only and not is_assert:
            continue
        for m in rx.finditer(body):
            e = ((m.groupdict().get("entity") or m.groupdict().get("entity2")
                  or m.groupdict().get("entity3") or "")).strip() or speaker
            if e.lower() in ("the", "an", "a"):
                e = speaker
            key = (typ, e, is_assert)
            if key not in out:
                out.append(key)
    return out


def _event_window(content: str) -> Optional[Tuple[datetime.date, datetime.date]]:
    """事件日期窗口: 时间锚行解析区间 (相对词日历) 优先; 无 → 消息 [date:] 日。
    同实体+同事件+窗口周重叠 → 去重只计一次。"""
    anchors = ta.find_anchors(content)
    if anchors:
        a = anchors[0]
        return a.start, a.end
    d = ta.message_anchor_date(content)
    if d is not None:
        return d, d
    return None


def _week_key(window: Tuple[datetime.date, datetime.date]) -> str:
    """窗口归一到 ISO 周 (去重粒度: 同周视为同事件实例 — conv-42 拒信口径)。"""
    s = window[0]
    iso = s.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def aggregate(numbered_docs: Iterable[Tuple[int, str]]
              ) -> Tuple[dict, dict]:
    """编号消息 [(N, content)] → (拥有值聚合, 事件实例聚合)。

    拥有值聚合: {predicate: {value: [msg_nos]}} — 完整列表 (消息号引用)。
    事件实例聚合: {(event_type, entity, week): [msg_nos]} — 同实体+同事件+
    同日期窗口合并; 回指表述 ('the Acme rejection', 无动作动词) 并入同
    (type, entity) 的既有实例, 不产生新实例 (去重, conv-42#q0042 修复)。
    两者均不改原文。"""
    own: dict = {}
    events: dict = {}
    # pass 1: 实义表述建实例 (窗口 = 锚解析或消息日期)
    for n, content in numbered_docs:
        for pred, cat, v in _value_lines(content):
            own.setdefault(pred, {}).setdefault(v, [])
            if n not in own[pred][v]:
                own[pred][v].append(n)
        for typ, e, _is_assert in _event_instances(content, assertion_only=True):
            w = _event_window(content)
            wk = _week_key(w) if w else "no-date"
            events.setdefault((typ, e, wk), [])
            if n not in events[(typ, e, wk)]:
                events[(typ, e, wk)].append(n)
    # pass 2: 回指表述并入已有实例 (同 type+entity); 无既有实例 → 忽略
    # (纯回顾无实义消息说明事件本体不在 DIRECT EVIDENCE 内, 不虚构实例)。
    for n, content in numbered_docs:
        for typ, e, _is_assert in _event_instances(content, assertion_only=False):
            for wk in list(events.keys()):
                if wk[0] == typ and wk[1] == e:
                    if n not in events[wk]:
                        events[wk].append(n)
                    break
    return own, events


def _plural(count: int, category: str) -> str:
    if category == "pet":
        return "pets" if count != 1 else "pet"
    if category == "child":
        return "children" if count != 1 else "child"
    return "items"


def render_cluster_block(numbered_docs: Iterable[Tuple[int, str]]) -> str:
    """编号消息 → [FACT CLUSTERS] 段文本; 无可聚合事实 → '' (不输出空段)。

    拥有谓词行: 'owns_pet: 3 pets: Toby (msg 1), Sara (msg 2), Mango (msg 3)'
    事件实例行: 'rejection: 1 instance from Acme (msg 4, 5)' — 同实体+同周去重。
    行文本只陈述聚合事实 + 消息号指针, 无祈使。"""
    docs = [(n, c) for n, c in numbered_docs]
    own, events = aggregate(docs)
    lines: List[str] = []
    for pred in ("owns_pet", "owns_child"):
        vals = own.get(pred)
        if not vals:
            continue
        cat = "pet" if pred == "owns_pet" else "child"
        entries = [f"{v} (msg {', '.join(str(x) for x in ns)})"
                   for v, ns in sorted(vals.items(), key=lambda kv: str(kv[0]))]
        count = len(vals)
        lines.append(f"{pred}: {count} {_plural(count, cat)}: " + ", ".join(entries))
    # 事件实例: 按键 (type, entity) 汇总周窗口去重后的次数
    by_type_entity: dict = {}
    for (typ, e, wk), ns in sorted(events.items()):
        by_type_entity.setdefault((typ, e), []).append(ns)
    for (typ, e), ns_groups in sorted(by_type_entity.items()):
        flat = sorted({x for group in ns_groups for x in group})
        k = len(ns_groups)
        lines.append(
            f"{typ}: {k} instance{'s' if k != 1 else ''} from {e} "
            f"(msg {', '.join(str(x) for x in flat)})")
    if not lines:
        return ""
    return "[FACT CLUSTERS]\n" + "\n".join(lines)


def append_fact_clusters(ctx: str, max_docs: int = 30) -> str:
    """组织路径 (a): ctx 尾部追加 '[FACT CLUSTERS]' 段 (引用 DIRECT EVIDENCE
    编号消息; 只扫编号 raw 行 — 组织段/块摘要不参与聚合)。无聚合事实 → 原 ctx。
    与 append_anchor_block 同形态: 附加段在 ctx 尾部, 原文消息零改动 (AC4)。"""
    items: List[Tuple[int, str]] = []
    for ln in ctx.splitlines():
        hit = ta.split_message_line(ln)
        if hit is not None:
            items.append(hit)
    items = items[:max_docs]
    block = render_cluster_block(items)
    if not block:
        return ctx
    return ctx + "\n\n" + block
