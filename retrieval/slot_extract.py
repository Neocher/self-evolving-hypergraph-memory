"""R8 Step3a 确定性槽证据抽取 — retrieval/slot_extract.py (纯规则, 零 IO / 零 LLM / 零系统钟).

本体论验收句: 评测面原文 (EpisodeNode/会话消息) 如何确定性到达 (entity×slot) 证据?
    → 归属过滤 (自述/他述含名/他述直呼) + 触发词边界过滤 + 模态门
      + 触发词宾语窗轻量值切分 (≤8 tokens, 尽力而为) → 过。

词表一律由调用方注入 (本模块不读文件); ts 一律入参注入 (不取系统时钟)。
依赖 core.modality_rules.classify (五值模态门) + retrieval.slot_facts
(SlotFact/SlotIndex), 自身零 env 零 IO。

归属过滤 (句级, v1):
    - 自述:     句 speaker==entity
    - 他述含名:  句含 entity 整词 (如 "John's favorite game..." → John)
    - 他述直呼:  非 entity 说话人句含第二人称 you/your (双人会话定向; oracle
                 D16:19 Dave 对 Calvin 吉他 "Why did you make it so shiny?")
模态门: gate="did" → classify∈{planned,wish,inferred} 剔除; gate="offer" →
wish 剔除 (in_talks/actual 保留); gate="plan" → 不过滤。

值切分 (v1 诚实边界, 轻量尽力而为): 触发词命中 (词首边界) 后的宾语窗
(≤8 tokens, 修剪 a/the/my/our/new 等冠词/修饰前缀), 按 and/,/;/ but/then
切多值, 每片段=value; 窗空/纯触发词 → 回退整句文本 (strip 后)。
完整 NP 值解析不在 v1 (值级文法留 v2)。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.modality_rules import classify
from retrieval.slot_facts import SlotFact, SlotIndex

# 触发词词首边界 (允许屈折: "mix"→"mixes", "lab"→"Lab"; 不要求词尾边界)。
_TRIG_LEFT_BOUNDARY = r"(?<![a-z0-9])"
# 他述直呼第二人称 (双人会话定向)。
_SECOND_PERSON_RE = re.compile(r"\b(?:you|your|yours|yourself|yourselves)\b")
# 宾语窗前导冠词/修饰/连接词前缀 (修剪 a/the/my/our/new 等)。
_PREFIX_TRIM: frozenset = frozenset({
    "a", "an", "the", "this", "that", "these", "those",
    "my", "our", "your", "his", "her", "its", "their",
    "new", "some", "any", "and", "but", "then", "or",
})
# 宾语窗切多值分隔符: and/,/;/ but/then。
_VALUE_SEP_RE = re.compile(r"\s+\b(?:and|but|then)\b\s+|[,;]")
# 句切分: 标点后随空白 (消息可能含多句; 模态按句判, 避免跨句污染)。
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# 宾语窗 token 上限。
_MAX_WINDOW_TOKENS = 8


def extract_slot_facts(
    msgs: Sequence[Mapping[str, Any]],
    entity: str,
    slot: str,
    triggers: Sequence[str],
    gate: str = "did",
) -> List[SlotFact]:
    """会话内消息 (时间序) → (entity×slot) 确定性槽证据事实列表 (纯规则)。

    msgs: [{dia, session_id, speaker, text, ts(可比较)}], 会话内时间序;
    每命中句按值切分产出 0..n 条 SlotFact (session_id/msg_ref 取句所在消息,
    ep_id=消息在 msgs 内序号 1-based, ts=消息 ts)。
    """
    facts: List[SlotFact] = []
    for idx, msg in enumerate(msgs, start=1):
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        session_id = msg.get("session_id")
        ts = msg.get("ts")
        dia = msg.get("dia")
        speaker = msg.get("speaker") or ""
        for sent in _split_sentences(text):
            if not _attributed(sent, speaker, entity):
                continue
            lower = sent.lower()
            hits = _trigger_hits(lower, triggers)
            if not hits:
                continue
            if _modality_gated(sent, gate):
                continue
            anchor = _pick_anchor(hits)
            for value in _split_object_values(sent, lower, anchor):
                facts.append(SlotFact(
                    session_id=session_id,
                    ep_id=idx,
                    ts=ts,
                    entity=entity,
                    slot=slot,
                    value=value,
                    msg_ref=dia,
                ))
    return facts


def build_slot_index(
    msgs_all: Sequence[Mapping[str, Any]],
    lexicon: Mapping[str, Mapping[str, Any]],
    triggers: Mapping[str, Sequence[str]],
) -> SlotIndex:
    """多 (entity×slot) 批量抽取 → SlotIndex (供 Step3b 装配 retrieve_slot)。

    lexicon:  {qkey: {entities: [...], slot, gate}} — qkey 形如 "conv-50#q0001";
    有 conv 前缀时按 qkey 前缀裁剪 msgs_all 到该会话族, 避免跨会话同名 "you"
    直呼误归; triggers: {slot: [触发词]}。
    """
    all_facts: List[SlotFact] = []
    for qkey, item in lexicon.items():
        slot = item.get("slot") or ""
        gate = item.get("gate") or "did"
        trigs = list(triggers.get(slot, []) or []) if triggers else []
        conv = None
        if isinstance(qkey, str) and "#" in qkey:
            conv = qkey.partition("#")[0]
        for entity in item.get("entities") or []:
            if conv is not None:
                scoped = [
                    m for m in msgs_all
                    if str(m.get("session_id") or "").startswith(conv + "-")
                ]
                # [R8-SLOT-SCOPE-1 2026-09-11] session_id 两种真实形态:
                #   ① 标签串 "conv-26-s1" (旧测试/标签化调用方) → 前缀裁剪有效, 行为不变;
                #   ② int conversation_idx 0..9 (评测链路: bench _r8_conv_msgs ← 评测库
                #      e.session_id) → "0".startswith("conv-26-") 恒 False → 前缀裁剪恒空。
                # 曾因此静默零注入 (R8_SLOT=1 的 A/B ON 臂与 OFF 逐字节等价, 无日志无异常)。
                # 命中为空时回落调用方传入的 msgs_all: bench L1693-1697 已按本题会话裁剪
                # (episode_cache.session_id == ci), 故回落 == "本题会话范围", 不引入跨会话误归。
                scope_msgs = scoped if scoped else list(msgs_all)
            else:
                scope_msgs = list(msgs_all)
            all_facts.extend(
                extract_slot_facts(scope_msgs, entity, slot, trigs, gate=gate)
            )
    return SlotIndex.from_facts(all_facts)


# ── 归属过滤 ───────────────────────────────────────────────────────────────

def _attributed(sent: str, speaker: str, entity: str) -> bool:
    """句归属: 自述 (speaker==entity) / 他述含 entity 整词 / 他述直呼 (第二人称)。"""
    ent = (entity or "").strip()
    if not ent:
        return False
    if (speaker or "").strip().lower() == ent.lower():
        return True
    low = sent.lower()
    # 他述含名: entity 以整词出现 ("John's favorite..." → 'john' 后接 ' 非词符)。
    if re.search(
        r"(?<![a-z0-9])" + re.escape(ent.lower()) + r"(?!\w)", low
    ):
        return True
    # 他述直呼: 双人会话中非 entity 说话人句含 you/your → 定向 entity。
    return _SECOND_PERSON_RE.search(low) is not None


# ── 触发词边界过滤 ─────────────────────────────────────────────────────────

def _trigger_hits(lower: str, triggers: Sequence[str]) -> List[Dict[str, Any]]:
    """词首边界命中 (大小写不敏感; 词表已小写): [{trigger, start, end}]。"""
    hits: List[Dict[str, Any]] = []
    for trig in triggers:
        t = (trig or "").strip().lower()
        if not t:
            continue
        for m in re.finditer(_TRIG_LEFT_BOUNDARY + re.escape(t), lower):
            hits.append({"trigger": trig, "start": m.start(), "end": m.end()})
    return hits


def _pick_anchor(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """宾语窗锚点 = 最左命中 (同起点取更长, 令窗越过后缀屈折)。"""
    return min(hits, key=lambda h: (h["start"], -len(h["trigger"])))


# ── 模态门 ─────────────────────────────────────────────────────────────────

def _modality_gated(sent: str, gate: str) -> bool:
    """gate → 是否剔除该句 (classify 五值闭域; plan 门不过滤)。"""
    label = classify(sent)
    if gate == "plan":
        return False
    if gate == "offer":
        return label == "wish"
    # gate == "did" (缺省): planned/wish/inferred 剔除, actual/in_talks 保留。
    return label in ("planned", "wish", "inferred")


# ── 值切分 (轻量尽力而为, v1) ─────────────────────────────────────────────

def _split_object_values(sent: str, lower: str,
                         anchor: Mapping[str, Any]) -> List[str]:
    """锚触发词后的宾语窗 (≤8 tokens) → 值片段列表; 窗空/纯触发词 → 整句回退。

    修饰窗: 修剪冠词/连接词前缀 → 按 and/,/;/ but/then 切多值 → strip。
    """
    window_start = _window_start(sent, anchor["start"], anchor["end"])
    tail = sent[window_start:].strip()
    if not tail:
        return [sent.strip()]
    toks = tail.split()
    if len(toks) > _MAX_WINDOW_TOKENS:
        toks = toks[:_MAX_WINDOW_TOKENS]
    window = " ".join(toks)

    # 修剪窗前缀 (a/the/my/our/new/and...), 再切多值; 空 → 回退整句。
    stripped = _strip_prefix_tokens(window)
    if not stripped:
        return [sent.strip()]
    parts = [p.strip() for p in _VALUE_SEP_RE.split(stripped)]
    parts = [p for p in parts if p]
    if not parts:
        return [sent.strip()]
    # 每片段再剥一次前缀 (窗级只剥一次不够: "a car and the truck")。
    out: List[str] = []
    for p in parts:
        p2 = _strip_prefix_tokens(p)
        if p2:
            out.append(p2)
    return out or [sent.strip()]


def _window_start(sent: str, hit_start: int, hit_end: int) -> int:
    """锚命中结束 → 其所在 token 结束后的下一 token 起点 (跨过词尾屈折/标点)。"""
    n = len(sent)
    i = hit_end
    while i < n and not sent[i].isspace():
        i += 1
    while i < n and sent[i].isspace():
        i += 1
    return i


def _strip_prefix_tokens(text: str) -> str:
    """剥除前导冠词/修饰/连接词 token (小写/strip 后判), 返回余下文本。"""
    toks = text.split()
    while toks:
        probe = toks[0].strip(".,!?;:'\"").lower()
        if probe in _PREFIX_TRIM:
            toks = toks[1:]
        else:
            break
    return " ".join(toks).strip()


# ── 句切分 ─────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """消息文本 → 句列表 (标点后随空白切分; 保留句末标点, 丢弃空句)。"""
    norm = " ".join(text.split())
    return [s.strip() for s in _SENT_SPLIT_RE.split(norm) if s.strip()]
