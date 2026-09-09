"""R8 E2 模态/归属一等化 — core/modality_rules.py (纯函数).

本体论验收句: 模态与主体是否被确定性标注?
    → 规则模块五值闭域标注 + speaker/recent_entity 归属规则 → 过。

`classify(text)` → MODALITIES 五值闭域 {actual, planned, wish, inferred, in_talks}:
    纯规则序贯匹配 (先硬特征后软词, 确定性无 LLM):
    ① "had planned to" (完成时已执行) → actual;
    ② "in talks" → in_talks (q0004 洽谈态判别轴, 与 wish 可分);
    ③ 修辞条件 "if you want" 剥离 (不构成愿望);
    ④ "going to"/"plan…to" → planned;
    ⑤ wish/hope/aspire/would → wish;
    ⑥ "i think"/推测词 → inferred;
    ⑦ 默认 → actual。

`extract_subject(text, speaker, recent_entity)` → 归属主体:
    双人消息 he/she/they/their → recent_entity (非说话人);
    i/my/we → speaker; 具名 recent_entity → recent_entity; 无主语省略句默认 speaker。
"""
from __future__ import annotations

import re
from typing import Optional

# 五值闭域 (定序元组: 供成员判定 + 保序遍历)。
MODALITIES: tuple[str, ...] = ("actual", "planned", "wish", "inferred", "in_talks")

# 修辞条件: "if you want ..." 尾从句不构成愿望。
_RHETORICAL_COND_RE = re.compile(r"\bif\s+(?:you|they|we|anyone)\s+want\w*\b.*$")
# 计划: going to / plan(s|ned|ning) to
_PLAN_RE = re.compile(r"\b(?:going to|plan\w*\s+to)\b")
# 愿望: wish/hope/aspire/would (含屈折变体)
_WISH_RE = re.compile(r"\b(?:wish\w*|hope\w*|aspire\w*|would)\b")
# 推断标记
_INFERRED_RE = re.compile(
    r"\b(?:i think|i believe|i guess|i suppose|maybe|probably|"
    r"seems?|appears?|likely|could be)\b"
)
# 主语归属: 第三人称代词 / 第一人称代词
_THIRD_PERSON_RE = re.compile(
    r"\b(?:he|she|they|them|their|theirs|him|his|her|hers)\b"
)
_FIRST_PERSON_RE = re.compile(r"\b(?:i|me|my|mine|we|us|our|ours)\b")


def classify(text: str) -> str:
    """文本 → 五值闭域模态 (确定性规则, 输出恒落 MODALITIES)。"""
    t = (text or "").lower()
    if not t.strip():
        return "actual"
    # ① 完成时已执行: "had planned to" (过去完成时, 计划已执行) → actual
    if re.search(r"\bhad\s+planned\b", t):
        return "actual"
    # ② 洽谈态 (q0004 判别轴): "in talks" → in_talks
    if "in talks" in t:
        return "in_talks"
    # ③ 剥离修辞条件 "if you want ..." (不构成愿望)
    core_t = _RHETORICAL_COND_RE.sub("", t)
    # ④ 计划
    if _PLAN_RE.search(core_t):
        return "planned"
    # ⑤ 愿望
    if _WISH_RE.search(core_t):
        return "wish"
    # ⑥ 推断
    if _INFERRED_RE.search(core_t):
        return "inferred"
    # ⑦ 默认 actual (事实/能力陈述)
    return "actual"


def _has_name(text: str, name: Optional[str]) -> bool:
    """具名实体是否以整词形式出现在文本中 (大小写不敏感)。"""
    if not name:
        return False
    return re.search(r"\b" + re.escape(str(name).lower()) + r"\b", text) is not None


def extract_subject(text: str, speaker: str,
                    recent_entity: Optional[str]) -> Optional[str]:
    """双人消息归属主体: he/she→recent_entity, i/my→speaker, 具名→recent, 省略→speaker。"""
    t = (text or "").lower()
    # ① 第三人称代词 (he/she/they/their/...) → 非说话人 (recent_entity)
    if _THIRD_PERSON_RE.search(t):
        return recent_entity if recent_entity is not None else speaker
    # ② 第一人称代词 (i/my/we/...) → 说话人
    if _FIRST_PERSON_RE.search(t):
        return speaker
    # ③ 具名提及 recent_entity / speaker
    if _has_name(t, recent_entity):
        return recent_entity
    if _has_name(t, speaker):
        return speaker
    # ④ 无主语省略句 → 默认归属 speaker
    return speaker
