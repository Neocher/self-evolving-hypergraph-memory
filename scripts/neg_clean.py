# -*- coding: utf-8 -*-
"""达摩院 R6-C 摘要清洁 — ctx 装配层否定式表述过滤 + 组织段题面实体优先 (纯 stdlib).

实证依据: 达摩院 r5 研究 (cat4 126 错: 拒答/未覆盖 55.6% 含拒答标记 29; new_wrong
118 主回退 ~43% 样本) — reader 看到 [MEMORY BLOCK]/[FACTS] 摘要里 "unspecified/
无证据/cannot be determined" 式否定表述 → 后退拒答; 另有选错事件因组织段未按
题面实体收窄。R6-C 在 ctx 装配层 (不改 LLM 摘要生成 prompt — prompt 层覆辙
R2c -8.3pp; 改装配后处理 = 结构层):
  1. 摘要否定句过滤: [MEMORY BLOCK]/[ENTITY]/[FACT TYPES] 等摘要段中的否定式
     表述整句剔除, 只留陈述原文事实的句子 + 消息号指针; 无事实可说 → 输出空段
     而非否定句。raw 消息行 (编号 '[N] ...') 逐字不动 (AC4)。
  2. 组织段题面实体优先: 题面含实体名 (人名/宠物名/活动名) 时, 该实体的
     [ENTITY ...] 段在组织段中前置排列 (会话内), 其次才全量。

开关 env NEG_CLEAN=0|1 (默认 0) — 默认 off 与 v6.17.0 逐字节等价 (零回归锚点)。
组织段文本只陈述事实, 无祈使。
"""
from __future__ import annotations

import re
from typing import List, Optional

# 否定式表述标记 (整句剔除; 特征化短语, 不误伤 'not' 正常否定叙事)
_NEG_MARKERS = [
    r"cannot\s+be\s+determined", r"cannot\s+determine",
    r"unable\s+to\s+determine", r"could\s+not\s+be\s+determined",
    r"not\s+mentioned", r"not\s+specified", r"unspecified",
    r"no\s+evidence", r"not\s+available", r"not\s+stated",
    r"not\s+indicated", r"does\s+not\s+say", r"did\s+not\s+mention",
    r"没有提到", r"未提及", r"无证据", r"无法确定", r"未指定",
    r"无法判断", r"无从得知", r"不清楚",
]
_NEG_RE = re.compile("|".join(_NEG_MARKERS), re.IGNORECASE)
# 摘要行前缀 (标题行/内容行; raw 编号行 '[N]' 除外)
_MEMORY_BLOCK_RE = re.compile(r"^\[MEMORY BLOCK (\d+)\]\s*(.*)$")
# 组织段通用标题 (ENTITY/RELATIONS/FACT TYPES/GLOBAL CONTEXT/DIRECT EVIDENCE/
# ROUND2 SUPPLEMENTAL/TIME ANCHORS — 段边界判定; '[ENTITY: X]' 另设精确正则)
_ANY_HEAD_RE = re.compile(
    r"^\[(?:ENTITY:.*|RELATIONS|FACT TYPES[^\]]*|GLOBAL CONTEXT[^\]]*|"
    r"DIRECT EVIDENCE|ROUND2 SUPPLEMENTAL EVIDENCE|TIME ANCHORS[^\]]*)\]$")
_SECTION_HEAD_RE = _ANY_HEAD_RE
_ENTITY_HEAD_RE = re.compile(r"^\[ENTITY:\s*(.*)\]$")

# 题面专名提取: 停用词表 (疑问词/介词/冠词等, 句首大写词过滤)
_PROPER_STOP = {
    "which", "what", "when", "where", "who", "whom", "whose", "why", "how",
    "the", "a", "an", "in", "on", "at", "of", "for", "from", "to", "with",
    "and", "or", "but", "did", "does", "do", "is", "are", "was", "were",
    "has", "have", "had", "it", "its", "this", "that", "there", "their",
    "they", "he", "she", "him", "her", "his", "we", "our", "you", "your",
    "i", "my", "me", "name", "names", "many", "much", "any", "all", "some",
}


def is_negative(text: str) -> bool:
    """文本是否含否定式表述标记 (unspecified/无证据类)。"""
    return bool(_NEG_RE.search(text or ""))


def _split_sentences(text: str) -> List[str]:
    """英文/混合摘要按句切分 (粗略: 句号/问号/感叹号后)。"""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in (x.strip() for x in parts) if p]


def filter_negative_sentences(text: str) -> str:
    """摘要文本 → 剔除含否定式表述的整句, 保留陈述事实的句子。

    全否定 → '' (调用方输出空段, 不输出否定句)。只处理传入的摘要文本,
    不改任何原始消息 (AC4)。"""
    if not text:
        return ""
    kept = [s for s in _split_sentences(text) if not is_negative(s)]
    return " ".join(kept)


def clean_summary_line(line: str) -> str:
    """单条 '[MEMORY BLOCK N] <text>' 行 → 否定句剔除后的行 (或 '' 表示空)。

    只剔除该行正文中的否定句; 行首前缀 '[MEMORY BLOCK N]' 保留。非摘要行原样。"""
    m = _MEMORY_BLOCK_RE.match(line)
    if not m:
        return line
    body = filter_negative_sentences(m.group(2))
    if not body:
        return ""  # 无事实可说 → 空行 (装配层删行, 输出空段而非否定句)
    return f"[MEMORY BLOCK {m.group(1)}] {body}"


def clean_ctx(ctx: str) -> str:
    """ctx 装配层摘要否定句过滤 (NEG_CLEAN=1 才调用)。

    只处理摘要行 ('[MEMORY BLOCK N] ...' / '[ENTITY: X]' 段内事实行) — 否定句
    整句剔除; 编号 raw 消息行 ('[N] [date: ...] ...') 逐字不动 (AC4 原文不变)。
    段内全部为空 → 删除该段 (标题行若其后无内容行也删), 输出空段而非否定句。"""
    if not ctx:
        return ctx
    out: List[str] = []
    in_entity_or_section = False  # ENTITY/FACT TYPES/GLOBAL CONTEXT 段内
    for ln in ctx.splitlines():
        if re.match(r"^\[\d+\]\s", ln):
            in_entity_or_section = False  # raw 消息行: 逐字保留, 段边界结束
            out.append(ln)
            continue
        if _SECTION_HEAD_RE.match(ln) or _ENTITY_HEAD_RE.match(ln):
            out.append(ln)
            in_entity_or_section = True
            continue
        if _MEMORY_BLOCK_RE.match(ln):
            cleaned = clean_summary_line(ln)
            if cleaned:
                out.append(cleaned)
            continue
        if in_entity_or_section:
            # 段内事实行 ('- <fact>' / '  · <attr>: <val>'): 否定句剔除
            cleaned = filter_negative_sentences(ln)
            if cleaned:
                out.append(cleaned)
            continue
        out.append(ln)
    return "\n".join(out) + ("\n" if ctx.endswith("\n") else "")


# ── 组织段题面实体优先 (R6-C 2) ───────────────────────────────────────────

def extract_proper_names(question: str) -> List[str]:
    """题面专名候选: 连续大写词序列 (过滤停用词/句首疑问词)。

    人名/宠物名/活动名通常是句中大写专名; 句首疑问词 (Which/What/...) 与
    小写词排除。无 → [] (调用方原样, 零回归)。"""
    if not question:
        return []
    # 句首词剔除 (疑问词等大写开头但非专名)
    no_lead = re.sub(r"^\s*[A-Z][a-z]+\b", "", question)
    found, seen = [], set()
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+\b", no_lead):
        w = m.group(0)
        if w.lower() in _PROPER_STOP:
            continue
        if w not in seen:
            seen.add(w)
            found.append(w)
    return found[:5]


def entity_first_ctx(ctx: str, question: Optional[str] = None) -> str:
    """组织段题面实体优先 (NEG_CLEAN=1 才调用): 题面含实体名时, 命中实体的
    '[ENTITY: X]' 段在全部 ENTITY 段中前置 (按题面出现序), 未命中段随其后;
    ENTITY 区前后内容 (RELATIONS/FACT TYPES/raw 行等) 保持原序与原文 (AC4)。

    ENTITY 区 = 首个 [ENTITY: X] 标题起至末个 ENTITY 块结束 (下个非 ENTITY
    标题或 ctx 尾为止) 的连续区 (ontology_organize 的 ENTITY 段本就相邻)。
    无题面实体 / ENTITY 段不足 2 个 / 已有序 → 原 ctx (零回归)。标题可能带
    conv URI 前缀 ('conv-41/Caroline'), 取末段与题面匹配。"""
    if not ctx or not question:
        return ctx
    names = extract_proper_names(question)
    if not names:
        return ctx
    lines = ctx.splitlines()
    heads = [i for i, ln in enumerate(lines) if _ENTITY_HEAD_RE.match(ln)]
    if len(heads) < 2:
        return ctx
    first = heads[0]
    last_head = heads[-1]
    end = len(lines)
    for j in range(last_head + 1, len(lines)):
        if _SECTION_HEAD_RE.match(lines[j]):
            end = j
            break

    def _key(b: List[str]) -> int:
        m = _ENTITY_HEAD_RE.match(b[0])
        if not m:
            return 10 ** 9
        tail = m.group(1).strip().split("/")[-1]
        for i, n in enumerate(names):
            if n.lower() == tail.lower():
                return i
        return 10 ** 9

    bounds = list(heads) + [end]
    blocks = [lines[bounds[k]:bounds[k + 1]] for k in range(len(heads))]
    ordered = sorted(blocks, key=_key)
    if ordered == blocks:
        return ctx
    prefix = lines[:first]
    suffix = lines[end:]
    out = list(prefix)
    for k, b in enumerate(ordered):
        if k:
            out.append("")
        out.extend(b)
    out.extend(suffix)
    return "\n".join(out).rstrip("\n") + ("\n" if ctx.endswith("\n") else "")
