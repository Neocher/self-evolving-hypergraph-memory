"""用户画像层 User-Profile (v5.39.0)
====================================
对标 Profile-Graph Memory（2606.06036）：显式用户画像层。
从用户直述（source_type=direct）节点提取稳定偏好/身份/工作，
检索时画像值命中 → score ×1.2 加分（复用 core-boost 模式，乘正因子单调），
search_profile 旁路返回画像上下文块（prepend 到 prompt，不参与主排序）。

持久化: data/user_profile.json（data/ 整体 gitignore，缺失/损坏 → 空 dict 降级）。
不用 GraphLite Hyperedge（≥2 成员硬约束 + 检索不可见 + N+1 反查）。
仅依赖 re/json/os/tempfile + fact_track.CORE_KEYWORDS（复用不另起词表，防漂移）。

【P2-单租户语义】画像为内存单例（app.py 全库扫描构建，query_router 模块级
全局持有），检索加分/旁路 context 不感知 namespace——当前产品单租户部署，
跨 namespace 画像共享为已知接受语义；多租户需按 {namespace: profile} 键控。
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile

from core.fact_track import CORE_KEYWORDS

logger = logging.getLogger(__name__)

# 复用 fact_track.CORE_KEYWORDS 的值提取型子集：过滤「一直/经常/始终/总是」
# 等无宾语值的修饰词，其余映射到分组 + 增量补「工作/工作于」——不另起词表，防漂移。
_MODIFIER_WORDS: frozenset[str] = frozenset({"一直", "经常", "始终", "总是"})
_GROUP_OF: dict[str, str] = {
    kw: ("identity" if kw in {"我是", "我住", "住在"}
         else "work" if kw == "职业"
         else "preferences")
    for kw in CORE_KEYWORDS
    if kw not in _MODIFIER_WORDS
}
_GROUP_OF["工作"] = "work"
_GROUP_OF["工作于"] = "work"
# 词序 = 匹配优先级：长模式优先（「住在」先于「我住」，防「我住在X」残留「在」；
# 「工作于」先于「工作」，防「我工作于X」残留「于」）
_PRIORITY: tuple[str, ...] = (
    "住在", "我是", "我住", "职业", "工作于", "工作",
    "喜欢", "偏好", "爱好", "擅长", "讨厌", "习惯",
)
_ZH_PATTERNS: tuple[tuple[str, str], ...] = tuple(
    (kw, _GROUP_OF[kw]) for kw in _PRIORITY if kw in _GROUP_OF
)

# 英文低误报词（I am/I like 必须绑 source 门控，不上词表）
_EN_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bi live in\b", re.I), "identity"),
    (re.compile(r"\bi work at\b", re.I), "work"),
    (re.compile(r"\bmy favorite\b", re.I), "preferences"),
)

# 句首第一人称前缀（防裸子串误匹配；画像对用户可见，容错要求高）
# my\b 使「My favorite ...」可达（低误报英文词，见 _EN_PATTERNS）
_SENT_START = re.compile(r"^(我|我的|my\b|i\b)", re.I)

# source_type 信任权重（direct 1.0 / tool 0.7 / inferred 0.5）
_SOURCE_WEIGHTS: dict[str, float] = {"direct": 1.0, "tool": 0.7, "inferred": 0.5}

# 宾语值归一：去空白/标点/「的」；前导「是」剥离（「我工作是老师」→「老师」）
_VALUE_CLEAN = re.compile(r"[\s，。！？、；：,.!?;:'\"()（）【】\[\]的]+")
_SEG_SPLIT = re.compile(r"[，。！？、；,.!?;]")

_DEFAULT_PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "user_profile.json",
)


def _normalize_value(raw: str) -> str:
    """模式词后宾语值归一：去标点/空格/「的」→ 紧凑串。"""
    v = _VALUE_CLEAN.sub("", raw or "").strip()
    if v.startswith("是"):
        v = v[1:]
    return v


def _value_from_tail(tail: str) -> str:
    """模式词后内容 → 取首个语义单元（到分隔标点为止）→ 归一。"""
    value = _normalize_value(_SEG_SPLIT.split(tail, 1)[0])
    return value if len(value) >= 2 else ""


def scan_preference_candidates(nodes: list[dict]) -> list[dict]:
    """扫描节点提取画像候选。

    硬门控（防 agent 报告误识别）：
      - source_type == "direct"（只有 source == "user" 能落 direct，models.py 保证）
      - 句首第一人称前缀 ^(我|我的|i\\b)
      - 中文复用 CORE_KEYWORDS 值提取子集 + 补「工作/工作于」；英文仅低误报词

    Returns:
        [{"value": str, "group": "preferences|identity|work", "source_type": "direct"}, ...]
    """
    candidates: list[dict] = []
    for node in nodes or []:
        content = str(node.get("content") or "").strip()
        if node.get("source_type") != "direct":
            continue
        if not content or not _SENT_START.match(content):
            continue
        # 中文模式词 first-match-wins（「工作于」先于「工作」）
        for kw, group in _ZH_PATTERNS:
            idx = content.find(kw)
            if idx >= 0:
                value = _value_from_tail(content[idx + len(kw):])
                if value:
                    candidates.append({"value": value, "group": group, "source_type": "direct"})
                break
        else:
            # 英文低误报词
            for pat, group in _EN_PATTERNS:
                m = pat.search(content)
                if m:
                    value = _value_from_tail(content[m.end():])
                    if value:
                        candidates.append({"value": value, "group": group, "source_type": "direct"})
                    break
    return candidates


def aggregate(candidates: list[dict]) -> dict:
    """按偏好值去重归一 + source_type 权重累积 + 多源计数加分。

    Returns:
        {"preferences": {value: {"weight", "sources"}}, "identity": {...}, "work": {...}}
    """
    acc: dict[str, dict[str, dict]] = {"preferences": {}, "identity": {}, "work": {}}
    for cand in candidates or []:
        value = _normalize_value(cand.get("value"))
        if len(value) < 2:
            continue
        group = cand.get("group") if cand.get("group") in acc else "preferences"
        entry = acc[group].setdefault(value, {"weight": 0.0, "sources": 0})
        entry["weight"] += _SOURCE_WEIGHTS.get(str(cand.get("source_type") or "inferred"), 0.5)
        entry["sources"] += 1
    # 多源计数加分：同一值 ≥2 个来源，每多 1 源 +0.1
    for entries in acc.values():
        for entry in entries.values():
            if entry["sources"] >= 2:
                entry["weight"] += 0.1 * (entry["sources"] - 1)
    return acc


def build_profile(candidates: list[dict]) -> dict:
    """产画像 dict（preferences/identity/work 分组）；空候选 → 空 dict 不 crash。"""
    grouped = aggregate(candidates)
    return {g: entries for g, entries in grouped.items() if entries}


def _decode_b64_content(s: str) -> str:
    """GraphLite 对 UTF-8 内容做 {b64}<base64> 透明编解码，解码回明文（同 system.py 语义）。"""
    if isinstance(s, str) and s.startswith("{b64}"):
        try:
            import base64
            return base64.b64decode(s[5:]).decode("utf-8", errors="replace")
        except Exception:
            return s
    return s


def scan_rows(rows: list) -> list[dict]:
    """query_cypher 返回行 → 节点候选（兼容三种返回格式）。

    - 深层嵌套 {"e": {"Node": {...}}} → _flatten_row（含 b64 解码）
    - 别名扁平 {"e.content": ...} → 手工解码 {b64}；缺失 source_type 默认
      inferred（防默认 direct 绕过来源门控）
    - 旧格式 [[id, content]] → 仅取 content，source_type 默认 inferred
    """
    nodes: list[dict] = []
    for row in rows or []:
        if isinstance(row, dict) and "e" in row:
            try:
                from graph.graphlite_store import GraphLiteStore
                nodes.append(GraphLiteStore._flatten_row(row, "e"))
            except Exception:
                continue
        elif isinstance(row, dict) and "e.content" in row:
            nodes.append({
                "content": _decode_b64_content(str(row.get("e.content") or "")),
                "source_type": str(row.get("e.source_type") or "inferred"),
            })
        elif isinstance(row, (list, tuple)) and len(row) > 1:
            nodes.append({"content": str(row[1]), "source_type": "inferred"})
    return nodes


def rebuild_or_keep(existing: dict, nodes: list[dict]) -> dict:
    """扫描候选重建画像；重建结果为空 → 保留已有画像（GraphLite 失败/无数据防空覆盖）。"""
    rebuilt = build_profile(scan_preference_candidates(nodes))
    return rebuilt if rebuilt else (existing or {})


def load_profile(path: str) -> dict:
    """读 JSON；缺失/损坏/非 dict → 空 dict（降级不抛错，同 ontology_evolution 模式）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_profile(profile: dict, path: str) -> bool:
    """temp + rename 原子写；失败仅告警不抛（与 ontology_evolution 同语义）。"""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.warning("User profile persist failed: %s", e)
        return False


def profile_values(profile: dict) -> set[str]:
    """画像全部值（检索命中检测用）。"""
    values: set[str] = set()
    for group in (profile or {}).values():
        if isinstance(group, dict):
            values.update(str(v) for v in group.keys())
    return values


def profile_hit(content: str, values: set[str]) -> bool:
    """content 是否含任一画像值（中文子串匹配）。"""
    return bool(content) and any(v and v in content for v in values)
