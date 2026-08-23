"""
OntologyEvolution — Schema 自演化（v5.38.0 Ontology-Evolution）
===============================================================
对标 MindMemOS：ontology 随数据生长。梦境 SYNTHESIZE 后聚合全部社区
topics/report → 1 次 LLM 调用（复用 llm_client.chat，temperature=0.1，
response_format=json_object）→ new_type / merge_existing / skip 三选一。

持久化: data/ontology_extended.json（gitignore，缺失/损坏 → 空 dict 降级）。
不硬改 ONTOLOGY_TYPES 模块全局：函数/实例级合并，原生类型优先（不覆盖）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from typing import Any, Optional

from core.ontology_validator import ONTOLOGY_TYPES

logger = logging.getLogger(__name__)

# 泛词守卫（复用 skill_bridge _STOPWORDS 思路）：conflict_keys 全泛词 → skip。
# 中英混合覆盖：英文停用词 + 中文高频虚词/无判别力词。
_GENERIC_KEYS: frozenset[str] = frozenset({
    # 英文
    "a", "an", "the", "of", "for", "to", "in", "on", "with", "at", "by",
    "from", "and", "or", "is", "are", "be", "was", "were", "it", "this",
    "that", "these", "those", "about", "into", "over", "under", "will",
    "can", "should", "would", "could", "may", "might", "must", "not",
    "no", "very", "just", "also", "then", "than", "data", "info",
    "information", "fact", "facts", "content", "report", "summary",
    # 中文
    "信息", "数据", "事实", "内容", "摘要", "报告", "情况", "关于",
    "相关", "一个", "这个", "那个", "什么", "如何", "可以", "进行",
})

# 保留键：extended JSON 顶层非类型键（attr_aliases 别名表）。类型决策命名空间
# 不得把它当类型渲染/merge/new_type —— 否则别名表被当类型 dict 污染 conflict_keys。
# 【v5.50.0 P1-2】
_RESERVED_KEYS: frozenset[str] = frozenset({"attr_aliases"})

# 新类型默认 contradiction_pattern（与原生 same_entity_diff_value 对齐）
_DEFAULT_PATTERN = "same_entity_diff_value"

_DEFAULT_EXTENDED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ontology_extended.json",
)


# ─── 加载 / 合并 ─────────────────────────────────────────────


def load_extended(path: str) -> dict:
    """读 JSON；缺失/损坏/非 dict → 空 dict（降级不抛错）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def merged_types(extended: Optional[dict] = None) -> dict:
    """合并 extended 到原生类型 — 原生优先（不覆盖）。

    语义: {**extended, **ONTOLOGY_TYPES}（后展开者优先）→ 即使 extended 文件
    被污染含原生同名键，原生定义仍保持。正常路径 extended 只含新类型，
    与 {**ONTOLOGY_TYPES, **extended} 完全等价。
    """
    return {**(extended or {}), **ONTOLOGY_TYPES}


def _extended_only(merged: dict) -> dict:
    """从合并结果里只取 extended 部分（原生类型不写回文件）。"""
    return {k: v for k, v in merged.items() if k not in ONTOLOGY_TYPES}


# ─── 守卫 ────────────────────────────────────────────────────


def _is_generic_key(key: Any) -> bool:
    """key 是否泛词：空 / 白名单 / 无判别力（有效字符 < 2）。"""
    k = str(key).strip().lower()
    if not k or k in _GENERIC_KEYS:
        return True
    effective = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", k)
    return len(effective) < 2


def _conflict_key_clash(new_keys: list, existing: dict) -> Optional[str]:
    """新类型 keys 与任一已有类型 conflict_keys 重叠 → 返回该类型名。

    跨类型 key 冲突 → skip（first-match-wins：新类型 key 不得与任何
    已有类型重叠）。
    """
    new_set = set(new_keys)
    for otype, info in existing.items():
        if new_set & set(info.get("conflict_keys", [])):
            return otype
    return None


# ─── 原子写 ──────────────────────────────────────────────────


def _atomic_write(path: str, extended: dict) -> bool:
    """temp + rename 原子写；失败仅告警不抛（与 LLM 失败 skip 同语义）。"""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(extended, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.warning("Ontology evolution persist failed: %s", e)
        return False


# ─── 决策应用（纯函数，写盘由调用方 to_thread 落盘）──────────────


def _proposal_from(parsed: dict, raw_type: Any) -> Optional[dict]:
    """把 LLM 输出里的一种 type 形态规整成 proposal dict。"""
    if isinstance(raw_type, dict):
        name = str(raw_type.get("name") or raw_type.get("type") or "").strip()
        if not name:
            return None
        return {
            "name": name,
            "description": str(raw_type.get("description") or "").strip(),
            "conflict_keys": [
                str(k).strip() for k in (raw_type.get("conflict_keys") or []) if str(k).strip()
            ],
        }
    name = str(raw_type or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "description": str(parsed.get("description") or "").strip(),
        "conflict_keys": [
            str(k).strip() for k in (parsed.get("conflict_keys") or []) if str(k).strip()
        ],
    }


def _register_new_type(prop: dict, current: dict):
    """注册单个新类型（守卫全过才写 current）。返回 (current, result)。"""
    name = prop.get("name", "")
    if not name:
        return None, {"action": "skip", "reason": "bad_type_name"}
    if name in _RESERVED_KEYS:
        return None, {"action": "skip", "reason": "type_reserved"}
    if name in current:
        return None, {"action": "skip", "reason": "type_exists"}
    keys = prop.get("conflict_keys") or []
    # 守卫：新类型需 ≥2 个非泛 conflict_keys
    non_generic = [k for k in keys if not _is_generic_key(k)]
    if len(non_generic) < 2:
        return None, {"action": "skip", "reason": "not_enough_specific_keys"}
    # 守卫：跨类型 key 冲突 → skip（first-match-wins）
    clash = _conflict_key_clash(non_generic, current)
    if clash is not None:
        return None, {"action": "skip", "reason": f"key_clash_with_{clash}"}
    current[name] = {
        "description": prop.get("description", ""),
        "conflict_keys": non_generic,
        "contradiction_pattern": _DEFAULT_PATTERN,
    }
    return current, {"action": "new_type", "type": name}


def _apply_new_type(parsed: dict, current: dict):
    """new_type 分支：最多注册 1 个新类型/轮 — 顺序尝试全部提案，注册第一个通过守卫的。"""
    proposals: list[dict] = []
    single = parsed.get("type") or parsed.get("name")
    if single:
        prop = _proposal_from(parsed, single)
        if prop:
            proposals.append(prop)
    for item in parsed.get("new_types") or []:
        if isinstance(item, dict):
            prop = _proposal_from(item, item.get("name") or item.get("type"))
            if prop:
                proposals.append(prop)
    if not proposals:
        return None, {"action": "skip", "reason": "no_type_proposal"}
    last = None
    for prop in proposals:
        new_current, result = _register_new_type(prop, current)
        if result["action"] == "new_type":
            return new_current, result
        last = result
    return None, last or {"action": "skip", "reason": "guard_rejected"}


def _apply_merge(parsed: dict, current: dict):
    """merge_existing 分支：仅合并 extended 类型 — 目标为原生类型 → skip。

    原生类型禁止 merge：current 中原生 value 与全局 ONTOLOGY_TYPES 共享引用
    （merged_types 只浅拷贝顶层），原地改 conflict_keys 会污染全局；且
    _extended_only 落盘时原生键被滤掉，导致"落盘为空却上报成功"的静默 no-op。
    """
    type_name = str(parsed.get("type") or parsed.get("merge_type") or "").strip()
    if not type_name or type_name not in current:
        return None, {"action": "skip", "reason": "merge_target_missing"}
    if type_name in _RESERVED_KEYS:
        return None, {"action": "skip", "reason": "merge_target_reserved"}
    if type_name in ONTOLOGY_TYPES:
        return None, {"action": "skip", "reason": "merge_target_native"}
    new_keys = [str(k).strip() for k in (parsed.get("conflict_keys") or []) if str(k).strip()]
    if not new_keys:
        return None, {"action": "skip", "reason": "no_new_keys"}
    existing = current[type_name]
    existing["conflict_keys"] = list(dict.fromkeys(
        [*(existing.get("conflict_keys") or []), *new_keys]
    ))
    return current, {"action": "merge_existing", "type": type_name}


def _apply_attr_ops(parsed: dict, current: dict,
                    distinct_attrs: Optional[list]) -> tuple[Optional[dict], dict]:
    """attr_ops 分支（v5.50.0 P2）：属性别名合并，与类型决策正交（可同轮发生）。

    LLM 响应 attr_ops 数组 [{"op": "merge_alias", "canonical": str, "aliases": [str]}]：
    - 守卫：max 1 attr_op/轮（顺序尝试第一个过守卫的）；canonical/alias 均过
      _is_generic_key（非泛词）；canonical 必须 ∈ distinct_attrs（否则孤儿
      alias 无消费方 → skip）
    - 写入：extended JSON 顶层 "attr_aliases": {"<canonical>": ["<alias1>", ...]}
      （canonical 对齐 {relation}_{key} 格式）。attr_aliases 非原生类型键，
      经 _extended_only 落盘保留；别名去重保序，canonical 自身不进 alias 表。
    """
    ops = parsed.get("attr_ops")
    if distinct_attrs is None or not isinstance(ops, list) or not ops:
        return None, {"action": "skip", "reason": "no_attr_op"}
    distinct_set = {str(a).strip() for a in distinct_attrs if str(a).strip()}
    last: Optional[dict] = None
    for op in ops:
        if not isinstance(op, dict):
            continue
        if str(op.get("op", "")).strip() != "merge_alias":
            continue
        canonical = str(op.get("canonical", "")).strip()
        aliases = [str(a).strip() for a in (op.get("aliases") or []) if str(a).strip()]
        # 守卫 1：canonical 非泛词 + ∈ distinct_attrs（孤儿 alias 无消费方 → skip）
        if _is_generic_key(canonical) or canonical not in distinct_set:
            last = {"action": "skip", "reason": "attr_canonical_missing_or_generic"}
            continue
        # 守卫 2：至少 1 个非泛 alias（泛词 alias 全滤后为空 → skip）
        non_generic = [a for a in aliases if not _is_generic_key(a)]
        # 【P3-1】canonical 自身不进 alias 表（自别名无消费方且污染归一表）
        non_generic = [a for a in non_generic if a.lower() != canonical.lower()]
        if not non_generic:
            last = {"action": "skip", "reason": "attr_aliases_generic"}
            continue
        aliases_map = current.setdefault("attr_aliases", {})
        if not isinstance(aliases_map, dict):
            aliases_map = {}
            current["attr_aliases"] = aliases_map
        existing = aliases_map.get(canonical) or []
        aliases_map[canonical] = list(dict.fromkeys([*existing, *non_generic]))
        return current, {"action": "attr_op", "canonical": canonical, "aliases": non_generic}
    return None, last or {"action": "skip", "reason": "attr_op_guard_rejected"}


# ─── LLM prompt ──────────────────────────────────────────────


def _build_prompt(communities: list, current: dict,
                  distinct_attrs: Optional[list] = None) -> str:
    """聚合社区 topics/report + 注入当前类型名/description + 属性名清单。"""
    type_blocks = "\n".join(
        f"- {name}: {info.get('description', '') or '; '.join(info.get('conflict_keys', []))}"
        for name, info in current.items()
        if name not in _RESERVED_KEYS
    )
    attr_lines = "\n".join(f"- {a}" for a in (distinct_attrs or [])) or "(none)"
    comm_blocks = []
    for c in communities:
        if not isinstance(c, dict):
            continue
        topics = c.get("topics") or []
        if not isinstance(topics, list):
            topics = []
        report = (c.get("report") or "")[:500]
        comm_blocks.append(f"### community\n- topics: {topics[:3]}\n- report: {report}")
    combined = "\n".join(comm_blocks)[:6000]
    return f"""You are the ontology curator of a self-evolving hypergraph memory system.
After dream consolidation, decide whether the schema needs to evolve.
Existing ontology types (name: description):
{type_blocks}

Existing property attribute names (attr_name; candidates for alias canonicalization):
{attr_lines}

Aggregated community summaries from this dream round:
{combined}

Decide ONE of three actions:
1. "new_type": propose a NEW ontology type if communities reveal a recurring semantic
   category NOT covered above.
   - "type": new type name (snake_case, distinct from all existing names)
   - "description": one-line description
   - "conflict_keys": at least 2 specific discriminative keywords
     (avoid generic words like data/info/fact/content)
2. "merge_existing": merge new conflict keywords into an EXISTING type.
   Use its exact name in "type", list only the new keys in "conflict_keys".
3. "skip": nothing worthwhile this round.

OPTIONALLY, also provide "attr_ops" to normalize synonymous attribute names:
[{{"op": "merge_alias", "canonical": "<attr_name from list above>",
  "aliases": ["<alias1>", ...]}}]
- canonical MUST be one of the existing attribute names listed above.
- aliases must be specific discriminative attribute words (not generic).
- at most 1 attr_op is applied per round; omit "attr_ops" (or set []) if none.

Respond with a single JSON object only:
{{"action": "new_type"|"merge_existing"|"skip", "type": "<name>",
  "description": "<text>", "conflict_keys": ["k1", "k2"],
  "attr_ops": [{{"op": "merge_alias", "canonical": "<name>", "aliases": ["a1"]}}]}}
Do not add any text outside the JSON object."""


# ─── 主入口 ──────────────────────────────────────────────────


async def evolve_once(summaries: list, llm_client, extended_path: str,
                      distinct_attrs: Optional[list] = None) -> dict:
    """聚合 → 1 次 LLM → new_type / merge_existing / skip（+ attr_ops 别名合并）。

    distinct_attrs: 现有属性名清单（get_distinct_attr_names 产物）。None → 跳过
    attr_ops（仅类型决策）；非 None 时 attr_ops 与类型决策正交，可同轮发生。

    Returns:
        {"action": "new_type", "type": str} |
        {"action": "merge_existing", "type": str} |
        {"action": "attr_op", "canonical": str, "aliases": [str]} |
        {"action": "skip", "reason": str}
    """
    if llm_client is None:
        return {"action": "skip", "reason": "no_llm_client"}
    current = merged_types(load_extended(extended_path))
    prompt = _build_prompt(summaries, current, distinct_attrs)
    try:
        response = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
    except Exception:
        logger.warning("Ontology evolution LLM failed — skip", exc_info=True)
        return {"action": "skip", "reason": "llm_failed"}
    if not response:
        return {"action": "skip", "reason": "llm_failed"}
    try:
        parsed = json.loads(response)
    except Exception:
        logger.warning("Ontology evolution JSON parse failed — skip")
        return {"action": "skip", "reason": "parse_failed"}
    if not isinstance(parsed, dict):
        return {"action": "skip", "reason": "parse_failed"}

    action = str(parsed.get("action", "skip")).strip().lower()
    mutation, result = None, None
    if action == "merge_existing":
        mutation, result = _apply_merge(parsed, current)
    elif action == "new_type":
        mutation, result = _apply_new_type(parsed, current)
    elif action == "skip":
        mutation, result = None, {"action": "skip", "reason": "llm_decided"}
    else:
        result = {"action": "skip", "reason": f"unknown_action:{action}"}
        _log_drift_if_skip(result)
        return result

    # 【v5.50.0 P2】attr_ops 与类型决策正交（可同轮发生）；类型 skip 但 attr_op
    # 通过守卫时仍落盘（base 回落 current）。类型决策成功优先返回，仅类型 skip
    # 时 attr_op 结果才作为返回值（attr_op 仍作为正交副作用落盘）。
    if distinct_attrs is not None:
        base = mutation if mutation is not None else current
        mutation, attr_result = _apply_attr_ops(parsed, base, distinct_attrs)
        if attr_result is not None and attr_result.get("action") == "attr_op" \
                and result.get("action") == "skip":
            result = attr_result

    # 落盘：asyncio.to_thread（不进事件循环；失败不声称成功）
    if mutation is not None:
        ok = await asyncio.to_thread(_atomic_write, extended_path, _extended_only(mutation))
        if not ok:
            logger.warning("Ontology evolution persist failed — %s NOT saved",
                           result.get("type") or result.get("canonical"))
            return {"action": "skip", "reason": "persist_failed"}
        if result["action"] in ("new_type", "merge_existing"):
            logger.info("Ontology evolution: %s → %s",
                        result["action"], result.get("type"))
        elif result["action"] == "attr_op":
            logger.info("Ontology evolution: attr_op → %s (%d aliases)",
                        result.get("canonical"), len(result.get("aliases", [])))
    _log_drift_if_skip(result)
    return result


# 【2026-08-23 漂移检测】借鉴《本体论增强问数》持续运营：新 SQL/模式无法被本体
# 覆盖时告警触发增补。SHM 语义 = new_type 被守卫拒绝（无法覆盖新模式）→ 打
# WARNING 提示检查 extended_types.yaml 治理（防本体静默腐烂/与数据脱节）。
_DRIFT_REASONS = ("not_enough_specific_keys", "key_clash", "guard_rejected",
                  "type_reserved", "bad_type_name", "no_type_proposal",
                  "unknown_action")


def _log_drift_if_skip(result: Optional[dict]) -> None:
    if not isinstance(result, dict):
        return
    if result.get("action") != "skip":
        return
    reason = str(result.get("reason", ""))
    if any(reason.startswith(r) for r in _DRIFT_REASONS):
        logger.warning(
            "Ontology drift: 新模式无法被本体覆盖 (reason=%s) — "
            "建议检查 ontology_extended.json 治理或增补", reason)


def classify_with_extended(text: str, entities: list, extended: Optional[dict] = None) -> str:
    """合并后分类：遍历 {**extended, **ONTOLOGY_TYPES}，first-match-wins。"""
    text_lower = (text or "").lower()
    for otype, info in merged_types(extended).items():
        if any(k in text_lower for k in info.get("conflict_keys", [])):
            return otype
    return "generic_fact"


class OntologyEvolution:
    """Schema 自演化器（v5.38.0）— 持 extended_path + llm_client，供 app.py 接线。

    evolve() 复用传入 llm_client.chat()（temperature=0.1, json_object）；
    llm_client 为空 → skip 不阻塞；JSON 落盘 asyncio.to_thread。
    """

    def __init__(self, extended_path: Optional[str] = None, llm_client=None,
                 graphlite_store=None):
        self.extended_path = extended_path or _DEFAULT_EXTENDED_PATH
        self.llm_client = llm_client
        self.graphlite_store = graphlite_store

    def load(self) -> dict:
        return load_extended(self.extended_path)

    def merged(self) -> dict:
        return merged_types(self.load())

    async def evolve(self, summaries: list, llm_client=None) -> dict:
        """1 次 LLM 调用入口（llm_client 空 → skip 不阻塞）。

        【v5.50.0 P0-1】生产路径取 distinct 属性清单：从 graphlite_store 拉取
        PropertyVerNode distinct attr_name 传给 evolve_once，触发 attr_ops（此前
        恒 distinct_attrs=None → attr_ops 死代码）。无 store / 拉取失败 → None
        （仅类型决策，跳过 attr_ops）。
        """
        distinct_attrs = None
        store = getattr(self, "graphlite_store", None)
        if store is not None and hasattr(store, "get_distinct_attr_names"):
            try:
                distinct_attrs = store.get_distinct_attr_names()
            except Exception:
                distinct_attrs = None
        return await evolve_once(summaries, llm_client or self.llm_client,
                                 self.extended_path, distinct_attrs=distinct_attrs)

    def classify(self, text: str, entities: list) -> str:
        return classify_with_extended(text, entities, self.load())
