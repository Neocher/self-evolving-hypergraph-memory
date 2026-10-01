"""
CJK 事实抽取模块（P1a）
======================

为 SHM 梦境管道提供「抽取器本体」——从中文（及中英混合）文本中确定性抽取
SPO 事实三元组，为 P1b 接线（写路径接入事实抽取）铺路。

本模块是**纯抽取器**，只产出 ``{subject, predicate, object, valid_time}``
事实字典，不落库、不接写路径、不碰引擎/schema/ontology（那些是 P1b 的事）。

两个确定性入口 + 一个 LLM 入口：

- ``extract_facts_cjk_rules(content)``：同步，4 类中文规则（端口/版本/路径/
  状态变更），零外部依赖（仅 re），幂等，上限 10 条。
- ``extract_facts_llm(llm, content, timeout)``：async，鸭子类型调用
  ``llm.chat(...)``；None / 短内容 / 非法 JSON / 超时 / 字段缺失 全降级为 []，
  不抛异常。
- ``extract_facts(llm, content)``：async，规则路 ∪ LLM 路，按
  (subject.lower(), predicate.lower(), object.lower()) 去重，规则路优先。

开关：``_enabled()`` 读环境变量 ``SHM_FACT_EXTRACT``，"0"/"false" → False，
默认 True（开关由 P1b hook 消费，extract 本身仍可直调）。

模块顶层仅导入 stdlib（re/os/asyncio/json/logging），无 ``core.*`` 导入，
从根上规避循环导入风险。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# 事实上限：规则路 / LLM 路 / 合并去重后统一截断到 ≤10。
_MAX_FACTS = 10

# a) 端口/代理：host:port（host 必填，避免把 ":8083" 裸端口误抽）。
#    host 支持 127.0.0.1 / localhost / 一般主机名。
_PORT_PAT = re.compile(
    r"(?<![0-9A-Za-z])"
    r"(?P<host>127\.0\.0\.1|localhost|[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)"
    r":(?P<port>\d{2,5})\b"
)

# b) 版本号：版本/version/v/V 前缀 + 数字版本（至少两位），不吞前置 v。
_VERSION_PAT = re.compile(
    r"(?<![0-9A-Za-z])"
    r"(?:版本|version|v|V)"
    r"\s*[vV]?\s*"
    r"(?P<ver>\d+\.\d+(?:\.\d+)*)"
    r"(?![0-9A-Za-z])"
)

# c) 路径/文件：以 ~/ / ./ ../ 开头 + 路径字符集；后用 _looks_like_path
#    过滤（含 "." 扩展名或第二个 "/"），避免把 "/1081" 误抽成路径。
_PATH_PAT = re.compile(r"(?:~/|/|\./|\.\./)[A-Za-z0-9._~/-]+")

# d) 状态变更动词表（扩展超集，覆盖基准语料 #5 "污染为" 等）。
_STATE_VERB_PAT = re.compile(
    r"(?:已)?"
    r"(?:改为|变为|变成|升级为|污染为|切换为|切换到|下线|停止|停用|停服|暂停|恢复)"
)


def _enabled() -> bool:
    """读环境变量 ``SHM_FACT_EXTRACT``："0"/"false" → False，默认 True。"""
    value = os.environ.get("SHM_FACT_EXTRACT", "").strip().lower()
    return value not in ("0", "false")


def _looks_like_path(path: str) -> bool:
    """路径候选过滤：命中段含 "."（扩展名/隐藏文件）或第二个 "/" 才算路径。

    用于排除 ":8083/:1081" 里的 "/1081" 这类裸数字噪声。
    """
    return "." in path or path.count("/") >= 2


def _extract_state_subject(prefix: str) -> str:
    """取动词前最近实体短语作为 subject，去尾随被动标记，空则 "系统"。"""
    subject = prefix.rstrip()
    while subject.endswith("被") or subject.endswith("已"):
        subject = subject[:-1].rstrip()
    return subject if subject else "系统"


def _extract_state_object(verb: str, rest: str) -> str:
    """取动词后的目标文本作为 object（变更描述），空则用动词本身。"""
    rest = rest.lstrip()
    m = re.match(r"[^\s,，。；;！!？?]+", rest)
    if m:
        return m.group(0)
    return verb


def extract_facts_cjk_rules(content: str) -> list[dict]:
    """从文本中确定性抽取 CJK 事实三元组（4 类规则），上限 10 条，幂等。"""
    if not content:
        return []
    out: list[dict] = []

    # a) 端口/代理
    for m in _PORT_PAT.finditer(content):
        if len(out) >= _MAX_FACTS:
            break
        out.append({
            "subject": "代理端口",
            "predicate": "是",
            "object": f"{m.group('host')}:{m.group('port')}",
            "valid_time": "",
        })

    # b) 版本号
    for m in _VERSION_PAT.finditer(content):
        if len(out) >= _MAX_FACTS:
            break
        out.append({
            "subject": "版本",
            "predicate": "是",
            "object": m.group("ver"),
            "valid_time": "",
        })

    # c) 路径/文件
    for m in _PATH_PAT.finditer(content):
        if len(out) >= _MAX_FACTS:
            break
        path = m.group(0)
        if not _looks_like_path(path):
            continue
        out.append({
            "subject": "路径",
            "predicate": "是",
            "object": path,
            "valid_time": "",
        })

    # d) 状态变更
    for m in _STATE_VERB_PAT.finditer(content):
        if len(out) >= _MAX_FACTS:
            break
        raw = m.group(0)
        verb = raw[1:] if raw.startswith("已") else raw
        out.append({
            "subject": _extract_state_subject(content[:m.start()]),
            "predicate": "状态",
            "object": _extract_state_object(verb, content[m.end():]),
            "valid_time": "",
        })

    return out


_FACT_PROMPT = (
    "你是记忆事实抽取器。从给定文本中抽取显式事实三元组，只输出 JSON 数组，"
    "每个元素形如 {\"subject\": \"主语\", \"predicate\": \"谓词\", \"object\": \"宾语\", "
    "\"valid_time\": \"时间(可选)\"}。最多 10 条，无事实则输出 []。"
    "不要输出任何 JSON 数组以外的内容。\n\n文本：\n"
)


async def extract_facts_llm(llm, content: str, timeout: float = 8.0) -> list[dict]:
    """调用 LLM 抽取事实三元组，全降级不抛异常。

    - ``llm is None`` → []
    - ``len(content) < 80`` → []（不调用 LLM）
    - 超时 / 非法 JSON / ``chat`` 返回 None / 字段缺失 → []
    - 合法 JSON 数组 → 解析为事实字典，补齐缺失的 ``valid_time=""``。

    鸭子类型：仅调用 ``llm.chat(messages=[...], temperature=0, max_tokens=1024)``，
    不 import ``core.llm_client``。
    """
    if llm is None:
        return []
    if not content or len(content) < 80:
        return []

    messages = [{"role": "user", "content": _FACT_PROMPT + content}]
    try:
        raw = await asyncio.wait_for(
            llm.chat(messages=messages, temperature=0, max_tokens=1024),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.debug("fact_extract: LLM 调用超时，降级为空结果")
        return []
    except Exception:
        logger.debug("fact_extract: LLM 调用异常，降级为空结果", exc_info=True)
        return []

    if not raw:
        return []

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.debug("fact_extract: LLM 返回非法 JSON，降级为空结果")
        return []

    # 容错：若顶层是 dict，尝试取 facts/result 数组
    if isinstance(data, dict):
        data = next(
            (data[k] for k in ("facts", "result") if isinstance(data.get(k), list)),
            None,
        )
    if not isinstance(data, list):
        return []

    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        predicate = item.get("predicate")
        object_ = item.get("object")
        if not (
            isinstance(subject, str) and isinstance(predicate, str)
            and isinstance(object_, str)
            and subject.strip() and predicate.strip() and object_.strip()
        ):
            continue
        valid_time = item.get("valid_time")
        if not isinstance(valid_time, str):
            valid_time = ""
        out.append({
            "subject": subject.strip(),
            "predicate": predicate.strip(),
            "object": object_.strip(),
            "valid_time": valid_time,
        })
        if len(out) >= _MAX_FACTS:
            break
    return out


async def extract_facts(llm, content: str) -> list[dict]:
    """规则路 ∪ LLM 路，按 (subject, predicate, object) 去重，规则路优先，≤10。"""
    rules = extract_facts_cjk_rules(content)
    llm_facts = await extract_facts_llm(llm, content)

    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    for fact in rules:
        key = (fact["subject"].lower(), fact["predicate"].lower(), fact["object"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(fact)

    for fact in llm_facts:
        if len(out) >= _MAX_FACTS:
            break
        key = (fact["subject"].lower(), fact["predicate"].lower(), fact["object"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(fact)

    return out[:_MAX_FACTS]
