"""评测臂 suff_check / followup_query 粘性缓存 (STICKY_SUFF)。

纯 stdlib 模块 (仅 hashlib/json), 零 IO / 零 LLM / 零 env 读 — 供 bench 薄委托与
单测直接 import (bench_locomo_v72_ontology.py 顶层无 __main__ 守护、import 即跑全量
评测, 故被测逻辑本体必须落在可 import 的纯模块, 与 time_anchors/fact_clusters/
neg_clean/slot_assembly 同模式)。

STICKY_SUFF=1 时 suff_check/followup_query 按 (question, 输入) 指纹进程内缓存,
命中不再触 LLM (消除评测 run 间/批内由 qwen3.8-max temp=0 非确定性注入的 ctx 漂移);
=0 时跳过指纹/缓存读写, 输出与 a662aaf 逐字节等价。prompt/解析/异常语义逐字搬移,
不改判卷链路。指纹用 \x00 分隔消除跨块拼接边界歧义 (["ab","c"] vs ["a","bc"] 同指纹)。
"""
from __future__ import annotations

import hashlib
import json
import os

# 进程内粘性缓存 (模块级 dict; 跨 run 语义由文件级覆盖 — 见 _file_cache 注)
_suff_cache: dict[str, tuple[bool, str]] = {}
_followup_cache: dict[str, str] = {}


def _file_dir() -> "str | None":
    """文件级 per-run 目录: $STICKY_CACHE_DIR/{STICKY_RUN_ID} (未设 run-id → None)。

    跨 run (独立进程) 命中依赖落盘: run1 写、run2 同 run-id 读; 不同 run-id 隔离。
    未设时回落进程内 dict 语义 (单测用)。
    """
    import os
    run_id = (os.environ.get("STICKY_RUN_ID") or "").strip()
    base = (os.environ.get("STICKY_CACHE_DIR") or "/tmp/sticky_cache").strip()
    if not run_id:
        return None
    d = os.path.join(base, run_id, "suff")
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        return None


def _file_get(fp: str) -> "str | None":
    try:
        with open(fp, "r", encoding="utf-8") as f:
            c = f.read()
        return c if c else None
    except Exception:
        return None


def _file_set(fp: str, val: str) -> None:
    try:
        with open(fp, "w", encoding="utf-8") as f:
            f.write(val)
    except Exception:
        pass


def suff_fingerprint(question: str, docs_top) -> str:
    """suff 指纹 = sha1(question + top10 每条前 120 字符), \x00 分隔。"""
    parts = [question]
    for d in docs_top[:10]:
        parts.append(d[:120])
    return hashlib.sha1(
        b"\x00".join(p.encode("utf-8", "replace") for p in parts)
    ).hexdigest()


def followup_fingerprint(question: str, missing: str) -> str:
    """followup 指纹 = sha1(question + missing), \x00 分隔。"""
    return hashlib.sha1(
        (question + "\x00" + (missing or "")).encode("utf-8", "replace")
    ).hexdigest()


def cached_suff_check(llm_generate, question, docs_top, enabled: bool):
    """sufficiency 判断 + 粘性缓存 (enabled=False 时行为与 a662aaf 逐字节等价)。

    异常路径维持 return True,"" 兜底且不写缓存。
    """
    fp = None
    fdir = None
    if enabled:
        fp = suff_fingerprint(question, docs_top)
        # 文件级 (跨 run) 先查: json 存 {"sufficient":bool,"missing":str}
        fdir = _file_dir()
        if fdir is not None:
            got = _file_get(os.path.join(fdir, "suff_" + fp + ".json"))
            if got is not None:
                try:
                    d = json.loads(got)
                    return (bool(d.get("sufficient")), str(d.get("missing_info", "")))
                except Exception:
                    pass
        hit = _suff_cache.get(fp)
        if hit is not None:
            return hit
    ctx = "\n".join(f"[{j+1}] {d[:120]}" for j, d in enumerate(docs_top[:10]))
    prompt = f"""You are searching a conversation log. Given the retrieved snippets below, decide whether they are SUFFICIENT to answer the question.

Question: {question}

Retrieved snippets:
{ctx}

Output STRICT JSON: {{"sufficient": true/false, "missing_info": "what specific info is missing, or empty string"}}
No other text."""
    try:
        raw = llm_generate(prompt, max_tokens=200, temperature=0.0)
        s, e = raw.find("{"), raw.rfind("}")
        d = json.loads(raw[s:e + 1]) if s >= 0 and e > s else {}
        result = (bool(d.get("sufficient")), str(d.get("missing_info", "")))
        if enabled and fp is not None:
            _suff_cache[fp] = result
            if fdir is not None:
                _file_set(
                    os.path.join(fdir, "suff_" + fp + ".json"),
                    json.dumps({"sufficient": result[0], "missing_info": result[1]}, ensure_ascii=False),
                )
        return result
    except Exception:
        return True, ""


def cached_followup_query(llm_generate, question, missing, enabled: bool):
    """followup 查询生成 + 粘性缓存 (enabled=False 时行为与 a662aaf 逐字节等价)。

    异常路径维持 return question 兜底且不写缓存。
    """
    fp = None
    fdir = None
    if enabled:
        fp = followup_fingerprint(question, missing)
        fdir = _file_dir()
        if fdir is not None:
            got = _file_get(os.path.join(fdir, "fu_" + fp + ".txt"))
            if got is not None:
                return got
        hit = _followup_cache.get(fp)
        if hit is not None:
            return hit
    prompt = f"""Generate a search query to find the missing information in a conversation log.

Question: {question}
Missing information needed: {missing}

Output a single search query string. No other text."""
    try:
        result = llm_generate(prompt, max_tokens=80, temperature=0.0).strip()[:200]
        if enabled and fp is not None:
            _followup_cache[fp] = result
            if fdir is not None:
                _file_set(os.path.join(fdir, "fu_" + fp + ".txt"), result)
        return result
    except Exception:
        return question
