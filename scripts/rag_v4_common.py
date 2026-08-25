"""rag_v4_common 兼容垫片 — LLM 调用走 DeepSeek API (DEEPSEEK_API_KEY from ~/.bashrc)。

为 bench_locomo_* 系列脚本提供 llm_generate / llm_judge / rerank / get_reranker。
"""
import json
import os
import re
from pathlib import Path

import urllib.request


def _key():
    k = os.environ.get("DEEPSEEK_API_KEY", "")
    if k:
        return k
    m = re.search(r'DEEPSEEK_API_KEY="([^"]+)"',
                  Path(os.path.expanduser("~/.bashrc")).read_text())
    return m.group(1) if m else ""


def _chat(prompt: str, max_tokens: int, temperature: float) -> str:
    key = _key()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        })
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.loads(resp.read())
    return d["choices"][0]["message"]["content"]


def llm_generate(prompt: str, max_tokens: int = 500, temperature: float = 0.2) -> str:
    return _chat(prompt, max_tokens, temperature)


def llm_judge(question: str, ground_truth: str, prediction: str) -> dict:
    """LLM-as-judge：对预测答案打分 (对齐 LoCoMo 口径)。"""
    prompt = (
        f"判断以下回答是否正确。\n\n问题: {question}\n标准答案: {ground_truth}\n"
        f"待评回答: {prediction}\n\n"
        "只输出 JSON: {\"correct\": true/false, \"reason\": \"一句话\"}"
    )
    raw = _chat(prompt, 100, 0.0)
    i = raw.find("{")
    try:
        d = json.loads(raw[i:] if i >= 0 else "{}")
        return {"correct": bool(d.get("correct")), "reason": d.get("reason", "")}
    except Exception:
        return {"correct": False, "reason": f"judge parse fail: {raw[:80]}"}


def rerank(query: str, docs: list, top_k: int = 10):
    """bge-reranker 不可用时的降级：原序截断（评测脚本内部已有 fallback）。"""
    return docs[:top_k]


def get_reranker():
    return None
