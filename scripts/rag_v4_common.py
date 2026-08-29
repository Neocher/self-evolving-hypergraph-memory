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
    m = re.search(r'DEEPSEEK_API_KEY\s*=\s*"?([^"\s]+)"?',
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
    # 显式禁代理（对齐 retrieval/hyde.py）：Hermes 会话注入 ALL_PROXY=socks5h://127.0.0.1:1081
    # 会劫持 urllib——昨 00:49 WARP 半开时 socks5 双向等待致评测挂死 2h；禁代理直连成功
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=120) as resp:
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


_reranker_model = None


def get_reranker():
    """bge-reranker cross-encoder（懒加载；失败返回 None 走降级）"""
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            name = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
            _reranker_model = CrossEncoder(name, max_length=512)
            print(f"[reranker] 已加载 {name}", flush=True)
        except Exception as e:
            print(f"[reranker] 加载失败, 走原序降级: {e}", flush=True)
            _reranker_model = False
    return _reranker_model or None


def rerank(query: str, docs: list, top_k: int = 10, top_n: int = None):
    """bge-reranker cross-encoder 重排；不可用时原序截断。兼容 top_k/top_n 两种调用。"""
    model = get_reranker()
    k = top_n or top_k
    if not model or not docs:
        return docs[:k]
    try:
        pairs = [[query, d[:1000]] for d in docs]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        ranked = sorted(zip(docs, scores), key=lambda x: -float(x[1]))
        return [d for d, s in ranked][:k]
    except Exception as e:
        print(f"[reranker] 推理失败, 原序降级: {e}", flush=True)
        return docs[:k]
