"""rag_v4_common 兼容垫片 — 双后端: MAAS (阿里云百炼 qwen3-14b, 评测口径) / DeepSeek (通用)。

为 bench_locomo_* 系列脚本提供 llm_generate / llm_judge / rerank / get_reranker。
优先 MAAS (设置 MAAS_API_KEY 即走 MAAS, 保持与 locomo-refined 官方评测口径一致);
否则回退 DeepSeek (DEEPSEEK_API_KEY from ~/.bashrc)。
"""
import json
import os
import re
import time
from pathlib import Path
import urllib.error
import urllib.request

MAAS_BASE = os.environ.get("MAAS_API_BASE", "https://llm-kegm398o6acjmcer.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
MAAS_MODEL = os.environ.get("MAAS_MODEL", "qwen3-14b")

# 达摩院 R2-0 工程纪律 (2026-09-03): LLM 瞬态失败自动重试 — 评测长跑网络抖动/
# 5xx/429/超时不再整题失败 (round1 32B reader 83% 失败根因 = 单次 300s 超时无重试
# 即 [gen err] 静默 skip)。确定性失败 (缺 key / 4xx 鉴权/坏请求) 不重试直接抛。
LLM_RETRIES = int(os.environ.get("LLM_RETRIES", "2"))  # 额外重试次数; 总尝试 = LLM_RETRIES + 1 = 3
_LLM_RETRY_DELAYS = (0.5, 1.0)  # 第 i 次重试前等待 (秒, 指数退避)
_LLM_TIMEOUT_BASE = {"maas": 300, "deepseek": 120}  # 单次请求超时基线 (s); 重试时按尝试 ×1/×2/×3


def _retryable(exc: Exception) -> bool:
    """瞬态失败判定: HTTP 429/5xx/超时/网络错误可重试; 缺 key / 4xx 确定性失败不重试。

    urllib HTTPError 同时是 URLError 子类且带 .code — 先按 code 判定再落网络层。
    """
    if isinstance(exc, RuntimeError) and "API_KEY" in str(exc):
        return False
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code == 429 or code >= 500
    return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError, ConnectionError))


def _key():
    k = os.environ.get("DEEPSEEK_API_KEY", "")
    if k:
        return k
    m = re.search(r'DEEPSEEK_API_KEY\s*=\s*"?([^"\s]+)"?',
                  Path(os.path.expanduser("~/.bashrc")).read_text())
    return m.group(1) if m else ""


def _chat(prompt: str, max_tokens: int = 500, temperature: float = 0.0, timeout: float | None = None) -> str:
    """单次 chat/completions 调用。timeout=None → 后端默认超时基线 (MAAS 300s / DeepSeek 120s);
    R2-0 重试路径由 llm_generate 显式传递增预算 (300/600/900 或 120/240/360)。"""
    maas_key = os.environ.get("MAAS_API_KEY", "")
    # 显式禁代理（对齐 retrieval/hyde.py）：Hermes 会话注入 ALL_PROXY=socks5h://127.0.0.1:1081
    # 会劫持 urllib——昨 00:49 WARP 半开时 socks5 双向等待致评测挂死 2h；禁代理直连成功
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t = timeout if timeout is not None else (
        _LLM_TIMEOUT_BASE["maas"] if maas_key else _LLM_TIMEOUT_BASE["deepseek"])
    if maas_key:
        body = json.dumps({
            "model": MAAS_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "enable_thinking": False,  # Qwen3 非流式调用必须关闭 thinking
        }).encode()
        req = urllib.request.Request(
            f"{MAAS_BASE}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {maas_key}"})
        with opener.open(req, timeout=t) as resp:
            d = json.loads(resp.read())
        return d["choices"][0]["message"]["content"]

    key = _key()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set (且未设 MAAS_API_KEY)")
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
    with opener.open(req, timeout=t) as resp:
        d = json.loads(resp.read())
    return d["choices"][0]["message"]["content"]


def llm_generate(prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
    """生成文本（用于记忆压缩、假设生成等）。

    达摩院 R2-0 工程纪律 (2026-09-03): 瞬态失败自动重试 LLM_RETRIES=2 次 (总 3 次
    尝试), 每次尝试的超时预算递增 (基线 ×1/×2/×3 — MAAS 300s → 300/600/900s,
    DeepSeek 120s → 120/240/360s), 重试间指数退避; 确定性失败 (缺 key/4xx) 不重试
    直接抛, 由调用方落错误行 — 评测长跑不再整题静默失败。温度保持 0.0 (R1 研究
    §0.2 事实: bench 传 0.2 但本函数原实现即硬编码 0.0, 评测确定性口径不动)。
    """
    base = _LLM_TIMEOUT_BASE["maas"] if os.environ.get("MAAS_API_KEY") else _LLM_TIMEOUT_BASE["deepseek"]
    last_exc = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            return _chat(prompt, max_tokens, 0.0, timeout=base * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            if attempt >= LLM_RETRIES or not _retryable(exc):
                raise
            time.sleep(_LLM_RETRY_DELAYS[min(attempt, len(_LLM_RETRY_DELAYS) - 1)])
    raise last_exc  # pragma: no cover — 循环内必 return 或 raise


def append_predict_error(out_path: str, qa_id: str, error) -> None:
    """R2-0: 预测失败错误行落盘 (jsonl 含 qa_id + 原因) — LLM 重试耗尽后不静默丢题。

    断点续跑 (RESUME 跳过成功行, 错误行仍在) 与事后归因 (round1 72.43% 分析缺每题
    失败现场) 均依赖该文件; 写失败抛给调用方打印, 不二次静默。
    """
    reason = str(error).replace("\n", " ").replace("\r", " ")[:500]
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"qa_id": str(qa_id), "error": reason, "ts": time.time()},
                           ensure_ascii=False) + "\n")


def ctx_composition(ctx: str) -> dict:
    """CTX_DUMP 归因: 最终 ctx 字符串的来源组成 (供翻转归因/证据现场分析)。

    返回: raw 直接证据条数与字符占比 + 组织段 (blocks 摘要/entity 段/relations/
    fact types) 条目与字符占比。ctx 为空 → 全 0 (诊断不抛)。
    """
    if not ctx:
        return {"n_direct": 0, "n_mem_block": 0, "n_entity_seg": 0, "n_rel_seg": 0,
                "n_fact_seg": 0, "chars_total": 0, "chars_direct": 0,
                "share_direct": 0.0, "share_org": 0.0}
    total = len(ctx)
    dsec = re.search(r"\[DIRECT EVIDENCE\]\n?(.*)$", ctx, re.S)
    direct_txt = dsec.group(1) if dsec else ""
    return {
        "n_direct": len(re.findall(r"(?m)^\[\d+\] ", direct_txt)),
        "n_mem_block": ctx.count("[MEMORY BLOCK"),
        "n_entity_seg": ctx.count("[ENTITY:"),
        "n_rel_seg": ctx.count("[RELATIONS]"),
        "n_fact_seg": ctx.count("[FACT TYPES"),
        "chars_total": total,
        "chars_direct": len(direct_txt),
        "share_direct": round(len(direct_txt) / total, 3),
        "share_org": round(1 - len(direct_txt) / total, 3),
    }


def llm_judge(question: str, ground_truth: str, prediction: str) -> dict:
    """LLM-as-judge：对预测答案打分（对齐 LoCoMo 评测口径）。

    返回: {"correct": bool, "reason": "判断理由"}
    """
    prompt = (
        f"判断以下回答是否正确。\n\n"
        f"问题: {question}\n"
        f"标准答案: {ground_truth}\n"
        f"待评回答: {prediction}\n\n"
        "只输出 JSON: {\"correct\": true/false, \"reason\": \"一句话理由\"}"
    )
    raw = _chat(prompt, 100, 0.0)
    i = raw.find("{")
    try:
        d = json.loads(raw[i:] if i >= 0 else "{}")
        return {"correct": bool(d.get("correct")), "reason": d.get("reason", "")}
    except Exception:
        return {"correct": False, "reason": f"judge parse fail: {raw[:80]}"}


_reranker_model = None


def _cuda_oom(exc: Exception) -> bool:
    """CUDA OOM 判定 (bge-reranker 显存不足): PyTorch 'out of memory' 特征。"""
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


def _empty_cuda_cache() -> None:
    """OOM 降级后释放 reranker 占用显存 (无 CUDA/清理失败均静默)。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
            if _cuda_oom(e):
                # 2026-09-03 达摩院 R1: OOM 静默降级 → 显式标记 + 清理显存 (不动排序语义)
                print("[RERANK] CUDA OOM → 原序降级 (bge-reranker 加载)", flush=True)
                _empty_cuda_cache()
            else:
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
        if _cuda_oom(e):
            # 2026-09-03 达摩院 R1: CUDA OOM 显式降级 (原静默走原序) + empty_cache; 排序不变
            print("[RERANK] CUDA OOM → 原序降级", flush=True)
            _empty_cuda_cache()
        else:
            print(f"[reranker] 推理失败, 原序降级: {e}", flush=True)
        return docs[:k]
