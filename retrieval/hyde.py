"""HyDE 假设文档生成（P3b）
==========================
生产 QueryRouter 的 HyDE 增强通道：LLM 生成假设段落 → 与原始查询一起参与
融合检索（dual 双路）或替代原始向量（replace 单路）。

与评测脚本（/tmp/bench_locomo_prod.py L252-256）等价：
  - urllib 同步直连 https://api.deepseek.com/chat/completions（无 /v1 前缀）
  - 显式禁代理（ProxyHandler({})）——评测实测 httpx 直连返回 Vercel HTML
    404、curl 直连成功，独立同步通道最可靠
  - key 从环境变量 DEEPSEEK_API_KEY 读（不新建 .env 加载器）；模型
    deepseek-chat，DEEPSEEK_MODEL 可覆盖；max_tokens=150 temperature=0.3

失败降级（防每查询重试烧预算）：
  - 确定性失败（缺 key / HTTP 401/403）→ _PERM_FAILED 永久跳过标记
  - 瞬时失败（超时 / 5xx / 网络错 / 其余 4xx）→ 60s 冷却窗口
    （last_fail_ts，窗口内直接返回 None）
  - 任何异常 → 返回 None（检索路径永不因 HyDE 抛错，零回归）

缓存：模块级 OrderedDict LRU（容量 256 / TTL 3600s，threading.Lock 保护），
key = 原始 query（评测 200 问不重复 LLM 调用）。
【P3b R1 P2】single-flight：per-key 进行中 Event（_inflight）——并发相同未缓存
query 只触发一次 LLM 调用，其余线程等 Event 置位后读缓存结果（网络调用在锁外）。
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Optional

from observability.logger import get_logger

logger = get_logger(__name__)

_API_URL = "https://api.deepseek.com/chat/completions"
_DEFAULT_MODEL = "deepseek-chat"
_PROMPT_TEMPLATE = (
    "Based on the question below, write a short factual paragraph "
    "(3-5 sentences) that would contain the answer. "
    "Make it concrete and specific. "
    # 【P3b R1 P2】语言一致性约束：中文查询生成中文假设段落（语种匹配提升向量相似度；
    # v6.1 bge-m3 多语言可跨语种，但同语言假设仍是最佳实践）；追加在指令段末尾，
    # Question:/Hypothetical passage: 结构不变（评测脚本子串断言与等效性不变）。
    "Write in the same language as the question.\n\n"
    "Question: {query}\n\n"
    "Hypothetical passage:"
)

_CACHE_CAPACITY = 256
_CACHE_TTL_S = 3600.0
_COOLDOWN_S = 60.0

# 模块级状态（进程常驻；单实例服务共享）
_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
_lock = threading.Lock()
# 【P3b R1 P2】single-flight 进行中表：key=原始 query → Event（拥有者 finally 置位）。
_inflight: dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()
_PERM_FAILED = False
_last_fail_ts = 0.0


def _sanitize_timeout(timeout: float) -> float:
    try:
        t = float(timeout)
    except (TypeError, ValueError):
        return 2.0
    return t if t > 0 else 2.0


def generate_hypothesis(query: str, timeout: float = 2.0) -> Optional[str]:
    """生成查询的假设文档段落；任何失败 → None（检索路径零回归）。

    【P3b R1 P2】single-flight：并发相同未缓存 query 只触发一次 LLM 调用——
    后到线程登记到 _inflight 后等待拥有者完成（Event 置位）再读缓存，不重复触网。

    Args:
        query: 原始查询文本（未归一化；缓存 key 即原文）
        timeout: HTTP 超时（秒）

    Returns:
        假设段落（去空白）或 None（未启用/失败/缓存过期重建失败）。
    """
    global _PERM_FAILED, _last_fail_ts
    q = (query or "").strip()
    if not q:
        return None
    now = time.time()
    if _PERM_FAILED or now - _last_fail_ts < _COOLDOWN_S:
        return None

    def _cache_hit(q: str) -> Optional[str]:
        """缓存命中（TTL 内）返回段落；TTL 过期视为 miss。"""
        with _lock:
            hit = _cache.get(q)
            if hit is not None and time.time() - hit[0] < _CACHE_TTL_S:
                _cache.move_to_end(q)
                return hit[1]
        return None

    hit = _cache_hit(q)
    if hit is not None:
        return hit

    # 【P3b R1 P2】single-flight 登记：已有线程在生成 → 等待其完成（网络调用在锁外）
    with _inflight_lock:
        ev = _inflight.get(q)
        if ev is None:
            ev = threading.Event()
            _inflight[q] = ev
            is_owner = True
        else:
            is_owner = False
    if not is_owner:
        # 防御性超时：拥有者线程异常消亡时防无限等待（正常路径拥有者 finally 必置位）。
        # 【R2 P2】收紧到检索预算同量级（timeout + 2s ≈ 3.5s，外层 3s/5s wait_for），
        # 防慢故障（DNS 悬挂/拥有者消亡）长时间占用 to_thread worker。
        ev.wait(timeout=_sanitize_timeout(timeout) + 2.0)
        return _cache_hit(q)

    try:
        # 拥有者：等锁期间可能已被并发完成，二次查缓存
        hit = _cache_hit(q)
        if hit is not None:
            return hit

        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            with _lock:
                _PERM_FAILED = True
            logger.warning("HyDE disabled: DEEPSEEK_API_KEY not set (permanent skip)")
            return None

        model = os.environ.get("DEEPSEEK_MODEL", "").strip() or _DEFAULT_MODEL
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": _PROMPT_TEMPLATE.format(query=q)}],
            "max_tokens": 150,
            "temperature": 0.3,
        }
        req = urllib.request.Request(
            _API_URL,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        # 显式禁代理：评测脚本实测走系统代理（Vercel HTML 404），禁代理直连成功
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=_sanitize_timeout(timeout)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            hypo = ((data.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "")
            hypo = (hypo or "").strip()
            if not hypo:
                raise ValueError("empty hypothesis from LLM")
            with _lock:
                _cache[q] = (time.time(), hypo)
                while len(_cache) > _CACHE_CAPACITY:
                    _cache.popitem(last=False)
            return hypo
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                with _lock:
                    _PERM_FAILED = True
                logger.warning("HyDE disabled: HTTP %d (permanent skip)", e.code)
            else:
                # 5xx / 429 / 其他 4xx：瞬时失败走冷却窗口
                with _lock:
                    _last_fail_ts = time.time()
                logger.warning("HyDE transient failure: HTTP %d (cooldown %.0fs)",
                               e.code, _COOLDOWN_S)
            return None
        except Exception:
            # 超时（URLError/TimeoutError）/ 网络错 / JSON 解析错 → 冷却窗口
            with _lock:
                _last_fail_ts = time.time()
            logger.debug("HyDE failure (cooldown %.0fs)", _COOLDOWN_S)
            return None
    finally:
        # 无论成败都释放 flight 并唤醒等待者（等待者读缓存或返回 None）
        with _inflight_lock:
            _inflight.pop(q, None)
        ev.set()
