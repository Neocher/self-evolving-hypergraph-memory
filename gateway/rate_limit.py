"""
gateway/rate_limit.py — 速率限制
===============================
基于 IP 的内存 token bucket。
配置: SHM_RATE_LIMIT="1000/m,10000/d"
"""

from __future__ import annotations

import collections
import os
import re
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    tokens: float
    max_tokens: int
    refill_rate: float  # tokens/second
    last_refill: float = field(default_factory=time.time)


class RateLimiter:
    """基于 IP 的 token bucket 速率限制器。"""

    def __init__(self, config: str | None = None):
        cfg = config or os.environ.get("SHM_RATE_LIMIT", "1000/m,10000/d")
        self._limits = self._parse_config(cfg)
        self._buckets: dict[str, dict[str, TokenBucket]] = collections.defaultdict(dict)
        self._last_cleanup = time.time()

    def check(self, ip: str, cost: int = 1) -> bool:
        """检查请求是否允许。返回 True=允许, False=超限。"""
        self._maybe_cleanup()
        now = time.time()
        for name, limit in self._limits.items():
            if ip not in self._buckets:
                self._buckets[ip] = {}
            if name not in self._buckets[ip]:
                self._buckets[ip][name] = TokenBucket(
                    tokens=limit["max"],
                    max_tokens=limit["max"],
                    refill_rate=limit["rate"],
                )
            bucket = self._buckets[ip][name]
            elapsed = now - bucket.last_refill
            bucket.tokens = min(bucket.max_tokens, bucket.tokens + elapsed * bucket.refill_rate)
            bucket.last_refill = now
            if bucket.tokens < cost:
                return False
            bucket.tokens -= cost
        return True

    def _parse_config(self, cfg: str) -> dict:
        """解析 '1000/m,10000/d' → {minute: {max, rate}, ...}"""
        limits = {}
        for part in cfg.split(","):
            part = part.strip()
            m = re.match(r"(\d+)/([smhd])", part)
            if m:
                count = int(m.group(1))
                unit = m.group(2)
                mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
                limits[f"{unit}"] = {"max": count, "rate": count / mult}
        return limits

    def _maybe_cleanup(self):
        """每 5 分钟清理一次 >1h 未活动的 IP。"""
        now = time.time()
        if now - self._last_cleanup < 300:
            return
        self._last_cleanup = now
        inactive = []
        for ip, buckets in self._buckets.items():
            all_old = all(
                now - b.last_refill > 3600 for b in buckets.values()
            )
            if all_old:
                inactive.append(ip)
        for ip in inactive:
            del self._buckets[ip]
