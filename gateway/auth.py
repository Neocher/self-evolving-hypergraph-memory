"""
gateway/auth.py — 认证系统
==========================
Token 文件: ~/.shm/auth.tokens (JSON, chmod 600)
Token 存储使用 SHA-256 哈希，不存明文。
DEV_MODE=true 跳过认证。
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import time
from typing import Optional

DEFAULT_TOKEN_DIR = os.path.expanduser("~/.shm")


class TokenManager:
    """基于文件的 Token 认证管理器。支持 TTL 和 scope。
    
    【安全】Token 以 SHA-256 哈希存储，原始 token 仅创建时返回一次。
    验证时比对哈希值，防止文件泄露导致凭证失窃。
    """

    DEFAULT_TTL = 30 * 24 * 3600  # 默认 30 天

    def __init__(self, token_path: Optional[str] = None):
        self._path = token_path or os.path.join(DEFAULT_TOKEN_DIR, "auth.tokens")
        self._tokens: dict[str, dict] = {}
        self._load()

    @staticmethod
    def _hash_token(token: str) -> str:
        """对 token 做 SHA-256 哈希，不可逆。"""
        return hashlib.sha256(token.encode()).hexdigest()

    def validate(self, token: str) -> Optional[dict]:
        """验证 token 有效性，返回 token 信息或 None（不含原始 token 值）。"""
        now = time.time()
        token_hash = self._hash_token(token)
        for name, info in self._tokens.items():
            stored_hash = info.get("token_hash", "")
            if not secrets.compare_digest(token_hash, stored_hash):
                continue
            expires_at = info.get("expires_at", 0)
            if expires_at and now > expires_at:
                self.revoke_key(name)
                return None
            return {"name": name, "scope": info.get("scope", ""),
                    "created_at": info.get("created_at", 0)}
        return None

    def create_key(self, name: str, ttl: int = DEFAULT_TTL, scope: str = "admin") -> str:
        """创建新 token。
        Args:
            name: token 名称标识
            ttl: 过期时间（秒），默认 30 天
            scope: 权限范围（admin/readonly）
        Returns:
            str: 原始 token（仅在创建时返回一次，不会持久化到文件）
        """
        raw_token = "shm_" + secrets.token_hex(16)
        self._tokens[name] = {
            "token_hash": self._hash_token(raw_token),
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
            "name": name,
            "scope": scope,
        }
        self._save()
        return raw_token

    def revoke_key(self, name: str) -> bool:
        """撤销 token。"""
        if name in self._tokens:
            del self._tokens[name]
            self._save()
            return True
        return False

    def list_keys(self) -> list[dict]:
        """列出所有 token（不含 token 值和哈希）。"""
        return [
            {"name": k, "created_at": v["created_at"],
             "expires_at": v.get("expires_at", 0), "scope": v.get("scope", "")}
            for k, v in self._tokens.items()
        ]

    def count(self) -> int:
        return len(self._tokens)

    def __repr__(self) -> str:
        return f"TokenManager(path={self._path!r}, keys={len(self._tokens)})"

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path) as f:
                    data = json.load(f)
                self._tokens = data.get("tokens", {})
                # 清理过期 token
                now = time.time()
                expired = [k for k, v in self._tokens.items()
                           if v.get("expires_at", 0) and now > v.get("expires_at", 0)]
                for k in expired:
                    del self._tokens[k]
                if expired:
                    self._save()
            else:
                self._tokens = {}
        except (json.JSONDecodeError, OSError):
            self._tokens = {}

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump({"tokens": self._tokens}, f, indent=2)
        os.chmod(self._path, stat.S_IRUSR | stat.S_IWUSR)


def create_auth_middleware(dev_mode: bool = False, skip_paths: Optional[list] = None,
                           token_manager: Optional[TokenManager] = None):
    """创建 FastAPI 中间件工厂 — 认证 + 速率限制。

    WARNING: dev_mode=True disables authentication. NEVER use in production.
    """
    from fastapi.responses import JSONResponse
    from gateway.rate_limit import RateLimiter

    if dev_mode:
        import logging
        logging.getLogger(__name__).warning(
            "DEV_MODE=true: authentication is DISABLED. "
            "Set DEV_MODE=false in production."
        )

    skip = set(skip_paths if skip_paths is not None else ["/health"])
    tm = token_manager or (None if dev_mode else TokenManager())
    rl = RateLimiter()

    async def middleware(request, call_next):
        if dev_mode:
            return await call_next(request)

        if request.url.path in skip:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        if not token:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        info = tm.validate(token)
        if not info:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})

        # scope enforcement: readonly tokens can only use safe HTTP methods
        scope = info.get("scope", "")
        if scope != "admin" and request.method not in ("GET", "HEAD", "OPTIONS"):
            return JSONResponse(status_code=403, content={"error": "insufficient_scope",
                                                           "required": "admin scope for write operations", "got": scope})

        ip = request.client.host if request.client else "127.0.0.1"
        if not rl.check(ip):
            return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"},
                                headers={"Retry-After": "60"})

        return await call_next(request)

    return middleware


def is_dev_mode() -> bool:
    mode = os.environ.get("DEV_MODE", "false").lower() == "true"
    if mode:
        import logging
        logging.getLogger(__name__).warning(
            "DEV_MODE=true: authentication is DISABLED. "
            "Set DEV_MODE=false in production."
        )
    return mode
