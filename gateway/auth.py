"""
gateway/auth.py — 认证系统
==========================
Token 文件: ~/.shm/auth.tokens (JSON, chmod 600)
DEV_MODE=true 跳过认证
"""

from __future__ import annotations

import json
import os
import secrets
import stat
import time
from typing import Optional

DEFAULT_TOKEN_DIR = os.path.expanduser("~/.shm")


class TokenManager:
    """基于文件的 Token 认证管理器。支持 TTL 和 scope。"""

    DEFAULT_TTL = 30 * 24 * 3600  # 默认 30 天

    def __init__(self, token_path: Optional[str] = None):
        self._path = token_path or os.path.join(DEFAULT_TOKEN_DIR, "auth.tokens")
        self._tokens: dict[str, dict] = {}
        self._load()

    def validate(self, token: str) -> Optional[dict]:
        """验证 token 有效性，返回 token 信息或 None。"""
        now = time.time()
        for name, info in self._tokens.items():
            if info.get("token") == token:
                expires_at = info.get("expires_at", 0)
                if expires_at and now > expires_at:
                    self.revoke_key(name)
                    return None
                return {"name": name, **info}
        return None

    def create_key(self, name: str, ttl: int = DEFAULT_TTL, scope: str = "admin") -> str:
        """创建新 token。
        Args:
            name: token 名称标识
            ttl: 过期时间（秒），默认 30 天
            scope: 权限范围（admin/readonly）
        """
        token = "shm_" + secrets.token_hex(16)
        self._tokens[name] = {
            "token": token,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
            "name": name,
            "scope": scope,
        }
        self._save()
        return token

    def revoke_key(self, name: str) -> bool:
        """撤销 token。"""
        if name in self._tokens:
            del self._tokens[name]
            self._save()
            return True
        return False

    def list_keys(self) -> list[dict]:
        """列出所有 token（不含 token 值）。"""
        return [
            {"name": k, "created_at": v["created_at"],
             "expires_at": v.get("expires_at", 0), "scope": v.get("scope", "")}
            for k, v in self._tokens.items()
        ]

    def count(self) -> int:
        return len(self._tokens)

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


def create_auth_middleware(dev_mode: bool = True, skip_paths: Optional[list] = None,
                           token_manager: Optional[TokenManager] = None):
    """创建 FastAPI 中间件工厂 — 认证 + 速率限制。"""
    from fastapi.responses import JSONResponse
    from gateway.rate_limit import RateLimiter

    skip = set(skip_paths or ["/health", "/docs", "/openapi.json", "/"])
    tm = token_manager or (None if dev_mode else TokenManager())
    rl = RateLimiter()

    async def middleware(request, call_next):
        if request.url.path in skip:
            return await call_next(request)

        if not dev_mode and tm is not None:
            auth = request.headers.get("Authorization", "")
            token = auth[7:] if auth.startswith("Bearer ") else ""
            if not token or not tm.validate(token):
                return JSONResponse(status_code=401, content={"error": "unauthorized"})

        ip = request.client.host if request.client else "127.0.0.1"
        if not rl.check(ip):
            return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"},
                                headers={"Retry-After": "60"})

        return await call_next(request)

    return middleware


def is_dev_mode() -> bool:
    return os.environ.get("DEV_MODE", "false").lower() == "true"
