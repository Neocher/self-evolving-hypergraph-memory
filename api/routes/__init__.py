"""
SHM API 路由 — 模块化路由包
============================
从 _routes.py 提供服务。
待拆分子模块: memory, retrieval, hyperedges, communities, dream, system
"""
from __future__ import annotations

from api._routes import (
    router,
    init_services,
    Services,
    flush_faiss_buffer,
    incremental_faiss_update,
    rebuild_index,
    logger,
    get_services,
)

__all__ = [
    "router", "init_services", "Services", "flush_faiss_buffer",
    "incremental_faiss_update", "rebuild_index", "logger", "get_services",
]
