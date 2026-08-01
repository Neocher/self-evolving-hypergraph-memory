"""
Dashboard 仪表盘
================
提供 Web 可视化界面，监控系统状态：
- /dashboard (HTML) — 仪表盘主页面
- /dashboard/api/overview (JSON) — 概览统计数据
- /dashboard/api/memories (JSON) — 记忆列表
- /dashboard/api/dreams (JSON) — 梦境报告
- /dashboard/api/logs (JSON) — 近期日志
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from api._routes import Services, get_services
from observability.logger import get_logger

logger = get_logger(__name__)

# 日志路径：优先从配置获取，其次 data/logs/，最后兜底
_DASHBOARD_LOG_DIR = os.environ.get("SHM_LOG_DIR", "")
if not _DASHBOARD_LOG_DIR:
    _DASHBOARD_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "logs")

# Dashboard API Key 认证（从环境变量读取，空字符串时禁用）
_DASHBOARD_API_KEY = os.environ.get("SHM_DASHBOARD_API_KEY", "")

async def _dashboard_auth(request: Request) -> None:
    """Dashboard API 认证守卫（可选，通过 SHM_DASHBOARD_API_KEY 环境变量配置）。"""
    if not _DASHBOARD_API_KEY:
        return
    api_key = request.headers.get("X-API-Key", "") or request.query_params.get("api_key", "")
    if api_key != _DASHBOARD_API_KEY:
        from fastapi.responses import JSONResponse
        from fastapi import HTTPException
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing dashboard API key. Set via SHM_DASHBOARD_API_KEY env var."
        )

dashboard_router = APIRouter()

# Jinja2 模板目录
_HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))


# ─── 辅助函数 ──────────────────────────────────────────────


def _now() -> float:
    return time.time()


# ─── 页面路由 ──────────────────────────────────────────────


@dashboard_router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_page(request: Request, _auth=Depends(_dashboard_auth)):
    """仪表盘主页面 (HTML)。"""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "now": _now()},
    )


# ─── API 路由 ──────────────────────────────────────────────


@dashboard_router.get("/dashboard/api/overview")
async def api_overview(
    deps: Services = Depends(get_services),
    _auth=Depends(_dashboard_auth),
) -> dict[str, Any]:
    """概览统计：记忆总数 / FAISS 状态 / 梦境状态 / 系统健康。"""
    stats: dict[str, Any] = {
        "timestamp": _now(),
        "graphlite_connected": deps.graphlite_store is not None,
        "encoder_loaded": deps.encoder is not None,
    }

    # 记忆统计
    memory_count = 0
    if deps.graphlite_store is not None:
        try:
            rows = deps.graphlite_store.query_cypher(
                "MATCH (e:EpisodeNode) RETURN count(*) AS cnt"
            )
            if rows:
                row = rows[0]
                memory_count = (
                    row[0]
                    if isinstance(row, (list, tuple))
                    else row.get("cnt", 0)
                )
        except Exception as e:
            logger.warning("Dashboard: memory count query failed: %s", e)
    stats["memory_count"] = memory_count

    # FAISS 统计
    faiss_size = 0
    faiss_dim = 0
    if deps.faiss_index is not None:
        try:
            faiss_size = deps.faiss_index.ntotal
        except Exception:
            pass
    if hasattr(deps, "faiss_dim"):
        faiss_dim = deps.faiss_dim
    stats["faiss_size"] = faiss_size
    stats["faiss_dim"] = faiss_dim

    # 梦境状态
    dream_status = "inactive"
    last_dream = None
    dream_run_count = 0
    if deps.dream_scheduler is not None:
        try:
            ds = deps.dream_scheduler
            dream_status = "running" if ds.is_running else "idle"
            dream_run_count = ds.run_count
            last_dream = ds.last_run_time if ds.last_run_time > 0 else None
        except Exception:
            pass
    stats["dream_status"] = dream_status
    stats["dream_run_count"] = dream_run_count
    stats["last_dream"] = last_dream

    # 超边 & 社区统计
    hyperedge_count = 0
    community_count = 0
    if deps.graphlite_store is not None:
        try:
            rows = deps.graphlite_store.query_cypher(
                "MATCH (h:HyperedgeNode) RETURN count(*) AS cnt"
            )
            if rows:
                row = rows[0]
                hyperedge_count = (
                    row[0]
                    if isinstance(row, (list, tuple))
                    else row.get("cnt", 0)
                )
        except Exception:
            pass
        try:
            rows = deps.graphlite_store.query_cypher(
                "MATCH (c:CommunityNode) RETURN count(*) AS cnt"
            )
            if rows:
                row = rows[0]
                community_count = (
                    row[0]
                    if isinstance(row, (list, tuple))
                    else row.get("cnt", 0)
                )
        except Exception:
            pass
    stats["hyperedge_count"] = hyperedge_count
    stats["community_count"] = community_count

    # 检索成功率趋势（最近 N 次）
    retrieval_trend: list[dict[str, Any]] = []
    try:
        from observability.metrics import get_metrics_text
        # 尝试从 Prometheus 指标中解析检索成功率
        metrics_text = get_metrics_text()
        stats["metrics_sample"] = metrics_text[:500] if metrics_text else ""
    except Exception:
        stats["metrics_sample"] = ""
    stats["retrieval_trend"] = retrieval_trend

    return stats


@dashboard_router.get("/dashboard/api/memories")
async def api_memories(
    limit: int = 50,
    offset: int = 0,
    deps: Services = Depends(get_services),
    _auth=Depends(_dashboard_auth),
) -> dict[str, Any]:
    """记忆列表（按时间倒序）。"""
    items: list[dict[str, Any]] = []
    total = 0

    if deps.graphlite_store is not None:
        try:
            # GraphLite 参数化 LIMIT 不生效 (实测返回超量) → 字面量插值 + int 校验
            _lim = int(limit)
            _off = int(offset)
            rows = deps.graphlite_store.query_cypher(
                "MATCH (e:EpisodeNode) RETURN e.id, e.content, "
                "e.source, e.created_at, e.tau_initial "
                f"ORDER BY e.created_at DESC LIMIT {_lim} OFFSET {_off}",
            )
            for row in rows:
                if isinstance(row, dict):
                    items.append({
                        "id": row.get("e.id", ""),
                        "content": (row.get("e.content", "") or "")[:200],
                        "source": row.get("e.source", ""),
                        "created_at": row.get("e.created_at", 0.0),
                        "tau": row.get("e.tau_initial", 1.0),
                    })
                elif isinstance(row, (list, tuple)):
                    items.append({
                        "id": str(row[0]) if len(row) > 0 else "",
                        "content": (str(row[1]) if len(row) > 1 else "")[:200],
                        "source": str(row[2]) if len(row) > 2 else "",
                        "created_at": float(row[3]) if len(row) > 3 else 0.0,
                        "tau": float(row[4]) if len(row) > 4 else 1.0,
                    })
            # 获取总数
            count_rows = deps.graphlite_store.query_cypher(
                "MATCH (e:EpisodeNode) RETURN count(*) AS cnt"
            )
            if count_rows:
                r = count_rows[0]
                total = r[0] if isinstance(r, (list, tuple)) else r.get("cnt", 0)
        except Exception as e:
            logger.warning("Dashboard: memory list query failed: %s", e)

    return {"items": items, "total": total, "limit": limit, "offset": offset}


@dashboard_router.get("/dashboard/api/dreams")
async def api_dreams(
    limit: int = 20,
    deps: Services = Depends(get_services),
    _auth=Depends(_dashboard_auth),
) -> dict[str, Any]:
    """梦境报告列表（从候选存储读取）。"""
    candidates: list[dict[str, Any]] = []

    store = getattr(deps, "dream_candidate_store", None)
    if store is not None:
        try:
            raw = store.list_candidates(limit=limit)
            candidates = [
                {
                    "dream_id": r.get("dream_id", ""),
                    "created_at": r.get("created_at", 0.0),
                    "trigger_mode": r.get("trigger_mode", "unknown"),
                    "community_count": r.get("community_count", 0),
                    "prune_count": r.get("prune_count", 0),
                    "conflict_count": r.get("conflict_count", 0),
                    "stats": r.get("stats", {}),
                }
                for r in raw
            ]
        except Exception as e:
            logger.warning("Dashboard: dream list query failed: %s", e)

    # 补充当前梦境状态
    current_dream = None
    if deps.dream_scheduler is not None:
        try:
            ds = deps.dream_scheduler
            current_dream = {
                "running": ds.is_running,
                "run_count": ds.run_count,
                "last_dream_time": ds.last_run_time if ds.last_run_time > 0 else None,
            }
        except Exception:
            pass

    return {"candidates": candidates, "total": len(candidates), "current": current_dream}


@dashboard_router.get("/dashboard/api/logs")
async def api_logs(
    lines: int = 100,
    _auth=Depends(_dashboard_auth),
) -> dict[str, Any]:
    """近期的系统日志。"""
    log_entries: list[dict[str, Any]] = []
    log_dir = Path(_DASHBOARD_LOG_DIR)
    if log_dir.is_dir():
        log_paths = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    else:
        log_paths = []

    for log_path in log_paths:
        if not log_path.exists():
            continue
        try:
            # tail -n 实现：使用 deque 避免全量读入
            from collections import deque
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail_lines = deque(f, maxlen=lines)
            for line in tail_lines:
                stripped = line.rstrip("\n")
                if stripped:
                    log_entries.append({
                        "message": stripped[:500],
                        "source": log_path.name,
                    })
        except Exception as e:
            log_entries.append({"message": f"Error reading {log_path}: {e}", "source": "error"})

    return {"logs": log_entries[:lines], "total": len(log_entries)}
