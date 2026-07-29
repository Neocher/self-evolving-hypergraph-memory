"""
深度健康检查
===========
提供系统各组件可用性和完整性检测：
- RyuGraph 连接状态 + 断路器状态
- FAISS 索引状态（大小、可搜索）
- BLAKE3 溯源链完整性
- 梦境调度器状态
- 系统资源使用情况
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class HealthCheckResult:
    """健康检查结果（避免与 api/models.py 中的 HealthStatus 命名冲突）。"""

    status: str  # 'ok' | 'degraded' | 'error'
    graph_connected: bool
    faiss_loaded: bool
    faiss_index_size: int = 0
    chain_verified: bool = True
    dream_scheduler_running: bool = False
    dream_run_count: int = 0  # 【FIX】梦境运行次数
    uptime_seconds: float = 0.0
    node_count: int = 0
    hyperedge_count: int = 0
    last_dream_time: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class _ChainVerificationCache:
    """缓存审计链验证结果，避免每次健康检查都遍历 244MB 审计链。"""

    def __init__(self, max_age: float = 300.0) -> None:
        self._last_result: bool = True
        self._last_check: float = 0.0
        self._max_age = max_age

    def get_or_refresh(self, audit_chain) -> bool:
        now = time.time()
        if now - self._last_check > self._max_age:
            try:
                self._last_result = audit_chain.verify_chain()
            except Exception:
                self._last_result = False
            self._last_check = now
        return self._last_result


# 模块级共享缓存实例，跨请求复用
_CHAIN_CACHE = _ChainVerificationCache(max_age=300.0)


class HealthChecker:
    """深度健康检查器，聚合各组件状态生成统一的健康检查报告。"""

    def __init__(
        self,
        graph_store=None,
        faiss_index=None,
        audit_chain=None,
        dream_scheduler=None,
    ) -> None:
        self.kuzu_store = graph_store  # kuzu_store 作为属性名保持向后兼容
        self.faiss_index = faiss_index
        self.audit_chain = audit_chain
        self.dream_scheduler = dream_scheduler
        self._start_time = time.time()

    def check(self) -> HealthCheckResult:
        """执行完整的深度健康检查。"""
        result = HealthCheckResult(
            status="ok",
            graph_connected=self._check_graph(),
            faiss_loaded=self._check_faiss(),
            uptime_seconds=time.time() - self._start_time,
        )

        if self.faiss_index is not None:
            try:
                result.faiss_index_size = self.faiss_index.ntotal
            except Exception:
                result.faiss_index_size = 0

        # 查询真实节点数和超边数
        if self.kuzu_store is not None:
            try:
                node_rows = self.kuzu_store.query_cypher(
                    "MATCH (n) RETURN count(*) AS cnt"
                )
                if node_rows and len(node_rows) > 0:
                    # RyuGraph returns [[298]] not [{"cnt": 298}]
                    row = node_rows[0]
                    if isinstance(row, (list, tuple)):
                        result.node_count = int(row[0])
                    elif isinstance(row, dict):
                        result.node_count = int(row.get("cnt", 0))
            except Exception:
                result.node_count = 0
            try:
                he_rows = self.kuzu_store.query_cypher(
                    "MATCH (h:HyperedgeNode) RETURN count(*) AS cnt"
                )
                if he_rows and len(he_rows) > 0:
                    row = he_rows[0]
                    if isinstance(row, (list, tuple)):
                        result.hyperedge_count = int(row[0])
                    elif isinstance(row, dict):
                        result.hyperedge_count = int(row.get("cnt", 0))
            except Exception:
                result.hyperedge_count = 0

        if self.audit_chain is not None:
            result.chain_verified = _CHAIN_CACHE.get_or_refresh(self.audit_chain)

        if self.dream_scheduler is not None:
            result.dream_scheduler_running = getattr(
                self.dream_scheduler, "_is_running", False
            )
            # 【FIX】从调度器读取上次梦境运行时间
            last_run = getattr(self.dream_scheduler, "_last_run_time", 0.0)
            if last_run > 0.0:
                result.last_dream_time = last_run
            # 【FIX】读取梦境运行次数
            result.dream_run_count = getattr(self.dream_scheduler, "_dream_run_count", 0)

        if not result.graph_connected:
            result.status = "error"
        elif not result.chain_verified or not result.faiss_loaded:
            result.status = "degraded"

        result.details = {
            "circuit_breaker": self._check_circuit_breaker(),
            "memory_usage": self._get_memory_usage(),
        }

        return result

    def _check_graph(self) -> bool:
        """检查 RyuGraph 连接状态。"""
        if self.kuzu_store is None:
            return False
        try:
            self.kuzu_store.query_cypher("RETURN 1 AS test")
            return True
        except Exception:
            return False

    def _check_faiss(self) -> bool:
        """检查 FAISS 索引状态。"""
        if self.faiss_index is None:
            return False
        try:
            _ = self.faiss_index.ntotal
            return True
        except Exception:
            return False

    def _check_circuit_breaker(self) -> Dict[str, Any]:
        """检查断路器状态（适配滑动窗口接口）。"""
        if self.kuzu_store is None:
            return {"state": "unknown"}
        cb = getattr(self.kuzu_store, "circuit_breaker", None)
        if cb is None:
            return {"state": "not_configured"}

        window = getattr(cb, "_window", [])
        window_size = len(window)
        recent_failures = sum(1 for r in window if not r) if window_size > 0 else 0

        return {
            "state": cb.state.value if hasattr(cb.state, "value") else str(cb.state),
            "window_size": window_size,
            "recent_failures": recent_failures,
            "success_rate": (
                ((window_size - recent_failures) / window_size) * 100
                if window_size > 0
                else 100.0
            ),
        }

    def _get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况。"""
        try:
            import psutil

            proc = psutil.Process()
            mem = proc.memory_info()
            return {
                "rss_mb": round(mem.rss / 1024 / 1024, 2),
                "vms_mb": round(mem.vms / 1024 / 1024, 2),
            }
        except ImportError:
            return {"info": "psutil not available"}
