"""
Prometheus 指标收集
==================
收集以下指标：
- 请求总数 (counter)
- 请求延迟分布 (histogram)
- 错误率 (counter)
- 梦境执行计数 (counter)
- 断路器状态 (gauge)
- 索引大小 (gauge)
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, REGISTRY, generate_latest

REQUEST_COUNT = Counter(
    "shm_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "shm_request_latency_seconds",
    "Request latency distribution",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)

ERROR_COUNT = Counter(
    "shm_errors_total",
    "Total number of errors",
    ["error_type", "module"],
)

CIRCUIT_BREAKER_STATE = Gauge(
    "shm_circuit_breaker_state",
    "Circuit breaker state (0=CLOSED, 1=HALF_OPEN, 2=OPEN)",
    ["component"],
)

DREAM_CYCLE_COUNT = Counter(
    "shm_dream_cycles_total",
    "Total number of dream cycles",
    ["trigger_mode"],
)

DREAM_DURATION = Histogram(
    "shm_dream_duration_seconds",
    "Dream cycle duration distribution",
    buckets=(1, 5, 10, 30, 60, 120, 300),
)

INDEX_SIZE = Gauge(
    "shm_index_size",
    "Number of vectors in FAISS index",
    ["index_type"],
)


def record_request(method: str, endpoint: str, status: str, duration: float) -> None:
    """记录一次 API 请求的指标。"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def record_error(error_type: str, module: str) -> None:
    """记录一次错误。"""
    ERROR_COUNT.labels(error_type=error_type, module=module).inc()


def record_circuit_breaker(component: str, state: int) -> None:
    """记录断路器状态变化。"""
    CIRCUIT_BREAKER_STATE.labels(component=component).set(state)


def record_dream_cycle(trigger_mode: str, duration: float) -> None:
    """记录一次梦境执行。"""
    DREAM_CYCLE_COUNT.labels(trigger_mode=trigger_mode).inc()
    DREAM_DURATION.observe(duration)


def set_index_size(index_type: str, size: int) -> None:
    """更新 FAISS 索引大小指标。"""
    INDEX_SIZE.labels(index_type=index_type).set(size)


def get_metrics() -> bytes:
    """获取 Prometheus 格式的指标数据（text/plain）。"""
    return generate_latest(REGISTRY)
