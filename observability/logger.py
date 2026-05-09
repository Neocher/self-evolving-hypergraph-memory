"""
结构化日志系统
=============
使用 structlog 实现结构化日志，每个请求注入 trace_id（UUID）。

若 structlog 初始化失败，自动降级到标准 logging 模块。
日志输出格式：JSON 行，便于日志聚合系统（ELK、Loki）解析。
"""

from __future__ import annotations

import uuid
import logging as _stdlib_logging
from contextvars import ContextVar
from typing import Optional

import structlog

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """获取当前上下文的 trace_id。"""
    return _trace_id_var.get()


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """设置当前上下文的 trace_id，None 则自动生成 UUID。"""
    tid = trace_id or str(uuid.uuid4())
    _trace_id_var.set(tid)
    return tid


def _inject_trace_id(_logger, _method_name, event_dict):
    """structlog processor: 注入 trace_id 到事件字典。"""
    event_dict["trace_id"] = get_trace_id()
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """
    配置结构化日志。

    若 structlog 初始化失败，自动降级到标准 logging 模块。
    """
    try:
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                _inject_trace_id,
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    except Exception as exc:
        _stdlib_logging.basicConfig(
            level=getattr(_stdlib_logging, log_level.upper(), _stdlib_logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        _stdlib_logging.getLogger(__name__).warning(
            "structlog unavailable, fell back to stdlib logging: %s", exc
        )


def get_logger(name: Optional[str] = None):
    """
    获取带 trace_id 注入的 logger 实例。

    若 structlog 不可用，降级返回标准 logging.Logger。
    """
    try:
        return structlog.get_logger(name or __name__)
    except Exception:
        return _stdlib_logging.getLogger(name or __name__)
