# -*- coding: utf-8 -*-
"""启动预热（v6.32 检索冷启动消除）验收测试。

对齐任务书 AC2..AC6（AC1 端到端首条延迟在真实进程实测，见交付报告）：
  AC2 预热确实执行：固定前缀短查询跑完整 FUSION 检索 + 日志打印耗时。
  AC3 fail-soft：预热路径抛异常不向上传播（服务仍可完成初始化）。
  AC4 可关闭：warmup_retrieval_enabled=False → 不执行预热（retrieve 不被调用）。
  AC5 无缓存污染：预热后结果缓存被清空，预热查询结果不泄漏给真实查询。

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_warmup.py -q
"""
from __future__ import annotations

import types

import api.app as app_module
from api.routes import _deps
from config.settings import RetrievalConfig
from retrieval.query_router import RetrievalLevel


def _fake_settings(enabled: bool):
    return types.SimpleNamespace(
        retrieval=types.SimpleNamespace(warmup_retrieval_enabled=enabled),
    )


class _RecordingRouter:
    """记录 retrieve 调用；可注入异常模拟预热失败（AC3）。"""

    def __init__(self, result=None, exc=None):
        self.calls: list[tuple[str, dict]] = []
        self._result = result if result is not None else []
        self._exc = exc

    def retrieve(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        if self._exc is not None:
            raise self._exc
        return self._result


class _RecordingLogger:
    """记录 info/warning 调用（structlog 不经 stdlib，caplog 抓不到 → 直接打桩）。"""

    def __init__(self):
        self.info_msgs: list[str] = []
        self.warning_msgs: list[str] = []

    @staticmethod
    def _fmt(msg, args):
        try:
            return msg % args if args else msg
        except Exception:
            return str(msg)

    def info(self, msg, *args, **kwargs):
        self.info_msgs.append(self._fmt(msg, args))

    def warning(self, msg, *args, **kwargs):
        self.warning_msgs.append(self._fmt(msg, args))

    def exception(self, msg, *args, **kwargs):
        self.warning_msgs.append(self._fmt(msg, args))

    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class TestWarmupConfig:
    """AC4 — 配置项默认开。"""

    def test_default_enabled(self):
        assert RetrievalConfig().warmup_retrieval_enabled is True

    def test_warmup_query_uses_non_colliding_prefix(self):
        """预热文本必须带不可能与真实查询碰撞的专属前缀。"""
        assert app_module._WARMUP_QUERY.startswith(app_module._WARMUP_QUERY_PREFIX)
        assert "warmup" in app_module._WARMUP_QUERY_PREFIX


class TestRunRetrievalWarmup:
    @staticmethod
    def _svc(router):
        return types.SimpleNamespace(query_router=router)

    def test_warmup_runs_full_fusion_and_logs(self, monkeypatch):
        """AC2 — 预热确实执行：FUSION 全链路检索 + 耗时日志。"""
        router = _RecordingRouter()
        fake_logger = _RecordingLogger()
        monkeypatch.setattr(app_module, "get_settings", lambda: _fake_settings(True))
        monkeypatch.setattr(app_module, "logger", fake_logger)

        app_module._run_retrieval_warmup(self._svc(router))

        assert len(router.calls) == 1, "预热应恰好执行一次完整检索"
        query, kwargs = router.calls[0]
        assert query.startswith(app_module._WARMUP_QUERY_PREFIX)
        assert kwargs.get("level") == RetrievalLevel.FUSION, "预热必须覆盖 FUSION 全链路"
        assert any(m.startswith("Retrieval warmup done in") for m in fake_logger.info_msgs), (
            "预热耗时未打印日志"
        )

    def test_fail_soft(self, monkeypatch):
        """AC3 — 预热抛异常不向上传播（服务仍可完成初始化）。"""
        router = _RecordingRouter(exc=RuntimeError("boom"))
        fake_logger = _RecordingLogger()
        monkeypatch.setattr(app_module, "get_settings", lambda: _fake_settings(True))
        monkeypatch.setattr(app_module, "logger", fake_logger)

        app_module._run_retrieval_warmup(self._svc(router))  # 不得抛出

        assert len(router.calls) == 1
        assert any("Retrieval warmup failed" in m for m in fake_logger.warning_msgs), (
            "预热失败须记 WARNING（fail-soft 可观测）"
        )

    def test_disabled_skips_warmup(self, monkeypatch):
        """AC4 — warmup_retrieval_enabled=False → 不执行预热。"""
        router = _RecordingRouter()
        fake_logger = _RecordingLogger()
        monkeypatch.setattr(app_module, "get_settings", lambda: _fake_settings(False))
        monkeypatch.setattr(app_module, "logger", fake_logger)

        app_module._run_retrieval_warmup(self._svc(router))

        assert router.calls == [], "关闭开关后不得执行预热检索"

    def test_clears_result_cache(self, monkeypatch):
        """AC5 — 预热后结果缓存被清空（预热查询结果不泄漏给真实查询）。"""
        router = _RecordingRouter()
        monkeypatch.setattr(app_module, "get_settings", lambda: _fake_settings(True))
        with _deps._result_cache_lock:
            _deps._result_cache.clear()
            _deps._result_cache[f"{app_module._WARMUP_QUERY_PREFIX}:5"] = object()

        app_module._run_retrieval_warmup(self._svc(router))

        with _deps._result_cache_lock:
            assert _deps._result_cache == {}, "预热后结果缓存须被显式清空"
