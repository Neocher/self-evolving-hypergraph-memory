"""根 conftest：为 pytest 提供 ``--timeout`` 参数的兜底实现。

背景：验收命令
    python3 -m pytest tests/ -x -q --timeout=120
在缺少第三方插件 ``pytest-timeout`` 的环境中，会在收集阶段即报
``error: unrecognized arguments: --timeout=120`` 而整体失败。

策略（零冲突）：
* 启动时探测 ``pytest_timeout`` 是否已安装；
* 未安装 → 本模块注册 ``--timeout`` 选项，并用 SIGALRM 对每个用例的
  整个 runtest 协议（setup/call/teardown 合并计时）强制超时；
  超时以显式异常呈现为 FAILED/ERROR，绝不静默吞掉；
* 已安装 → 本模块不定义任何钩子，完全交由官方插件处理，
  避免 "--timeout" 选项被重复定义导致的启动冲突。

注意：本文件只做参数与超时兜底，不触碰业务逻辑；
所有用例仍必须经由公共入口 retrieve()/write 端点验证行为。
"""

from __future__ import annotations

import signal

import pytest

try:
    import pytest_timeout  # type: ignore[import-not-found]  # noqa: F401

    _OFFICIAL_TIMEOUT_PRESENT = True
except Exception:
    _OFFICIAL_TIMEOUT_PRESENT = False


if not _OFFICIAL_TIMEOUT_PRESENT:

    class _FallbackTimeout(Exception):
        """超时时抛出的内部异常，由 pytest 呈现为显式失败而非静默跳过。"""


    def _make_alarm_handler(seconds: float):
        def _handler(signum, frame):  # pragma: no cover - 由信号异步触发
            raise _FallbackTimeout(
                f"Timeout: test exceeded {seconds:g} seconds "
                "(fallback --timeout active; pytest-timeout 未安装)"
            )

        return _handler


    def pytest_addoption(parser) -> None:
        group = parser.getgroup(
            "fallback-timeout", "--timeout 兜底实现（仅在缺 pytest-timeout 时生效）"
        )
        group.addoption(
            "--timeout",
            action="store",
            type=float,
            default=None,
            metavar="SECONDS",
            help=(
                "每个用例的超时秒数（取整）。"
                "仅当未安装 pytest-timeout 插件时由本兜底实现生效。"
            ),
        )


    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(item, nextitem):
        seconds = item.config.getoption("timeout", default=None)

        # 未配置超时，或平台不支持 SIGALRM → 完全走默认流程
        if seconds is None or not hasattr(signal, "SIGALRM"):
            return (yield)

        prev_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _make_alarm_handler(float(seconds)))
        signal.alarm(max(1, int(seconds)))
        try:
            return (yield)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)
