"""回归测试：验收命令的 ``--timeout=120`` 参数必须被正确识别。

复刻验收调用形态（子进程 + collect-only，避免整套用例被双重执行），
同时断言：
1. 输出中不得出现 ``unrecognized arguments``；
2. 子进程退出码为 0（参数解析链路完整、收集成功）。

无论环境装了官方 pytest-timeout，还是走本仓库 conftest 的兜底实现，
本测试都必须通过。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_acceptance_command_accepts_timeout_flag():
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--collect-only",
        "-q",
        "--timeout=120",
        "-p",
        "no:cacheprovider",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=110,  # 低于外层 120s 预算
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "unrecognized arguments" not in combined, combined
    assert proc.returncode == 0, combined
