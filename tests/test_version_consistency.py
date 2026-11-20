"""SHM 版本元数据一致性测试 (v5.31.0)。"""

import tomllib
from pathlib import Path

from shm._version import (
    __version__,
    __version_info__,
    __version_name__,
    __release_date__,
    VERSION_SUMMARY,
)

ROOT = Path(__file__).resolve().parents[1]


def _first_v_line(summary: str) -> str:
    """VERSION_SUMMARY 首个 v 开头行（去空白）。"""
    for line in summary.splitlines():
        stripped = line.strip()
        if stripped.startswith("v"):
            return stripped
    raise AssertionError("VERSION_SUMMARY 无 v 开头行")


def test_version_internal_consistency():
    """__version_info__ 与 __version__ 一致；VERSION_SUMMARY 首块标题一致。

    标题允许带副标题（如 ``— 检索性能总纲（…）``），因此校验前缀与冒号结尾，
    不再要求与 ``{name}:`` 逐字符相等（v6.28 起标题确实带了副标题）。
    """
    assert __version_info__ == tuple(map(int, __version__.split(".")))
    header = _first_v_line(VERSION_SUMMARY)
    expected_prefix = f"v{__version__} ({__release_date__}) {__version_name__}"
    assert header.startswith(expected_prefix), (
        f"VERSION_SUMMARY 首行必须以 {expected_prefix!r} 开头: {header!r}"
    )
    assert header.endswith(":"), f"VERSION_SUMMARY 首行必须以 ':' 结尾: {header!r}"


def test_version_files_sync():
    """__version__ == pyproject.toml project.version == VERSION 文件内容。"""
    with open(ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    assert pyproject["project"]["version"] == __version__

    version_file = (ROOT / "VERSION").read_text().strip()
    assert version_file == __version__
