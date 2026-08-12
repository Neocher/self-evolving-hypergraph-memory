#!/usr/bin/env python3
"""
MCP (Model Context Protocol) 共享工具服务器 — mcp SDK 2.0.0 版
================================================================
所有 Agent (Claude Code / OpenCode / Codex) 通过此 MCP 服务器共享工具。

当前提供:
- read_file: 读取文件内容
- search_files: 搜索文件内容或文件名
- terminal: 执行shell命令 (白名单限制)
- get_project_info: 获取项目结构信息

使用方式 (mcp SDK >= 2.0.0):
  mcp run mcp_server.py
  或 python mcp_server.py
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from pathlib import Path

from mcp.server import MCPServer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("shm-mcp-server")

# 工作目录
WORKDIR = Path(__file__).parent

# 仅保留只读检查命令；解释器/执行器 (python/pip/npm/node/git/make/pytest/ruff/mypy)
# 可被用作 RCE 通道，一律移出白名单
ALLOWED_COMMANDS = {
    "ls", "cat", "head", "tail", "wc", "find", "grep", "rg",
    "tree", "du", "df", "file", "stat",
}

# read_file 单文件读取上限 (1 MB)，防止整文件读入内存无上限
MAX_FILE_BYTES = 1_000_000


def _minimal_env() -> dict[str, str]:
    safe_keys = {
        "PATH", "HOME", "USER", "TERM", "LANG", "LC_ALL",
        "VIRTUAL_ENV", "PYTHONPATH", "PIP_INDEX_URL",
    }
    # 保留 _API_KEY 过滤作为纵深防御，防止后续扩展 safe_keys 时泄漏密钥
    return {k: v for k, v in os.environ.items()
            if k in safe_keys and not k.endswith("_API_KEY")}


mcp = MCPServer(
    "shm-mcp-server",
    version="2.0.0",
    instructions=(
        "SHM 项目共享工具服务器。read_file/search_files/terminal 均以 "
        "项目根目录为基准路径。terminal 仅允许白名单命令。"
    ),
)


@mcp.tool()
def read_file(path: str) -> str:
    """读取项目中的文件内容。

    Args:
        path: 相对于项目根目录的文件路径
    """
    root = WORKDIR.resolve()
    full_path = (WORKDIR / path).resolve()
    if not full_path.is_relative_to(root):
        return json.dumps({"error": "path escapes project root"}, ensure_ascii=False)
    if not full_path.exists() or not full_path.is_file():
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)
    if full_path.stat().st_size > MAX_FILE_BYTES:
        return json.dumps(
            {"error": f"File too large (>{MAX_FILE_BYTES} bytes): {path}"},
            ensure_ascii=False,
        )
    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
        return json.dumps({"content": content, "path": str(full_path)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def search_files(pattern: str, target: str = "content") -> str:
    """在项目中搜索文件内容或按文件名查找。

    Args:
        pattern: 搜索模式 (content 模式为正则, files 模式为文件名 glob)
        target: content=搜内容, files=按文件名查找
    """
    try:
        if target == "content":
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--", pattern, str(WORKDIR)],
                capture_output=True, text=True, timeout=10,
            )
        else:
            result = subprocess.run(
                ["find", str(WORKDIR), "-name", pattern, "-type", "f"],
                capture_output=True, text=True, timeout=10,
            )
        lines = [l for l in result.stdout.split("\n") if l.strip()]
        return json.dumps({"matches": lines[:50], "total": len(lines)}, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "search timed out"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def terminal(command: str, timeout: int = 15) -> str:
    """在项目目录中执行shell命令（受白名单限制）。

    Args:
        command: 要执行的命令, 首个词必须在白名单内
        timeout: 超时秒数 (默认 15)
    """
    try:
        cmd_parts = shlex.split(command)
        if not cmd_parts:
            return json.dumps({"error": "empty command"}, ensure_ascii=False)
        base = os.path.basename(cmd_parts[0])
        if base not in ALLOWED_COMMANDS:
            return json.dumps({"error": f"command not allowed: {base}"}, ensure_ascii=False)
        result = subprocess.run(
            cmd_parts, capture_output=True, text=True,
            timeout=timeout, cwd=str(WORKDIR),
            env=_minimal_env(),
        )
        return json.dumps({
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-1000:],
            "exit_code": result.returncode,
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"command timed out after {timeout}s"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def get_project_info() -> str:
    """获取当前项目的基本信息（目录结构、文件数量）。"""
    py_files = list(WORKDIR.rglob("*.py"))
    dirs = set(f.parent.relative_to(WORKDIR) for f in py_files)
    return json.dumps({
        "root": str(WORKDIR),
        "python_files": len(py_files),
        "directories": sorted(str(d) for d in dirs if str(d) != "."),
        "directory_count": len(dirs),
    }, ensure_ascii=False)


if __name__ == "__main__":
    logger.info("SHM MCP Server starting (stdio mode, mcp SDK 2.0.0)...")
    mcp.run()
