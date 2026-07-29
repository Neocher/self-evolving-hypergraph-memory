"""
MCP (Model Context Protocol) 共享工具服务器
========================================
所有 Agent (Claude Code / OpenCode / Codex) 通过此 MCP 服务器共享工具。

当前提供:
- read_file: 读取文件内容
- search_files: 搜索文件内容或文件名
- terminal: 执行shell命令
- get_project_info: 获取项目结构信息

使用方式:
  python mcp_server.py
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("mcp-server")

# 工作目录
WORKDIR = Path(__file__).parent


def handle_read_file(path: str) -> dict:
    """读取文件"""
    full_path = WORKDIR / path
    if not full_path.exists() or not full_path.is_file():
        return {"error": f"File not found: {path}"}
    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
        return {"content": content, "path": str(full_path)}
    except Exception as e:
        return {"error": str(e)}


def handle_search_files(pattern: str, target: str = "content") -> dict:
    """搜索文件"""
    try:
        if target == "content":
            result = subprocess.run(
                ["grep", "-rn", pattern, "--include=*.py", str(WORKDIR)],
                capture_output=True, text=True, timeout=10,
            )
        else:
            result = subprocess.run(
                ["find", str(WORKDIR), "-name", pattern, "-type", "f"],
                capture_output=True, text=True, timeout=10,
            )
        lines = [l for l in result.stdout.split("\n") if l.strip()]
        return {"matches": lines[:50], "total": len(lines)}
    except subprocess.TimeoutExpired:
        return {"error": "search timed out"}
    except Exception as e:
        return {"error": str(e)}


def handle_terminal(command: str, timeout: int = 15) -> dict:
    """执行命令"""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(WORKDIR),
        )
        return {
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-1000:],
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"command timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


def handle_project_info() -> dict:
    """获取项目信息"""
    py_files = list(WORKDIR.rglob("*.py"))
    dirs = set(f.parent.relative_to(WORKDIR) for f in py_files)
    return {
        "root": str(WORKDIR),
        "python_files": len(py_files),
        "directories": sorted(str(d) for d in dirs if str(d) != "."),
        "directory_count": len(dirs),
    }


TOOLS = {
    "read_file": {
        "description": "读取项目中的文件内容",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "handler": handle_read_file,
    },
    "search_files": {
        "description": "在项目中搜索文件内容或按文件名查找",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "target": {"type": "string", "enum": ["content", "files"]},
            },
            "required": ["pattern"],
        },
        "handler": handle_search_files,
    },
    "terminal": {
        "description": "在项目目录中执行shell命令",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 15},
            },
            "required": ["command"],
        },
        "handler": handle_terminal,
    },
    "get_project_info": {
        "description": "获取当前项目的基本信息（目录结构、文件数量）",
        "input_schema": {"type": "object", "properties": {}},
        "handler": handle_project_info,
    },
}


def handle_request(request: dict) -> dict:
    """处理 JSON-RPC 请求"""
    req_id = request.get("id", 0)
    method = request.get("method", "")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                },
                "serverInfo": {
                    "name": "shm-mcp-server",
                    "version": "1.0.0",
                },
            },
        }
    elif method == "tools/list":
        tool_list = [
            {
                "name": name,
                "description": info["description"],
                "inputSchema": info["input_schema"],
            }
            for name, info in TOOLS.items()
        ]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tool_list},
        }
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        if tool_name in TOOLS:
            result = TOOLS[tool_name]["handler"](**arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]},
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
        }

    return {"jsonrpc": "2.0", "id": req_id, "result": None}


def main():
    """MCP 服务器主循环 (stdio 模式)"""
    logger.info("SHM MCP Server starting (stdio mode)...")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")


if __name__ == "__main__":
    main()
