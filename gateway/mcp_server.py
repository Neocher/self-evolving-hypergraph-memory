"""
MCP Server — SHM 记忆网关 + 三体协奏管道
===========================================
通过 Model Context Protocol (SSE/stdio) 暴露 SHM 核心能力和 Agent 调度。
对已有 SHM 服务 (:8000) 和 ACP 桥 (:8770) 通过 HTTP 访问，不创建新连接。

运行方式:
    # SSE 模式（推荐，供外部 MCP 客户端连接）
    python -m gateway.mcp_server --port 8002

    # stdio 模式
    python -m gateway.mcp_server
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("gateway.mcp")

_AC_BRIDGE = "http://127.0.0.1:8770"
_SHM_API = "http://127.0.0.1:8000"
_AGENT_MAP = {
    "cc": "claude-code", "claude-code": "claude-code",
    "oc": "opencode", "opencode": "opencode",
    "codex": "codex",
}


async def _acp_dispatch(agent: str, prompt: str) -> dict:
    """通过 ACP 桥发送任务并轮询等待结果。"""
    target = _AGENT_MAP.get(agent, agent)
    async with httpx.AsyncClient(timeout=620) as client:
        r = await client.post(
            f"{_AC_BRIDGE}/dispatch",
            json={"target_agent": target, "prompt": prompt},
        )
        r.raise_for_status()
        task_id = r.json()["task_id"]
        for _ in range(60):
            await asyncio.sleep(5)
            r = await client.get(f"{_AC_BRIDGE}/tasks/{task_id}")
            data = r.json()
            if data["status"] in ("completed", "failed"):
                return data
        return {"status": "timeout", "output": "", "error": "poll timed out"}


def _fmt_result(agent: str, result: dict) -> str:
    out = f"Agent: {agent}\nStatus: {result['status']}\nElapsed: {result.get('elapsed', 'N/A')}s\n"
    if result.get("output"):
        out += f"\nOutput:\n{result['output'][:2000]}"
    if result.get("error"):
        out += f"\nError: {result['error'][:500]}"
    return out


def register_tools(mcp: FastMCP) -> None:
    """向 FastMCP 注册所有工具（通过 HTTP 调用已有服务）。"""

    @mcp.tool(name="shm_health", description="SHM 健康检查 — 返回核心组件状态。")
    async def shm_health() -> str:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                h = (await c.get(f"{_SHM_API}/health")).json()
            s = h.get("stats", {})
            return (
                f"Status: {h.get('status', '?')}\n"
                f"Graph: {h.get('graph_connected', '?')}\n"
                f"Version: {s.get('version', '?')} — {s.get('version_name', '?')}\n"
                f"Nodes: {s.get('node_count', '?')} | Hyperedges: {s.get('hyperedge_count', '?')}\n"
                f"FAISS: {s.get('faiss_index_size', '?')} vectors\n"
                f"Dream runs: {s.get('dream_run_count', '?')} | Uptime: {s.get('uptime_seconds', 0):.0f}s"
            )
        except Exception as e:
            return f"SHM unavailable: {e}"

    @mcp.tool(
        name="pipeline_dispatch",
        description="向指定 Agent 发任务（cc/claude-code / opencode/oc / codex），等待完成。",
    )
    async def pipeline_dispatch(agent: str, prompt: str) -> str:
        return _fmt_result(agent, await _acp_dispatch(agent, prompt))

    @mcp.tool(
        name="pipeline_trio",
        description="完整三体协奏：CC 设计 → OpenCode 实现 → Codex 审核。返回三段结果。",
    )
    async def pipeline_trio(prompt: str) -> str:
        parts = []
        for agent, role in [("cc", "设计"), ("opencode", "实现"), ("codex", "审核")]:
            parts.append(f"─── {role} ({agent}) ───")
            result = await _acp_dispatch(agent, prompt)
            parts.append(f"Status: {result['status']} ({result.get('elapsed', 'N/A')}s)")
            if result.get("output"):
                parts.append(result["output"][:1500])
            if result.get("error"):
                parts.append(f"Error: {result['error'][:300]}")
            parts.append("")
        return "\n".join(parts)

    @mcp.tool(
        name="pipeline_status",
        description="查询 ACP 桥的所有 Agent 健康状态和活跃任务数。",
    )
    async def pipeline_status() -> str:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{_AC_BRIDGE}/agents")
            agents = r.json()
            lines = ["Agent Health:", "─" * 40]
            for name, info in agents.items():
                lines.append(
                    f"  {name:12s} success={info['success']:3d}"
                    f" failure={info['failure']:2d}"
                    f" degraded={'YES' if info['degraded'] else 'no'}"
                )
            try:
                h = (await client.get(f"{_AC_BRIDGE}/health")).json()
                lines.append(f"\nActive tasks: {h.get('tasks', '?')}")
            except Exception:
                pass
            return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHM MCP Gateway Server")
    parser.add_argument("--port", type=int, default=0, help="HTTP SSE port (omit for stdio mode)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    mcp = FastMCP(
        name="SHM MCP Gateway",
        instructions="SHM (Self-evolving Hypergraph Memory) MCP server. "
                     "Access SHM memory and dispatch Trio Concerto pipeline (CC/OpenCode/Codex).",
        debug=args.debug,
        port=args.port,
    )
    register_tools(mcp)

    if args.port:
        logger.info("Starting SHM MCP Gateway on SSE :%d", args.port)
        mcp.run(transport="sse", mount_path="/")
    else:
        logger.info("Starting SHM MCP Gateway (stdio mode)...")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
