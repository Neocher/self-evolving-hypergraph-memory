# ── Trio Concerto Pipeline 工具 ──────────────────────────────────────
# 通过 MCP 协议暴露 Agent 调度能力
# 底层走 ACP 桥 (:8770) 管理子进程

import httpx

_AC_BRIDGE = "http://127.0.0.1:8770"
_AGENT_MAP = {
    "cc": "claude-code", "claude-code": "claude-code",
    "oc": "opencode", "opencode": "opencode",
    "codex": "codex",
}


async def _acp_dispatch(agent: str, prompt: str) -> dict:
    """通过 ACP 桥发送任务并等待结果。"""
    target = _AGENT_MAP.get(agent, agent)
    async with httpx.AsyncClient(timeout=600) as client:
        r = await client.post(
            f"{_AC_BRIDGE}/dispatch",
            json={"target_agent": target, "prompt": prompt},
        )
        r.raise_for_status()
        task_id = r.json()["task_id"]

        # 轮询直到完成或超时
        for _ in range(60):
            await asyncio.sleep(5)
            r = await client.get(f"{_AC_BRIDGE}/tasks/{task_id}")
            data = r.json()
            if data["status"] in ("completed", "failed"):
                return data
        return {"status": "timeout", "output": "", "error": "poll timed out"}


def register_pipeline_tools(mcp: "FastMCP") -> None:
    """向 FastMCP 注册管道工具。"""

    @mcp.tool(
        name="pipeline_dispatch",
        description="向指定 Agent 发送任务（CC/OpenCode/Codex），等待完成。返回执行结果。",
    )
    async def pipeline_dispatch(agent: str, prompt: str) -> str:
        """向指定 Agent 发送任务。"""
        result = await _acp_dispatch(agent, prompt)
        out = f"Agent: {agent}\nStatus: {result['status']}\nElapsed: {result.get('elapsed', 'N/A')}s\n"
        if result.get("output"):
            out += f"\nOutput:\n{result['output'][:2000]}"
        if result.get("error"):
            out += f"\nError: {result['error'][:500]}"
        return out

    @mcp.tool(
        name="pipeline_trio",
        description="运行完整三体协奏管道：CC 设计 -> OpenCode 实现 -> Codex 审核。返回三段执行结果。",
    )
    async def pipeline_trio(prompt: str) -> str:
        """三段式编排：设计→实现→审核。"""
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
        description="查询 ACP 桥的所有 Agent 健康状态和当前任务数。",
    )
    async def pipeline_status() -> str:
        """查询 Agent 健康状态。"""
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
                r2 = await client.get(f"{_AC_BRIDGE}/health")
                h = r2.json()
                lines.append(f"\nActive tasks: {h.get('tasks', '?')}")
            except Exception:
                pass
            return "\n".join(lines)
