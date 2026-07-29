#!/usr/bin/env python3
"""通过 MCP SSE 调用 pipeline_status 测试"""
import asyncio, httpx

async def main():
    async with httpx.AsyncClient(timeout=30) as c:
        # 1. SSE 获取 session
        async with c.stream("GET", "http://127.0.0.1:8003/sse") as resp:
            session = None
            async for line in resp.aiter_lines():
                if line.startswith("data: /messages/?session_id="):
                    session = line.split("session_id=")[1].strip()
                    break
        if not session:
            print("Failed to get session")
            return

        # 2. 调用 pipeline_status
        r = await c.post(
            f"http://127.0.0.1:8003/messages/?session_id={session}",
            json={"jsonrpc": "2.0", "method": "tools/call",
                  "params": {"name": "pipeline_status", "arguments": {}}, "id": 1},
        )
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print(r.text[:500])

        # 3. 调用 pipeline_dispatch (CC)
        r2 = await c.post(
            f"http://127.0.0.1:8003/messages/?session_id={session}",
            json={"jsonrpc": "2.0", "method": "tools/call",
                  "params": {"name": "pipeline_dispatch",
                            "arguments": {"agent": "cc", "prompt": "输出 Hello from MCP"}},
                  "id": 2},
        )
        print(f"\nDispatch: {r2.status_code}")
        if r2.status_code == 200:
            print(r2.text[:500])

asyncio.run(main())
