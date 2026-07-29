#!/usr/bin/env python3
"""Codex MCP 客户端——集成到 ACP Bridge"""
import asyncio
import json
import os
import sys


class CodexMCPClient:
    def __init__(self, cwd="/home/admin/shm"):
        self.cwd = cwd
        self.proc = None

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            "codex", "mcp-server",
            "-c", "model=deepseek-chat",
            "-c", "approval=never",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env={"DEEPSEEK_API_KEY": "sk-<REDACTED>",
                 "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/admin/.hermes/node/bin:/home/admin/.local/bin",
                 "HOME": "/home/admin"},
        )

    async def run_codex(self, prompt: str, timeout: int = 300) -> str:
        """运行 Codex 会话，返回文本结果"""
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "codex", "arguments": {
                   "prompt": prompt,
                   "model": "deepseek-chat",
                   "approval-policy": "never",
               }}}
        self.proc.stdin.write((json.dumps(req) + "\n").encode())
        await self.proc.stdin.drain()

        # 持续读直到收到最终响应
        while True:
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=timeout)
            data = json.loads(line.decode())
            if "id" in data:  # JSON-RPC response (not notification)
                content = data.get("result", {}).get("content", [])
                texts = [c["text"] for c in content if c.get("type") == "text"]
                if texts:
                    return "\n".join(texts)
                if data.get("result", {}).get("isError"):
                    return f"[Codex Error] {texts[0] if texts else json.dumps(data)[:200]}"
                return json.dumps(data)[:500]

    async def close(self):
        if self.proc:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.proc.kill()


async def main():
    client = CodexMCPClient()
    await client.start()
    try:
        prompt = sys.argv[1] if len(sys.argv) > 1 else "只输出数字42"
        print(f"Running codex (prompt: {prompt[:50]}...)")
        result = await client.run_codex(prompt)
        print(f"\n=== RESULT ({len(result)} chars) ===")
        print(result[:500])
    except asyncio.TimeoutError:
        print("TIMEOUT: Codex session did not complete")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
