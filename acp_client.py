#!/usr/bin/env python3
"""
ACP Agent Client — CC/Codex 的 ACP 协议适配器
===========================================
通过 ACP Bridge (:8770) 接收任务、执行、回传结果。

用法:
  python3 acp_client.py --agent claude-code --bridge http://127.0.0.1:8770
  python3 acp_client.py --agent codex --bridge http://127.0.0.1:8770
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request


def register_agent(bridge_url: str, name: str, caps: list[str]):
    """在 ACP Bridge 上注册自己"""
    data = json.dumps({
        "name": name,
        "capabilities": caps,
        "status": "idle",
        "endpoint": f"stdio://{name}",
    }).encode()
    req = urllib.request.Request(
        f"{bridge_url}/api/acp/register",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=5)
    return json.loads(resp.read())


def fetch_task(bridge_url: str, agent_name: str) -> dict | None:
    """轮询获取分配给自己的任务"""
    try:
        resp = urllib.request.urlopen(f"{bridge_url}/api/acp/tasks?limit=10", timeout=5)
        all_tasks = json.loads(resp.read()).get("tasks", [])
        for t in all_tasks:
            if t["target"] == agent_name and t["status"] == "running":
                return t
    except Exception:
        pass
    return None


def report_result(bridge_url: str, task_id: str, agent_name: str,
                  status: str, output: str = "", error: str = ""):
    """回传任务结果"""
    data = json.dumps({
        "task_id": task_id,
        "agent_name": agent_name,
        "status": status,
        "output": output,
        "error": error,
    }).encode()
    req = urllib.request.Request(
        f"{bridge_url}/api/acp/callback",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=5)


def execute_task(task: dict) -> tuple[str, str, str]:
    """根据 agent 类型执行任务"""
    agent = task["target"]
    prompt = task["prompt"]
    context = task.get("context", {})

    full_prompt = prompt
    if context:
        ctx_lines = "\n".join(f"{k}: {v}" for k, v in context.items())
        full_prompt = f"{ctx_lines}\n\n{prompt}"

    try:
        if agent == "claude-code":
            result = subprocess.run(
                ["claude", "-p", "-", "--print"],
                input=full_prompt.encode(),
                capture_output=True, text=True, timeout=50,
                cwd="/home/admin/shm",
            )
            if result.returncode == 0:
                return ("completed", result.stdout.strip(), "")
            return ("failed", "", result.stderr.strip())

        elif agent == "codex":
            result = subprocess.run(
                ["codex", "exec", "-c", "approval=never",
                 "-c", 'sandbox_permissions=["full"]', full_prompt],
                capture_output=True, text=True, timeout=50,
                cwd="/home/admin/shm",
            )
            if result.returncode == 0:
                return ("completed", result.stdout.strip(), "")
            return ("failed", "", result.stderr.strip())

        else:
            return ("failed", "", f"Unknown agent: {agent}")

    except subprocess.TimeoutExpired:
        return ("failed", "", "Task timed out")
    except FileNotFoundError as e:
        return ("failed", "", f"Command not found: {e}")
    except Exception as e:
        return ("failed", "", str(e))


def main():
    parser = argparse.ArgumentParser(description="ACP Agent Client")
    parser.add_argument("--agent", required=True,
                        choices=["claude-code", "codex", "opencode"])
    parser.add_argument("--bridge", default="http://127.0.0.1:8770")
    parser.add_argument("--poll-interval", type=int, default=3)
    args = parser.parse_args()

    caps = {
        "claude-code": ["analyze", "plan", "review"],
        "codex": ["code", "execute", "review"],
        "opencode": ["code", "edit", "execute"],
    }

    # 注册
    register_agent(args.bridge, args.agent, caps.get(args.agent, []))
    print(f"[ACP:{args.agent}] Registered with bridge at {args.bridge}")

    # 轮询任务
    print(f"[ACP:{args.agent}] Waiting for tasks (poll every {args.poll_interval}s)...")
    while True:
        task = fetch_task(args.bridge, args.agent)
        if task:
            print(f"[ACP:{args.agent}] Received task: {task['id']}")
            print(f"  Prompt: {task['prompt'][:100]}...")

            status, output, error = execute_task(task)
            report_result(args.bridge, task["id"], args.agent, status, output, error)
            print(f"[ACP:{args.agent}] Task {task['id']}: {status}")

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
