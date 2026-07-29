#!/usr/bin/env python3
"""Transcribe audio via DashScope - task wrapper format."""
import json
import urllib.request
import time

api_key = "sk-REDACTED"
file_id = "eaae73dd-8102-407d-941e-080efaca1d74"

# ASR with task wrapper
body = {
    "model": "qwen3-asr-flash",
    "input": {
        "file_ids": [file_id]
    },
    "parameters": {
        "result_format": "text"
    }
}

# Try with task wrapper
task_body = {
    "task": "asr",
    "model": "qwen3-asr-flash",
    "input": {
        "file_ids": [file_id]
    },
    "parameters": {
        "result_format": "text"
    }
}

for name, b in [("no task", body), ("with task", task_body)]:
    print(f"\n尝试 {name} 格式...")
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/api/v1/services/asr/transcriptions",
        data=json.dumps(b).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            print(f"  ✅ 成功: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
            
            # Check for task_id
            task_id = result.get("output", {}).get("task_id", "") or \
                      result.get("data", {}).get("task_id", "")
            if task_id:
                print(f"\n  ⏳ 任务ID: {task_id}, 等待结果...")
                for i in range(30):
                    time.sleep(2)
                    poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
                    poll_req = urllib.request.Request(
                        poll_url,
                        headers={"Authorization": f"Bearer {api_key}"}
                    )
                    try:
                        with urllib.request.urlopen(poll_req, timeout=10) as poll_resp:
                            poll = json.loads(poll_resp.read())
                            status = poll.get("output", {}).get("task_status", "UNKNOWN")
                            print(f"    状态: {status}", end="\r")
                            if status == "SUCCEEDED":
                                print(f"\n  ✅ 完成!")
                                output = poll.get("output", {})
                                result_data = output.get("result", output)
                                print(json.dumps(result_data, indent=2, ensure_ascii=False)[:1000])
                                break
                            elif status in ("FAILED", "CANCELED"):
                                print(f"\n  ❌ {status}: {poll.get('output', {}).get('message', '')}")
                                break
                    except urllib.error.HTTPError as pe:
                        err = pe.read().decode()
                        print(f"\n  ❌ 轮询错误: {err[:200]}")
                        break
            break
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"  ❌ Error {e.code}: {err[:300]}")
