#!/usr/bin/env python3
"""Transcribe audio using uploaded file with DashScope ASR."""
import json
import urllib.request
import time

api_key = "sk-REDACTED"
file_id = "eaae73dd-8102-407d-941e-080efaca1d74"

# Create transcription task
body = {
    "model": "qwen3-asr-flash",
    "input": {
        "file_ids": [file_id]
    },
    "parameters": {
        "result_format": "text"
    }
}

print("🎤 创建转录任务...")
req = urllib.request.Request(
    "https://dashscope.aliyuncs.com/api/v1/services/asr/transcriptions",
    data=json.dumps(body).encode(),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    method="POST"
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        print(json.dumps(result, indent=2, ensure_ascii=False)[:1000])
        
        # Check for task_id for polling
        task_id = result.get("output", {}).get("task_id", "")
        if task_id:
            print(f"\n⏳ 任务ID: {task_id}, 轮询结果...")
            for i in range(30):
                time.sleep(2)
                poll_req = urllib.request.Request(
                    f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                with urllib.request.urlopen(poll_req, timeout=10) as poll_resp:
                    poll = json.loads(poll_resp.read())
                    status = poll.get("output", {}).get("task_status", "")
                    print(f"  状态: {status}")
                    if status == "SUCCEEDED":
                        print("\n📝 转写结果:")
                        print(poll.get("output", {}).get("result", {}).get("transcription", ""))
                        break
                    elif status == "FAILED":
                        print(f"❌ 失败: {poll.get('output', {}).get('message', '')}")
                        break
        
except urllib.error.HTTPError as e:
    err = e.read().decode()
    print(f"❌ Error {e.code}: {err[:500]}")
