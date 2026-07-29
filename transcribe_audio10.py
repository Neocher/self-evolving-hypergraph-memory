#!/usr/bin/env python3
"""Try DashScope multimodal-generation for ASR."""
import json
import urllib.request

api_key = "sk-REDACTED"
file_id = "eaae73dd-8102-407d-941e-080efaca1d74"

body = {
    "model": "qwen3-asr-flash",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "请转写这段音频的内容。"},
                    {"file_id": file_id}
                ]
            }
        ]
    }
}

req = urllib.request.Request(
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    data=json.dumps(body).encode(),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    method="POST"
)
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
except urllib.error.HTTPError as e:
    err = e.read().decode()
    print(f"Error {e.code}: {err[:500]}")
