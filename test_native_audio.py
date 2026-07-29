#!/usr/bin/env python3
"""Test DashScope native multimodal API for audio."""
import base64
import json
import urllib.request

api_key = "sk-REDACTED"

with open("/tmp/audio_test.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# DashScope native multimodal-generation API format
body = {
    "model": "qwen3.5-omni-plus",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "请完整转写这段音频的内容。"},
                    {"audio": f"data:audio/mp3;base64,{audio_b64}"}
                ]
            }
        ]
    },
    "parameters": {
        "result_format": "message"
    }
}

data = json.dumps(body).encode()
print(f"请求大小: {len(data)//1024}KB")

req = urllib.request.Request(
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    data=data,
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    method="POST"
)
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
        print(json.dumps(result, indent=2, ensure_ascii=False)[:3000])
except urllib.error.HTTPError as e:
    err = e.read().decode()
    print(f"Error {e.code}: {err[:500]}")
