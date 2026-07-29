#!/usr/bin/env python3
"""Transcribe audio using DashScope file-based ASR API."""
import base64
import json
import urllib.request
import time

path = "/home/admin/.hermes/cache/documents/doc_ed61b6665be4_阳神修炼与天心合一.m4a"
api_key = "sk-REDACTED"

# Step 1: Upload file
print("📤 上传音频文件...")
with open(path, "rb") as f:
    audio_data = f.read()

file_body = {
    "model": "qwen3-asr-flash",
    "audio": base64.b64encode(audio_data).decode(),
}

req = urllib.request.Request(
    "https://dashscope.aliyuncs.com/api/v1/services/asr/transcriptions",
    data=json.dumps(file_body).encode(),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    method="POST"
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
        print("Result:", json.dumps(result, indent=2, ensure_ascii=False)[:1000])
except urllib.error.HTTPError as e:
    err = e.read().decode()
    print(f"Error {e.code}: {err[:500]}")
except Exception as e:
    print(f"Exception: {e}")
