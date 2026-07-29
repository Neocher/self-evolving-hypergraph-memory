#!/usr/bin/env python3
"""Transcribe audio via DashScope - full task format."""
import json
import urllib.request
import time

api_key = "sk-REDACTED"
file_id = "eaae73dd-8102-407d-941e-080efaca1d74"

# Try different API formats
endpoints = [
    ("v1/services/asr/transcriptions", {
        "task": "asr",
        "model": "qwen3-asr-flash",
        "input": {"file_ids": [file_id]},
        "parameters": {"result_format": "text"}
    }),
    ("v1/services/asr/transcriptions/create", {
        "model": "qwen3-asr-flash",
        "input": {"file_ids": [file_id]}
    }),
    ("v1/services/audio/asr/transcriptions", {
        "model": "qwen3-asr-flash",
        "input": {"audio_files": [{"file_id": file_id}]}
    }),
]

for endpoint, body in endpoints:
    url = f"https://dashscope.aliyuncs.com/api/{endpoint}"
    print(f"\n📡 POST {endpoint}")
    req = urllib.request.Request(
        url,
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
            print(f"  ✅ {json.dumps(result, indent=2, ensure_ascii=False)[:300]}")
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"  ❌ {e.code}: {err[:200]}")
