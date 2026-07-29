#!/usr/bin/env python3
"""Transcribe using qwen-audio via correct API format."""
import json
import urllib.request

api_key = "sk-REDACTED"

# Minimal transcription request
body = {
    "model": "paraformer-realtime-v2",
    "input": {
        "audio_file": "https://dashscope.oss-cn-hangzhou.aliyuncs.com/samples/audio/test.m4a"
    }
}

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
        print(json.dumps(result, indent=2, ensure_ascii=False)[:500])
except urllib.error.HTTPError as e:
    err = e.read().decode()
    print(f"Error {e.code}: {err[:300]}")
