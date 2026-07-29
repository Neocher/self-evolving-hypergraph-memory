#!/usr/bin/env python3
"""Transcribe audio using qwen3-asr-flash via OpenAI-compatible API."""
import base64
import json
import urllib.request

path = "/home/admin/.hermes/cache/documents/doc_ed61b6665be4_阳神修炼与天心合一.m4a"
api_key = "sk-REDACTED"

with open(path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Try ASR via compatible-mode
body = {
    "model": "qwen3-asr-flash",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请转写这段音频"},
                {"type": "audio", "audio": f"data:audio/m4a;base64,{audio_b64}"}
            ]
        }
    ],
    "max_tokens": 2000
}

req = urllib.request.Request(
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    method="POST"
)
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read())
        if "choices" in result:
            print(result["choices"][0]["message"]["content"])
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False)[:1000])
except urllib.error.HTTPError as e:
    err = e.read().decode()
    print(f"Error {e.code}: {err[:500]}")
