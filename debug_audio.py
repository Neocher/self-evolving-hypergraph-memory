#!/usr/bin/env python3
"""Debug DashScope audio API format."""
import base64
import json
import urllib.request

api_key = "sk-REDACTED"

# Read mp3
with open("/tmp/audio_test.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Test possible formats
tests = [
    # Format 1: type=audio with data URL
    {"model": "qwen3.5-omni-plus", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "转写"},
        {"type": "audio", "audio": f"data:audio/mp3;base64,{audio_b64}"}
    ]}]},
    # Format 2: type=audio_url
    {"model": "qwen3.5-omni-plus", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "转写"},
        {"type": "audio_url", "audio_url": {"url": f"data:audio/mp3;base64,{audio_b64}"}}
    ]}]},
    # Format 3: type=audio with file_id
    {"model": "qwen3.5-omni-plus", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "转写"},
        {"type": "file", "file": {"data": audio_b64, "format": "mp3"}}
    ]}]},
]

for i, body in enumerate(tests):
    print(f"\n📡 Format {i+1}...")
    data = json.dumps(body).encode()
    print(f"    body大小: {len(data)//1024}KB")
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            print(f"    ✅ {json.dumps(result, indent=2, ensure_ascii=False)[:300]}")
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"    ❌ {e.code}: {err[:200]}")
