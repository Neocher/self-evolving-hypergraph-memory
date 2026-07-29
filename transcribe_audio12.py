#!/usr/bin/env python3
"""Test various audio content formats for DashScope API."""
import json
import urllib.request

api_key = "sk-REDACTED"
file_id = "eaae73dd-8102-407d-941e-080efaca1d74"

formats = [
    [{"text": "转写音频"}, {"audio": file_id}],
    [{"text": "转写音频"}, {"audio_id": file_id}],
    [{"text": "转写音频"}, {"audio_file_id": file_id}],
]

for i, fmt in enumerate(formats):
    body = {
        "model": "qwen3-asr-flash",
        "input": {
            "messages": [
                {"role": "user", "content": fmt}
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
            print(f"✅ Format {i}: {json.dumps(result, indent=2, ensure_ascii=False)[:300]}")
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"❌ Format {i} ({list(fmt[1].keys())[0]}): {e.code} - {err[:150]}")
