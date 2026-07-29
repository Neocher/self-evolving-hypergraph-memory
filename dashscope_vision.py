#!/usr/bin/env python3
"""Analyze image via DashScope multimodal API directly."""
import base64
import json
import urllib.request
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/admin/.hermes/cache/images/img_42ba29b72f60.jpg"

with open(path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

body = {
    "model": "qwen3.5-omni-plus",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请详细描述这张图片的内容"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }
    ],
    "max_tokens": 500
}

req = urllib.request.Request(
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={
        "Authorization": "Bearer sk-REDACTED",
        "Content-Type": "application/json"
    },
    method="POST"
)
with urllib.request.urlopen(req, timeout=120) as resp:
    result = json.loads(resp.read())
    if "choices" in result:
        print(result["choices"][0]["message"]["content"])
    else:
        print("Error:", json.dumps(result, indent=2)[:300])
