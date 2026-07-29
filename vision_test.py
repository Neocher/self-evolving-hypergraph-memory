#!/usr/bin/env python3
"""Send image to multi-modal model for analysis."""
import base64
import json
import urllib.request

with open("/home/admin/.hermes/cache/images/img_42ba29b72f60.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

body = {
    "model": "qwen/qwen3.5-omni-plus",
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
    "http://127.0.0.1:8082/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={
        "Authorization": "Bearer admin",
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
