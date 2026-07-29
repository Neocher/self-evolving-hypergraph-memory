#!/usr/bin/env python3
"""Transcribe audio: upload file then transcribe via DashScope."""
import json
import urllib.request
import os
import time

path = "/home/admin/.hermes/cache/documents/doc_ed61b6665be4_阳神修炼与天心合一.m4a"
api_key = "sk-REDACTED"

# Step 1: Upload file as binary
print("📤 上传文件...")
import http.client

# Use multipart upload
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
filename = os.path.basename(path)
file_size = os.path.getsize(path)

with open(path, "rb") as f:
    file_data = f.read()

body_bytes = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
    f"Content-Type: audio/m4a\r\n\r\n"
).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(
    "https://dashscope.aliyuncs.com/api/v1/files",
    data=body_bytes,
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    },
    method="POST"
)
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
        file_id = result.get("output", {}).get("file_id", result.get("file_id", ""))
        print(f"✅ 上传成功: file_id={file_id}")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:500])
except urllib.error.HTTPError as e:
    err = e.read().decode()
    print(f"❌ 上传失败 {e.code}: {err[:500]}")
