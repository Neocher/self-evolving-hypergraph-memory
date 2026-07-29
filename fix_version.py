#!/usr/bin/env python3
"""Push CLAUDE.md version fix to GitHub."""
import base64, json, os, urllib.request

GH_TOKEN = os.environ.get("GH_TOKEN", "")
headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}
REPO = "Neocher/smart-router"

with open("/home/admin/shm/CLAUDE.md", "rb") as f:
    content_b64 = base64.b64encode(f.read()).decode()

req = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/contents/CLAUDE.md",
    headers=headers)
with urllib.request.urlopen(req, timeout=15) as resp:
    sha = json.loads(resp.read())["sha"]

req = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/contents/CLAUDE.md",
    data=json.dumps({"message": "🐛 修复版本号: SHM v3.0 → v5.9", "content": content_b64, "sha": sha, "branch": "main"}).encode(),
    headers={**headers, "Content-Type": "application/json"},
    method="PUT")
with urllib.request.urlopen(req, timeout=15) as resp:
    print(f"✅ CLAUDE.md 已更新: {json.loads(resp.read())['content']['sha'][:10]}")
