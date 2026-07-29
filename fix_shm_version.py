#!/usr/bin/env python3
"""Fix SHM CLAUDE.md version on GitHub."""
import base64, json, os, urllib.request

GH_TOKEN = os.environ.get("GH_TOKEN", "")
headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}
REPO = "Neocher/self-evolving-hypergraph-memory"

with open("/home/admin/shm/CLAUDE.md", "rb") as f:
    content_b64 = base64.b64encode(f.read()).decode()
    # Verify v5.9 is in content
    assert b"v5.9" in base64.b64decode(content_b64), "v5.9 not found in content"

# Get SHA of existing CLAUDE.md
req = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/contents/CLAUDE.md",
    headers=headers)
with urllib.request.urlopen(req, timeout=15) as resp:
    sha = json.loads(resp.read())["sha"]

# Update
data = json.dumps({
    "message": "🐛 修复版本号: SHM v3.0 → v5.9",
    "content": content_b64, "sha": sha, "branch": "main"
}).encode()
req = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/contents/CLAUDE.md",
    data=data,
    headers={**headers, "Content-Type": "application/json"},
    method="PUT")
with urllib.request.urlopen(req, timeout=15) as resp:
    print(f"✅ SHM CLAUDE.md 已更新: {json.loads(resp.read())['content']['sha'][:10]}")
