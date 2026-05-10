#!/usr/bin/env python3
"""Upload SHM v4 code to GitHub via Content API"""
import json, base64, os, subprocess, sys

TOKEN = os.popen("cat ~/.git-credentials | sed 's|https://[^:]*:\\([^@]*\\)@.*|\\1|'").read().strip()
OWNER = "Neocher"
REPO = "self-evolving-hypergraph-memory"
BASE = f"https://api.github.com/repos/{OWNER}/{REPO}"

def api(method, path, data=None):
    """Call GitHub API"""
    cmd = f'curl -s -X {method} -H "Authorization: token {TOKEN}" -H "Content-Type: application/json"'
    if data:
        tmp = "/tmp/gh_payload.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        cmd += f' -d @{tmp}'
    cmd += f' "{BASE}{path}"'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    try:
        return json.loads(result.stdout)
    except:
        print(f"API error: {result.stdout[:500]}")
        return {"error": result.stdout[:500]}

def read_b64(path):
    with open(os.path.join("/home/admin/shm", path), "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Files to upload (path_in_repo, source_path)
files = [
    ("README.md", "README.md"),
    (".gitignore", ".gitignore"),
    ("Makefile", "Makefile"),
    ("pyproject.toml", "pyproject.toml"),
    ("requirements.txt", "requirements.txt"),
    ("run_server.py", "run_server.py"),
    ("benchmark.py", "benchmark.py"),
    ("api/__init__.py", "api/__init__.py"),
    ("api/app.py", "api/app.py"),
    ("api/models.py", "api/models.py"),
    ("api/routes.py", "api/routes.py"),
    ("config/__init__.py", "config/__init__.py"),
    ("config/settings.py", "config/settings.py"),
    ("config/defaults.yaml", "config/defaults.yaml"),
    ("core/__init__.py", "core/__init__.py"),
    ("core/audit_chain.py", "core/audit_chain.py"),
    ("core/dream_pipeline.py", "core/dream_pipeline.py"),
    ("core/dream_scheduler.py", "core/dream_scheduler.py"),
    ("core/hebbian.py", "core/hebbian.py"),
    ("core/retry.py", "core/retry.py"),
    ("core/ssm_gate.py", "core/ssm_gate.py"),
    ("core/tau_decay.py", "core/tau_decay.py"),
    ("embedding/__init__.py", "embedding/__init__.py"),
    ("embedding/encoder.py", "embedding/encoder.py"),
    ("graph/__init__.py", "graph/__init__.py"),
    ("graph/hyperedge.py", "graph/hyperedge.py"),
    ("graph/kuzu_store.py", "graph/kuzu_store.py"),
    ("observability/__init__.py", "observability/__init__.py"),
    ("observability/health.py", "observability/health.py"),
    ("observability/logger.py", "observability/logger.py"),
    ("observability/metrics.py", "observability/metrics.py"),
    ("retrieval/__init__.py", "retrieval/__init__.py"),
    ("retrieval/coarse_to_fine.py", "retrieval/coarse_to_fine.py"),
    ("retrieval/community_report.py", "retrieval/community_report.py"),
    ("retrieval/query_router.py", "retrieval/query_router.py"),
]

# Upload files one by one using Content API
sha_map = {}  # Track SHAs for subsequent commits

for i, (repo_path, local_path) in enumerate(files):
    content = read_b64(local_path)
    
    payload = {
        "message": f"Add {repo_path}",
        "content": content,
        "branch": "main"
    }
    if repo_path in sha_map:
        payload["sha"] = sha_map[repo_path]
    
    result = api("PUT", f"/contents/{repo_path}", payload)
    
    if "content" in result:
        sha_map[repo_path] = result["content"]["sha"]
        print(f"  [{i+1}/{len(files)}] {repo_path} OK")
    elif "message" in result and "sha" in result.get("error", {}):
        sha_map[repo_path] = result["error"]["sha"]
        print(f"  [{i+1}/{len(files)}] {repo_path} (already exists)")
    else:
        print(f"  [{i+1}/{len(files)}] {repo_path} FAIL: {result.get('message', str(result)[:200])}")

print(f"\nUploaded {len(sha_map)}/{len(files)} files")
