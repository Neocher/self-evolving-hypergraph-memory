#!/usr/bin/env python3
"""Arxiv 自动追踪 — 智能体记忆系统每日新论文"""
import json, os, subprocess, sys
from datetime import datetime, timedelta

PROXY = "socks5://172.21.0.1:1081"
CURL = ["curl", "--socks5-hostname", "172.21.0.1:1081", "-s", "--max-time", "20"]
BASE = "https://export.arxiv.org/api/query"

QUERIES = {
    "self-evolving-memory": "all:%22self-evolving+memory%22+OR+all:%22self-modifying+memory%22",
    "agent-memory-consolidation": "all:%22memory+consolidation%22+AND+all:%22agent%22",
    "hypergraph-memory": "all:%22hypergraph+memory%22+AND+all:%22agent%22",
    "sleep-dream-consolidation": "all:%22sleep+consolidation%22+AND+all:%22LLM%22",
    "graph-agent-memory": "all:%22graph+memory%22+AND+all:%22LLM+agent%22",
}

def fetch(query_id, query):
    url = f"{BASE}?search_query={query}&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending"
    r = subprocess.run(CURL + [url], capture_output=True, text=True, timeout=25)
    if r.returncode != 0:
        return []
    import re
    titles = re.findall(r'<title>(.*?)</title>', r.stdout)
    ids = re.findall(r'arxiv.org/abs/(\d+\.\d+)', r.stdout)
    summaries = re.findall(r'<summary>(.*?)</summary>', r.stdout, re.DOTALL)
    papers = []
    for i, (t, aid) in enumerate(zip(titles, ids)):
        if i == 0 and "arXiv Query" in t:
            continue
        s = summaries[i].strip()[:200] if i < len(summaries) else ""
        papers.append({"title": t.strip(), "id": aid, "summary": re.sub(r'\s+', ' ', s)})
    return papers

def main():
    results = {}
    for qid, q in QUERIES.items():
        try:
            papers = fetch(qid, q)
            results[qid] = papers
        except Exception as e:
            results[qid] = [{"title": f"ERROR: {e}", "id": "", "summary": ""}]
    output = f"# Arxiv Agent Memory Tracker — {datetime.now():%Y-%m-%d}\n\n"
    for qid, papers in results.items():
        if not papers:
            continue
        output += f"## {qid}\n"
        for p in papers:
            aid = f" (arXiv:{p['id']})" if p['id'] else ""
            output += f"- **{p['title']}**{aid}\n"
            if p.get('summary'):
                output += f"  {p['summary'][:150]}\n"
        output += "\n"
    print(output)

if __name__ == "__main__":
    main()
