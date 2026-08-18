"""
SHM CLI — 开发者命令行工具
=========================
通过 HTTP 连接 SHM API 服务（默认 :8000），在终端直接调用 SHM 能力。

用法:
    # 写入感觉缓冲区
    python -m gateway.cli write "今天学到了MCP协议"

    # 写入情节节点
    python -m gateway.cli episode "重要会议纪要" --source meeting --namespace proj_x

    # 检索记忆
    python -m gateway.cli retrieve "MCP协议" --top-k 5

    # 纯向量检索
    python -m gateway.cli search "MCP" --top-k 10

    # 健康检查
    python -m gateway.cli health

    # 触发梦境
    python -m gateway.cli dream

    # 查看社区
    python -m gateway.cli communities

    # 超边列表
    python -m gateway.cli hyperedges --limit 20

    # Cypher 查询
    python -m gateway.cli cypher "MATCH (n:EpisodeNode) RETURN n.content LIMIT 5"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class SHMClient:
    """通过 HTTP 与 SHM API 通信的轻量客户端。"""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("SHM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")

    def _post(self, path: str, payload: dict) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: dict | None = None) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def write(self, content: str, source: str = "api", namespace: str | None = None) -> Dict[str, Any]:
        return self._post("/memories/sensory", {
            "content": content,
            "source": source,
            "namespace": namespace,
        })

    def episode(self, content: str, source: str = "user",
                namespace: str | None = None,
                force_promote: bool = False) -> Dict[str, Any]:
        return self._post("/memories/episodes", {
            "content": content,
            "source": source,
            "namespace": namespace,
            "force_promote": force_promote,
        })

    def retrieve(self, query: str, top_k: int = 20,
                 namespace: str | None = None,
                 include_shared: bool = True,
                 strategy: str | None = "auto") -> Dict[str, Any]:
        return self._post("/memories/retrieve", {
            "query": query,
            "top_k": top_k,
            "namespace": namespace,
            "include_shared": include_shared,
            "strategy": strategy,
        })

    def search_vector(self, query: str, limit: int = 10) -> Dict[str, Any]:
        return self._post("/search/vector", {
            "query": query,
            "limit": limit,
        })

    def health(self) -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/health", timeout=10).json()

    def dream(self) -> Dict[str, Any]:
        return self._post("/memories/dream/trigger", {})

    def communities(self, limit: int = 50) -> List[Dict[str, Any]]:
        data = self._get("/communities", {"limit": limit})
        return data.get("communities", [])

    def hyperedges(self, limit: int = 50, node_id: str | None = None) -> Dict[str, Any]:
        if node_id:
            return self._get(f"/nodes/{node_id}/hyperedges")
        return self._get("/hyperedges", {"limit": limit})

    def cypher(self, query: str, params: dict | None = None) -> Dict[str, Any]:
        return self._post("/query", {"query": query, "params": params or {}})


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gateway.cli",
        description="SHM 开发者命令行工具",
    )
    parser.add_argument(
        "--base-url", "-u",
        default=os.environ.get("SHM_BASE_URL", DEFAULT_BASE_URL),
        help=f"SHM API 地址 (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="以 JSON 格式输出",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # write
    p_write = sub.add_parser("write", help="写入感觉缓冲区 (Layer1)")
    p_write.add_argument("content", help="记忆内容")
    p_write.add_argument("--source", default="api", help="来源标识")
    p_write.add_argument("--namespace", "-n", default=None, help="命名空间")

    # episode
    p_ep = sub.add_parser("episode", help="创建情节节点 (Layer2)")
    p_ep.add_argument("content", help="情节内容")
    p_ep.add_argument("--source", default="user", help="来源标识")
    p_ep.add_argument("--namespace", "-n", default=None, help="命名空间")
    p_ep.add_argument("--force", action="store_true", default=False, help="强制提升")

    # retrieve
    p_ret = sub.add_parser("retrieve", help="三级融合检索")
    p_ret.add_argument("query", help="检索文本")
    p_ret.add_argument("--top-k", type=int, default=20, help="返回结果数")
    p_ret.add_argument("--namespace", "-n", default=None, help="限定命名空间")
    p_ret.add_argument("--strategy", default="auto", help="检索策略 (auto/hybrid)")

    # search
    p_src = sub.add_parser("search", help="纯向量检索")
    p_src.add_argument("query", help="查询文本")
    p_src.add_argument("--top-k", type=int, default=10, help="返回结果数")

    # health
    sub.add_parser("health", help="深度健康检查")

    # dream
    sub.add_parser("dream", help="触发梦境")

    # communities
    p_com = sub.add_parser("communities", help="列出社区")
    p_com.add_argument("--limit", type=int, default=50, help="返回上限")

    # hyperedges
    p_he = sub.add_parser("hyperedges", help="列出超边")
    p_he.add_argument("--limit", type=int, default=50, help="返回上限")
    p_he.add_argument("--node-id", default=None, help="按节点过滤")

    # cypher
    p_cy = sub.add_parser("cypher", help="执行只读 Cypher 查询")
    p_cy.add_argument("query", help="Cypher 查询语句")
    p_cy.add_argument("--params", default=None, help="查询参数 (JSON)")

    return parser


def _format_results(results: list, top_k: int) -> str:
    if not results:
        return "  (no results)"
    lines = []
    for i, r in enumerate(results[:top_k], 1):
        content = r.get("content", "")[:200].replace("\n", " ")
        score = r.get("score", 0)
        lines.append(f"  {i:2d}. [{score:.3f}] {content}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    client = SHMClient(base_url=args.base_url)
    use_json = args.json

    try:
        match args.command:
            case "write":
                data = client.write(args.content, source=args.source, namespace=args.namespace)
                if use_json:
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    print(f"  record_id: {data['record_id']}")
                    print(f"  buffer_usage: {data['buffer_usage']}")

            case "episode":
                data = client.episode(
                    args.content, source=args.source,
                    namespace=args.namespace, force_promote=args.force,
                )
                if use_json:
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    print(f"  episode_id: {data['episode_id']}")
                    print(f"  status: {data['status']}")
                    print(f"  tau_initial: {data['tau_initial']}")

            case "retrieve":
                data = client.retrieve(
                    args.query, top_k=args.top_k,
                    namespace=args.namespace,
                    strategy=args.strategy,
                )
                if use_json:
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    print(f"Query: {args.query!r}")
                    print(f"Strategy: {data.get('strategy_used', '?')}  "
                          f"Degraded: {data.get('degraded', '?')}  "
                          f"Latency: {data.get('latency_ms', 0):.0f}ms")
                    print(f"Results ({data.get('total_found', 0)}):")
                    print(_format_results(data.get("results", []), args.top_k))

            case "search":
                data = client.search_vector(args.query, limit=args.top_k)
                if use_json:
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    print(f"Vector search: {args.query!r}")
                    print(f"Degraded: {data.get('degraded', '?')}  "
                          f"Latency: {data.get('latency_ms', 0):.0f}ms")
                    print(f"Results ({data.get('total_found', 0)}):")
                    print(_format_results(data.get("results", []), args.top_k))

            case "health":
                data = client.health()
                if use_json:
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    stats = data.get("stats", {})
                    print(f"Status:               {data.get('status', '?')}")
                    print(f"Graph connected:      {data.get('graph_connected', '?')}")
                    print(f"FAISS loaded:         {data.get('faiss_loaded', '?')}")
                    print(f"Dream scheduler:      {data.get('dream_scheduler_running', '?')}")
                    print(f"Node count:           {stats.get('node_count', 'N/A')}")
                    print(f"Hyperedge count:      {stats.get('hyperedge_count', 'N/A')}")
                    print(f"Chain verified:       {stats.get('chain_verified', 'N/A')}")
                    print(f"Uptime:               {stats.get('uptime_seconds', 0):.0f}s")
                    print(f"FAISS index size:     {stats.get('faiss_index_size', 'N/A')}")
                    print(f"Last dream:           {stats.get('last_dream_time', 'N/A')}")
                    print(f"Dream runs:           {stats.get('dream_run_count', 'N/A')}")

            case "dream":
                data = client.dream()
                if use_json:
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    print(f"Dream triggered: accepted={data.get('accepted')}, "
                          f"message={data.get('message')}")

            case "communities":
                communities = client.communities(limit=args.limit)
                if use_json:
                    print(json.dumps(communities, ensure_ascii=False, indent=2))
                else:
                    print(f"Communities ({len(communities)}):")
                    for c in communities:
                        kw = ", ".join(c.get("keywords", [])[:5]) if c.get("keywords") else ""
                        print(f"  [{c.get('id', '?')[:8]}] {c.get('name', '')}  "
                              f"members={c.get('member_count', '?')}  "
                              f"score={c.get('leiden_score', 0):.2f}")
                        if kw:
                            print(f"        keywords: {kw}")

            case "hyperedges":
                data = client.hyperedges(limit=args.limit, node_id=args.node_id)
                if use_json:
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    hyperedges = data.get("hyperedges", [])
                    print(f"Hyperedges ({len(hyperedges)}):")
                    for h in hyperedges:
                        members = ", ".join(m[:8] for m in h.get("member_ids", []))
                        print(f"  [{h.get('id', '?')[:8]}] type={h.get('type', '?')}  "
                              f"gate={h.get('gate_value', 1.0):.2f}  members=[{members}]")

            case "cypher":
                params = json.loads(args.params) if args.params else None
                data = client.cypher(args.query, params=params)
                if use_json:
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    rows = data.get("rows", [])
                    error = data.get("error")
                    if error:
                        print(f"Error: {error}")
                    else:
                        print(f"Cypher result: {data.get('count', 0)} rows")
                        for row in rows[:20]:
                            print(f"  {row}")

            case _:
                parser.print_help()
                return 1

    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to SHM at {args.base_url}", file=sys.stderr)
        print("Is the SHM server running?", file=sys.stderr)
        return 1
    except requests.exceptions.Timeout:
        print("Error: Request timed out", file=sys.stderr)
        return 1
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP {e.response.status_code} — {e.response.text}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
