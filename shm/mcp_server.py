"""
SHM MCP Server — Model Context Protocol 接入
============================================
让 Claude Desktop / Cursor / 任何 MCP 客户端直接连接 SHM 作为记忆后端。

用法:
  python -m shm.mcp_server                    # 通过 stdio 运行（MCP 标准模式）
  python -m shm.mcp_server --http :8001        # 通过 HTTP/SSE 运行（调试模式）

协议: MCP (Model Context Protocol) — JSON-RPC over stdio
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
import urllib.error
from typing import Any

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("shm-mcp")

SHM_BASE_URL = os.environ.get("SHM_BASE_URL", "http://127.0.0.1:8000")


# ─── MCP JSON-RPC 工具函数 ──────────────────────────────────

def _json_rpc(id: int, method: str, params: dict[str, Any] | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}}


def _result(id: int, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _error(id: int, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# ─── SHM HTTP 客户端 ────────────────────────────────────────

def _shm_post(path: str, data: dict) -> dict:
    """调用 SHM REST API"""
    url = f"{SHM_BASE_URL}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def _shm_get(path: str) -> dict:
    """调用 SHM REST API (GET)"""
    url = f"{SHM_BASE_URL}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


# ─── MCP 工具实现 ────────────────────────────────────────────

TOOLS = [
    {
        "name": "shm_add",
        "description": "向 SHM 记忆系统添加一条记忆",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "记忆内容文本"},
                "source": {"type": "string", "description": "来源标识（如 user, assistant, system）", "default": "user"},
                "namespace": {"type": "string", "description": "命名空间（可选）", "default": ""},
            },
            "required": ["content"],
        },
    },
    {
        "name": "shm_search",
        "description": "搜索 SHM 记忆系统，返回最相关的记忆",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询文本"},
                "top_k": {"type": "integer", "description": "返回结果数", "default": 5},
                "namespace": {"type": "string", "description": "命名空间过滤（可选）", "default": ""},
            },
            "required": ["query"],
        },
    },
    {
        "name": "shm_stats",
        "description": "查询 SHM 记忆系统状态统计",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "shm_health",
        "description": "检查 SHM 服务健康状态",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def _handle_call(tool_name: str, args: dict) -> dict:
    """执行 MCP tool 调用"""
    if tool_name == "shm_add":
        result = _shm_post("/memories/episodes", {
            "content": args["content"],
            "source": args.get("source", "user"),
            "namespace": args.get("namespace", ""),
        })
        if "error" in result:
            return {"content": [{"type": "text", "text": f"❌ {result['error']}"}]}
        return {"content": [{"type": "text", "text":
            f"✅ 记忆已添加 (id={result.get('episode_id','?')}, τ={result.get('tau_initial','?'):.3f})"}]}

    if tool_name == "shm_search":
        payload = {
            "query": args["query"],
            "top_k": args.get("top_k", 5),
        }
        if args.get("namespace"):
            payload["namespace"] = args["namespace"]
        result = _shm_post("/memories/retrieve", payload)
        if "error" in result:
            return {"content": [{"type": "text", "text": f"❌ {result['error']}"}]}
        results = result.get("results", [])
        if not results:
            return {"content": [{"type": "text", "text": "没有找到相关记忆"}]}
        lines = [f"📝 找到 {len(results)} 条相关记忆：\n"]
        for i, r in enumerate(results[:10], 1):
            content = r.get("content", "")[:200]
            score = r.get("score", 0)
            eid = r.get("episode_id", "?")[:8]
            lines.append(f"  [{i}] (score={score:.3f}, id={eid}) {content}")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    if tool_name == "shm_stats":
        result = _shm_get("/health")
        if "error" in result:
            result = _shm_get("/health")
        stats = result.get("stats", result)
        return {"content": [{"type": "text", "text":
            f"📊 SHM 状态\n"
            f"  版本: {stats.get('version','?')}\n"
            f"  节点: {stats.get('node_count',0)}\n"
            f"  超边: {stats.get('hyperedge_count',0)}\n"
            f"  FAISS: {stats.get('faiss_index_size',0)}\n"
            f"  断路器: {stats.get('circuit_breaker',{}).get('state','unknown')}\n"
            f"  溯源链: {'✅' if stats.get('chain_verified') else '❌'}",
        }]}

    if tool_name == "shm_health":
        result = _shm_get("/health")
        status = result.get("status", "error")
        return {"content": [{"type": "text", "text":
            f"{'✅' if status=='ok' else '❌'} SHM 状态: {status}"}]}

    return {"content": [{"type": "text", "text": f"未知工具: {tool_name}"}]}


# ─── MCP 协议主循环（stdio JSON-RPC） ───────────────────────

def _send(msg: dict) -> None:
    """发送 JSON-RPC 消息到 stdout"""
    line = json.dumps(msg, ensure_ascii=False)
    sys.stdout.write(f"Content-Length: {len(line.encode('utf-8'))}\r\n\r\n{line}")
    sys.stdout.flush()


def _read_message() -> dict | None:
    """从 stdin 读取 JSON-RPC 消息"""
    headers = {}
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    content_length = int(headers.get("content-length", 0))
    if content_length == 0:
        return None
    body = sys.stdin.read(content_length)
    return json.loads(body)


def run_stdio() -> None:
    """通过 stdio 运行 MCP 服务器（Claude Desktop 标准模式）"""
    request_id = 0
    initialized = False

    while True:
        msg = _read_message()
        if msg is None:
            break

        rid = msg.get("id", 0)
        method = msg.get("method", "")
        params = msg.get("params", {})

        # ---- 初始化 ----
        if method == "initialize":
            initialized = True
            _send(_result(rid, {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "shm-mcp", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            }))
            continue

        if method == "notifications/initialized":
            continue

        if method == "notifications/cancelled":
            continue

        # ---- tools/list ----
        if method == "tools/list":
            _send(_result(rid, TOOLS))
            continue

        # ---- tools/call ----
        if method == "tools/call":
            result = _handle_call(params.get("name", ""), params.get("arguments", {}))
            _send(_result(rid, result))
            continue

        # ---- 未知方法 ----
        _send(_error(rid, -32601, f"Method not found: {method}"))


def run_http(host: str = "0.0.0.0", port: int = 8001) -> None:
    """通过 HTTP/SSE 运行 MCP 服务器（调试模式）"""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class MCPHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "server": "shm-mcp"}).encode())
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if self.path == "/mcp":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode()
                msg = json.loads(body)
                rid = msg.get("id", 0)
                method = msg.get("method", "")

                if method == "tools/list":
                    resp = _result(rid, TOOLS)
                elif method == "tools/call":
                    params = msg.get("params", {})
                    result = _handle_call(params.get("name", ""), params.get("arguments", {}))
                    resp = _result(rid, result)
                else:
                    resp = _error(rid, -32601, f"Method not found: {method}")

                resp_body = json.dumps(resp, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, fmt, *args):
            pass  # 静默

    server = HTTPServer((host, port), MCPHandler)
    logger.warning("SHM MCP HTTP server listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


# ─── main ────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--http" in sys.argv:
        idx = sys.argv.index("--http")
        addr = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ":8001"
        host, _, port_str = addr.partition(":")
        port = int(port_str) if port_str else 8001
        run_http(host or "0.0.0.0", port)
    else:
        run_stdio()
