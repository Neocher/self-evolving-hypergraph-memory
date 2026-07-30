"""
SHM MCP Server v2 — Model Context Protocol 完整实现
=====================================================
基于 MCP Python SDK (FastMCP)，完整的工具注册与自动发现。
比 v1 (shm/mcp_server.py) 的改进：
  - 使用官方 MCP SDK FastMCP（正确 JSON-RPC、取消/通知/错误）
  - 工具从 SHM REST API 自动映射（6 个工具，覆盖全功能）
  - Streamable HTTP + stdio 双传输
  - 多Agent支持：agent_id → SHM namespace

用法:
  python -m shm.mcp_server_v2                 # stdio 模式（Claude Desktop）
  python -m shm.mcp_server_v2 --http :8222     # HTTP 模式（远程/调试）

Claude Desktop 配置 (~/Library/Application Support/Claude/claude_desktop_config.json):
  {\"mcpServers\": {\"shm\": {\"command\": \"python3\", \"args\": [\"-m\", \"shm.mcp_server_v2\"]}}}

依赖:
  pip install mcp>=1.0.0 httpx uvicorn
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("shm-mcp-v2")

from config.settings import get_settings

SHM_BASE_URL = get_settings().shm_client.base_url

# ─── SHM HTTP Client (async with httpx) ──────────────────────

_httpx_client = None

async def _get_client():
    global _httpx_client
    if _httpx_client is None:
        import httpx
        _httpx_client = httpx.AsyncClient(timeout=15.0)
    return _httpx_client


async def _shm_post(path: str, data: dict) -> dict:
    c = await _get_client()
    try:
        r = await c.post(f"{SHM_BASE_URL}{path}", json=data)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


async def _shm_get(path: str) -> dict:
    c = await _get_client()
    try:
        r = await c.get(f"{SHM_BASE_URL}{path}")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─── MCP Server Setup (FastMCP) ──────────────────────────────

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "shm-mcp-v2",
    instructions="SHM 自演化超图记忆系统 — MCP 接入。提供记忆写入、语义搜索、状态查询、梦境触发、本体查看、溯源审计6大工具。多Agent支持：通过 source/namespace 字段区分不同Agent身份。",
)


# ─── Tools ────────────────────────────────────────────────────

@mcp.tool()
async def shm_add_memory(
    content: str,
    source: str = "mcp-client",
    namespace: str = "",
    visibility: str = "private",
) -> str:
    """向 SHM 写入一条记忆。自动记录来源Agent、命名空间和可见性。

    Args:
        content: 记忆内容文本（任意长度）
        source: 来源标识（agent名称，如 hermes/codex/claude/cursor）
        namespace: 命名空间（用于数据隔离）。空=全局可见
        visibility: 可见性：private(仅当前namespace)/shared(所有Agent可检索)
    """
    result = await _shm_post("/memories/episodes", {
        "content": content,
        "source": source,
        "namespace": namespace,
        "visibility": visibility,
    })
    if "error" in result:
        return f"❌ 写入失败: {result['error']}"
    eid = result.get("episode_id", "?")[:12]
    tau = result.get("tau_initial", 1.0)
    return f"✅ 记忆已写入 (id={eid}, τ={tau:.3f})"


@mcp.tool()
async def shm_search_memory(
    query: str,
    top_k: int = 10,
    namespace: str = "",
    include_shared: bool = True,
) -> str:
    """搜索 SHM 记忆库。三层检索（超图→向量→关键词降级），支持跨Agent搜索。

    Args:
        query: 搜索查询文本。支持自然语言
        top_k: 返回结果数上限 (1-50)
        namespace: 限定检索命名空间。空=搜全部+shared
        include_shared: 是否包含其他Agent共享的记忆
    """
    payload: dict = {"query": query, "top_k": min(top_k, 50)}
    if namespace:
        payload["namespace"] = namespace
    if not include_shared:
        payload["include_shared"] = False

    result = await _shm_post("/memories/retrieve", payload)
    if "error" in result:
        return f"❌ 搜索失败: {result['error']}"

    results = result.get("results", [])
    if not results:
        return "📭 未找到相关记忆"

    lines = [f"📝 找到 {len(results)} 条相关记忆："]
    for i, r in enumerate(results[:min(top_k, 10)], 1):
        content = r.get("content", "")[:150]
        score = r.get("score", 0)
        nid = r.get("node_id", "?")[:12]
        level = r.get("retrieval_level", "?")
        source = r.get("source", "?")
        lines.append(f"\n  [{i}] score={score:.3f} level={level}")
        lines.append(f"      source={source} id={nid}")
        lines.append(f"      {content}")
    return "\n".join(lines)


@mcp.tool()
async def shm_get_stats() -> str:
    """查询 SHM 运行状态：版本、节点数、超边、FAISS索引、断路器、溯源完整性。"""
    result = await _shm_get("/health")
    if "error" in result:
        return f"❌ {result['error']}"
    s = result.get("stats", {})
    cb = s.get("circuit_breaker", {})
    return (
        f"📊 SHM 运行状态\n"
        f"━━━━━━━━━━━━━━━\n"
        f"  版本: v{s.get('version','?')} — {s.get('version_name','?')}\n"
        f"  状态: {result.get('status','?')}\n"
        f"  节点: {s.get('node_count',0)} | 超边: {s.get('hyperedge_count',0)}\n"
        f"  FAISS: {s.get('faiss_index_size',0)} 条\n"
        f"  梦境: {s.get('dream_run_count',0)} 次\n"
        f"  断路器: {cb.get('state','?')} (成功率 {cb.get('success_rate',0)}%)\n"
        f"  溯源: {'✅ 完整' if s.get('chain_verified') else '❌ 异常'}"
    )


@mcp.tool()
async def shm_trigger_dream() -> str:
    """显式触发 SHM 梦境调度器：社区检测、知识融合、冲突解决、剪枝。"""
    result = await _shm_post("/memories/dream/trigger", {})
    if "error" in result:
        return f"❌ {result['error']}"
    accepted = result.get("accepted", False)
    msg = result.get("message", "")
    return f"{'✅' if accepted else '⚠️'} {msg}"


@mcp.tool()
async def shm_list_ontology() -> str:
    """列出 SHM 已注册的实体类型和边类型（本体schema）。"""
    entities = await _shm_get("/ontology/types")
    edges = await _shm_get("/ontology/edges")
    lines = ["📐 SHM 本体类型\n━━━━━━━━━━━━━━━"]

    if "error" not in entities:
        ets = entities.get("entity_types", [])
        lines.append(f"\n  实体类型 ({len(ets)}):")
        for et in ets[:12]:
            lines.append(f"    · {et.get('name','?')} — {et.get('description','')[:60]}")

    if "error" not in edges:
        edgs = edges.get("edge_types", [])
        lines.append(f"\n  边类型 ({len(edgs)}):")
        for ed in edgs[:12]:
            src = ", ".join(ed.get("source_types", [])[:2])
            tgt = ", ".join(ed.get("target_types", [])[:2])
            lines.append(f"    · {ed.get('name','?')} ({src}→{tgt})")
    return "\n".join(lines)


@mcp.tool()
async def shm_audit_node(node_id: str) -> str:
    """查询记忆节点的完整溯源链（BLAKE3哈希验证的不可篡改审计记录）。

    Args:
        node_id: 记忆节点ID（从 shm_search_memory 结果中获取）
    """
    result = await _shm_get(f"/memories/audit/{node_id}")
    if "error" in result:
        return f"❌ {result['error']}"

    ops = result.get("operations", [])
    verified = result.get("chain_verified", False)
    blocks = result.get("total_blocks", 0)
    lines = [
        f"🔍 溯源链: {node_id[:12]}",
        f"━━━━━━━━━━━━━━━",
        f"  BLAKE3验证: {'✅ 完整' if verified else '❌ 异常'} | 区块数: {blocks}",
        f"  操作记录 ({len(ops)}):",
    ]
    for op in ops[:15]:
        lines.append(f"    [{op.get('op_type','?')}] {op.get('reason','?')}")
        if op.get("old_value"):
            lines.append(f"      旧值→新值: {op['old_value'][:40]} → {op.get('new_value','')[:40]}")
    return "\n".join(lines)


# ─── Entry Points ─────────────────────────────────────────────

def run_stdio():
    """通过 stdio 运行 MCP 服务器（Claude Desktop / Cursor 标准模式）。"""
    mcp.run(transport="stdio")


def run_http(host: str = "127.0.0.1", port: int = 8222):
    """通过 Streamable HTTP 运行 MCP 服务器（远程/调试）。"""
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.log_level = "warning"
    logger.warning("SHM MCP v2 HTTP server on http://%s:%s", host, port)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    if "--http" in sys.argv:
        idx = sys.argv.index("--http")
        addr = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ":8222"
        host, _, port_str = addr.partition(":")
        port = int(port_str) if port_str else 8222
        import uvicorn
        logger.warning("SHM MCP v2 HTTP server on http://%s:%s", host, port)
        run_http(host or "127.0.0.1", port)
    else:
        run_stdio()
