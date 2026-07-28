"""
MCP Server — SHM 记忆网关
=========================
通过 Model Context Protocol (stdio 传输) 暴露 SHM 核心能力。
供 Claude Desktop / Cursor / Windsurf 等 MCP 客户端使用。

运行方式:
    # stdio 模式（默认，用于 Claude Desktop 等）
    python -m gateway.mcp_server

    # 或通过 SSE 运行在 :8002
    python -m gateway.mcp_server --port 8002

Claude Desktop 配置:
    {
        "mcpServers": {
            "shm": {
                "command": "python3",
                "args": ["-m", "gateway.mcp_server"],
                "env": {}
            }
        }
    }
"""

from __future__ import annotations

import argparse
import logging
import sys

from mcp.server.fastmcp import FastMCP

from gateway.gateway_api import GatewayAPI

logger = logging.getLogger("gateway.mcp")


def _build_gateway() -> GatewayAPI:
    """初始化 SHM 服务并返回 GatewayAPI 实例。"""
    from api._routes import init_services
    from api.app import _init_services

    svc = _init_services()
    init_services(svc)
    logger.info("SHM services initialized for MCP gateway")
    return GatewayAPI(svc)


def register_tools(mcp: FastMCP, api: GatewayAPI) -> None:
    """向 FastMCP 注册所有 SHM 工具。"""

    @mcp.tool(
        name="shm_write",
        description="将原始文本写入 SHM Layer1 感觉缓冲区（环形缓冲区）。返回 record_id 和 buffer_usage。",
    )
    async def shm_write(
        content: str,
        source: str = "api",
        namespace: str | None = None,
    ) -> str:
        """写入感觉缓冲区。"""
        result = await api.write_sensory(content=content, source=source, namespace=namespace)
        return (
            f"Written to sensory buffer: record_id={result.record_id}, "
            f"buffer_usage={result.buffer_usage}"
        )

    @mcp.tool(
        name="shm_store_episode",
        description="直接创建 Layer2 情节节点（记忆），含 τ 衰减计算、SSM 门控过滤和命名空间链接。",
    )
    async def shm_store_episode(
        content: str,
        source: str = "user",
        namespace: str | None = None,
        metadata: str | None = None,
        force_promote: bool = False,
        visibility: str = "private",
    ) -> str:
        """创建情节记忆节点。"""
        import json
        meta = json.loads(metadata) if metadata else None
        result = await api.store_episode(
            content=content,
            source=source,
            namespace=namespace,
            metadata=meta,
            force_promote=force_promote,
            visibility=visibility,
        )
        return (
            f"Episode created: id={result.episode_id}, status={result.status}, "
            f"tau={result.tau_initial:.3f}"
        )

    @mcp.tool(
        name="shm_retrieve",
        description="三级融合检索（语义向量 + 关键词 + 图），含 Cypher 兜底和去重。返回格式化为文本的结果列表。",
    )
    async def shm_retrieve(
        query: str,
        top_k: int = 20,
        namespace: str | None = None,
        include_shared: bool = True,
    ) -> str:
        """融合检索记忆。"""
        result = await api.retrieve(
            query=query,
            top_k=top_k,
            namespace=namespace,
            include_shared=include_shared,
        )
        if not result.results:
            return f"No results found for query: {query!r} (latency={result.latency_ms:.0f}ms)"

        lines = [
            f"Retrieved {result.total_found} results for {query!r} "
            f"(strategy={result.strategy_used}, degraded={result.degraded}, "
            f"latency={result.latency_ms:.0f}ms):"
        ]
        for i, r in enumerate(result.results[:top_k], 1):
            content = r.content[:200].replace("\n", " ")
            lines.append(f"  {i:2d}. [{r.score:.3f}] {content}")
        return "\n".join(lines)

    @mcp.tool(
        name="shm_search_vector",
        description="纯向量检索（直通 FAISS），不经过三级融合管道。适合精确语义匹配。",
    )
    async def shm_search_vector(
        query: str,
        top_k: int = 10,
    ) -> str:
        """纯向量检索。"""
        result = await api.search_vector(query=query, limit=top_k)
        if not result.results:
            return f"No vector results for query: {query!r} (latency={result.latency_ms:.0f}ms)"

        lines = [
            f"Vector search: {result.total_found} results for {query!r} "
            f"(degraded={result.degraded}, latency={result.latency_ms:.0f}ms):"
        ]
        for i, r in enumerate(result.results[:top_k], 1):
            content = r.content[:150].replace("\n", " ")
            lines.append(f"  {i:2d}. [{r.score:.4f}] {content}")
        return "\n".join(lines)

    @mcp.tool(
        name="shm_health",
        description="SHM 深度健康检查 — 返回所有核心组件的运行状态：Kuzu 图库、FAISS 索引、溯源链、梦境调度器。",
    )
    async def shm_health() -> str:
        """深度健康检查。"""
        h = await api.health()
        return (
            f"Status: {h.status}\n"
            f"Graph connected: {h.graph_connected}\n"
            f"FAISS loaded: {h.faiss_loaded}\n"
            f"Dream scheduler: {h.dream_scheduler_running}\n"
            f"Node count: {h.stats.get('node_count', 'N/A')}\n"
            f"Hyperedge count: {h.stats.get('hyperedge_count', 'N/A')}\n"
            f"Chain verified: {h.stats.get('chain_verified', 'N/A')}\n"
            f"Uptime: {h.stats.get('uptime_seconds', 0):.0f}s\n"
            f"FAISS index size: {h.stats.get('faiss_index_size', 'N/A')}\n"
            f"Last dream: {h.stats.get('last_dream_time', 'N/A')}\n"
            f"Dream runs: {h.stats.get('dream_run_count', 'N/A')}"
        )

    @mcp.tool(
        name="shm_dream_trigger",
        description="显式触发 SHM 梦境整合管道 — 对记忆进行社区发现、压缩、剪枝、冲突消解。",
    )
    async def shm_dream_trigger(mode: str = "auto") -> str:
        """触发梦境管道。"""
        result = await api.trigger_dream(mode=mode)
        return f"Dream trigger: accepted={result.accepted}, message={result.message}"


def main() -> None:
    """MCP 服务器入口。"""
    parser = argparse.ArgumentParser(description="SHM MCP Gateway Server")
    parser.add_argument(
        "--port", type=int, default=0,
        help="HTTP SSE port (omit for stdio mode)",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    api = _build_gateway()

    mcp = FastMCP(
        name="SHM Memory Gateway",
        instructions="SHM (Self-evolving Hypergraph Memory) MCP server — "
                     "write, retrieve, search, and manage memories.",
        debug=args.debug,
    )

    register_tools(mcp, api)

    if args.port:
        logger.info("Starting SHM MCP Gateway on SSE :%d", args.port)
        mcp.run(transport="sse", mount_path="/")
    else:
        logger.info("Starting SHM MCP Gateway (stdio mode)...")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
