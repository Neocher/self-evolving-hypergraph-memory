"""
A2A Server — SHM Agent-to-Agent HTTP JSON RPC 网关
===================================================
通过 HTTP JSON RPC 风格暴露 SHM 核心记忆操作，供其他 AI Agent 直接调用。

运行方式:
    python -m gateway.a2a_server          # 默认 :8001
    python -m gateway.a2a_server --port 8001 --host 0.0.0.0

端点:
    POST /memory/write          写入感觉缓冲区
    POST /memory/store_episode  创建情节记忆
    POST /memory/retrieve       三级融合检索
    POST /memory/search         纯向量检索
    GET  /memory/health         健康检查
    POST /memory/dream          触发梦境管道
    GET  /.well-known/agent-card.json  AgentCard 元数据
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from gateway.gateway_api import GatewayAPI

logger = logging.getLogger("gateway.a2a")

# ── 认证配置 ──
A2A_API_KEY = os.environ.get("A2A_API_KEY", "")
if not A2A_API_KEY:
    logger.warning(
        "A2A_API_KEY not set — authentication disabled. "
        "Set A2A_API_KEY environment variable to enable Bearer token auth."
    )
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

# ── Pydantic 请求／响应模型 ──────────────────────────────────────────


class WriteRequest(BaseModel):
    content: str = Field(..., description="写入的原始文本")
    source: str = Field("api", description="来源标识")
    namespace: Optional[str] = Field(None, description="命名空间")


class StoreEpisodeRequest(BaseModel):
    content: str = Field(..., description="情节文本")
    source: str = Field("user", description="来源标识")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据")
    namespace: Optional[str] = Field(None, description="命名空间")
    force_promote: bool = Field(False, description="强制提升为情节")
    visibility: str = Field("private", description="可见性")


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="检索查询")
    top_k: int = Field(20, description="返回结果数")
    namespace: Optional[str] = Field(None, description="命名空间过滤")
    include_shared: bool = Field(True, description="是否包含共享记忆")
    session_ts: Optional[float] = Field(
        None, description="session 时间锚（相对时间词解析基准，None 回落墙钟）",
    )


class SearchVectorRequest(BaseModel):
    query: str = Field(..., description="查询文本")
    top_k: int = Field(10, description="返回结果数")


class DreamRequest(BaseModel):
    mode: str = Field("auto", description="梦境模式 (auto / force)")


# ── AgentCard 模型 ──────────────────────────────────────────────────


class AgentCapability(BaseModel):
    name: str
    description: str


class AgentCard(BaseModel):
    """A2A Agent Card — 遵循 Agent-to-Agent 协议规范。"""
    name: str = "SHM Memory Agent"
    description: str = (
        "Self-evolving Hypergraph Memory — 让 AI Agent 拥有类人记忆能力："
        "写入、检索、演化的全周期记忆管理。"
    )
    url: str = "http://localhost:8001"
    version: str = "5.10.0"
    capabilities: List[AgentCapability] = [
        AgentCapability(name="memory.write", description="将原始文本写入 Layer1 感觉缓冲区"),
        AgentCapability(name="memory.store_episode", description="直接创建 Layer2 情节记忆节点"),
        AgentCapability(name="memory.retrieve", description="三级融合检索（语义+关键词+图）"),
        AgentCapability(name="memory.search", description="纯向量检索（FAISS）"),
        AgentCapability(name="memory.health", description="深度健康检查"),
        AgentCapability(name="memory.dream", description="显式触发梦境整合管道"),
    ]


# ── 服务构建 ─────────────────────────────────────────────────────────


_gateway_instance: Optional[GatewayAPI] = None


def _build_gateway() -> GatewayAPI:
    """初始化 SHM 服务并返回 GatewayAPI 实例（单例）。"""
    global _gateway_instance
    if _gateway_instance is not None:
        return _gateway_instance
    from api._routes import init_services
    from api.app import _init_services

    svc = _init_services()
    init_services(svc)
    _gateway_instance = GatewayAPI(svc)
    logger.info("SHM services initialized for A2A gateway")
    return _gateway_instance


# ── 认证中间件 ──────────────────────────────────────────────────────


async def _verify_auth(request: Request, call_next):
    if A2A_API_KEY:
        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        elif auth_header.startswith("ApiKey "):
            token = auth_header[7:]
        else:
            token = request.headers.get("X-API-Key", "")
        if token != A2A_API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Set A2A_API_KEY env var or provide Authorization: Bearer <key> header."}
            )
    return await call_next(request)


async def _body_size_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body too large (max {MAX_UPLOAD_SIZE // 1024 // 1024} MB)"}
        )
    return await call_next(request)


# ── 路由注册 ─────────────────────────────────────────────────────────


def register_routes(app: FastAPI, api: GatewayAPI) -> None:
    """向 FastAPI 应用注册所有 A2A 端点。"""

    @app.post("/memory/write", summary="写入感觉缓冲区")
    async def memory_write(req: WriteRequest) -> dict:
        """将原始文本写入 Layer1 环形缓冲区。返回 record_id 和 buffer_usage。"""
        result = await api.write_sensory(
            content=req.content,
            source=req.source,
            namespace=req.namespace,
        )
        return {
            "record_id": result.record_id,
            "buffer_usage": result.buffer_usage,
        }

    @app.post("/memory/store_episode", summary="创建情节记忆")
    async def memory_store_episode(req: StoreEpisodeRequest) -> dict:
        """直接创建 Layer2 情节节点，含 τ 衰减、SSM 门控和命名空间链接。"""
        result = await api.store_episode(
            content=req.content,
            source=req.source,
            metadata=req.metadata,
            namespace=req.namespace,
            force_promote=req.force_promote,
            visibility=req.visibility,
        )
        return {
            "episode_id": result.episode_id,
            "status": result.status,
            "tau_initial": result.tau_initial,
            "created_at": result.created_at,
            "source": result.source,
        }

    @app.post("/memory/retrieve", summary="三级融合检索")
    async def memory_retrieve(req: RetrieveRequest) -> dict:
        """粗到精三级融合检索，含 Cypher 兜底和去重。"""
        result = await api.retrieve(
            query=req.query,
            top_k=req.top_k,
            namespace=req.namespace,
            include_shared=req.include_shared,
            session_ts=req.session_ts,
        )
        return {
            "query": result.query,
            "strategy_used": result.strategy_used,
            "results": [
                {
                    "node_id": r.node_id,
                    "content": r.content,
                    "score": r.score,
                    "retrieval_level": r.retrieval_level,
                }
                for r in result.results
            ],
            "total_found": result.total_found,
            "latency_ms": result.latency_ms,
            "degraded": result.degraded,
        }

    @app.post("/memory/search", summary="纯向量检索")
    async def memory_search(req: SearchVectorRequest) -> dict:
        """直通 FAISS 的纯向量检索。"""
        result = await api.search_vector(query=req.query, limit=req.top_k)
        return {
            "query": result.query,
            "results": [
                {
                    "node_id": r.node_id,
                    "content": r.content,
                    "score": r.score,
                    "faiss_id": r.faiss_id,
                }
                for r in result.results
            ],
            "total_found": result.total_found,
            "latency_ms": result.latency_ms,
            "degraded": result.degraded,
        }

    @app.get("/memory/health", summary="深度健康检查")
    async def memory_health() -> dict:
        """返回所有核心组件的运行状态。"""
        h = await api.health()
        return {
            "status": h.status,
            "graph_connected": h.graph_connected,
            "faiss_loaded": h.faiss_loaded,
            "dream_scheduler_running": h.dream_scheduler_running,
            "stats": h.stats,
            "timestamp": h.timestamp,
        }

    @app.post("/memory/dream", summary="触发梦境管道")
    async def memory_dream(req: DreamRequest = DreamRequest()) -> dict:
        """显式触发梦境整合管道 — 社区发现、压缩、剪枝、冲突消解。"""
        result = await api.trigger_dream(mode=req.mode)
        return {
            "accepted": result.accepted,
            "message": result.message,
        }

    @app.get("/.well-known/agent-card.json", summary="AgentCard 元数据")
    async def agent_card() -> AgentCard:
        """返回 A2A AgentCard，描述本 Agent 的能力。"""
        return AgentCard()


# ── 入口 ────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期 — 启动时预热初始化。"""
    _build_gateway()
    yield


def create_app() -> FastAPI:
    """构建并返回配置好的 FastAPI 应用（供测试或 uvicorn 使用）。"""
    app = FastAPI(
        title="SHM A2A Gateway",
        description="Agent-to-Agent HTTP JSON RPC gateway for Self-evolving Hypergraph Memory",
        version="5.10.0",
        lifespan=lifespan,
    )

    if A2A_API_KEY:
        app.middleware("http")(_verify_auth)
    app.middleware("http")(_body_size_limit)

    register_routes(app, _build_gateway())
    return app


def main() -> None:
    """A2A 服务器入口。"""
    parser = argparse.ArgumentParser(description="SHM A2A Gateway Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8001, help="监听端口")
    parser.add_argument("--debug", action="store_true", default=False, help="启用调试日志")
    parser.add_argument("--api-key", type=str, default="", help="API Key（覆盖 A2A_API_KEY 环境变量）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    global A2A_API_KEY
    if args.api_key:
        A2A_API_KEY = args.api_key

    import uvicorn
    app = create_app()
    logger.info("Starting SHM A2A Gateway on %s:%d", args.host, args.port)
    if A2A_API_KEY:
        logger.info("A2A API key authentication enabled")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
