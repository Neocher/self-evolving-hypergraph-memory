"""
ACP Adapter — 为 ACP Bridge 添加 SHM 记忆操作
=============================================
将 SHM 的记忆能力注册为 ACP 动作（action），
使 Hermes → OpenCode / Codex / Claude 等 Agent
能通过 ACP 协议直接调用 SHM 的写入、检索、健康检查功能。

用法:
    import acp_bridge
    from gateway.gateway_api import GatewayAPI
    from gateway.acp_adapter import SHMACPAdapter

    adapter = SHMACPAdapter(acp_bridge, gateway_api)

之后即可通过:
    POST /action/shm:write   {"params": {"content": "..."}}
    POST /action/shm:retrieve  {"params": {"query": "..."}}
    POST /action/shm:health    {"params": {}}
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from gateway.gateway_api import GatewayAPI

logger = logging.getLogger("gateway.acp-adapter")


class SHMACPAdapter:
    """为 ACP Bridge 注册 SHM 记忆操作。

    通过 ``acp_bridge.register_action()`` 注册三个动作：
    ``shm:write`` / ``shm:retrieve`` / ``shm:health``。

    使用前确保 ``acp_bridge`` 模块已导入且 ``register_action`` 可用。
    """

    def __init__(self, bridge_module: Any, gateway: GatewayAPI) -> None:
        """初始化适配器并注册动作。

        Args:
            bridge_module: ``acp_bridge`` 模块（或任何有 ``register_action`` 的对象）。
            gateway: 已初始化的 ``GatewayAPI`` 实例。
        """
        self._bridge = bridge_module
        self._gateway = gateway
        self._register_all()

    def _register_all(self) -> None:
        """注册所有 SHM 动作到 ACP Bridge。"""
        register = self._bridge.register_action

        register("shm:write", self._handle_write)
        register("shm:retrieve", self._handle_retrieve)
        register("shm:health", self._handle_health)
        logger.info("Registered 3 SHM actions on ACP bridge")

    async def _handle_write(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 shm:write — 写入感觉缓冲区。"""
        content = params.get("content", "")
        if not content:
            return {"status": "error", "message": "Missing required param: content"}
        source = params.get("source", "acp")
        namespace = params.get("namespace")
        result = await self._gateway.write_sensory(
            content=content,
            source=source,
            namespace=namespace,
        )
        return {
            "status": "ok",
            "record_id": result.record_id,
            "buffer_usage": result.buffer_usage,
        }

    async def _handle_retrieve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 shm:retrieve — 三级融合检索。"""
        query = params.get("query")
        if not query:
            return {"status": "error", "message": "Missing required param: query"}
        top_k = params.get("top_k", 20)
        namespace = params.get("namespace")
        result = await self._gateway.retrieve(
            query=query,
            top_k=top_k,
            namespace=namespace,
        )
        return {
            "status": "ok",
            "query": result.query,
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

    async def _handle_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 shm:health — 深度健康检查。"""
        h = await self._gateway.health()
        return {
            "status": "ok",
            "health_status": h.status,
            "graph_connected": h.graph_connected,
            "faiss_loaded": h.faiss_loaded,
            "dream_scheduler_running": h.dream_scheduler_running,
            "stats": h.stats,
        }
