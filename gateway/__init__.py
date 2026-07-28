"""
SHM 多协议网关
=============
为 SHM (Self-evolving Hypergraph Memory) 提供统一的协议无关接口。

子模块:
    gateway_api    — 核心 GatewayAPI，直接封装业务逻辑
    mcp_server     — MCP (Model Context Protocol) stdio 服务器
    a2a_server     — A2A (Agent-to-Agent) HTTP JSON RPC 服务器
    acp_adapter    — ACP (Agent Communication Protocol) 动作适配器
    cli            — 开发者命令行工具 (HTTP → SHM)
"""

# GatewayAPI 按需导入（避免 CLI 等轻量入口触发 numpy/faiss 等重依赖）
#   from gateway import GatewayAPI
def __getattr__(name):
    if name == "GatewayAPI":
        from gateway.gateway_api import GatewayAPI as _cls
        return _cls
    if name == "SHMACPAdapter":
        from gateway.acp_adapter import SHMACPAdapter as _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["GatewayAPI", "SHMACPAdapter"]
