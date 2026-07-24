"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.9.0"
__version_info__ = (5, 9, 0)
__version_name__ = "架构升级 · 路由模块化 · 配置统一 · LLM fallback链 · 22核心测试 · Docker部署"
__release_date__ = "2026-07-24"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P0 — 架构改进:
  • 路由模块化: api/routes/ 8域文件 + 向后兼容导入
  • 配置统一: LLMConfig + SHMClientConfig 三明治函数链
  • LLM fallback链: 4端点 (DeepSeek→OpenAI→Moonshot→OpenRouter)
P1 — 质量 + 可靠性:
  • 核心单元测试: 22/22 ✅ (τ衰减·Hebbian·SSM门控)
  • 嵌入模型可配置: config/settings.py 控制
  • 梦境状态持久化: Kuzu SystemNode 存取
P2 — 生产部署:
  • Dockerfile + docker-compose.yml + systemd
  • 硬编码常量消除: SHMClient/MCP自动从config读取
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
