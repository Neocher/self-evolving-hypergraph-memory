"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.19.2"
__version_info__ = (5, 19, 2)
__version_name__ = "fix: structlog日志丢失 + FAISS rebuild GraphLite格式兼容 + uptime基准修复"
__release_date__ = "2026-07-31"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心变更 — 遗留问题修复轮 (2026-07-31):
  • observability/logger.py: structlog 日志全部丢失修复
    (root logger 无 StreamHandler，INFO/WARNING 不可见 → 绑定 handler + 级别)
  • api/routes/system.py: rebuild_index 兼容 GraphLite 深层嵌套返回格式
    (嵌套 Node properties → _flatten_row，此前 FAISS 永远 0)
  • observability/health.py: uptime 恒≈0 修复
    (HealthChecker 每次请求新建 → 改用模块导入时刻作进程级基准)

📌 前置 (v5.19.x):
  • v5.19.1: cdlib 社区检测兼容 + Louvain next_comm 修复
  • v5.19.0: OWL 导出 + 本体匹配 + LLM 关系抽取

⚙️ 部署: deploy.sh 一键部署 | Docker 就绪"""
