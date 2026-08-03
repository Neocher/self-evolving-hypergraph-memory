"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.20.1"
__version_info__ = (5, 20, 1)
__version_name__ = "OpenSource-Ready"
__release_date__ = "2026-08-03"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心变更 — 熔断器+重试落地 (2026-08-03):
  • CircuitBreaker 状态机: closed→open→half_open, 滑动窗口+
    threading.RLock+探针时间武装 (graphlite_store.py)
  • SDK 异常适配: _INFRA_EXCEPTIONS 匹配 SDK 自有 QueryError/
    GraphLiteConnectionError (与内置类无继承关系, 此前熔断永不触发)
  • with_retry 双模式: 同步/异步包装器, 读路径重试 2 次
  • 写路径中立: execute_cypher 不计数, 坏 GQL 不污染读窗口
  • 全局 CircuitBreakerOpen→503 handler, L1→L2 级联可触发
  • SE 检索失败类型契约修复: 异常返回 [] 而非 dict (消除 500 路径)

📌 验证:
  • 全量测试 334 passed (300 基线 + 34 新增)
  • 真实 SDK QueryError 计数/重试/跳闸/恢复/级联闭环
  • health 显示 circuit_breaker=CLOSED (原 not_configured)
  • 变异验证证明测试真实守护 P1-1 (改坏代码测试即失败)
  • v5.19.1: cdlib 社区检测兼容

⚙️ 部署: deploy.sh 一键部署 | Docker 就绪"""
