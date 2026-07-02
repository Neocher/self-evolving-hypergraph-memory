"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "4.2.0"
__version_info__ = (4, 2, 0)
__version_name__ = "稳定加固版 · OOM防护 + 优雅关闭 + 心跳优化"
__release_date__ = "2026-05-11"

VERSION_SUMMARY = f"""
SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Phase 0-3: 安全底线 + 功能修复 + 质量加固 + 性能提升
• C1: FAISS 增量重建 — 梦境后自动清理被剪枝向量
• C3: 梦境并行化 — Leiden 聚类拆分量并行
• ✅ 67 个单元测试全部通过（τ衰减/断路器/Hebbian/SSM/Kuzu/FAISS）
• 服务端: FastAPI + Kuzu + FAISS + SSM门控 + BLAKE3溯源
• 插件端: Hermes MemoryProvider HTTP桥接
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
