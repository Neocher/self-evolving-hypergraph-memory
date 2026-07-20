"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.6.0"
__version_info__ = (5, 6, 0)
__version_name__ = "命名空间图隔离 · 图能力升级"
__release_date__ = "2026-07-19"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• P0: 命名空间图隔离 — SessionNode + SESSION_MEMBER 实现图隔离
• P0: 写入/检索/删除 — sensory/episodes 支持 namespace 参数
• P0: DELETE /memories/namespace/{name} — 批量清理模拟数据
• P1: Kuzu DETACH DELETE — 自动处理所有关联边
• P1: MERGE ON CREATE — 修复 Kuzu 主键冲突
• 新功能: MiroFish 适配器 — 替代 Zep Cloud，对接本地 SHM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
