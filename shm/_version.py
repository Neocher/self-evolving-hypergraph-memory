"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.8.3"
__version_info__ = (5, 8, 3)
__version_name__ = "多Agent支持 · visibility共享策略 · source字段放宽"
__release_date__ = "2026-07-24"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• source字段从Enum放宽为str，支持任意agent标识（hermes/codex/claude/cursor）
• 新增 visibility 参数 (private/shared)，实体跨Agent共享
• 检索端点增加 include_shared 过滤
• EpisodeNode schema 增加 visibility STRING 列
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
