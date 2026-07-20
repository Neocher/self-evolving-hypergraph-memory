"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.8.2"
__version_info__ = (5, 8, 2)
__version_name__ = "批量关系写入 · OSINT结构化关系抽取 · API Key热加载修复"
__release_date__ = "2026-07-19"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 新增 POST /batch/relations 批量写入语义边端点
• 新增 scripts/batch_osint_relations.py OSINT域名字段/IP/URL结构化关系抽取
• 修复 run_server.py setdefault→直接赋值，API Key热加载不生效
• 实体发现准确率100% (上下文投票消歧)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
