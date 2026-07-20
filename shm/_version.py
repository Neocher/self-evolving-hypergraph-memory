"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.8.0"
__version_info__ = (5, 8, 0)
__version_name__ = "本体自发现 · 关系抽取 · 置信度累积 · 实体消歧"
__release_date__ = "2026-07-19"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• P0: 实体自动发现 — 扫描数据自动发现候选实体+类型，API热注册
• P0: 关系抽取细化 — 共现边→语义谓词边(FOUNDED/LEADS/ACQUIRED等)
• P0: 置信度累积 — evidence_count追踪同一事实的多源确认
• P0: 实体消歧 — 上下文消歧(Apple公司vs水果)+别名Alias链接
• P0: 指代消解 — 英文/中文代词自动替换("He"→"Elon Musk")
• P1: 证据边fallback — 语义关系优先，共现fallback
• P1: Kuzu On Merge — 动态创建实体节点+关系边
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
