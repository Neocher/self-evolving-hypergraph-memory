"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.7.0"
__version_info__ = (5, 7, 0)
__version_name__ = "Ontology v2 动态类型系统 · 本体能力升级"
__release_date__ = "2026-07-19"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• P0: Ontology v2 动态类型系统 — API注册实体/边类型，无需重启
• P0: 属性类型系统 — 8种类型(string/int/float/boolean/date/string[]/text_embedding/entity_ref)
• P0: 边约束 — 源/目标实体类型白名单，对称边支持
• P0: 类型继承 — Person ← Agent 子类型体系
• P0: 写时验证 — 集成到 episodes 路由，属性类型+必填+范围校验
• P1: REST API — 9个 CRUD 端点管理本体
• P1: 基线预载 — 11种实体类型 + 7种边类型（兼容旧版 ENTITY_TYPE_MAP）
• 对标 Zep: 动态类型注册 + 属性系统 + 边约束
• 超越 Zep: 类型继承 + Kuzu 持久化 + 写时+读时验证
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
