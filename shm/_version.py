"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.3.0"
__version_info__ = (5, 3, 0)
__version_name__ = "本体增强版 · 实体扩军180 + CJK匹配 + OSINT类型"
__release_date__ = "2026-07-06"

VERSION_SUMMARY = f"""
SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• P0: 本体实体扩军 95→180 (公司/人物/中文术语/向量DB)
• P1: CJK-aware 实体匹配 (中文不再因 ASCII 边界漏检)
• P3: 拓扑路径增强 (共享实体间 RELATES_TO 检测)
• P4: OSINT 矛盾检测类型 (domain_info/ip_address/url_link)
• P5: 动态本体学习 (大写候选实体自动发现)
• Kuzu 修复: MERGE 只对 name 主键, SET 其余属性
• ✅ 85 个单元测试通过 (2 个预知分数公式偏差)
• 服务端: FastAPI + Kuzu 本体图 + FAISS 1112 向量
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
