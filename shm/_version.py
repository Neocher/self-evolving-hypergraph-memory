"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.4.0"
__version_info__ = (5, 4, 0)
__version_name__ = "热同步API Key · 多provider自动切换"
__release_date__ = "2026-07-17"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• P6: 热同步API Key — 运行时检测~/.hermes/.env变更自动切换
• P6: 多provider轮询 — 支持DEEPSEEK/OPENAI/ANTHROPIC/KIMI等
• P6: reasoning_content回退 — 兼容thinking模型空content问题
• P0: 启动时共享Hermes配置源 — 与主智能体同源管理
• P5: 动态本体学习 (大写候选实体自动发现)
• Kuzu 修复: MERGE 只对 name 主键, SET 其余属性
• ✅ 85 个单元测试通过 (2 个预知分数公式偏差)
• 服务端: FastAPI + Kuzu 本体图 + FAISS 1120 向量
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
