"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.5.1"
__version_info__ = (5, 5, 1)
__version_name__ = "LLM异步化 · 事件循环修复"
__release_date__ = "2026-07-17"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• P0: 三层Embedding — Cloud API→sentence-transformers→TF-IDF降级
• P0: Dream自动应用 — 候选积压≥20自动触发apply+清理
• P0: 加速社区发现 — 轮询60s+候选数>10触发
• P0: LongMemEval基准 — 11项评测: precision/recall/时序/去重
• P1: 多信号检索 — 向量(0.5)+BM25(0.3)+实体匹配(0.2)融合
• P1: 时序推理 — 指数衰减加权，24h内数据权重翻倍
• P1: 实体链接 — LLM命名实体识别+跨节点实体关联
• P6: 热同步API Key — 运行时检测~/.hermes/.env变更自动切换
• P6: 多provider轮询 — 支持DEEPSEEK/OPENAI/ANTHROPIC/KIMI等
• ✅ 174 个单元测试通过 (2 个预知分数公式偏差)
• 服务端: FastAPI + Kuzu 本体图 + FAISS 384维向量
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
