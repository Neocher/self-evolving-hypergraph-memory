"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.21.3"
__version_info__ = (5, 21, 3)
__version_name__ = "P0-Stability"
__release_date__ = "2026-08-09"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心变更 — P0 静默失败修复 (2026-08-09):
  • SSM gate learn 闭环: 写路径接线 learn() 正负信号 (outcome-gate_value
    方向) + reward 连续非负 + alpha 上界 clamp + fail-open 容差 ≤1e-9 —
    修复 warmup 后约半数正常写入被静默过滤 (数据丢失)
  • 5 个幽灵方法实现 (graphlite_store): create/get_visual_node,
    get_visual_nodes, get_or_create_session, link_session_member —
    修复视觉记忆/会话关联静默失败 + /memories/visual 500
  • toLower(e.content) CONTAINS 死代码: 4 处删除 + b64 中文限制契约
    文档化 (GraphLite lexer 不支持 UTF-8, 中文 L4 兜底依赖向量/BM25)
  • 测试: 386 passed 1 skipped (真实 GraphLite 集成测试 8 条)

v5.21.2 (2026-08-05) BM25-Harden:
  • BM25 空语料日志降噪 + bm25_build_timeout 双语义拆分
  • Embedding 升级 BAAI/bge-m3 (1024维) + FAISS 维度动态适配

📌 验证:
  • 全量测试 386 passed 1 skipped (21s)
  • g_mlp 正样本学习 0.6164→0.6470 (修复前反降 0.5603)
  • GraphLite 集成: visual roundtrip / session 幂等 / text_search 契约

⚙️ 部署: deploy.sh 一键部署 | systemd shm-server | Docker 就绪"""
