"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.17.2"
__version_info__ = (5, 17, 2)
__version_name__ = "安全加固 — 7个P0修复 + 子进程隔离 + 密钥哈希存储 + 全局历史清理"
__release_date__ = "2026-07-30"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
研究差距闭环 (2026-07-28, 30+论文):
  参考: EvolveMem·Retain or Consolidate·MemTX·AdaMem·MemClaw·Language Models Need Sleep·TMA-NM·WorldDB·RoMem

P0 — 检索自演化 (SelfEvolvingRetrieval):
  • FailureLogger → DiagnosisEngine → EvolutionGuard
  • 7个可演化参数: 融合权重/BM25/top-K/策略
  • 自动回滚(>15%退化) + 停滞探索(6x无变化)

P1 — 预算感知门控 + 过度巩固防护 + 事务性记忆
P2 — 可学习遗忘 + 多Agent共享 + SSM梦境深度升级

🌐 多协议网关 (v5.14.1):
  • MCP  :8002 | A2A  :8001 | ACP  :8770 | HTTP :8000 | CLI 终端
  • GatewayAPI 570行 — 统一内部接口

📡 第一梯队 — 研究深度:
  • 时序相位旋转 (RoMem): S = α·τ(t) + (1-α)·(Φ+1)/2
  • 记忆投毒防御 (TMA-NM): 5条规则 + 隔离区 (FAISS/梦境/检索三跳)
  • 写入消解 (WorldDB): OCC乐观锁 + LWW/Merge/Additive + 冲突日志
  • 多模态记忆: CLIP/Whisper懒加载 + MediaStore

🛡 第二梯队 — 生产就绪:
  • 认证: Bearer Token (DEV_MODE=true跳过)
  • 速率限制: IP Token Bucket (1000/min)
  • Dashboard: FastAPI+Jinja2+Chart.js (概览/记忆/梦境/日志)
  • VectorDB可插拔: BaseVectorStore ABC + FaissStore

⚙️ 部署: deploy.sh 一键部署 | Docker 就绪
📊 测试: 239/243 ✅ (4个kuzu存量问题)
📖 文档: README全面更新 | 技术博客 | 小红书"""
