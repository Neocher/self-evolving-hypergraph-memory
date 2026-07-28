"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.14.0"
__version_info__ = (5, 14, 0)
__version_name__ = "完整差距闭环 — P0检索自演化 + P1预算/校准/事务 + P2遗忘/共享/梦境"
__release_date__ = "2026-07-28"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
全局研究闭环 (2026-07-28, 30+论文, 4份报告):
  参考论文: EvolveMem·Retain or Consolidate?·MemTX·
           Manufactured Confidence·Language Models Need Sleep·
           AdaMem·MemClaw

P0 — 检索自演化 (SelfEvolvingRetrieval):
  • FailureLogger → DiagnosisEngine → EvolutionGuard
  • 7个可演化参数: 融合权重/BM25/top-K/策略
  • 自动回滚(>15%退化) + 停滞探索(6x无变化)
  • 零侵入集成到 QueryRouter

P1 — 预算感知门控 (Budget-Aware Gating):
  • α = f(budget_ratio): 预算充足→consolidate(SSM↑), 紧张→retain(MLP↓)
  • spend_budget(): consolidate 操作消耗预算
  • 参考: Retain or Consolidate? (arXiv:2607.17545)

P1 — 过度巩固防护 (Confidence Calibrator):
  • 复合信心 = 源权重 × exp(-γ×consolidation_count)
  • 源类型: direct=1.0 / inferred=0.7 / hearsay=0.4
  • 自动标记审查: 信心<0.1 或 整合>10次
  • 集成到 DreamPipeline Step 3b
  • 参考: Manufactured Confidence (arXiv:2606.29279)

P1 — 事务性记忆写入 (Transaction Manager):
  • 两阶段提交: 暂存→commit/rollback
  • 上下文管理器: with transaction() as tx
  • 异常自动回滚
  • 参考: MemTX (arXiv:2607.13157)

P2 — 可学习遗忘 (AdaptiveDecayLearner):
  • 为每条记忆学习个性化 τ decay_rate
  • SGD: L = (τ_pred - τ_desired)²
  • 反馈来源: 访问频率↑→τ_desired↑, 门控遗忘→τ_desired↓

P2 — 多Agent共享记忆 (MemClaw):
  • agent_scope: 'global' | agent_id | list[agent_id]
  • provenance: source_agent_id + source_timestamp
  • get_visible_hyperedges(agent_id) scope过滤

P2 — SSM梦境深度升级 (SSMDreamWrapper):
  • N轮SSM循环巩固: N轮 step() 使隐状态收敛到稳定表示
  • 收敛检测: |h_{{t-1}} - h_t| < threshold 提前停止
  • reset(): 清空KV cache式重置
  • 参考: Language Models Need Sleep (arXiv:2605.26099)

📊 测试覆盖:
  • 总计 239/243 ✅ (4个kuzu存量问题)
  • 新增: 7 calibrator + 16 self_evolving = 23测试
  • 三体协奏管道已验证3轮 (CC→OpenCode→Codex)

⚙️ 部署:
  • D+F 研究节点: Python3.11+Go1.25+LibreOffice
  • Arxiv追踪: 每日08:00, 5组搜索词, 推送到所有频道

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Private repository — Neocher/self-evolving-hypergraph-memory
tag: v5.14.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
