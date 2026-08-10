"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.21.12"
__version_info__ = (5, 21, 12)
__version_name__ = "Dream-Fix"
__release_date__ = "2026-08-10"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心变更 — 修正 embedding 声明漂移 (2026-08-09):
  • 修正：EmbeddingConfig.model_name 默认值 bge-m3→bge-small-zh-v1.5
    （YAML 早已回退，代码默认未同步）；FAISS 无"动态适配"（dimension 恒 512）

v5.21.7 (2026-08-09) CB-Config:
  • 熔断配置注入 (2026-08-09):
  • GraphLiteStore 构造传 cb_config=cfg.circuit_breaker
    (YAML/默认配置不再死配置, 熔断阈值可调)
  • conftest fixture cb_config 透传 + 注入断言测试
  • 测试: 398 passed

v5.21.6 (2026-08-09) Dream-Integrity:
  • 梦境候选孤儿社区 + 共享成员湮灭 + 外部边误删 双路径闭环

  • _persist_community_nodes 重写: 删 DETACH DELETE 全删 + 增量 upsert
    + COMMUNITY_MEMBER 边 (双阶段, 只删自己边) + execute_cypher 写路径
  • dream_pipeline 同源湮灭修复: Phase 1 限定 cid + Phase 3 max_community_by_member
  • F1 共享成员湮灭 + F2 外部边误删 双路径闭环
  • 测试: 398 passed (dream 15/15 + 幂等 + 外部边存活)

v5.21.5 (2026-08-09) Fallback-Fix:
  • LLM fallback 轮转 9→12 (openrouter 不再永不触达)

  • fallback 循环 range 9→12 (4端点×3次), url_idx 覆盖全部端点
  • 最后一个端点 openrouter 不再永不触达 ([3,3,3,0]→[3,3,3,3])
  • 日志计数 3→12 修正 + 4 个轮转测试 (序列/401/403/主端点)
  • 测试: 390 passed 1 skipped

v5.21.4 (2026-08-09) Param-Cleanup:
  • gate_threshold 死参数删除 (LSP 静默断裂修复)

  • EvolvableParams.gate_threshold 纯死参数删除 (从不被 _evolve 调节,
    同步目标 QueryRouter 无此属性) — LSP 静默断裂修复
  • SSM gate 阈值演化不受影响 (dual_gate.adapt_threshold 独立)
  • 测试: 386 passed 1 skipped

v5.21.3 (2026-08-09) P0-Stability:
  • SSM gate learn 闭环 + 5 幽灵方法 + toLower 契约

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
