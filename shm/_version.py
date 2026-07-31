"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.19.3"
__version_info__ = (5, 19, 3)
__version_name__ = "fix: SSM门控冷启动放行 + 写入链路await修复 + GraphLite别名扁平解析"
__release_date__ = "2026-07-31"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心变更 — 写入链路打通 (2026-07-31):
  • core/dual_gate.py: SSM 门控冷启动保护（fail-open）
    根因: 随机初始化权重对任意输入输出 ≈0.45 < 0.5 → 所有写入被过滤，
    EpisodeNode 永远为 0。warmup_steps=100 内默认放行，积累数据后门控生效
  • api/routes/write.py: defense_engine.pre_check() 缺 await →
    TypeError: cannot unpack coroutine（写入 100% 500）
  • api/routes/system.py: rebuild_index 兼容 GraphLite 别名扁平返回
    (e.id/e.content 格式 + b64 透明编解码 → FAISS 首次真正重建)

📌 验证:
  • 写入 3 条记忆成功 (episode_id 正常返回)
  • FAISS rebuild: indexed_count=3, dim=384 (此前永远 0)
  • 向量检索命中 3 条真实记忆
  • 梦境管线激活: CLUSTER 3 communities / SYNTHESIZE 3 reports / 18 keywords

📌 前置 (v5.19.x):
  • v5.19.2: structlog 日志 + FAISS GraphLite 兼容 + uptime 基准
  • v5.19.1: cdlib 社区检测兼容
  • v5.19.0: OWL 导出 + 本体匹配 + LLM 关系抽取

⚙️ 部署: deploy.sh 一键部署 | Docker 就绪"""
