"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.21.3"
__version_info__ = (5, 21, 3)
__version_name__ = "BM25-Harden"
__release_date__ = "2026-08-05"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心变更 — BM25 加固 + Embedding 升级 bge-m3 (2026-08-04):
  • BM25 空语料日志降噪: 首次 warning + 后续 debug (_bm25_empty_warned)
  • bm25_build_timeout 双语义拆分: 新增 bm25_retry_cooldown 冷却窗口
  • 冷却节奏补测试: test_bm25_retry_cooldown_gates_rebuild +
    test_bm25_empty_corpus_log_noise_reduced (359 passed)
  • Embedding 升级 BAAI/bge-m3 (1024维, 多语言, 中文C-MTEB领先):
    encoder 支持任意 HF 缓存模型 (通用 _find_model_snapshot),
    FAISS 维度动态适配 (512→1024, 启动自动重建索引)

v5.21.1 (2026-08-05) 修复:
  • test_bm25_empty_corpus_log_noise_reduced 缺 configure_logging()
    调用 — structlog 默认 PrintLogger 不走 stdlib logging,
    测试 handler 捕获不到日志 (358 passed + 1 env skip)
  • pyproject.toml / README 版本号同步 (5.20.1 → 5.21.1)

📌 验证:
  • 全量测试 359 passed (357 基线 + 2 新增)
  • bge-m3 CPU 加载 1.8s, 编码 1024维 0.1s/2句
  • 空库 warning 仅 1 次 (修复前每 30s 刷屏)
  • v5.20.0: 熔断器+重试落地 (334 passed)

⚙️ 部署: deploy.sh 一键部署 | Docker 就绪"""
