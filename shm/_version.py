"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.19.6"
__version_info__ = (5, 19, 6)
__version_name__ = "GraphLite-GQL-complete"
__release_date__ = "2026-08-01"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心变更 — GraphLite GQL 兼容性全打通 (2026-08-01):
  • BM25 中文检索: char_wb ngram(2,4) + raw_query 传原始中文
    (token_pattern 对 CJK 边界失效 + normalize 映射导致生产链路召回为空)
  • TfidfSearchIndex: char_wb 兼容中文单字, 移除重复 fit
  • 本体矛盾检测恢复: execute_cypher 返回 rows + _interpolate b64 编码中文
    (GraphLite Rust lexer UTF-8 bug, 中文直插 PANIC → 静默漏检)
  • 清除全部 GQL MERGE 残留 (10 处): GraphLite 不支持 MERGE,
    MATCH 存在性 + INSERT 逗号分隔替代 (社区/Hebbian/本体/ALIAS_OF 边)
  • 4 个真实引擎测试恢复 (271 passed + 0 skipped)

📌 验证:
  • 本体矛盾检测实测: 张三 1990 → 2000 检测 1 冲突, confidence 0.5
  • RELATES_TO/ALIAS_OF/Hebbian 边全部实测创建成功
  • SHM 重启 0 错误 0 警告, nodes=800+
  • 全量测试 271 passed (原 267+4skip)

📌 前置 (v5.19.x):
  • v5.19.5: 梦境聚类 Leiden 最优算法 (leidenalg)
  • v5.19.4: GraphLite 全新库初始化 + 双名兼容
  • v5.19.3: 写入链路 SSM 门控修复 (await + GraphLite 别名/b64)
  • v5.19.2: structlog 日志 + FAISS GraphLite 兼容
  • v5.19.1: cdlib 社区检测兼容

⚙️ 部署: deploy.sh 一键部署 | Docker 就绪"""
