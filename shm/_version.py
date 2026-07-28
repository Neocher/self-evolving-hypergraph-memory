"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.13.0"
__version_info__ = (5, 13, 0)
__version_name__ = "检索自演化 — SelfEvolvingRetrieval + 三体协奏全流程"
__release_date__ = "2026-07-28"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P0 — 架构骨架重构:
  • 路由模块化: api/_routes.py 2327行单体 → api/routes/ 8域独立文件
  • 配置统一: LLMConfig + SHMClientConfig 进入 config/settings.py
  • LLM fallback链: 4端点自动切换 (DeepSeek→OpenAI→Moonshot→OpenRouter)

P1 — 质量保障:
  • 核心单元测试: 22/22 ✅ (τ衰减·Hebbian·SSM门控·FAISS)
  • 嵌入模型可配置: config/settings.py → defaults.yaml 三明治函数链
  • 梦境状态持久化: save_state()/load_state() → Kuzu SystemNode

P2 — 生产部署:
  • Dockerfile (Python 3.11-slim, HEALTHCHECK)
  • docker-compose.yml (单服务 + 数据卷持久化)
  • shm.service → systemd (Restart=on-failure, MemoryHigh=2G)
  • install.sh 一键安装脚本
  • 硬编码常量消除: SHMClient/MCP 自动从 config 读取

⚡ 性能优化 (10.6x 写入提升):
  • 异步嵌入: 写入路径去除同步 _process_embed_queue (421ms→40ms)
  • 批量 Kuzu 操作: 关系抽取 3N次→2次
  • FAISS 调优: IVFFlat→FlatL2, nprobe 10→3, batch buffer 10→50
  • 短内容跳过: <50字不跑关系抽取, <80字不跑实体消歧
  • 嵌入 LRU 缓存: TextEncoder._cached_embed (max 512)
  • 检索结果缓存: 相同 query+top_k 命中 470ms→10ms (43x)
  • 并发 QPS: 2.5→25 (10x)

📊 测试覆盖:
  • 总计 212/214 ✅ (2个既有 ontology 已知问题)
  • 新增 38 测试: core(22) + retrieval(7) + routes(9)
  • 覆盖: τ衰减·Hebbian·SSM·FAISS·缓存·嵌入·路由

性能基准 (v5.9.0, 1660节点, 1150 FAISS):
  写入 P50:     421ms →  39ms (10.6x)
  搜索 (缓存):  884ms →  10ms (88x)
  搜索 (冷启动): 884ms → 470ms (1.9x)
  写入 QPS:       2.5 →  25  (10x)
  并发写入:    1360ms → 147ms (9.2x)
  内存 RSS:                   5.1 MB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
从 v5.8.4 (MCP v2) 升级 — 10 次提交, +1500/-160 行
架构评分: 5.7 → 7.8/10 (+2.1)
Private repository — Neocher/self-evolving-hypergraph-memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
