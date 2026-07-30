"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.18.0"
__version_info__ = (5, 18, 0)
__version_name__ = "图引擎替换 — RyuGraph → GraphLite (GQL)，修复UTF-8 + ALTER TABLE"
__release_date__ = "2026-07-30"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心变更 — 嵌入式图引擎替换 (2026-07-30):
  • 引擎: RyuGraph (Kuzu fork, 139⭐, 已停) → GraphLite (228⭐, Rust, ISO GQL)
  • 解决: ALTER TABLE不支持 → GQL动态属性，schema可运行时演进
  • 解决: 文件锁限制 → Sled单文件无锁并发
  • 解决: 电路断路器 → 直接GQL执行，零开销
  • 解决: UTF-8 Rust lexer panic → b64透明编解码
  • 适配器: graph/graphlite_store.py (~260行) → 替代 ryu_store.py (~750行)
  • 编译: Rust/Cargo via SJTUG镜像 (大陆可构建)

📦 依赖变化:
  • 移除: ryugraph, circuitbreaker
  • 新增: GraphLite (GitHub: GraphLite-AI/GraphLite, sdk-python)
  • 系统: Rust 1.95.0, libgraphlite_ffi.so (9.6MB)

🌐 多协议网关 (v5.14.1):
  • MCP  :8002 | A2A  :8001 | ACP  :8770 | HTTP :8000 | CLI 终端

⚙️ 部署: deploy.sh 一键部署 | Docker 就绪"""
