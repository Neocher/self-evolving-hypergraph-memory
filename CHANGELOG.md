# Changelog

## v5.7.0 (2026-07-19) — Ontology v2 动态类型系统

### 🆕 新增
- **Ontology v2**: 动态实体/边类型注册（API CRUD，无需重启）
- **属性类型系统**: 8种类型（string/int/float/boolean/date/string[]/text_embedding/entity_ref）
- **边约束**: 源/目标实体类型白名单 + 对称边支持
- **类型继承**: Person ← Agent 子类型体系
- **写时验证**: 集成到 episodes 路由，自动校验属性类型+必填+范围
- **9个 REST 端点**: 本体完整 CRUD
- **基线预载**: 11种实体类型 + 7种边类型

### 🔧 修复
- Kuzu MERGE 主键冲突: 改为 MERGE ... ON CREATE SET 模式
- Kuzu DETACH DELETE 自动处理所有关联边

### 📦 生态（同期新增）
- **MCP Server**: Model Context Protocol 接入（Claude Desktop/Cursor）
- **Python SDK**: `shm.client.SHMClient` 封装全部 API
- **Dockerfile + docker-compose.yml**: 一行命令部署
- **README 更新**: 完整 v5.7 文档

## v5.6.0 (2026-07-19) — 命名空间图隔离

### 🆕 新增
- 命名空间图隔离: sensory/episodes/retrieve 全面支持 `namespace` 参数
- SessionNode + SESSION_MEMBER: 利用已有 Kuzu 表结构实现零迁移图隔离
- DELETE /memories/namespace/{name}: 批量清理模拟数据
- MiroFish 适配器: 完整替换 Zep Cloud SDK，对接本地 SHM

## v5.5.2 (2026-07-18) — 检索优化

### 🔧 优化
- BM25 提权
- 新闻数据注入
- 超边构建优化

## v5.5.1 (2026-07-17) — LLM异步化

### 🔧 修复
- httpx sync → async 改造
- 事件循环阻塞修复

## v5.5.0 (2026-07-16) — P0/P1 升级

### 🆕 新增
- 三层Embedding: Cloud API → sentence-transformers → TF-IDF降级
- Dream自动应用: 候选积压≥20自动触发apply+清理
- 加速社区发现: 轮询60s+候选数>10触发
- LongMemEval基准: 11项评测
- 多信号检索: 向量(0.5)+BM25(0.3)+实体匹配(0.2)融合
- 时序推理: 指数衰减加权，24h内数据权重翻倍
- 实体链接: LLM命名实体识别+跨节点实体关联

## v5.0.0 (2026-07-01) — 热同步API Key

### 🆕 新增
- 热同步API Key: 运行时检测.env变更自动切换
- 多provider轮询: DEEPSEEK/OPENAI/ANTHROPIC/KIMI等

## v4.x (前身)

- **v4.0**: τ-Hebbian-梦境三核心 + SSM门控过滤 + BLAKE3区块链溯源
- **v3.0**: Kuzu 图数据库 + 超边概念 + 社区发现
- **v2.0**: FAISS 语义搜索 + 向量化升级
- **v1.0**: 原型验证，SQLite + TF-IDF 检索
