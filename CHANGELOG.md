# Changelog

## v5.8.2 (2026-07-20) — 批量关系写入 · OSINT关系抽取

### 🆕 新增
- **POST /batch/relations**: 批量写入语义边端点，支持大量关系一次性注入
- **OSINT结构化关系抽取**: `scripts/batch_osint_relations.py` 自动从域名/IP/URL字段推断关系类型
- **语义边增长**: 6→294 条，覆盖 OSINT 域全量关系

### 🔧 修复
- API Key热加载: `run_server.py` 中 `setdefault`→`os.environ[k]=v` 直接赋值，解决 Hermes 已有 Key 被跳过的问题
- Kuzu RELATES_TO schema 仅保留 `relation STRING`，兼容写入

## v5.8.1 (2026-07-20) — 实体发现准确率100%

### 🆕 新增
- **上下文投票消歧**: 实体发现增加共识投票机制，去伪存真
- **实体覆盖率**: 26/26 实体类型 100% 准确识别

### 🔧 修复
- 实体发现伪阳性: 上下文投票过滤单次偶然匹配
- Dream Scheduler 稳定性: 候选积压检测阈值调优

## v5.8.0 (2026-07-19) — 本体自发现 · 关系抽取 · 置信度累积 · 实体消歧

### 🆕 新增
- **实体自动发现**: 从文本输入中自动识别并注册新实体类型，无需预定义
- **关系抽取管道**: 10 种谓词类型，基于 LLM 的语义关系自动抽取
- **置信度累积**: BLAKE3 证据追踪 + 出现次数加权，置信度递增加密可审计
- **实体消歧管道**: 同义合并 + 指代消解 + 别名归并，三步清洗
- **write-time pipeline**:
  1. ontology validate — 类型校验
  2. relation extract — 10 predicate types
  3. evidence track — BLAKE3 hash + count
  4. entity resolve — disambiguation + coreference + alias

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
