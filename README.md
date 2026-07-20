# SHM — 自演化超图记忆系统

**Self-evolving Hypergraph Memory v5.7.0**

> 让 AI Agent 拥有像人一样的记忆——会遗忘、会强化、会做梦、会溯源。
>
> 不只是存储，而是**演化**。

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-ready-orange)](https://modelcontextprotocol.io)

---

## ✨ 快速开始

### 方式一：Docker（推荐）

```bash
git clone https://github.com/Neocher/self-evolving-hypergraph-memory.git
cd self-evolving-hypergraph-memory
docker compose up -d
# SHM 运行在 http://localhost:8000
```

### 方式二：Python 原生

```bash
pip install -r requirements.txt
python run_server.py
```

### 方式三：Python SDK

```python
from shm.client import SHMClient

client = SHMClient()
# 写入记忆
client.add_episode("Elon Musk founded SpaceX in 2002.", source="user")
# 检索
results = client.search("Who founded SpaceX?")
for r in results:
    print(f"  [{r['score']:.2f}] {r['content']}")
# 查看状态
print(client.stats())
```

---

## 🏗 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                       │
│  POST /memories/sensory    POST /memories/episodes               │
│  POST /memories/retrieve   POST /memories/visual                 │
│  POST /ontology/types      POST /ontology/edges                  │
│  POST /hyperedges          POST /index/rebuild                   │
│                       ... 43 个端点                              │
├──────────────────────────────────────────────────────────────────┤
│                        Core Engine                               │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐   │
│  │ τ-Decay    │  │ Hebbian    │  │ SSM Gate │  │ Dream      │   │
│  │ 遗忘引擎   │  │ 强化引擎   │  │ 门控过滤 │  │ 自演化引擎 │   │
│  └────────────┘  └────────────┘  └──────────┘  └────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│                        Memory Layers (5层)                        │
│  Layer1: Sensory Buffer (环缓冲区)                                │
│  Layer2: Episodic Nodes (Kuzu EpisodeNode + τ衰减)               │
│  Layer3: Communities (Leiden社区聚类 + 摘要+关键词)               │
│  Layer4: Hyperedges (episode/semantic/temporal 超边)             │
│  Layer5: Dream Integration (社区发现+剪枝+冲突消解+压缩)         │
├──────────────────────────────────────────────────────────────────┤
│                    Storage Backends                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Kuzu DB  │  │ FAISS    │  │ BM25     │  │ BLAKE3 AuditChain│ │
│  │ 图数据库 │  │ 向量索引 │  │ 关键词   │  │ 区块链溯源      │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🌟 核心能力

### 🔍 三级融合检索
| 信号 | 算法 | 权重 |
|:----|:-----|:----|
| 语义向量 | FAISS 384维 (IVFFlat) | 0.5 |
| 关键词 | BM25 (自定义实现) | 0.3 |
| 实体匹配 | LLM NER + 正则 | 0.2 |

**自动降级链:** FAISS不可用 → BM25 → 关键词（永不空返回）

### 🔗 图能力
- **Kuzu 嵌入式图数据库** — 零运维，单进程启动
- **超边 (Hyperedge)** — 3种类型：episode/semantic/temporal
- **社区发现** — Leiden算法，自动聚类
- **命名空间隔离** — SessionNode + SESSION_MEMBER（v5.6）

### 🧬 本体系统 (v5.7)
| 能力 | 实现 |
|:----|:------|
| 动态类型注册 | `POST /ontology/types` API热注册，无需重启 |
| 属性类型系统 | 8种: string/int/float/boolean/date/string[]/text_embedding/entity_ref |
| 边约束 | 源/目标实体类型白名单 |
| 类型继承 | Person ← Agent |
| 写时验证 | 属性类型+必填+范围校验 |
| 预载基线 | 11实体类型 + 7边类型 |

### 🔄 自演化（梦境管道）
SHM 独有的记忆自演化机制—10步管道：
```
Gather → Cluster(Leiden) → Synthesize(LLM摘要) → Entity Link(NER)
→ Compress → Prune(剪枝) → Resolve(冲突消解) → Persist
```
其他记忆系统（Mem0/Zep/MemGPT）无此能力。

### 🕒 时序感知
- **τ 指数衰减** — 半衰期30分钟，旧记忆自动降权
- **访问刷新** — 检索到即刷新生效时间
- **24h 权重加倍** — 近期记忆在检索中权重更高

### 🔐 区块链溯源
- BLAKE3 哈希链记录所有写入操作
- `GET /memories/audit/{id}` 查完整溯源链
- 链完整性验证 + 区块回滚

### 🧠 SSM 状态空间门控
- 低价值内容自动过滤（不持久化）
- 基于 hidden_state + feature_vector 的决策

---

## 🚀 MCP 集成

SHM 内置 MCP Server，可直接作为 Claude Desktop / Cursor 的记忆后端：

### Claude Desktop 配置

编辑 `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "shm": {
      "command": "python3",
      "args": ["-m", "shm.mcp_server"],
      "env": {
        "SHM_BASE_URL": "http://127.0.0.1:8000"
      }
    }
  }
}
```

### MCP 工具

| 工具名 | 功能 |
|:-------|:-----|
| `shm_add` | 添加记忆 |
| `shm_search` | 搜索记忆 |
| `shm_stats` | 系统状态 |
| `shm_health` | 健康检查 |

---

## 📡 API 总览（43 个端点）

| 分类 | 端点 | 功能 |
|:----|:-----|:-----|
| **写入** | `POST /memories/sensory` | 感觉缓冲区写入 |
| | `POST /memories/episodes` | 直接创建情节节点 |
| | `POST /memories/visual` | 创建视觉记忆 |
| **检索** | `POST /memories/retrieve` | 三级融合检索 |
| | `POST /search/vector` | 纯向量检索 |
| | `POST /query` | Cypher 查询 |
| **本体** | `POST/GET/DELETE /ontology/types` | 实体类型 CRUD |
| | `POST/GET/DELETE /ontology/edges` | 边类型 CRUD |
| | `GET /ontology/stats` | 本体统计 |
| **图** | `POST/GET /hyperedges` | 超边 CRUD |
| | `GET /hyperedges/by-node/{id}` | 查询节点超边 |
| | `GET /communities` | 社区列表 |
| **梦境** | `POST /memories/dream/trigger` | 触发梦境 |
| | `GET /dream/candidates` | 梦境候选列表 |
| **运维** | `GET /health` | 健康检查 |
| | `GET /metrics` | Prometheus 指标 |
| | `POST /index/rebuild` | 重建 FAISS 索引 |
| **命名空间** | `DELETE /memories/namespace/{name}` | 批量删除 |

完整 OpenAPI 规范：启动服务后访问 `http://localhost:8000/openapi.json`

---

## 📦 Python SDK

```python
from shm.client import SHMClient

client = SHMClient(base_url="http://localhost:8000")

# 写入 & 检索
client.add_episode("今天讨论的项目架构需要重构", source="user", namespace="meeting_01")
results = client.search("项目重构", top_k=3, namespace="meeting_01")

# 本体管理
client.register_entity_type("Meeting", attributes=[
    {"name": "topic", "type": "text_embedding", "required": True},
    {"name": "attendees", "type": "integer", "min_value": 1},
])

# 系统管理
stats = client.stats()
client.trigger_dream()
client.rebuild_index()
```

---

## 🐳 Docker 部署

```bash
# 构建并启动
docker compose build
docker compose up -d

# 查看日志
docker compose logs -f

# 健康检查
curl http://localhost:8000/health

# 停止
docker compose down
```

---

## 📊 竞品对比

| 能力 | SHM v5.7 | Mem0 | Zep | Neo4j+Vec |
|:----|:---------|:-----|:----|:----------|
| 检索融合 | ⭐ 三信号+降级 | 向量+元数据 | 图+向量 | Cypher+向量 |
| 自演化 | ⭐ 独有梦境管道 | ❌ | ❌ | ❌ |
| 区块链溯源 | ⭐ 独有 | ❌ | ❌ | ❌ |
| 超边 | ⭐ 3种 | ❌ | 仅二元边 | ❌ |
| 本体系统 | ✅ 8种属性 | ❌ | 1种 | ❌ |
| 社区发现 | ✅ Leiden | ❌ | ❌ | Louvain |
| 时序衰减 | ✅ τ指数 | ❌ | ❌ | ❌ |
| 部署 | 单进程+嵌入式 | SDK | 云端 | 独立服务 |

---

## 🗺 版本历史

| 版本 | 亮点 |
|:----|:------|
| **v5.7.0** | Ontology v2 动态类型系统 · 8种属性 · 边约束 · 写时验证 |
| **v5.6.0** | 命名空间图隔离 · SessionNode + SESSION_MEMBER |
| **v5.5.0** | 三层Embedding降级 · 多信号检索 · Dream自动应用 |
| **v5.0.0** | 热同步API Key · 多provider轮询 |
| **v4.x** | τ-Hebbian-梦境三核心 · SSM门控 · BLAKE3溯源链 |

---

## 📝 License

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 PR！
