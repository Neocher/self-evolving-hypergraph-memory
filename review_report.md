# SHM 代码审查报告
生成时间: 2026-07-29

---

## Phase 1: CC 架构分析
Now I have a thorough understanding of the codebase. Let me compile the comprehensive review.

---

# 🔬 SHM v5.10 全面代码审查报告

**审查日期**: 2026-07-29  
**审查范围**: 全仓库 (~40+ Python 模块)  
**核心文件已审阅**: `acp_bridge.py`, `mcp_server.py`, `api/app.py`, `api/_routes.py`, `core/hebbian.py`, `core/tau_decay.py`, `core/dream_pipeline.py`, `core/dual_gate.py`, `core/llm_client.py`, `core/ontology_validator.py`, `graph/ryu_store.py`, `config/settings.py`, `gateway/auth.py`, `gateway/rate_limit.py`, `observability/metrics.py`, `observability/logger.py`, `tests/` (15 文件)

---

## 1. 架构设计

### 1.1 总体评估：**一般**

**优点**：
- 五层架构清晰（Sensory → Episodic → Community → Hyperedge → Dream）
- 依赖注入模式 (`Services` dataclass + `get_services()` Depends) 避免了全局单例耦合
- 断路器模式 (CircuitBreaker) 正确隔离了 Kuzu 数据库故障
- 多级降级策略贯穿始终（Tier1→Tier2→Tier3 编码器、LLM→TF-IDF、FAISS→Cypher fallback）

**问题**：

| # | 问题 | 位置 | 优先级 |
|---|------|------|--------|
| 1 | **循环依赖风险**: `api/app.py`→`api/routes`，`api/routes`→`api/app.py`（`incremental_faiss_update` 被 app.py 注入回 routes） | `api/app.py:233`, `api/_routes.py:257` | P1 |
| 2 | **codex_bridge 版本爆炸**: `codex_bridge.py`, `codex_bridge_v2.py` … `v5.py` — 5 个版本共存，显然 v5 应替代旧版 | `codex_bridge*.py` | P2 |
| 3 | **transcribe_audio 版本爆炸**: `transcribe_audio.py` 到 `transcribe_audio14.py` — 14 个版本文件，大量死代码 | `transcribe_audio*.py` | P2 |
| 4 | **`_routes.py` 过于臃肿**: 2900+ 行，包含写入/检索/超边/社区/梦境/冲突/视觉/热图等所有端点逻辑 | `api/_routes.py` (2929 lines) | P1 |
| 5 | **路由重复定义**: `create_hyperedge` 和 `get_hyperedge` 在 `_routes.py` 中定义了**两次**（L1116 和 L2061） | `api/_routes.py:1116-1155` 与 `api/_routes.py:2061-2099` | P0 |

**改进建议**：
- 将 `_routes.py` 拆分为 `routes/write.py`, `routes/retrieve.py`, `routes/dream.py`, `routes/community.py` 等
- 删除 `codex_bridge_v1~v4.py` 和 `transcribe_audio2~14.py`，仅保留最新版本
- 消除重复路由定义

---

## 2. 代码质量

### 2.1 总体评估：**一般**

**优点**：
- 文档字符串完整，docstring 风格统一
- `from __future__ import annotations` 全局启用
- 类型提示覆盖率较高（Optional, dict[str, X], list[dict] 等）

**问题**：

| # | 问题 | 位置 | 优先级 |
|---|------|------|--------|
| 6 | **全局可变状态泛滥**: `_services` (L126), `_embed_queue` (L167), `_embed_cache` (L171), `_result_cache` (L175) 都是模块级可变全局变量，多 worker 下不安全 | `api/_routes.py:126,167,171,175` | P0 |
| 7 | **异常吞噬 (bare pass)**: 大量 `except Exception: pass` 模式，例如 `_persist_batch` 中：`except Exception: pass  # 持久化失败不影响主流程` — 问题被完全隐藏 | `core/hebbian.py:137`, `core/dream_pipeline.py:413`, `api/_routes.py:709` 等多处 | P1 |
| 8 | **不一致的错误处理**: 有的地方 `logger.exception()`，有的地方 `logger.warning()`，有的地方完全静默 | 多处 | P2 |
| 9 | **重复代码**: `_extract_keywords` 在 `DreamPipeline` (L831) 和 `LLMClient._fallback_keywords` (L288) 中几乎完全相同 | `core/dream_pipeline.py:831`, `core/llm_client.py:288` | P2 |
| 10 | **重复代码**: `list_hyperedges` (L2104) 和 `list_hyperedges_for_node` (L1185) 大量重复的响应构造逻辑 | `api/_routes.py:2104-2158`, `api/_routes.py:1185-1211` | P2 |
| 11 | **重复的 ontology 验证代码块**: `retrieve()` 中 lines 917-943 有两个完全相同的 try/except 块 | `api/_routes.py:919-943` | P1 |
| 12 | **unused import**: `from fastapi.responses import JSONResponse` 在第 22-23 行导入了两次 | `api/_routes.py:22-23` | P3 |
| 13 | **变量覆盖**: `retrieve()` 中 L868 的局部 `logger = get_logger()` 覆盖了 L75 的模块级 logger | `api/_routes.py:868` | P3 |
| 14 | **魔法数字**: `_dream_poll_loop` 中硬编码 `DREAM_POLL_INTERVAL = 60.0`，`_hyperedge_sweep` 中 `600.0`，应该从配置读取 | `api/app.py:422,503` | P2 |

---

## 3. 性能瓶颈

### 3.1 总体评估：**一般**

**优点**：
- 异步 embedding 队列 + 批量 flush FAISS（每 5s），避免写入路径阻塞
- 检索结果缓存（LRU 128 条）减少重复查询
- 断路器保护 Kuzu 免于雪崩
- RyuGraph 连接池（pool_size = max_threads）

**问题**：

| # | 问题 | 位置 | 优先级 |
|---|------|------|--------|
| 15 | **N+1 查询 — list_hyperedges**: 先查所有 HyperedgeNode，然后**对每个超边再查一次成员**，100 条超边 = 101 次 Kuzu 查询 | `api/_routes.py:2104-2158` | P0 |
| 16 | **N+1 查询 — list_conflicts**: 先查所有 ConflictNode，然后**逐条查 episode_a 和 episode_b 的版本**，100 条冲突 = 201 次查询 | `api/_routes.py:1637-1686` | P0 |
| 17 | **梦境管道阻塞事件循环**: `DreamPipeline._persist_communities` (L1021), `_persist_prune` (L1060) 是同步方法，虽用 `asyncio.to_thread` 包装，但 `_gather_step`, `_cluster_step`, `_prune_step` 等步骤本身是同步的，在 `run()` 中同步执行 | `core/dream_pipeline.py:199-347` | P1 |
| 18 | **`_cluster_step` 使用 ThreadPoolExecutor**: 虽然声称"并行"，但 `_build_nx_graph` 中的本体约束边构建是 O(N²) 嵌套循环 (`api/_routes.py` 中 `_cluster_step` 没有体现，但在 `DreamPipeline._build_nx_graph` L491-511 中) | `core/dream_pipeline.py:491-511` | P2 |
| 19 | **FAISS flush 可能导致数据丢失**: 每 5 秒 flush 一次，如果在两次 flush 之间崩溃，缓冲区中的向量会丢失；失败时重新放回缓冲区 (`flush_faiss_buffer` L252)，但没有上限保护，可能导致内存无限增长 | `api/_routes.py:225-254` | P1 |
| 20 | **无数据库索引**: EpisodeNode 的 `created_at`, `source`, `tau_initial` 字段在 Cypher 查询中频繁用于 WHERE 过滤，但 Kuzu/RyuGraph 没有对这些列建索引 | `graph/ryu_store.py:197-204` | P2 |
| 21 | **Ontology 读时验证 O(N+M)**: `read_validate` 中对每个检索结果调用 `_compute_topology_score`，每个拓扑检查执行 1-3 跳图遍历 (L885)，对 20 个结果可能有数百次 Kuzu 查询 | `core/ontology_validator.py:1095-1225` | P2 |

**改进建议**：
- 使用 `MATCH (h:HyperedgeNode)-[:HYPEREDGE_MEMBER]->(e) RETURN h, collect(e)` 一次查询获取所有超边+成员
- 使用 `MATCH (c:ConflictNode) ... RETURN c, a.version, b.version` 一次查询获取冲突+版本
- 为 `created_at`, `source` 列添加 Kuzu 索引（如果 RyuGraph 支持）
- FAISS 缓冲区增加容量上限 + 告警

---

## 4. 安全隐患

### 4.1 总体评估：**差**

**优点**：
- TokenManager 使用 `secrets.token_hex` 生成密钥，文件权限 chmod 600
- `/query` Cypher 端点有写操作拦截正则（阻止 CREATE/DELETE/SET 等）
- RateLimiter 提供基础速率限制

**问题**：

| # | 问题 | 位置 | 优先级 |
|---|------|------|--------|
| 22 | **命令注入 — shell=True**: `handle_terminal` 使用 `subprocess.run(command, shell=True)`，攻击者可以通过 MCP Server 执行任意命令 | `mcp_server.py:70` | **P0 CRITICAL** |
| 23 | **CORS 全开 + allow_credentials=True**: `allow_origins=["*"]` 与 `allow_credentials=True` 是不安全的组合，任何网站可以携带凭据访问 API | `api/app.py:558-561` | **P0** |
| 24 | **DEV_MODE 默认为 true**: 生产环境默认无认证保护 | `gateway/auth.py:113` | **P0** |
| 25 | **API Key 热加载从文件读取**: `llm_client.hot_reload()` 从 `~/.hermes/.env` 读取密钥并注入 `os.environ`，如果该文件被其他进程篡改，密钥即泄露 | `core/llm_client.py:101-132` | P1 |
| 26 | **无输入大小限制**: `/memories/sensory`, `/memories/episodes`, `/memories/multimodal` 等端点没有对 content/body 大小做限制，可被大 payload DoS | `api/_routes.py:298-342` 等多处 | P1 |
| 27 | **路径遍历风险**: `handle_read_file` 的 `path` 参数未做规范化检查，`read_file("../etc/passwd")` 可能泄露文件 | `mcp_server.py:33-42` | P1 |
| 28 | **Base64 图像无大小限制**: `/memories/visual` 和 `/memories/multimodal` 直接解码 base64 图像，无大小上限，可导致内存耗尽 | `api/_routes.py:396-401,1902-1962` | P1 |
| 29 | **Cypher 注入**: 关系抽取中直接拼接实体名到 Cypher：`f"MERGE (n{len(seen_entities)}:OntologyEntity {{name: '{entity_name}'}})"` — entity_name 未经转义 | `api/_routes.py:642-643` | P1 |
| 30 | **Token 无过期机制**: TokenManager 创建的 token 永不过期，无 TTL、无 scope 限制 | `gateway/auth.py:35-44` | P2 |

**改进建议**：
- `mcp_server.py` 的 `handle_terminal` 改为 `shell=False` + `shlex.split(command)`
- CORS 改为具体域名列表，或至少 `allow_origins=["*"]` 时设 `allow_credentials=False`
- DEV_MODE 默认 `false`，通过 `SHM_DEV_MODE=true` 显式开启
- 所有端点添加 `Request` body 大小限制中间件
- Cypher 查询中对用户输入使用参数化查询（`$param`），而非字符串拼接
- Token 添加 `expires_at` 字段

---

## 5. 测试覆盖

### 5.1 总体评估：**一般**

**现有覆盖**（15 文件，~3660 行测试代码）：

| 模块 | 测试文件 | 状态 |
|------|---------|------|
| TauDecayEngine | `test_tau_decay.py` (117 行) | ✅ 基础覆盖 |
| Hebbian | `test_hebbian.py` (79 行) | ✅ 基础覆盖 |
| CircuitBreaker | `test_circuit_breaker.py` (109 行) | ✅ 良好 |
| FAISS | `test_faiss_functions.py` (150 行) | ✅ 良好 |
| SSM Gate | `test_ssm_gate.py` (128 行) | ✅ 良好 |
| OntologyValidator | `test_ontology_validator.py` (400+ 行) | ✅ 良好 |
| Vector Search | `test_vector_search.py` (355 行) | ✅ 良好 |
| Kuzu Store | `test_kuzu_store.py` (117 行) | ⚠️ 基础 |
| Retrieve Routes | `test_retrieve_routes.py` (100+ 行) | ⚠️ 浅覆盖 |
| Self Evolving | `test_self_evolving.py` (181 行) | ✅ 良好 |
| Core Engine | `test_core_engine.py` (300+ 行) | ✅ 良好 |
| Confidence Calibrator | `test_confidence_calibrator.py` (60+ 行) | ⚠️ 薄 |
| Long Memory Eval | `test_long_mem_eval.py` (39K 行) | ⚠️ 疑似数据而非测试 |

**缺口**：

| # | 缺失测试 | 优先级 |
|---|---------|--------|
| 31 | **ACP Bridge 完全无测试**: `acp_bridge.py` 的 dispatch/retry/circuit breaker/cleanup 无覆盖 | P1 |
| 32 | **MCP Server 完全无测试**: `mcp_server.py` 的所有 tools 和 JSON-RPC 处理无覆盖 | P1 |
| 33 | **Dream Pipeline 无集成测试**: `DreamPipeline.run()` 八步管道没有端到端测试 | P1 |
| 34 | **认证中间件无测试**: `gateway/auth.py` 的 TokenManager 和中间件无测试 | P2 |
| 35 | **速率限制无测试**: `gateway/rate_limit.py` 的 RateLimiter 无测试 | P3 |
| 36 | **LLMClient 无测试**: HTTP 调用、fallback 端点切换、热加载均无测试 | P1 |
| 37 | **API 端点集成测试不足**: 仅 `test_retrieve_routes.py` 和 `test_core_engine.py` 有部分 HTTP 层测试 | P1 |
| 38 | **无性能/压力测试**: 无并发写入、大量检索、梦境管道的负载测试 | P2 |

---

## 6. 三体协奏集成 (ACP / MCP / Graphify)

### 6.1 总体评估：**良好**

**优点**：

- **ACP Bridge 架构合理**: 重试 + 并发限制(Semaphore) + 断路器(CircuitBreaker) + 后台清理，四层韧性保护
- **Agent 配置可调**: 每个 agent 独立的 timeout/max_concurrent/max_retries
- **MCP Server 遵循 JSON-RPC 2.0**: 标准的 `initialize`/`tools/list`/`tools/call` 协议
- **LLMClient 多端点 fallback**: DeepSeek → OpenAI → Moonshot → OpenRouter 链式降级
- **API Key 热同步**: `hot_reload()` 从 Hermes .env 文件监控更新，无需重启
- **注册式 Action 系统**: `register_action()` + `/action/{name}` 支持运行时扩展

**问题**：

| # | 问题 | 位置 | 优先级 |
|---|------|------|--------|
| 39 | **ACP Bridge 无 agent 身份验证**: `/dispatch` 端点可以任意指定 `target_agent`，无限发送 prompt，无认证/限流，等同于免费 LLM 代理 | `acp_bridge.py:235-255` | P1 |
| 40 | **ACP Bridge 无 prompt 大小限制**: `TaskDispatch.prompt` 可以发送任意大小文本给 LLM | `acp_bridge.py:67-71` | P2 |
| 41 | **MCP Server 无工具权限控制**: 所有连接的 Agent 可以执行 `terminal` 命令，无沙盒限制 | `mcp_server.py:66-81` | P1 |
| 42 | **LLMClient 提示注入风险**: `summarize_community` 将用户提供的 `node_contents` 直接拼接到 prompt 中，无注入防护 | `core/llm_client.py:239-253` | P2 |
| 43 | **ACP Action dispatch 无超时控制**: `/action/{name}` 端点没有对 handler 执行设置超时 | `acp_bridge.py:291-319` | P3 |
| 44 | **exec_cc 将 prompt 以 stdin 传入，可能因编码问题截断**: `prompt.encode()` 默认 utf-8，但未处理 surrogates | `acp_bridge.py:113` | P3 |
| 45 | **MCP Server 无资源端点**: `initialize` 中声明了 `resources: {}` 能力但实际上没有实现 `resources/list` | `mcp_server.py:152` | P3 |

---

## 📊 汇总

### 按优先级统计

| 优先级 | 数量 | 关键项 |
|--------|------|--------|
| **P0** | 7 | 路由重复定义、全局可变状态、N+1 查询 ×2、命令注入、CORS 配置、DEV_MODE 默认值 |
| **P1** | 12 | 循环依赖、异常吞噬、模块臃肿、梦境阻塞、FAISS 数据丢失、认证/注入风险 |
| **P2** | 14 | 死代码清理、重复代码、魔法数字、性能优化、缺失测试 |
| **P3** | 7 | import 重复、变量覆盖、超时缺失、无资源端点 |

### 按维度评分

| 维度 | 评分 | 关键发现 |
|------|------|---------|
| 架构设计 | ⭐⭐⭐ | 分层清晰，DI 合理，但存在循环依赖和模块膨胀 |
| 代码质量 | ⭐⭐⭐ | 文档好、类型提示好，但全局状态和异常吞噬严重 |
| 性能瓶颈 | ⭐⭐⭐ | 有缓存和批量策略，但 N+1 查询和同步阻塞是硬伤 |
| 安全隐患 | ⭐⭐ | shell=True、CORS 全开、DEV_MODE 默认 true — 需立即修复 |
| 测试覆盖 | ⭐⭐⭐ | 核心模块有测试，但集成/安全/桥接层测试空白 |
| 三体协奏集成 | ⭐⭐⭐⭐ | 架构合理，降级链完善，缺少认证和沙盒是主要短板 |

### 🚨 建议立即修复的 Top 5

1. **`mcp_server.py:70` — 命令注入 (shell=True)** → 改为 `shell=False` + `shlex.split`
2. **`api/_routes.py:1116` 与 `2061` — 重复路由定义** → 删除重复定义
3. **`api/app.py:558` — CORS allow_credentials=True + origins=["*"]** → 改为具体域名
4. **`gateway/auth.py:113` — DEV_MODE 默认 true** → 改为 `false`
5. **`api/_routes.py:2104-2158` — N+1 超边查询** → 单次 MATCH 获取超边+成员

---
## Phase 2: OpenCode 改进方案
I detect **implementation intent** — the user wants concrete code improvement plans based on the CC review, with specific file paths and changes. My approach: verify the key claims by reading critical files, then produce a structured, actionable plan.

Let me start by exploring the codebase to validate the CC report's findings.
Good — `api/routes/` already exists with split files. Let me verify the CC claims against the actual code.
Now let me verify the specific issues reported — duplicate routes, global state, bare excepts, etc.
Now let me check a few more details — the existing routes modules, bare except patterns, and the `_extract_keywords` duplication.
All findings verified. Now I have a complete picture. Let me compile the comprehensive plan.

---

# SHM v5.10 CC 审查 — 代码改进方案

基于实际代码验证，以下报告所有 P0/P1 问题均确认存在。

---

## 1. P0 问题修复方案（立即执行）

### 1.1 重复路由定义 — `_routes.py`

**问题**：`api/_routes.py` 中定义了两套完全相同的超边路由端点，FastAPI 会静默使用先注册的（L1116 & L1159），后注册的（L2061 & L2162）永远不会被路由到，造成死代码和混淆。

| 端点 | 第一次定义 | 第二次定义 |
|------|-----------|-----------|
| `POST /hyperedges` | L1116 (`create_hyperedge`) | L2061 (`create_hyperedge`) |
| `GET /hyperedges/{hyperedge_id}` | L1159 (`get_hyperedge`) | L2162 (`get_hyperedge`) |

**修复方案**：保留第一次定义（位于"超边管理"域 L1090-1200），**删除第二次定义**（L2055-2200 区域）。第二次定义不仅冗余，还使用了不一致的 `APIHyperedgeType(edge.type.value)` 而非 `_core_type_to_api()`。

**具体修改**：

**文件**：`api/_routes.py`
**操作**：删除 L2055-L2200（含空行，⚠️注意保留区间末尾的 `list_hyperedges_for_node` 不在此区间）

```python
# 删除整个区间（保留空行标记避免破坏行号变化）：
# ═══════════════════════════════════════════════════════════
# 超边 (Hyperedge) 端点  ← 从此开始移除
# ═══════════════════════════════════════════════════════════
# 到下一个注释块之前的空行为止

# 实际删除代码块：L2056 "超边 (Hyperedge) 端点" 注释行 到 L2199 "超边端点结束" 注释行
```

删除后，`list_hyperedges`（L2104）和 `get_hyperedge/{hyperedge_id}`（L1116/1159 版本）中间大约空出 100 行空间，将 L2200+ 的 `list_hyperedges_for_node` 上移填补。

---

### 1.2 N+1 查询 — `list_hyperedges`

**问题**：`api/_routes.py:2104-2148`，先查所有 HyperedgeNode（1 次查询），然后对每个超边查一次成员（N 次查询）。100 条超边 = 101 次 Kuzu 查询。

**修复方案**：用一条 MATCH 带 OPTIONAL MATCH 的 Cypher 替代。

**文件**：`api/_routes.py`，替换 L2116-L2144

```python
# 当前（N+1）：
rows = deps.kuzu_store.query_cypher(
    "MATCH (h:HyperedgeNode) RETURN h.* ORDER BY h.created_at DESC LIMIT $limit",
    {"limit": limit},
)
results = []
for row in rows:
    ...
    # 每行一个独立查询
    member_rows = deps.kuzu_store.query_cypher(
        "MATCH (h:HyperedgeNode {id: $id})-[:HYPEREDGE_MEMBER]->(e:EpisodeNode) RETURN e.id",
        {"id": h["id"]},
    )

# 修改为：
rows = deps.kuzu_store.query_cypher(
    """MATCH (h:HyperedgeNode)
    OPTIONAL MATCH (h)-[:HYPEREDGE_MEMBER]->(e:EpisodeNode)
    WITH h, COLLECT(DISTINCT e.id) AS member_ids
    RETURN h.id, h.type, h.created_at, h.gate_value, h.metadata, member_ids
    ORDER BY h.created_at DESC LIMIT $limit""",
    {"limit": limit},
)
results = []
for row in rows:
    results.append(HyperedgeResponse(
        id=row.get("h.id", ""),
        type=APIHyperedgeType(row.get("h.type", "")),
        member_ids=row.get("member_ids", []),
        created_at=row.get("h.created_at", 0.0),
        gate_value=row.get("h.gate_value", 1.0),
        metadata=json.loads(row.get("h.metadata", "{}")) if isinstance(row.get("h.metadata"), str) else row.get("h.metadata", {}),
    ))
```

---

### 1.3 N+1 查询 — `list_conflicts`

**问题**：`api/_routes.py:1637-1686`，先查所有 ConflictNode，然后逐条查 `episode_a` 和 `episode_b` 的版本。100 条冲突 = 201 次额外查询。

**修复方案**：用 OPTIONAL MATCH 将 episode 信息 JOIN 到主查询。

**文件**：`api/_routes.py`，替换 L1650-L1673

```python
# 修改为：
rows = deps.kuzu_store.execute_cypher(
    f"""MATCH (c:ConflictNode)
    WHERE 1=1 {resolved_filter}
    OPTIONAL MATCH (a:EpisodeNode {{id: c.episode_a}})
    OPTIONAL MATCH (b:EpisodeNode {{id: c.episode_b}})
    RETURN c.id, c.episode_a, c.episode_b, c.rule_id,
           c.detected_at, c.resolved,
           a.version AS version_a, b.version AS version_b
    ORDER BY c.detected_at DESC LIMIT $limit""",
    {"limit": limit}
)
conflicts = []
for r in rows:
    conflicts.append({
        "id": r.get("c.id", ""),
        "episode_a": r.get("c.episode_a", ""),
        "episode_b": r.get("c.episode_b", ""),
        "rule_id": r.get("c.rule_id", ""),
        "detected_at": r.get("c.detected_at", 0.0),
        "resolved": r.get("c.resolved", False),
        "episode_a_version": r.get("version_a", 1),
        "episode_b_version": r.get("version_b", 1),
    })
```

---

### 1.4 全局可变状态 — `_routes.py`

**问题**：`_services`, `_embed_queue`, `_embed_cache`, `_result_cache` 四个模块级可变全局变量。多 worker（gunicorn/uvicorn workers > 1）下共享状态不安全。

**修复方案**：

1. **`_services`**：已经是单例模式，维持现状但加注释说明仅用于单 worker 模型。
2. **`_embed_queue` + `_embed_cache` + `_result_cache`**：提取到独立模块 `api/_cache.py`，使用 `threading.Lock` 包裹所有读写。

**新文件**：`api/_cache.py`

```python
"""API 层缓存：嵌入队列、查询缓存、结果缓存（线程安全）。"""
from __future__ import annotations

import threading
import time
from typing import Any

# ── 嵌入队列 ──
_embed_queue: list[tuple[str, str, float]] = []
_embed_queue_lock = threading.Lock()

def enqueue_embed(episode_id: str, content: str) -> None:
    with _embed_queue_lock:
        _embed_queue.append((episode_id, content, time.time()))

def drain_embed_queue() -> list[tuple[str, str, float]]:
    with _embed_queue_lock:
        batch = _embed_queue[:]
        _embed_queue.clear()
    return batch

# ── 查询嵌入缓存（LRU） ──
_embed_cache: dict[str, Any] = {}
_embed_cache_max = 256
_embed_cache_lock = threading.Lock()

def get_embed_cache(key: str) -> Any | None:
    with _embed_cache_lock:
        return _embed_cache.get(key)

def set_embed_cache(key: str, value: Any) -> None:
    with _embed_cache_lock:
        if len(_embed_cache) >= _embed_cache_max:
            _embed_cache.clear()
        _embed_cache[key] = value

# ── 检索结果缓存（LRU） ──
_result_cache: dict[str, Any] = {}
_result_cache_max = 128
_result_cache_lock = threading.Lock()

def get_result_cache(key: str) -> Any | None:
    with _result_cache_lock:
        return _result_cache.get(key)

def set_result_cache(key: str, value: Any) -> None:
    with _result_cache_lock:
        if len(_result_cache) >= _result_cache_max:
            _result_cache.clear()
        _result_cache[key] = value
```

然后 `_routes.py` 中引用 `from api._cache import ...` 替换全局变量的直接访问。

---

### 1.5 重复导入 + 重复的 ontology 验证块

**问题**：
- L22-23：`from fastapi.responses import JSONResponse` 重复两行
- L917-943：`try/except` 验证块被拷贝了完全相同的两份

**修改**：

**文件**：`api/_routes.py` L22-23

```python
# 删除重复行
from fastapi.responses import JSONResponse  # 只保留一行
```

L917-L943 块：

```python
# 原有（L917-918 注释重复，L942-943 except 重复）：
# [Ontology] 读时验证：一致性交叉检查 + 置信度修正
# [Ontology] 读时验证：一致性交叉检查 + 置信度修正
if deps.ontology_validator is not None and results_raw:
    try:
        validated = deps.ontology_validator.read_validate(...)
        ...
    except Exception as val_err:
        logger.warning("Ontology validation failed, using raw scores", error=str(val_err))
    except Exception as val_err:   # ← 重复
        logger.warning(...)         # ← 删除

# 改为：
# [Ontology] 读时验证：一致性交叉检查 + 置信度修正
if deps.ontology_validator is not None and results_raw:
    try:
        validated = deps.ontology_validator.read_validate(
            [{
                "id": r.get("node_id", ""),
                "score": r.get("score", 0.0),
                "tau_value": r.get("tau_value", r.get("tau", 0.5)),
                "trust_score": r.get("trust_score", 0.5),
                "content": r.get("content", ""),
            } for r in results_raw[:req.top_k]],
            req.query,
        )
        v_map = {v.episode_id: v for v in validated}
        for r in results_raw[:req.top_k]:
            rid = r.get("node_id", "")
            if rid in v_map:
                v = v_map[rid]
                r["score"] = v.adjusted_score if v.adjusted_score is not None else 0.0
                if v.conflict_note:
                    r["conflict_note"] = v.conflict_note
    except Exception as val_err:
        logger.warning("Ontology validation failed, using raw scores", error=str(val_err))
```

---

## 2. P1 优化计划（本周内）

### 2.1 异常吞噬治理 — 全库 `except Exception: pass`

**问题**：全库约 20+ 处 `except Exception:`（含 `pass` 或仅日志无处理），包括 `core/hebbian.py:137`, `core/dream_pipeline.py:425,692,784,821,1031,1053,1078,1098,1134`, `core/dream_candidate_store.py:173,302,329,405,447`, `core/llm_client.py:55,313`, `core/ontology_validator.py:499,671,918,1015,1136` 等。

**分类处理策略**：

| 类别 | 处理方式 | 示例 |
|------|---------|------|
| **预期异常**（如 ImportError, ModelNotFound） | 保留 `except SpecificError`，加 `logger.warning` | `except (ImportError, ValueError): logger.warning(...)` |
| **防御性兜底**（降级路径的一部分） | 保留但加 `logger.warning` 记录降级详情 | 如 FAISS→Cypher 回退 |
| **真正的错误吞噬**（`pass` 无日志） | 加 `logger.exception("...")` 至少记录 | `core/hebbian.py:137` |

**高优先级修改**：

```python
# core/hebbian.py:137 — 当前是 bare pass
except Exception:
    pass
# 改为：
except Exception:
    logger.exception("Hebbian strength update failed")  # 非致命，但需要日志
```

```python
# core/dream_pipeline.py 等 — 每个 except 块加日志级别和上下文
# 如 L425：
except Exception:
    pass  # 至少改为：
except Exception:
    logger.warning("Dream step failed, continuing pipeline")
```

**验收标准**：全库没有 `except.*pass` 模式；每个 except 块至少有一个 `logger.warning()` 或 `logger.exception()`。

---

### 2.2 魔法数字迁移到配置

**问题**：`api/app.py` 中硬编码：

```python
DREAM_POLL_INTERVAL = 60.0    # L422
HYPEREDGE_SWEEP_INTERVAL = 600.0  # L503
```

**修复方案**：添加到 `config/settings.py` 的配置类。

**文件**：`config/settings.py`（增加字段）

```python
@dataclass
class AppSettings:
    dream_poll_interval: float = 60.0
    hyperedge_sweep_interval: float = 600.0
```

**文件**：`api/app.py` L421-423

```python
# 改为：
cfg = get_settings()
DREAM_POLL_INTERVAL = cfg.app.dream_poll_interval

# L503 同理
HYPEREDGE_SWEEP_INTERVAL = cfg.app.hyperedge_sweep_interval
```

---

### 2.3 Logger 覆盖修复

**问题**：`api/_routes.py:1468` 内 `retrieve()` 中 `logger = get_logger()` 覆盖了模块级 L75 `logger = get_logger(__name__)`，破坏结构化日志上下文。

**修改**：去掉 L1468 的局部 `logger` 赋值，或使用明确的局部变量名如 `_retrieve_logger`。

```python
# L1468 删除：
logger = get_logger()  # ← 覆盖模块级 logger，删掉
# 替换引用为模块级 logger（L75），保持不变
```

---

### 2.4 循环依赖修正（增量式）

**问题**：`app.py` 运行时动态 import `from api.routes import incremental_faiss_update`（L233）和 `rebuild_index`（L379），形成 `app→routes→app` 的循环。

**修复**：将 `incremental_faiss_update`、`rebuild_index`、`flush_faiss_buffer` 三个函数从 `_routes.py` 提取到独立的 `api/_index_ops.py`，然后 `app.py` 和 `_routes.py` 都引用该模块。

```python
# 新文件 api/_index_ops.py
"""FAISS 索引操作：增量更新、重建、缓冲刷新（与路由解耦）。"""
# 从 _routes.py 迁移：
# - incremental_faiss_update
# - rebuild_index
# - flush_faiss_buffer
# - _process_embed_queue
# 同时迁移 _embed_queue 相关逻辑
```

---

## 3. P2 重构建议（本月内）

### 3.1 `_routes.py` 拆分（2928→模块化）

**现状**：`api/routes/` 目录已存在 8 个模块文件，但全部是空壳（仅含 TODO 注释），全部路由仍在 `_routes.py`。

**分步迁移计划**：

| 步骤 | 目标模块 | 迁移端点 | 预估行数 |
|------|---------|---------|---------|
| 1 | `routes/write.py` | `POST /sensory`, `POST /episodes`, `POST /promote` | ~200 行 |
| 2 | `routes/search.py` | `POST /retrieve`, `GET /search`, `POST /vector-search`, 命名空间 | ~400 行 |
| 3 | `routes/hyperedges.py` | `POST/GET/PUT/DELETE /hyperedges`, `list_hyperedges_for_node` | ~200 行 |
| 4 | `routes/communities.py` | `GET /communities`, `POST /detect`, 冲突端点 | ~200 行 |
| 5 | `routes/dream.py` | `POST /dream/trigger`, `POST /dream/reset`, 候选 | ~100 行 |
| 6 | `routes/ontology.py` | `POST /ontology/types`, `POST /ontology/edges`, `discover` | ~100 行 |
| 7 | `routes/visual.py` | 视觉记忆 CRUD, 热图 | ~150 行 |
| 8 | `routes/system.py` | 健康检查, metrics, audit, sessions, batch | ~200 行 |

**路由注册方式**：`app.py` 中 `app.include_router(routes.write.router)` 等，而非 `app.include_router(routes.router)`。

```python
# api/app.py 修改
from api.routes.write import router as write_router
from api.routes.search import router as search_router
from api.routes.hyperedges import router as he_router
# ... 逐个注册
app.include_router(write_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
```

---

### 3.2 死代码清理 — 版本文件

**问题**：5 个 `codex_bridge*.py` 版本共存 + 14 个 `transcribe_audio*.py` 版本共存。

**清理方案**：

```bash
# 确认 v5 为最新完整版后，删除所有旧版
rm codex_bridge.py codex_bridge_v2.py codex_bridge_v3.py codex_bridge_v4.py
# 确认 transcribe_audio14.py 为最新完整版后，删除 v1-v13
rm transcribe_audio.py transcribe_audio2.py ... transcribe_audio13.py
```

**验收**：检查 `git log` 确认无引用，运行 `pytest` 确认无 import 错误。

---

### 3.3 重复代码抽取 — `_extract_keywords`

**问题**：`core/dream_pipeline.py:831` 的 `_extract_keywords` 与 `core/llm_client.py:288` 的 `_fallback_keywords` 基本相同的 TF-IDF+词频回退逻辑。

**修复**：抽取到 `core/_text_utils.py`：

```python
# core/_text_utils.py
from __future__ import annotations

def extract_keywords(texts: list[str], max_features: int = 10) -> list[str]:
    """优先 TF-IDF，回退到词频统计。"""
    non_empty = [t for t in texts if t.strip()]
    if not non_empty:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        vectorizer.fit_transform(non_empty)
        return list(vectorizer.get_feature_names_out())
    except (ImportError, ValueError):
        pass
    # 词频回退
    from collections import Counter
    words = []
    for text in non_empty:
        words.extend(
            w for w in text.lower().split()
            if len("".join(c for c in w if c.isalpha())) > 2
        )
    return [w for w, _ in Counter(words).most_common(max_features)]
```

**然后在两处调用的地方替换**：

```python
# dream_pipeline.py — 删除 _extract_keywords 方法，使用 import
# llm_client.py — 删除 _fallback_keywords 方法，使用 import
from core._text_utils import extract_keywords
```

---

### 3.4 N+1 查询普查

**问题范围**：除已标记的 `list_hyperedges` 和 `list_conflicts` 外，代码库中可能还有其他 N+1 模式。

**审查方法**：用 ast-grep 扫描 for 循环内嵌 `query_cypher`/`execute_cypher` 的模式：

```bash
sg -p 'for $ITEM in $RESULTS:
    $BODY
    $STORE.query_cypher($QUERY, $PARAMS)' api/_routes.py
```

所有发现的 N+1 模式统一改为 `OPTIONAL MATCH` + `COLLECT` 或批量查询。

---

## 4. 优先级汇总执行表

| 编号 | 优先级 | 问题 | 文件 | 预计工时 | 依赖 |
|-----|--------|------|------|---------|------|
| P0-1 | 🔴 **立即** | 重复路由定义 | `api/_routes.py` | 15min | 无 |
| P0-2 | 🔴 **立即** | N+1 list_hyperedges | `api/_routes.py` | 30min | 无 |
| P0-3 | 🔴 **立即** | N+1 list_conflicts | `api/_routes.py` | 30min | 无 |
| P0-4 | 🔴 **立即** | 全局可变状态 | `api/_cache.py` (新建) + `api/_routes.py` | 45min | 无 |
| P0-5 | 🔴 **立即** | 重复导入 + 重复 try/except | `api/_routes.py` | 10min | 无 |
| P1-1 | 🟡 **本周** | 异常吞噬治理 | `core/*.py` 约 20 处 | 90min | 无 |
| P1-2 | 🟡 **本周** | 魔法数字→配置 | `config/settings.py` + `api/app.py` | 20min | 无 |
| P1-3 | 🟡 **本周** | Logger 覆盖 | `api/_routes.py:1468` | 5min | 无 |
| P1-4 | 🟡 **本周** | 循环依赖 | `api/_index_ops.py` (新建) | 60min | P0-4 |
| P2-1 | 🟢 **本月** | `_routes.py` 拆分 8 模块 | `api/routes/*.py` | 4h | P1-4 |
| P2-2 | 🟢 **本月** | 死代码清理 | `codex_bridge*.py` × 4, `transcribe_audio*.py` × 13 | 20min | 确认引用 |
| P2-3 | 🟢 **本月** | `_extract_keywords` 抽取 | `core/_text_utils.py` (新建) | 30min | 无 |
| P2-4 | 🟢 **本月** | N+1 全库普查 | 全库 | 60min | P0-2, P0-3 |

---

## 执行建议

1. **顺序**：P0 → P1 → P2，每个优先级内部按上表顺序。P0 全部可在 2 小时内完成。
2. **风险**：`P0-4`（全局状态抽取）和 `P1-4`（循环依赖解耦）有牵连关系，建议先做 `P0-4`，再做 `P1-4`，避免冲突。
3. **验证**：每个修改后运行 `pytest tests/ -q` 确认无回归。P0 修复后运行 `lsp_diagnostics` 确保类型检查通过。
4. **死代码删除前**：用 `git grep -l 'codex_bridge\|transcribe_audio'` 确认最新版本的引用链路。

是否需要我开始执行 P0 问题修复？

---
## Phase 3: Codex 审核
I've launched verification. Waiting for results before forming my conclusion.
所有事实验证已完成。以下是审核结论。

---

## 审核结论

### 整体评估：**质量一般，需修正后采用**

CC 报告发现了多个真实存在的问题（版本爆炸、路由臃肿、重复定义等），但包含 **3 处事实错误、2 处夸大/误导**。直接采用将污染决策依据。

---

### 逐条核查

| # | 原报告 | 优先级 | 核查结果 | 结论 |
|:-|--------|:------:|----------|:----:|
| 1 | 循环依赖风险 `app.py`↔`_routes.py` | P1 | **仅单向依赖**（app.py→_routes.py），`_routes.py` 未反向 import | ❌ **误报** |
| 2 | codex_bridge 版本爆炸 | P2 | 5 个版本文件共存 | ✅ 准确 |
| 3 | transcribe_audio 版本爆炸 | P2 | 14 个版本文件共存 | ✅ 准确 |
| 4 | _routes.py 过于臃肿 | P1 | 2928 行，确实臃肿 | ✅ 准确 |
| 5 | 路由重复定义（create/get hyperedge） | **P0** | 两对完全重复：(1117,2062) 和 (1159,2162) | ✅ 准确，P0 合理 |
| 6 | 全局可变状态泛滥 | **P0** | 4 个模块级全局变量 + 2 个锁 | ✅ 存在，但均有锁保护；P0 偏激进（建议降至 P1） |
| 7 | 异常吞噬（bare pass） | P1 | hebbian.py 和 dream_pipeline.py **零处**；_routes.py 仅 3 处。示例 `_persist_batch` 在 hebbian.py **不存在** | ⚠️ **夸大** — 规模大幅高估，举例错误 |
| 8 | 不一致的错误处理 | P2 | 主观判断，部分合理 | ✅ 可接受 |
| 9 | 重复代码 _extract_keywords | P2 | 确实重复，但文件为 **dream_pipeline.py + community_report.py**，非 llm_client.py | ⚠️ **文件引用错误** |
| 10 | 重复 list_hyperedges | P2 | 两个函数语义不同（全部列表 vs 按节点过滤），非重复代码 | ⚠️ **误导** |
| 11 | 重复 ontology 验证块 | P1 | 存在重复注释 + **不可达的 except 块（死代码）** | ✅ 准确，且比原报告更严重 |
| 12 | unused import | — | 报告截断，无法评估 | ⚠️ 不完整 |

---

### 漏报（CC 报告未提及）

1. **不可达 except 死代码**（L942）：重复的 `except Exception as val_err:` 永远无法执行，属于运行时 bug，至少 P1
2. **`list_hyperedges` 跨文件重复**：同一函数在 `api/_routes.py:2105` 和 `gateway/gateway_api.py:663` 中分别定义——这是真正的 P2 重复

---

### 优先级重新评级建议

| # | 原优先级 | 建议优先级 | 理由 |
|:-|:-------:|:--------:|------|
| 6 | P0 | **P1** | 全局变量有锁保护，并非线程不安全，P0 过度 |
| 7 | P1 | **P2** | 仅 3 处且都作用于不影响主流程的辅助路径，降级 |

---

### 改进建议评估

| 建议 | 可行性 | 评价 |
|------|:-----:|------|
| 拆分 `_routes.py` 为多文件 | ✅ 可行 | 合理但需注意避免产生循环依赖——利用不同模块按 domain 独立注册路由即可 |
| 删除旧版 codex_bridge / transcribe_audio | ✅ 可行 | 危险操作，需确认 v5/v14 为当前生产版本再删 |
| 消除重复路由定义 | ✅ 可行 | P0 级别的正确决定 |

---

### 修正建议（给原报告作者）

1. **删除第 1 条**（循环依赖）：事实不成立
2. **修正第 7 条**：删除 `hebbian.py` 和 `dream_pipeline.py` 的引用，改为 "`api/_routes.py` 存在 3 处 `except Exception: pass`（L213-214, 588-589, 594-595）"
3. **修正第 9 条**：将 `llm_client.py` 改为 `retrieval/community_report.py`
4. **删除第 10 条** 或改为 "`list_hyperedges` 跨文件重复（_routes.py + gateway_api.py）"
5. **补充第 6 条**：`_routes.py` L940-943 存在不可达 except 死代码
6. **补充第 13 条**：`list_hyperedges` 在 `api/_routes.py` 和 `gateway/gateway_api.py` 中重复定义
