# SHM v4.0 修复与升级方案

**版本：** v1.0  
**日期：** 2026-05-10  
**范围：** 基于三Agent审查发现的32个问题，按P0→P1→P2→P3分阶段修复

---

## Phase 0 — 地基修复（P0级，4个问题）

> 目标：恢复核心功能，堵住安全漏洞。每个修复独立可部署。

### 0.1 修复 Cypher 注入（2处）

| 文件 | 行号 | 问题 | 修复方式 |
|------|------|------|---------|
| `api/routes.py` | 339 | f-string拼接用户输入 | 改为参数化查询`$w0..$w4` |
| `retrieval/query_router.py` | 287 | 同上 | 改为参数化查询 |

**代码变更（routes.py:337-341）：**
```python
# 改前：
conditions = " OR ".join(f"e.content CONTAINS '{w}'" for w in words[:5])
cypher = f"MATCH (e:EpisodeNode) WHERE {conditions} RETURN e.id AS node_id, ..."

# 改后：
params = {f"w{i}": w for i, w in enumerate(words[:5])}
conditions = " OR ".join(f"e.content CONTAINS $w{i}" for i in range(len(words[:5])))
cypher = f"MATCH (e:EpisodeNode) WHERE {conditions} RETURN e.id AS node_id, ..."
```

query_router.py 做同样修改。

### 0.2 修复 `get_hyperedges_by_node` 返回 None

**文件：** `graph/kuzu_store.py:350`
```python
# 改前：
if dicts:
    return [_clean_kuzu_row(r) for r in dicts]
# 隐式返回 None

# 改后：
if dicts:
    return [_clean_kuzu_row(r) for r in dicts]
return []  # ← 加这一行
```

### 0.3 给梦境持久化方法加异常日志（5处）

**文件：** `core/dream_pipeline.py`
```python
# 改前（5处 except）：
except Exception:
    pass

# 改后：
except Exception:
    logger.warning("Community persist failed at record level", exc_info=True)
```

涉及方法：`_persist_communities:552`, `_persist_prune:596`, `_persist_merge:616`, `_persist_hyperedges:647`，以及成员边创建 `:574`。

### 0.4 修复 shutdown 释放资源

**文件：** `api/app.py:250-253`
```python
# 改前：
poll_task.cancel()
logger.info("SHM v4.0 shutting down")

# 改后：
poll_task.cancel()
if svc.kuzu_store:
    svc.kuzu_store.close()
logger.info("SHM v4.0 shutting down")
```

kuzu_store.py 的 `close()` 方法已存在（:355），只是从未被调用。

---

## Phase 1 — 核心功能修复（P1级，5个问题）

### 1.1 FAISS 检索错位修复

**根因：** FAISS 存储的是 EpisodeNode 的向量（用 `abs(hash(episode_id))` 做 ID），
但 `query_router.py:175` 把 FAISS 返回的 ID 当成 `hyperedge_id` 去查 `get_hyperedge_members()`。

**方案 A（推荐）— 改查询端：**
在 `query_router.py` 的 L1 检索中，FAISS 搜索后不走 `get_hyperedge_members()`，
而是直接返回 EpisodeNode 作为结果。将 `top_k_hyperedges` 重命名为 `top_k_episodes`。

```python
# query_router.py:163-179 改前：
distances, indices = self.faiss_index.search(query_embedding, self.config.top_k_hyperedges)
hyperedge_scores = list(zip(indices[0], distances[0]))
for he_id, score in hyperedge_scores:
    members = self.kuzu_store.get_hyperedge_members(str(he_id))

# 改后：
distances, indices = self.faiss_index.search(query_embedding, self.config.top_k_episodes)
episode_scores = list(zip(indices[0], distances[0]))
for ep_id, score in episode_scores:
    # 直接查 EpisodeNode（而不是把 ep_id 当 hyperedge_id 去查成员）
    episode = self.kuzu_store.get_episode(str(ep_id))
    if episode:
        results.append({"node_id": ep_id, "content": episode.get("content", ""), "score": float(score)})
```

**方案 B（备选）— 改 FAISS 写入端：**
在 `routes.py:179` 创建 FAISS 索引时，用真正的 Hyperedge UUID 做 ID。
但超边在写入时可能尚未创建，复杂度更高。

### 1.2 实现 `_sensory_buffer` 环形缓冲区

**文件：** `graph/kuzu_store.py` — 在 `__init__` 中增加：

```python
from collections import deque

class KuzuStore:
    def __init__(self, config):
        # ... 现有初始化 ...
        self._sensory_buffer: deque = deque(maxlen=config.sensory_buffer_size or 1000)
```

**配置项** 在 `config/defaults.yaml` 和 `config/settings.py` 中新增 `sensory_buffer_size: 1000`。

**文件：** `api/routes.py:115-123` — 改后代码自动工作（`getattr` 找到真正的 deque，`deque` 自带 `maxlen` 限制无需 `is_full()` 检查）：

```python
# 改前（11行复杂逻辑，buf可能为空）：
buf = getattr(deps.kuzu_store, "_sensory_buffer", None)
buffer_usage = 0
if buf is not None:
    buf.append({"id": record_id, ...})
    buffer_usage = len(buf)
    if hasattr(buf, "is_full") and buf.is_full():
        evicted = buf.evict_oldest()
        ...

# 改后（5行，deque自带maxlen自动挤出）：
buf = deps.kuzu_store._sensory_buffer  # 现在一定存在
buf.append({"id": record_id, ...})
buffer_usage = len(buf)
```

### 1.3 断路器 HALF_OPEN 状态修复

**文件：** `graph/kuzu_store.py`

```python
# 在 record_failure() 方法中增加 HALF_OPEN → OPEN 回退：
def record_failure(self) -> None:
    self._window.append(False)
    if len(self._window) < 2:
        return
    failures = sum(1 for r in self._window if not r)
    error_rate = failures / len(self._window)
    if error_rate > self.config.failure_threshold:
        self.state = CircuitState.OPEN
    # 新增：HALF_OPEN 下任何失败立即回退到 OPEN
    elif self.state == CircuitState.HALF_OPEN:
        self.state = CircuitState.OPEN
```

### 1.4 SSM 门控接入

**方案：** 在 `api/routes.py` 的 `create_episode()` 末尾调用 SSM 门控。
在写入 EpisodeNode 后，调用 `ssm_gate.step()` 更新状态，
并根据 `ssm_gate.should_keep()` 决定该节点是否值得保留。

```python
# routes.py, create_episode 函数末尾（在 return 之前）：
if deps.ssm_gate:
    features = np.array([
        len(str(req.content)),           # 内容长度
        _now() - start,                  # 处理耗时（新鲜度信号）
        float(req.source != "system"),   # 来源权重
    ], dtype=np.float32)
    hidden, gate = deps.ssm_gate.step(features, deps.ssm_gate.hidden_state)
    if not deps.ssm_gate.should_keep(gate, threshold=0.3):
        logger.debug("SSM gate filtered episode", episode_id=episode_id)
```

### 1.5 修复 `promote_to_episode` 数据源

**文件：** `api/routes.py:282-286`

```python
# 改前：从 EpisodeNode 表查 sensory_record_id（永远为空）
existing = deps.kuzu_store.get_episode(req.sensory_record_id)
if existing:
    content = existing.get("content", "")
else:
    content = "promoted_record"

# 改后：从 sensory_buffer 查，或要求客户端提供内容
content = req.content or "promoted_record"
# 可选：尝试从缓冲区恢复
if not content and deps.kuzu_store._sensory_buffer:
    for item in deps.kuzu_store._sensory_buffer:
        if item["id"] == req.sensory_record_id:
            content = item["content"]
            break
```

---

## Phase 2 — 质量加固（P2级，6个问题）

### 2.1 删除死代码

| 文件 | 删除内容 | 行数 |
|------|---------|:----:|
| `retrieval/coarse_to_fine.py` | 整个文件（无人调用） | 207行 |
| `core/ssm_gate.py` 中的`_extract_features` | 从未被调用 | 20行 |
| `core/tau_decay.py` 中的`_tau_cache` | 定义从未使用 | 声明+注释 |

### 2.2 修复 FAISS ID 稳定性

**文件：** `api/routes.py:177`

```python
# 改前（hash在进程间不一致）：
faiss_id = abs(hash(episode_id)) % (2**63)

# 改后（UUID稳定）：
faiss_id = int(uuid.UUID(episode_id).int % (2**63))
```

### 2.3 统一配置类（消除重复定义）

将 `core/tau_decay.py:TauDecayConfig` 和 `core/hebbian.py:HebbianConfig` 删除，
统一使用 `config/settings.py` 中的配置类。在 `app.py` 初始化时直接传 settings 对象。

### 2.4 提取重复代码

| 重复代码 | 位置1 | 位置2 | 提取为 |
|---------|-------|-------|--------|
| TF-IDF关键词提取 | `dream_pipeline.py:362` | `community_report.py:116` | `utils/tfidf.py` |
| Jaccard相似度 | `dream_pipeline.py:653` | —（仅一处） | 可暂时不动 |
| 清理Kuzu行 | `kuzu_store.py:249` | `hyperedge.py:132` | `utils/kuzu_helpers.py` |

### 2.5 FAISS 配置同步

**文件：** `api/app.py:67`

```python
# 改前：忽略 index_type / nlist / nprobe
base_index = faiss.IndexFlatL2(dim)

# 改后：如果配置了 IVFFlat 则使用
if cfg.faiss.index_type == "IVFFlat":
    quantizer = faiss.IndexFlatL2(dim)
    base_index = faiss.IndexIVFFlat(quantizer, dim, cfg.faiss.nlist)
else:
    base_index = faiss.IndexFlatL2(dim)
```

### 2.6 给 `Services` 容器加类型标注

**文件：** `api/routes.py:59-69`

```python
# 改前：
@dataclass
class Services:
    kuzu_store: Any = None
    faiss_index: Any = None

# 改后：
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from graph.kuzu_store import KuzuStore
    from embedding.encoder import TextEncoder
    ...

@dataclass
class Services:
    kuzu_store: Optional['KuzuStore'] = None
    faiss_index: Any = None  # FAISS没有好类型
    tau_engine: Optional['TauDecayEngine'] = None
```

---

## Phase 3 — 架构升级（P3级，5个问题）

### 3.1 添加测试覆盖

最低目标：覆盖 P0 修复 + 核心检索路径

```python
# tests/test_kuzu_store.py
# tests/test_query_router.py
# tests/test_dream_pipeline.py
```

验收标准：`pytest -v --cov=shm --cov-fail-under=20`

### 3.2 梦境同步 I/O 改异步

将 `dream_pipeline.py` 中的 `kuzu_store.query_cypher` 调用改为
`await asyncio.get_event_loop().run_in_executor(None, ...)`。

### 3.3 梦境任务加超时保护

**文件：** `core/dream_scheduler.py:100`

```python
# 改前：
self._running_task = asyncio.create_task(self._run_dream(...))

# 改后：
self._running_task = asyncio.create_task(
    asyncio.wait_for(self._run_dream(...), timeout=300)
)
```

### 3.4 修复 `query` 端点只读拦截可绕过

**文件：** `api/routes.py:743-748` — 改用 Kuzu `MATCH` 正则检查，
或直接拒绝非 `MATCH`/`RETURN` 开头的查询：

```python
upper_q = q.strip().upper()
if not (upper_q.startswith("MATCH") or upper_q.startswith("RETURN") or upper_q.startswith("UNWIND")):
    raise _HE(status_code=400, detail="Only MATCH/RETURN/UNWIND queries allowed")
```

### 3.5 梦境结果接入 FAISS 索引

在 `dream_pipeline.py` 的 `_persist_hyperedges` 方法末尾，
为新创建的 `HyperedgeNode` 生成向量并加入 FAISS 索引。

---

## 实施路线图

```
Phase 0 ─ 快速修复（估计：1天）
  0.1 Cypher注入修复  ← 最紧急，代码变更最小
  0.2 get_hyperedges返回None修复  ← 一行代码
  0.3 梦境异常日志  ← 5处替换，可并行
  0.4 shutdown释放资源  ← 一行代码
  └─ 验收：手动curl测试/query端点无注入、梦境日志可见

Phase 1 ─ 功能修复（估计：2-3天）
  1.1 FAISS检索错位修复  ← 核心修复，三Agent联手
  1.2 sensory_buffer实现  ← 新增类属性+配置项
  1.3 断路器HALF_OPEN修复  ← 3行代码
  1.4 SSM门控接入  ← 涉及新逻辑，需测试
  1.5 promote_to_episode修复  ← 边界情况，影响小
  └─ 验收：检索正确返回、感觉缓冲区正常工作

Phase 2 ─ 质量加固（估计：1-2天）
  2.1 删除死代码  ← 安全，可自动
  2.2 FAISS ID稳定性  ← 无副作用
  2.3 统一配置类  ← 可能影响初始化流程
  2.4 提取重复代码  ← 纯重构
  2.5 FAISS配置同步  ← 配置生效验证
  2.6 Services类型标注  ← 纯重构
  └─ 验收：测试全部通过、类型检查通过

Phase 3 ─ 架构升级（估计：3-5天）
  3.1 添加测试覆盖
  3.2 梦境异步化
  3.3 梦境超时保护
  3.4 查询端只读拦截加固
  3.5 梦境结果可检索
  └─ 验收：覆盖率≥20%、异步改造不退化功能
```

---

## 文件变更清单

| Phase | 文件 | 变更类型 | 估计行变更 |
|:----:|------|:--------:|:---------:|
| P0 | `api/routes.py:339` | 修改 | +3/-1 |
| P0 | `retrieval/query_router.py:287` | 修改 | +3/-1 |
| P0 | `graph/kuzu_store.py:350` | 修改 | +1 |
| P0 | `core/dream_pipeline.py:552,574,596,616,647` | 修改 | +5 |
| P0 | `api/app.py:252` | 修改 | +2 |
| P1 | `retrieval/query_router.py:163-179` | 重写 | ~20行 |
| P1 | `graph/kuzu_store.py:__init__` | 新增 | +3 |
| P1 | `config/settings.py` + `defaults.yaml` | 新增配置 | +2 |
| P1 | `graph/kuzu_store.py:record_failure` | 修改 | +3 |
| P1 | `api/routes.py:create_episode` | 修改 | +12 |
| P1 | `api/routes.py:promote_to_episode` | 修改 | +5/-5 |
| P2 | 删除 `coarse_to_fine.py` | 删除 | -207 |
| P2 | `api/routes.py:177` | 修改 | +1/-1 |
| P2 | `core/tau_decay.py:19` + `core/hebbian.py:19` | 删除类 | -30 |
| P2 | `config/settings.py` | 修改 | ±10 |
| P2 | 新建 `utils/tfidf.py` | 新增 | +40 |
| P2 | `api/routes.py:59-69` | 修改 | +15/-10 |
| P2 | `api/app.py:67` | 修改 | +6/-1 |
| P3 | 新建 `tests/` 目录 | 新增 | +200 |
| P3 | `core/dream_scheduler.py:100` | 修改 | +3 |
| P3 | `api/routes.py:743-748` | 修改 | +4/-2 |

**总计：** 17个文件修改，~500行变更，估计实施时间 **7-11天**
