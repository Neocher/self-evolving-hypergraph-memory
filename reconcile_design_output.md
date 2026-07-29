现在我已完整阅读了所有三个目标文件和相关代码。下面是完整设计方案。

---

## 设计方案：写时消解系统 (Write Reconciliation)

### 1. 问题分析

| 维度 | 现状 | 问题 |
|------|------|------|
| 并发写 | 后写覆盖先写（LWW） | Agent A 和 Agent B 同时写同一实体，A 的修改被静默丢弃 |
| 版本追踪 | 无 | 无法判断"谁基于哪个版本做了修改" |
| 冲突检测 | 仅 Ontology 层有 ConflictNode（语义矛盾） | 写-写冲突（同一字段被不同 agent 改）完全不感知 |
| 已有设施 | MemTX（两阶段提交）、MemClaw（hyperedge supersession_of） | 事务是结构性的，不管冲突；MemClaw 只在超边层，不到实体层 |

### 2. 架构概览

```
┌──────────────────────────────────────────────────┐
│                  API Layer (_routes.py)           │
│  POST /conflicts/reconcile     (新增)             │
│  GET  /conflicts/reconcile/log (新增)             │
│  GET  /conflicts (已有，扩展返回字段)              │
├──────────────────────────────────────────────────┤
│            WriteReconciler (新增)                  │
│  ┌────────────┐ ┌──────────────┐ ┌────────────┐  │
│  │ Conflict   │ │ Strategy     │ │ Conflict   │  │
│  │ Detector   │ │ Resolver     │ │ Logger     │  │
│  │ (OCC)      │ │ (3-way merge)│ │ (auditable)│  │
│  └────────────┘ └──────────────┘ └────────────┘  │
├──────────────────────────────────────────────────┤
│         TransactionManager (扩展)                  │
│  + version_check()   乐观锁版本检查                │
│  + conflict_log      冲突记录 ring buffer         │
├──────────────────────────────────────────────────┤
│            RyuStore (扩展)                         │
│  + WriteConflictNode   新节点表                   │
│  + EpisodeNode.version 版本号列                   │
│  + update_with_version()  OCC 安全写入             │
└──────────────────────────────────────────────────┘
```

### 3. 核心模块：`core/write_reconciler.py`（新增）

#### 3.1 数据结构

```python
class ConflictType(Enum):
    WRITE_WRITE = "write_write"    # 同一字段被两个 agent 并发修改
    SEMANTIC = "semantic"          # 已有：本体语义矛盾
    MERGE_NEEDED = "merge_needed"  # 同实体不同属性分别被修改

class Strategy(Enum):
    LWW = "lww"           # 默认：时间戳大的胜出
    MERGE = "merge"       # 属性级合并：同名取新，异名叠加
    ADDITIVE = "additive" # 实体级叠加：保留所有版本，建 supersession 链

@dataclass
class WriteIntent:
    """一次写入意图 — 消解的最小单元"""
    entity_type: str       # EpisodeNode | HyperedgeNode | etc.
    entity_id: str
    agent_id: str          # 写入方标识
    base_version: int      # 写入方看到的版本号（用于 OCC）
    fields: dict           # 要修改的字段 {field_name: new_value}
    timestamp: float

@dataclass
class ConflictRecord:
    """一次冲突记录"""
    conflict_id: str
    conflict_type: ConflictType
    entity_type: str
    entity_id: str
    intents: list[WriteIntent]       # 冲突的各方写入
    strategy_used: Strategy
    resolution: dict                 # 最终写入的数据
    loser_intents: list[WriteIntent] # 被丢弃/合并的写入
    resolved_at: float
    resolved_by: str                 # "auto" | agent_id
```

#### 3.2 冲突检测器 (ConflictDetector)

```python
class ConflictDetector:
    """
    乐观并发控制 (OCC) 检测器。
    每个实体维护一个单调递增的 version 号。
    写入时携带 base_version，提交时比对当前版本：
      - base_version == current_version → 无冲突，写入成功，version += 1
      - base_version < current_version  → 检测到冲突，进入消解流程
    """
    def check(self, intent: WriteIntent, current_version: int) -> bool:
        """返回 True 表示无冲突"""
        return intent.base_version == current_version

    def find_conflicting_writes(
        self, intent: WriteIntent, entity_history: list[dict]
    ) -> list[dict]:
        """找到 base_version 之后的所有写入记录"""
```

**并发安全**：使用 `threading.Lock` 字典（per entity_id），确保同一实体的 check-and-increment 是原子的：

```python
self._entity_locks: dict[str, threading.Lock] = {}
# 用 defaultdict 模式，key = f"{entity_type}:{entity_id}"
```

#### 3.3 消解策略 (Strategy Resolver)

```python
class StrategyResolver:
    def resolve(self, conflict: ConflictRecord, strategy: Strategy) -> ConflictRecord:
        if strategy == Strategy.LWW:
            return self._resolve_lww(conflict)
        elif strategy == Strategy.MERGE:
            return self._resolve_merge(conflict)
        elif strategy == Strategy.ADDITIVE:
            return self._resolve_additive(conflict)

    def _resolve_lww(self, c: ConflictRecord) -> ConflictRecord:
        """时间戳最大的胜出，其他写入丢弃但记录在 loser_intents"""
        winner = max(c.intents, key=lambda i: i.timestamp)

    def _resolve_merge(self, c: ConflictRecord) -> ConflictRecord:
        """
        属性级三路合并：
        - 同名属性 → 取时间戳最新的
        - 不同名属性 → 全部叠加（两个 agent 改了不同字段，不冲突）
        """

    def _resolve_additive(self, c: ConflictRecord) -> ConflictRecord:
        """
        实体级叠加：
        - 保留所有版本，通过 supersession_of 链关联
        - 最新版本作为主实体，旧版本标记 superseded
        - 类似 Git 的 keep-both 策略
        """
```

#### 3.4 冲突日志 (ConflictLogger)

```python
class ConflictLogger:
    """写入 Kuzu WriteConflictNode + 内存 ring buffer"""
    def log(self, record: ConflictRecord) -> None: ...
    def query(self, entity_id: str = None, limit: int = 50) -> list[ConflictRecord]: ...
    def stats(self) -> dict: ...  # 消解统计
```

### 4. 现有文件修改

#### 4.1 `core/transaction_manager.py` — 最小修改

```diff
+ from core.write_reconciler import WriteReconciler, WriteIntent, Strategy

  class TransactionManager:
      def __init__(self):
          ...
+         self._reconciler: Optional[WriteReconciler] = None
+         self._entity_versions: dict[str, int] = {}  # entity_key → version

+     def set_reconciler(self, reconciler: WriteReconciler) -> None: ...

+     def get_entity_version(self, entity_key: str) -> int:
+         """获取实体的当前版本号（OCC base）"""
+         return self._entity_versions.get(entity_key, 0)

+     def check_and_increment(self, entity_key: str, base_version: int) -> bool:
+         """原子检查并递增。返回 True = 无冲突，已递增"""
```

**不改** `MemoryTransaction.commit()` 的默认行为 — 后写覆盖依旧生效。仅在显式调用 `reconcile()` 时才走冲突检测路径。

#### 4.2 `graph/ryu_store.py` — 最小修改

```diff
  # 新增节点表
+ self.conn.execute(
+     "CREATE NODE TABLE IF NOT EXISTS WriteConflictNode ("
+     "id STRING, conflict_type STRING, entity_type STRING, entity_id STRING, "
+     "agent_ids STRING, strategy STRING, resolution STRING, "
+     "loser_intents STRING, resolved_at DOUBLE, resolved_by STRING, "
+     "PRIMARY KEY (id))"
+ )

  # EpisodeNode 不用 ALTER TABLE ADD COLUMN（因为 RyuGraph 0.11 不支持）
  # 改为在 _init_schema 的 CREATE NODE TABLE 中加 version 列：
  # EpisodeNode: + version INT64 DEFAULT 0

+ def update_episode_with_version(
+     self, episode_id: str, fields: dict,
+     expected_version: int, agent_id: str
+ ) -> bool:
+     """
+     OCC 安全更新。返回 True = 成功，False = 版本冲突。
+     内部用 Cypher WHERE version = expected_version 做 CAS。
+     """
```

**关键 SQL**（CAS 更新）：
```cypher
MATCH (e:EpisodeNode {id: $id})
WHERE e.version = $expected_version
SET e.content = $content, e.version = e.version + 1, e.last_agent = $agent_id
RETURN e.version AS new_version
```
如果返回空 → 版本冲突，返回 False。

#### 4.3 `api/_routes.py` — 新增端点

```python
# ═══════════════════════════════════════════════════════════
# 写时消解 (Write Reconciliation)
# ═══════════════════════════════════════════════════════════

class ReconcileRequest(BaseModel):
    """消解请求"""
    entity_id: str = Field(..., description="目标实体 ID")
    entity_type: str = Field(default="EpisodeNode")
    strategy: str = Field(default="lww", description="lww | merge | additive")
    agent_id: str = Field(default="system", description="发起消解的 agent")
    custom_fields: Optional[dict] = Field(default=None, description="手动合并结果")

class ReconcileResponse(BaseModel):
    conflict_id: str
    strategy_used: str
    winner_agent: str
    loser_agents: list[str]
    resolution: dict
    resolved_at: float

@router.post("/conflicts/reconcile", summary="消解写入冲突")
async def reconcile_conflict(req: ReconcileRequest, deps) -> ReconcileResponse:
    """
    对指定实体的写入冲突执行消解。
    支持三种策略：lww / merge / additive。
    如果 custom_fields 非空，跳过自动策略，直接使用手动结果。
    """

@router.get("/conflicts/reconcile/log", summary="查询消解历史")
async def list_reconcile_log(
    entity_id: Optional[str] = None,
    limit: int = 50,
    deps: Services = Depends(get_services),
) -> dict:
    """查询所有或指定实体的消解历史记录"""

@router.get("/conflicts/reconcile/log/{entity_id}", summary="单实体消解日志")
async def get_entity_reconcile_log(entity_id: str, deps) -> dict:
    """查询指定实体的完整消解链"""

@router.get("/conflicts/reconcile/stats", summary="消解统计")
async def reconcile_stats(deps) -> dict:
    """返回各策略使用次数、冲突率等统计"""
```

### 5. 数据流：并发写入消解全流程

```
Agent A (version=3)           Agent B (version=3)
      │                              │
      │ write_episode(id=X,          │ write_episode(id=X,
      │   content="A's edit",        │   content="B's edit",
      │   base_version=3)            │   base_version=3)
      │                              │
      ▼                              ▼
  ┌──────────────────────────────────────┐
  │  TransactionManager                   │
  │  check_and_increment("EpisodeNode:X") │
  │                                      │
  │  A wins race (CAS succeeds)          │
  │  → version becomes 4                 │
  │                                      │
  │  B loses (WHERE version=3 fails)     │
  │  → ConflictDetector fires            │
  │  → ConflictRecord created:           │
  │      intents: [A_write, B_write]     │
  │      base_version: 3                 │
  └──────────────────────────────────────┘
      │
      ▼ (default: LWW, winner = A)
  ┌──────────────────────────────────────┐
  │  StrategyResolver                     │
  │  → LWW: A wins, B recorded as loser  │
  │  → MERGE: if different fields→both   │
  │  → ADDITIVE: B's write → new entity  │
  │              with supersession_of=A   │
  └──────────────────────────────────────┘
      │
      ▼
  ┌──────────────────────────────────────┐
  │  ConflictLogger                       │
  │  → WriteConflictNode persisted        │
  │  → ring buffer entry                  │
  └──────────────────────────────────────┘
```

### 6. Schema 变更汇总

#### 6.1 RyuGraph 新表

```sql
CREATE NODE TABLE IF NOT EXISTS WriteConflictNode (
    id STRING,
    conflict_type STRING,       -- write_write | semantic | merge_needed
    entity_type STRING,         -- EpisodeNode | HyperedgeNode | ...
    entity_id STRING,
    agent_ids STRING,           -- JSON array of conflicting agent IDs
    strategy STRING,            -- lww | merge | additive
    resolution STRING,          -- JSON: final merged data
    loser_intents STRING,       -- JSON: discarded writes
    resolved_at DOUBLE,
    resolved_by STRING,         -- "auto" | agent_id
    PRIMARY KEY (id)
)
```

#### 6.2 EpisodeNode 列扩展（在 CREATE TABLE 时加入，非 ALTER）

```sql
-- EpisodeNode 新增列:
version INT64 DEFAULT 0,       -- OCC 版本号
last_agent STRING DEFAULT '',  -- 最后写入 agent
```

### 7. 并发安全性分析

| 场景 | 保障机制 |
|------|---------|
| 两个 agent 同时写同一实体 | `threading.Lock` per-entity + CAS WHERE version=N |
| OCC 版本号回绕 | INT64 可支持 2^63 次更新，实际永不回绕 |
| 锁争用 | 按 entity_id 粒度锁定，不同实体完全并行 |
| 死锁 | 单一锁，不存在循环等待 |
| 10 agents × 100 writes 并发 | 每次写入只锁一个实体，冲突率低时吞吐接近无锁 |

### 8. 向后兼容性

| 关注点 | 保证 |
|--------|------|
| 默认写入路径 | **不改** — `create_episode` 等现有方法行为不变 |
| 旧数据 | version=0 作为初始版本，OCC 对旧数据透明 |
| 未启用消解时 | `TransactionManager._reconciler` 为 None，走原 LWW 逻辑 |
| 现有 ConflictNode | 保留不动，WriteConflictNode 是新表，语义不同（前者是语义矛盾检测，后者是写-写冲突追踪）|

### 9. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `core/write_reconciler.py` | **新增** | 核心消解引擎：ConflictDetector + StrategyResolver + ConflictLogger |
| `core/transaction_manager.py` | 扩展 | +version tracking, +reconciler 注入, +per-entity locks |
| `graph/ryu_store.py` | 扩展 | +WriteConflictNode 表, +EpisodeNode version/last_agent 列, +update_with_version() |
| `api/_routes.py` | 新增端点 | POST `/conflicts/reconcile`, GET `/conflicts/reconcile/log`, GET `/conflicts/reconcile/log/{id}`, GET `/conflicts/reconcile/stats` |
| `api/models.py` | 新增模型 | ReconcileRequest, ReconcileResponse, ReconcileLogEntry 等 |
| `api/app.py` | 修改（1行） | init_services 中注入 WriteReconciler |

### 10. 验收标准

1. **默认兼容**：现有 `POST /memories/episodes` 写入行为不变，所有现有测试通过
2. **冲突检测**：两个并发写入同一实体能检测到冲突并记录
3. **三策略**：`lww` / `merge` / `additive` 均能正确执行
4. **日志可查**：`GET /conflicts/reconcile/log` 返回完整消解历史
5. **并发安全**：10 线程 × 100 次写入无数据丢失，冲突记录完整
6. **兼容 RyuGraph 0.11**：不依赖 `ALTER TABLE ADD COLUMN`，所有新列在 `CREATE NODE TABLE IF NOT EXISTS` 时声明

---

是否同意这个设计方案？有任何需要调整的地方请指出，确认后我开始编码实现。