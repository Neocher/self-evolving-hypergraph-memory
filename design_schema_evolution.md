任务书已确认。以下为《design_schema_evolution.md》全文。

---

# SHM Schema 自进化 P0-② —— 实体属性与关系的自我进化（设计文档）

> 版本：v1.0 ｜ 范围：EntityNode 属性/关系自进化闭环 ｜ 硬约束遵循：无 LLM、证据分区、OverGraph 无 ALTER 契约、梦境 write_queue、检索可读、公共入口测试

## 1. 模块 / 文件结构

### 1.1 新增文件

**`core/attribute_extractor.py`**（属性提取，纯规则）
```python
@dataclass(slots=True)
class PatternSpec:
    pattern: str                      # 正则（中英双列）
    partition: str                    # 模式组 = 分区键（防刷票的计票单元）
    entity_role: int                  # 0=主语/1=宾语/2=上下文最近实体
    attr_name: str
    local_conf: float                 # 单条证据局部置信度

@dataclass(slots=True)
class ExtractedAttribute:
    entity_id: str
    attr_name: str
    attr_value: str
    partition: str
    evidence_episode_id: str
    evidence_span: str                # 原文片段（溯源）
    local_conf: float

# entity_type -> attr_name -> [PatternSpec]
ATTRIBUTE_PATTERNS: dict[str, dict[str, list[PatternSpec]]]

def extract_attributes(
    episode_id: str,
    episode_content: str,
    entities: Sequence[EntityNode],
) -> list[ExtractedAttribute]: ...
```

**`core/schema_evolver.py`**（演化引擎：计票 + 阈值 + 固化 + 冲突）
```python
@dataclass(slots=True)
class AttrStat:
    attr_name: str
    value: str
    value_blake3: str
    votes: dict[str, int]             # partition -> 票数
    evidence: list[str]               # episode_id 去重列表
    confidence: float

def accumulate_votes(
    sidecar: dict,
    extracted: Sequence[ExtractedAttribute],
) -> dict: ...
    # 纯函数：blake3 证据键去重 → 分区累票 → 重算 confidence

def confidence(
    votes: Mapping[str, int],
    weights: Mapping[str, float] | None = None,
) -> float: ...
    # 分区计票置信度（见 §5）

def decide(
    stat: AttrStat,
    solidified: dict | None,
    *,
    t_solidify: float = 0.60,
    hyst: float = 0.15,
) -> Action: ...   # EMERGE / SOLIDIFY / STRENGTHEN / CORRECT / IGNORE

async def evolve_attributes(
    store: OverGraphStore,
    extracted: Sequence[ExtractedAttribute],
) -> EvolutionReport: ...

async def evolve_relations(
    store: OverGraphStore,
    extracted: Sequence[ExtractedRelation],
) -> EvolutionReport: ...
```

### 1.2 改动文件

**`graph/overgraph_store.py`**（新增 5 个方法，复用已有 `create_property_version`）
```python
async def locked_update_entity_props(
    self, entity_id: str, mutator: Callable[[dict], dict]
) -> dict: ...
    # 锁内读-改-写 EntityNode.props（复用 _locked_upsert_node 整包替换语义）
    # 解决侧车并发写丢失：merge 必须发生在锁内，不能读-写分离

async def create_rel_edge(
    self,
    src_entity_id: str,
    dst_entity_id: str,
    predicate: str,
    confidence: float,
    evidence_episode_ids: list[str],
) -> str: ...
    # label = f"REL_{predicate}"，elementKey = sha1(f"rel:{predicate}:{src}:{dst}")
    # MERGE 幂等；边不可原地更新（_ensure_edge:1630），权重演化只写侧车

async def get_rel_neighbors(
    self, entity_id: str, predicates: list[str] | None = None
) -> list[dict]: ...
    # 1 跳谓词邻居（检索通道 P2 用）

async def get_entity_attributes(self, entity_id: str) -> dict: ...
async def get_entity_relations(self, entity_id: str) -> dict: ...
```

**`core/relation_extractor.py`**（复用 10 谓词 + 新增 4 谓词，详见 §4）

**`core/dream_pipeline.py`**（新增 `_persist_schema_evolution`，接入 PERSIST，详见 §6）

**`retrieval/query_router.py`**（`_entity_expansion`/`_scope_retrieve` 增属性匹配 + 关系邻居，详见 §7）

**`api/routes/ontology.py`**（新增 `POST /ontology/evolve` 等，详见 §8）

---

## 2. 存储 Schema

### 2.1 设计取向

- **演化态（高频读写）存侧车**：EntityNode.props 内嵌 `attrs_json` / `rels_json` 两个 JSON 字段。无 ALTER TABLE、不新增表，全部走 `_locked_upsert_node` 整包替换。
- **固化态（低频、不可原地更新）落边 + 版本节点**：跨阈值后写 `REL_<谓词>` 边（一次性事实标记）+ `PropertyVerNode`（版本化属性值）。边/节点一旦创建不修改，靠 sha1 确定性 key 幂等。
- **证据键**：`blake3(f"{episode_id}:{attr_name}:{value}")` 取前 8 hex 作为 value 键与证据去重键。

### 2.2 EntityNode.props 侧车结构

```jsonc
{
  "type": "Person",
  "name": "张三",
  // ── 属性侧车 ──
  "attrs_json": {
    "title": {
      "candidates": {
        "9f3b2c1a": {                     // blake3(value) 前 8
          "value": "CEO",
          "votes": {                      // 分区计票（禁混池）
            "pattern_title_cn": 3,
            "pattern_title_en": 1
          },
          "evidence": ["ep_sha1_a", "ep_sha1_b"],
          "conf": 0.68,
          "first_seen": "2026-08-01T00:00:00Z",
          "last_seen": "2026-08-20T00:00:00Z"
        }
      },
      "solidified": {
        "value": "CEO",
        "value_blake3": "9f3b2c1a",
        "version": 2,
        "conf": 0.82,
        "pvn_key": "pvn:ent_zhangsan:title:9f3b2c1a:2",
        "active": true
      }
    }
  },
  // ── 关系侧车 ──
  "rels_json": {
    "FOUNDED": {
      "<dst_entity_id>": {
        "target_name": "某某科技",
        "votes": { "pattern_founded_cn": 4 },
        "evidence": ["ep_sha1_x"],
        "conf": 0.72,
        "solidified": true,
        "edge_key": "rel:FOUNDED:ent_zhangsan:ent_moumou"
      }
    }
  }
}
```

### 2.3 边 label 设计

| 元素 | 值 |
|---|---|
| label | `REL_<PREDICATE>`（如 `REL_FOUNDED` / `REL_LEADS` / `REL_WORK_AT` / `REL_LOCATED_IN` / `REL_ACQUIRED`） |
| 方向 | EntityNode(src) → EntityNode(dst) |
| elementKey | `sha1(f"rel:{PREDICATE}:{src_id}:{dst_id}")` |
| 边 props | `{predicate, confidence(固化时快照), evidence_episode_ids}` |

> 边创建后**不原地更新**：后续置信度强化只写 `rels_json`，边作为"已确认事实"标记存在一次。

### 2.4 PropertyVerNode 复用

复用现有 `create_property_version`（幂等，sha1 确定性 key）：

| 字段 | 说明 |
|---|---|
| elementKey | `sha1(f"pvn:{entity_id}:{attr_name}:{value_blake3}:{version}")` |
| props | `{entity_id, attr_name, attr_value, version, conf, evidence_episode_ids, active}` |
| 语义 | 每次修正/强化跨阈值固化一个新 version，旧版本保留作演化历史（无删除、无 ALTER） |

---

## 3. 属性提取规则

纯正则/词典/模板，无 LLM。按 `entity_type` 挂规则表，从 Episode 内容定位**实体锚点**（已知 entity 的 name/别名出现处），再在锚点上下文窗口（±80 字符）内匹配属性模式。

| entity_type | attr_name | 模式示例（中 / 英） | partition | local_conf |
|---|---|---|---|---|
| Person | title | `([\u4e00-\u9fa5]{2,4})，(?:现任|担任|出任)(.+?)(?:[，。]|$)` / `(.+?),?\s+(?:is|as)\s+the?\s+(CEO\|CTO\|COO\|CFO)` | pattern_title_cn / _en | 0.5 |
| Person | company | `(?:就职于\|加入\|任职于)(.+?公司)` / `(?:joined\|works? at) (.+? Inc\.\|Corp\|Ltd)` | pattern_company_cn / _en | 0.5 |
| Organization | industry | `(?:专注\|深耕)(.+?领域)` / `(?:focuses on\|specializes in) (.+?)` | pattern_industry_cn / _en | 0.5 |
| Organization | founded | `成立于(\d{4})年` / `founded in (\d{4})` | pattern_founded_cn / _en | 0.8 |
| Organization | location | `总部位于(.+?)[，。]` / `headquartered in (.+?)[,.]` | pattern_location_cn / _en | 0.6 |

**定位规则**（锚点消歧）：
1. 优先匹配已落库实体 `name`/别名（来自 `community.entity_links`）；同一句出现多个实体时，取**最近邻**锚点。
2. 属性模式命中但锚点缺失 → 丢弃（不臆造实体）。
3. 同一 `(entity, attr, value)` 在单条 Episode 内只取一次（blake3 证据键去重，重跑幂等）。

---

## 4. 关系抽取规则

**复用** `relation_extractor` 现有 10 谓词（FOUNDED / LEADS / ACQUIRED / LOCATED_IN / WORK_AT 等，中英双模式，纯正则）。复用方式：抽取器输出从"裸三元组"升级为 `ExtractedRelation`（带 `partition`/`evidence` 字段），规则本体不动。

```python
@dataclass(slots=True)
class ExtractedRelation:
    src_entity_id: str
    dst_entity_id: str
    predicate: str          # FOUNDED / LEADS / ...
    partition: str          # pattern_xxx_cn/_en
    evidence_episode_id: str
    evidence_span: str
    local_conf: float
```

**新增 4 谓词**（补中英覆盖，仍纯正则）：

| 谓词 | 中 / 英 模式 | partition |
|---|---|---|
| `PARTNER_WITH` | `(?:与\|和)(.+?)(?:达成\|建立)合作` / `partners? with (.+?)` | pattern_partner_cn/_en |
| `SUBSIDIARY_OF` | `(.+?)(?:旗下\|全资子公司)(.+?公司)` / `(.+?) is a subsidiary of (.+?)` | pattern_subsidiary_cn/_en |
| `MEMBER_OF` | `(.+?)(?:加入\|是…成员)(.+?[组织\|协会\|联盟])` / `(.+?) is a member of (.+?)` | pattern_member_cn/_en |
| `COMPETES_WITH` | `(.+?)(?:与\|和)(.+?)(?:竞争\|抗衡)` / `(.+?) competes? with (.+?)` | pattern_compete_cn/_en |

**覆盖约束**：每个新谓词必须同时提供中、英两列模式；`partition` 严格区分语言与模式组，禁止中文票与英文票混池计票。

---

## 5. 置信度累积策略

### 5.1 分区计票公式（防刷票）

分区 = 抽取模式组（`partition` 字段）。两层防刷票：
1. **证据键去重**：`blake3(episode_id:attr:value)` 幂等，同一证据重跑/重放不计二次票。
2. **单分区饱和**：同一分区重复投票收益递减，封顶 `CAP = 5`，杜绝单模式刷票。

```python
CAP = 5
MIN_PARTITIONS = 2

def confidence(votes, weights=None):
    weights = weights or {}
    # 强度分：各分区饱和值的加权平均
    sat = {p: min(n, CAP) / CAP for p, n in votes.items()}
    wsum = sum(weights.get(p, 1.0) for p in sat) or 1.0
    strength = sum(weights.get(p, 1.0) * s for p, s in sat.items()) / wsum
    # 多样性分：独立分区数 / MIN_PARTITIONS（≥2 个独立来源才满分）
    diversity = min(1.0, len(sat) / MIN_PARTITIONS)
    return 0.6 * diversity + 0.4 * strength   # 范围 [0, 1]
```

示例：1 分区 1 票 = 0.38；2 分区各 1 票 = 0.68；1 分区 5 票 = 0.70；2 分区各 5 票 = 1.0。

### 5.2 阈值与演化规则

| 常量 | 值 | 含义 |
|---|---|---|
| `T_EMERGE` | 0.35 | 出现：候选进入 sidecar |
| `T_SOLIDIFY` | 0.60 | 固化：写 PropertyVerNode / REL 边 |
| `HYST`（迟滞带） | 0.15 | 防翻转，见下 |

状态机：

- **出现 EMERGE**：`conf >= T_EMERGE` → 写入 `attrs_json.<attr>.candidates`（或 `rels_json`），不落边。
- **固化 SOLIDIFY**：`conf >= T_SOLIDIFY` 且（无 solidified 值 或 `conf >= solidified.conf + HYST`）→ 新建 `PropertyVerNode(version+1)`；关系则建 `REL_` 边；`solidified` 指向新版本，`active=true`。
- **强化 STRENGTHEN**：已固化值再获证据 → 仅更新 `solidified.conf`（侧车内），不动已创建的边/节点。
- **修正 CORRECT**：冲突值（不同 `value_blake3`）`conf >= solidified.conf + HYST` → 固化新版本（version+1），`solidified` 切到新值，旧版本 `active=false` 保留作历史。
- **忽略 IGNORE**：`conf < T_EMERGE` 或未跨迟滞带 → 仅累票，不改 solidified。

**迟滞带作用**：两个竞争值 A/B 置信度接近时，新值必须**领先 0.15** 才能替换旧值，避免证据抖动导致边/节点反复创建。旧值不物理删除（无 ALTER 契约），靠 `active=false` 在检索层过滤。

---

## 6. 梦境管道接入点

**位置**：`core/dream_pipeline.py` PERSIST 阶段，`_persist_entities`（P0-①，消费 `community.entity_links`）**之后**新增一步。

```python
async def _persist_schema_evolution(self, community: CommunityReport) -> None:
    """PERSIST 阶段：抽取实体属性/关系并排队落库（degraded 语义，失败不阻塞）。"""
    entities = await self._load_persisted_entities(community)      # 读已落库实体
    attrs = extract_attributes(community.episode_id, community.content, entities)
    rels  = extract_relations(community.episode_id, community.content, entities)

    # 走 write_queue，顺序：属性 → 关系（关系端点依赖实体已存在，属性依赖实体已存在）
    for a in attrs:
        self._persist_async(self._write_attr, a)      # 复用 _persist_async
    for r in rels:
        self._persist_async(self._write_rel, r)
```

**write_queue 顺序约束**（同一条目内）：
1. `_persist_entities`（MENTIONS 边，实体落库）→
2. `_write_attr`（属性侧车 + 可能的 PropertyVerNode）→
3. `_write_rel`（关系侧车 + 可能的 REL 边）。

**degraded 自愈**：任一 `_write_*` 失败仅标记 degraded（复用现有降级计数），不 raise、不阻塞梦境；下个梦境周期重放时靠 blake3 证据键 + sha1 elementKey 幂等去重，天然自愈。

---

## 7. 检索通道设计（P1/P2 分级）

| 级别 | 通道 | 接入点 | 作用 | 成本 |
|---|---|---|---|---|
| **P1** | 属性匹配 | `_entity_expansion` | 在按 name 召回实体后，再对**已固化属性**（title/industry/location 等）做 token 级匹配，扩展实体候选集 | 低（读 attrs_json / PropertyVerNode，无额外向量计算） |
| **P2** | 关系邻居召回 | `_scope_retrieve` | 对命中实体走 1 跳 `REL_` 边拉取邻居实体，作为补充召回候选，分数降权 | 中（图遍历 1 跳） |

**P1 细则**：查询词命中的 `attrs_json.<attr>.solidified.value`（`active=true`）→ 该 `entity_id` 进入扩展集，得分按 `solidified.conf` 加权。只读固化值，候选（未固化）不进检索，保证精度。

**P2 细则**：`get_rel_neighbors(entity_id, predicates=None)` 拉 1 跳邻居，邻居得分 = 命中实体分 × `edge.confidence` × `0.5`（降权系数），避免邻居喧宾夺主。P2 默认关，`query` 带 `scope=relations` 或召回不足（hit < k）时启用。

---

## 8. API 端点

| 方法/路径 | 说明 |
|---|---|
| `POST /ontology/evolve` | 触发属性/关系演化（见下） |
| `GET /ontology/entity/{entity_id}/attributes` | 读 `attrs_json` 侧车（候选 + 固化 + 证据） |
| `GET /ontology/entity/{entity_id}/relations` | 读 `rels_json` 侧车 + 已固化 REL 边 |
| `GET /ontology/entity/{entity_id}/relations/neighbors?predicates=FOUNDED,LEADS` | 1 跳谓词邻居 |

```jsonc
// POST /ontology/evolve
{
  "episode_ids": ["ep_sha1_a"] | null,   // null = 全量重放
  "entity_ids":  ["ent_sha1_x"] | null,
  "dry_run": false                        // true 只出报告不落库
}
// → EvolutionReport
{
  "entities_scanned": 12,
  "attrs_extracted": 47,
  "rels_extracted": 9,
  "solidified_attrs": 3,
  "solidified_rels": 2,
  "corrected": 1                          // 冲突修正次数
}
```

---

## 9. 测试计划 + AC 验收标准

**测试走公共入口**（单测禁直调 `_write_*` / `_locked_*` 内部方法，一律经 `evolve_*` / API / `_persist_schema_evolution`）。

| 用例 | 验证点 |
|---|---|
| 属性提取 | 中/英各 1 条样例，`extract_attributes` 正确产出 `(entity, attr, value, partition, evidence)` |
| 关系抽取 | 复用谓词 + 新增 4 谓词，中英双列覆盖，`partition` 不混池 |
| 置信度累积 | 1 分区 1 票不固化；2 分区各 1 票固化；单分区 5 票封顶（conf 不超 0.70） |
| 幂等重跑 | 同一 episode 重放两次，票数/证据不重复累加（blake3 去重） |
| 冲突处理 | 旧值 conf=0.70、新值 conf=0.86（领先 0.16 > HYST）→ 修正切换；领先 0.05 → 不切换（迟滞带） |
| 固化落库 | 跨阈值后 `REL_` 边 / PropertyVerNode 存在且 elementKey 幂等；再跑不新增重复边 |
| degraded | `_write_*` 注入失败 → 梦境不中断，degraded 计数 +1 |
| 检索 | P1 属性匹配召回含属性命中实体；P2 邻居召回含 1 跳邻居且降权 |

**AC（验收标准）**：
1. 无 LLM 调用；属性/关系抽取全部规则化，单条耗时 < 5ms（P50）。
2. 每条固化属性/关系含 ≥1 条 evidence_episode_id，且 `blake3` 证据键重跑幂等（重放结果与首跑一致）。
3. 跨阈值固化走 `REL_` 边 + `PropertyVerNode`，无 ALTER、无原地边更新。
4. 梦境集成走 `write_queue`，任一写入失败不阻塞梦境，degraded 自愈。
5. 检索通道：新增属性/关系可被 `_entity_expansion`/`_scope_retrieve` 利用（AC 检索用例通过）。
6. `pytest` 覆盖率 ≥ 80%，`ruff` + `mypy` 通过。

---

## 10. 一句话总结 + 实施量级估算

> **一句话**：属性/关系走"侧车演化 + 跨阈值固化"——`attrs_json`/`rels_json` 分区计票累积置信度（blake3 幂等 + 单分区封顶防刷票），跨 `T_SOLIDIFY=0.60` 且领先迟滞带 `0.15` 时固化 `REL_<谓词>` 边 + `PropertyVerNode`，由梦境 PERSIST 阶段经 write_queue 落库、检索通道 P1/P2 消费。

**量级估算**：

| 模块 | LOC | 工时(h) |
|---|---|---|
| `core/attribute_extractor.py`（新增） | ~230 | 4 |
| `core/schema_evolver.py`（新增） | ~220 | 4 |
| `core/relation_extractor.py`（改动，+4 谓词） | ~70 | 2 |
| `graph/overgraph_store.py`（改动，+5 方法） | ~160 | 3 |
| `core/dream_pipeline.py`（改动） | ~50 | 1.5 |
| `api/routes/ontology.py`（改动） | ~60 | 1.5 |
| `retrieval/query_router.py`（改动，P1/P2） | ~90 | 2 |
| 测试（`tests/`） | ~350 | 5 |
| **合计** | **~1230** | **~23** |

> 注：阈值 `T_EMERGE/T_SOLIDIFY/HYST` 与 `CAP/MIN_PARTITIONS` 需在测试阶段用样例数据校准一次，工时已含 2h 校准余量。