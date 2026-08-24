# AtomicFact 事实级中间层设计（P0-③）

> 2026-08-24 ｜ 基于 CC 理论对照 P0 建议 + v68 教训 + EverOS 参考
> 目标：Episode 拆原子事实（subject-predicate-object/time/evidence）→ dense+sparse 双索引 → 接入 FUSION

## 一、背景与动机

### 理论依据（CC 策划 + 自查验证）
- SHM 记忆粒度 = 消息级 EpisodeNode（叙事整体）——精确问答时检索粒度粗（整条消息进上下文）
- EverOS（93.05）核心 = **AtomicFact 事实级 MaxSim**（dense+sparse 双表 + hybrid RRF）
- 自查确认：SHM 生产代码**零事实级索引**（grep core/retrieval/graph 无 AtomicFact/事实表）

### v68 教训（实证）
- v68 评测层复刻事实级（塞池硬拼）→ **0pp**——根因：bge-small-zh 判别力不足 + 塞池竞争（v64 教训）
- **bge-m3 时代**（1024d 多语言，判别力显著提升）→ 值得重试**生产级**实现

### 设计原则
1. **生产级**（梦境拆事实 + 落库 + 检索通道），非评测脚本 hack（SHM P0-P3 推进原则 08-21）
2. **不塞池硬拼**（v64/v65 教训）——独立通道 + 独立打分 + 与现有 FUSION 通道分权
3. **渐进落地**：先事实抽取 → 落库 → 检索通道 → 评测验证 → 融合调权

## 二、数据模型

### AtomicFactNode（新节点类型）
```
AtomicFactNode {
  elementKey: "fact-{sha1(subject|predicate|object|valid_time)}"
  subject: str        # 主体实体名（对齐 EntityNode）
  predicate: str      # 谓词（对齐 PropertyVerNode 谓词规范）
  object: str         # 客体值
  valid_time: str     # 有效时间（可空；LoCoMo 时间题关键）
  source_episode: str # 来源 EpisodeNode id（证据链）
  confidence: float   # 抽取置信度
  embedded: vector    # 事实向量（dense 索引）
  sparse: dict        # 稀疏特征（BM25 索引，predicate+object 词项）
}
MENTIONS_EPISODE 边: (AtomicFactNode)-[:MENTIONS]->(EpisodeNode)
```

### 抽取（梦境新增步骤，P0-① 扩展）
```
梦境 Step 6.5: _extract_atomic_facts(episode_content)
- LLM 结构化抽取（subject/predicate/object/time）——低 cost prompt（对齐 ontology 抽取）
- 或正则/依存解析 fallback（无 LLM 时）
- 幂等：sha1 主键去重（同事实同版本不重复落库）
- 失败不阻塞（PERSIST degraded 语义）
```

## 三、检索通道（FUSION 扩展）

### 通道设计（独立打分，不塞池）
```
fact_channel:
  query → 拆实体+谓词 → 双索引检索：
    dense: fact_embedding 向量相似度（bge-m3）
    sparse: predicate/object BM25
  hybrid RRF(k=20) → 事实候选 → 经 MENTIONS 边映射回 EpisodeNode
  分数 = 独立 fact_score（不进 FUSION 池，单独排名段）
```

### 融合策略（渐进）
```
Phase A（评测验证）：fact 通道 top-5 与 FUSION top-40 合并（fact 段排后，观察增益）
Phase B（调权）：fact 独立权重 w_fact（0.2 起）与 FUSION 权重并列
Phase C（agentic）：fact 通道接入 agentic 锚点提取（sufficiency 不足时事实补充）
```

## 四、评测验证

### 单变量铁律
- 同脚本同池同 judge：基线 = sufficiency 全量结果（DeepSeek judge）
- 变量 = fact 通道开/关
- 30 问冒烟 → 200 问决定性

### 预期
- 精确属性/单跳题（cat1/cat4 属性类）↑
- 时间题（cat2）↑（valid_time 精确匹配）
- 整体 +1~3pp（保守，EverOS 差距 2-5pp 的一半）

## 五、风险与边界

| 风险 | 缓解 |
|---|---|
| 抽取噪声（LLM 幻觉事实） | confidence 阈值 + 仅固化高置信事实（对齐 property 固化逻辑） |
| 事实冗余（同事实多版本） | sha1 主键 + 版本链（对齐 PropertyVerNode 幂等） |
| 检索重复（事实→episode 映射重复） | MENTIONS 边去重 + existing_ids 排除 |
| 存储膨胀 | 事实节点精简（不存全文，只存三元组） |
| 评测回归 | 单变量冒烟先行，-1pp 即回滚 |

## 六、实施步骤（拆分）

1. **Step 1**：AtomicFactNode schema + 抽取函数（梦境 Step 6.5）
2. **Step 2**：fact 落库 + MENTIONS 边 + 幂等
3. **Step 3**：fact 检索通道（dense+sparse hybrid）接入 FUSION（Phase A 评测模式）
4. **Step 4**：30 问冒烟（fact 开/关单变量）
5. **Step 5**：200 问决定性 + 调权（Phase B/C）
6. **Step 6**：版本 bump v6.4.0 + 发布

## 七、与 P0 ② 评测校准的关系
- 评测校准（strict×3）先跑（等 sufficiency 全量完成）——拿校准基线
- AtomicFact 完成后用同校准口径复测——跨版本可比
