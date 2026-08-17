# P1 设计任务书 — 学习式 Dreaming + MESA（CC 审查）

## 背景

SHM 沿路线图推进：P0-1 v5.47（LoCoMo 69.5%）→ P0-2 v5.48 Agentic-Retrieval（55.3%）→ **P1 学习式 Dreaming+MESA（目标 88-92%）**。

MESA = Memory-Enhanced Search Architecture（记忆增强检索架构）：**Dreaming 产物反哺检索通道，形成"梦境→检索"闭环**。

## 现状（已实测确认，勿重跑）

### Dreaming 管道（core/dream_pipeline.py，1684 行）
- run() 流程：GATHER → CLUSTER → SYNTHESIZE(LLM 摘要) → COMPRESS → PRUNE → RESOLVE → AUDIT
- SYNTHESIZE: `_synthesize_step` (:714) 对社区 `summarize_community`（LLM，节点≥2 且 i<20）
- Ontology 自演化: `_ontology_evolution_step` (:786) LLM 判断 schema 演化
- SSM 巩固: `SSMDreamWrapper` (:93) 社区特征向量多轮 SSM step 调 confidence
- 候选落盘: dream_candidate_store.py，`community_summaries` 含 {id, member_count, member_ids, report, keywords, topics, patterns, contradictions}
- 生产数据实证（最近候选）：report/topics/patterns 真实生成（LLM 摘要已生效），但内容是探针/系统记录（数据形态问题，非代码问题）

### 检索侧（retrieval/query_router.py）
- `_community_expansion` (:1875, v5.41)：种子 → get_communities_by_seeds → BM25(query, summary) relevance ≥0.5 → get_community_members → 成员 append（扩展分 = relevance × min(种子分) × boost 0.6）
- **关键缺口**：社区摘要（report）只用于 relevance 判断扩召回，**摘要本身不进入检索结果**——梦境产出的结构化知识未被直接消费
- 融合通道：vector / bm25 / entity / property_temporal / community_expansion / visual_recall
- `_finish` 统一出口：community_expansion + visual_recall append 后 _deduplicate_and_sort

### 学习闭环现状
- 检索侧已有 self_evolving.py（v5.43）：FailureLogger → DiagnosisEngine(LLM) → ConfigEvolver（回滚保护）
- 梦境侧**无反馈闭环**：聚类质量/巩固效果/摘要价值无评估→调参机制（仅 retrieval_health_probe 单向探针）

## 设计目标

1. **MESA 核心**：梦境社区摘要（report/topics/keywords）作为"合成记忆节点"进入检索候选（level=mesa_synthesis，score 钳制低于原始节点），使"梦境→检索"闭环
2. **学习式 Dreaming**：梦境产物的检索消费反馈（摘要命中率/回报）驱动参数演化（阈值/权重/开关），复用 self_evolving 的 ConfigEvolver 模式（不重造）
3. **零回归**：默认开关关闭（MESA off / 梦境学习 off），打开时严格钳制不干扰主通道

## 审查要求（只做 read_file/grep 静态分析，不实测）

1. 评估 MESA 合成节点进检索的插入点：是否在 `_finish` 内 append（对齐 community_expansion），还是独立通道？
2. 社区摘要的 score 如何计算（不能高于原始节点；参考 community_expansion 的相对尾分缩放模式）
3. 合成节点的去重/生命周期：摘要更新后旧合成节点怎么办？检索结果如何标记（level 字段）？
4. 学习式 Dreaming 的最小可行闭环：什么信号做奖励（摘要被检索命中率？），参数演化范围（阈值/权重/开关），如何防漂移（回滚保护）
5. 复用 self_evolving 的 ConfigEvolver 还是独立实现？（Karpathy：最少代码，禁预留抽象）
6. 检查现有 config/settings.py CommunityExpansionConfig + defaults.yaml community_expansion 段的扩展方式

## 输出格式

- 设计决策表（每项：方案 + 理由 + 涉及文件）
- 推荐实施范围（P0/P1/P2 分级）
- 一句话方案摘要
- 明确标注"关键假设"（调用链证据，不臆测）
