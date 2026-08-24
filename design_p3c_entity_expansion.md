# P3c cat1 跨消息多跳增强 — CC 设计任务书

## 背景
P3b HyDE 已发布（v5.52.0，LoCoMo 56.8→60.0%）。超越路径最后一项 P3c：cat1 跨消息增强。当前 LoCoMo cat1 仅 **37.2%**（43 问 16 对），是全类别最低——v5.48 曾达 41.9%，说明还有结构性提升空间。

## cat1 失败模式（数据驱动，已从 locomo10.json 分析）
典型问题（证据跨多个会话 D1-D13）：
- "What activities does Melanie partake in?" — 证据 D5:4, D9:1, D1:12, D1:18（**4 条跨 3 会话**）
- "Where did Caroline move from 4 years ago?" — 证据 D3:13, D4:3（**跨会话 + 时间推断**）
- "What do Melanie's kids like?" — 证据 D6:6, D4:8（跨会话）
- "How many times has Melanie gone to the beach in 2023?" — 证据 D10:8, D6:16（**数量聚合 + 时间**）

**核心模式：实体为中心的跨会话聚合**——单问题围绕 1-2 个实体（Melanie/Caroline 等人名），答案信息分散在多个会话的多条消息。现有向量检索只召回语义最接近的 top-k，跨会话聚合型问题召回不全。

## 现状（已核实的证据，请 read_file/grep 确认）
1. `retrieval/query_router.py`：
   - `_entity_match` L1137：查询 unigram/bigram → OR CONTAINS EpisodeNode.content（**无实体规范化/别名、无跨会话扩展、仅词面匹配**）
   - 补充通道模式（可直接复用范式）：`_community_expansion` / `_mesa_synthesis` / `_visual_recall` / `_property_temporal_retrieve` 都是「补充非替代」——在 `_finish` 闭包内 append → `_deduplicate_and_sort` 单点去重 + boost + 钳制
   - `_finish` L1378 统一出口：所有通道 append 后去重排序
   - `QueryRouterConfig` L200：各通道开关（community_expansion.enabled / mesa.enabled / visual_recall.enabled）
2. `config/defaults.yaml` retrieval 段：各补充通道配置块
3. **中文限制**：GraphLite CONTAINS 中文 b64 无子串保持性 → entity 通道对中文基本失效（P0-3 已知），中文主通道是向量/BM25
4. **时间感知**：cat2 已有 session_ts 注入（`_property_temporal_retrieve`）；EpisodeNode 有 `created_at`

## 设计决策点（CC 逐项拍板，给结论+理由）
1. **实体识别方式**：
   A. 规则：英文专名检测（连续大写词/专名列表启发式）——零成本、评测可用
   B. LLM 实体抽取（梦境已有 NER 可复用）——准但每次检索 +LLM 延迟，与 HyDE 叠加成本
   C. 混合：规则为主 + 可选 LLM 增强
   推荐哪个？注意检索热路径（生产 3s 预算）约束。
2. **跨消息扩展机制**：种子消息（查询实体命中的消息）→ 扩展同实体/同主题的关联消息。选项：
   A. 新增独立补充通道 `_entity_expansion`（仿 community/mesa 范式：append + _finish 去重）——seed 消息按实体聚合，取同实体其他消息
   B. 增强现有 `_entity_match`：匹配后按实体分组 + 组内扩展（改动现有通道）
   C. 图遍历：EpisodeNode → 超边/关系 → 关联节点（需要实体图质量，当前稠密度不足）
   推荐哪个？给出扩展半径（同会话？跨会话？）与数量上限。
3. **实体索引**：是否需要离线实体索引（写入时抽取实体 → 实体→消息倒排），还是查询时动态 OR CONTAINS（现有）？考虑：5882 条评测库查询时 OR CONTAINS 单次 ~0.1-0.3s，长跑降级已知；离线索引质量更好但改动大。
4. **时间感知增强**：cat1 时间型问题（4 years ago / in 2023）是否复用 session_ts 过滤扩展消息（按时间锚排序/截断）？还是只做实体聚合不碰时间？
5. **中文兼容**：生产主场景中文 entity 通道失效——P3c 是否同时做中文实体链接（如中文专名检测 + 向量扩展）？还是本期只做英文（评测验证）+ 中文走向量兜底？
6. **开关与默认**：`entity_expansion.enabled` 默认？考虑到它是纯本地通道（无 LLM）延迟低，是否默认开（对比 hyde 默认关）？

## 约束
- 不碰 core/llm_client.py；不改变现有通道签名（新增补充通道走 _finish 范式）
- 版本 bump v5.53.0（代号 Cross-Message-Expansion）
- 新增测试走公共入口 retrieve()
- 关键假设必须给出接线点证据（read_file/grep 确认调用链）
- 参考 P3b 教训：生产入口 level 接线已通（auto→FUSION 默认），新通道只需 FUSION 生效即可被生产命中

## 输出格式
设计决策表（决策点→结论→理由）+ 关键实现点清单（文件/位置/改动量级）+ 风险与降级路径 + 一句话总结
