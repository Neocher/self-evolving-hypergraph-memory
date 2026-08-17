# P0-2 Agentic 检索规划 — CC 设计任务书

## 背景
SHM v5.47.0（P0-1 Entity-Property-Time 已完成）复测成绩：LoCoMo 69.5% / PersonaMem 98.3%。
LoCoMo 距 MindMemOS 94.03% 差 24.5pp，主要瓶颈（复测诊断实证）：
1. **时间推理类 QA 失败**（LoCoMo cat=2）：答案需跨消息推理（如 "7 May 2023" = session date_time "8 May" + 消息说 "yesterday"）
2. **跨消息关联失败**（cat=1 identity/attribute）：答案消息在 top-12 之外（rerank 区分度差），需多步检索
3. **单轮检索天花板**：现 FUSION = 向量+BM25+实体 三路并行一轮 → 无多步细化

## 目标
设计 P0-2 **Agentic 检索规划**（多步检索代理）：
- 第一步：理解查询 → 分解子查询/检索意图（时间？属性？身份？事件？）
- 第二步：按意图路由到专用检索通道（现有 L1-L4/FUSION/property_temporal/hypergraph）
- 第三步：多轮细化（首轮结果不足 → 基于证据消息的实体/时间锚点二次检索）
- 第四步：融合 + 重排 + 置信度

## 现有资产（Hermes 实证，可直接复用）
- `_property_temporal_retrieve`（P0-1 新增）：属性时间版本链通道
- `_hypergraph_retrieve`：超图检索（v5/v6 实体超图曾负贡献，需谨慎）
- `_fusion_retrieve`：三路并行（向量+BM25+实体）
- `_community_expansion` / `_visual_recall`：补充召回
- `detect_strategy(query)`：查询策略检测
- `_normalize_query`：中文术语归一化
- `retrieval/self_evolving.py`：FailureLogger→DiagnosisEngine→EvolutionGuard 自演化

## 设计要求
1. **架构**：在 retrieve() 主流程增加 Agentic 规划层（或独立通道），复用现有检索原语，不推倒重来
2. **查询意图分类**：LLM 分类 or 规则分类？权衡（LLM 延迟 vs 规则覆盖面）
3. **多步检索**：首轮 top-12 证据不足时，如何提取锚点（实体/时间/属性）发起第二轮？防止死循环/预算爆炸
4. **时间推理**：session 时间上下文注入（LoCoMo 诊断：date_time + "yesterday" 需推断）
5. **预算控制**：每查询最多 N 步、每步最多 M 个 LLM 调用（三体协奏教训：LLM 批量调用必须保护）
6. **评测闭环**：如何在 LoCoMo 200 问上验证提升（独立脚本，不污染生产）

## 约束
- 只做 read_file/grep 静态分析，不编译不实测
- 聚焦 query_router.py（2230 行）现有结构，最小侵入
- 输出：设计方案（文件/函数/接口级别）+ 关键权衡 + 推荐方案
- 参考 P0-1 教训：设计任务书不嵌重型实测；"无消费方 API = 半死代码"先问消费方

## 输出格式
1. 现状分析（检索链路图）
2. Agentic 层设计（2 种方案对比：LLM 规划器 vs 规则+LLM 混合）
3. 推荐方案 + 理由
4. 实施拆解（Phase 2 给 OpenCode 的最小改动集）
5. 风险与缓解
