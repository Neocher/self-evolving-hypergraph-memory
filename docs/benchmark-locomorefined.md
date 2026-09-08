# LoCoMo-Refined Benchmark — SHM v6.19.0 精卫收官评测（官方口径）

> 评测日期: 2026-09-08 · 数据: LoCoMo-Refined 官方 1382 问 · 判卷: **官方 Qwen3-14B refined 协议**（LoCoMo-Refined 官方仓库 evaluate.py）

## 结果

| 类别 | 题数 | 正确 | 准确率 |
|:--|:--|:--|:--|
| cat1 多证据聚合/事实问答 | 213 | 146 | 68.5% |
| cat2 时间/日期推理 | 299 | 244 | 81.6% |
| cat3 推断/偏好 | 68 | 53 | 77.9% |
| cat4 跨会话综合 | 802 | 706 | **88.0%** |
| **合计** | **1382** | **1149** | **83.14%** |

- 错误: 0（1382 全量零缺失零解析错误）
- **> SOTA 82.65%（MemoraX，同官方判卷口径）+0.49pp**
- 官方参考分（同表 re-score）: MemoraX 82.65% · MemOS 63.60% · EverMemOS 58.25% · MemPalace 58.68% · Mem0 48.91%

## 分数史（官方 qwen3-14b refined 口径，全部 1382 题全量）

| 版本 | 日期 | 分数 | 说明 |
|:--|:--|:--|:--|
| v6.10.1 | 2026-09-01 | 67.29% | 精卫基线 |
| v6.13.0 | 2026-09-03 | 72.43% | 数据面恢复（pkl blocks=393） |
| v6.17.0 | 2026-09-05 | 76.48% | qwen3.8-max reader + P0-a 会话作用域 + P0-b 确定性时间层 |
| **v6.19.0** | **2026-09-08** | **83.14%** | **+ R7-3 多模态 caption 证据通道 → 超 SOTA** |

## 方法（v6.19.0 生产管道）

### 证据检索与组织
- **SHM 引擎**: OverGraph EpisodeNode（5882 消息）+ FAISS dense + BM25 + entity graph 三通道融合
- **P0-a 会话作用域**: `QueryRouter.retrieve(scope=conversation_idx)` — 引擎级跨会话隔离（消除 M1 跨会话污染）
- **P0-b 确定性时间层**: 零 LLM 相对时间词 → 绝对区间日历算术（真日历 oracle 单测）
- **R7-3 多模态证据通道**: 图像消息 blip_caption 附加 `[img: ...]` 进证据文本（910/5882 消息）— 修复"图像承载答案"盲区并强化检索召回
- 组织段（ontology_organize）: ENTITY/RELATIONS/FACT TYPES 只陈述事实，无祈使指令
- 4 分片并行 ~5-6h 全量

### Reader
- **qwen3.8-max**（DashScope Token Plan 通道）· 温度 0.0 · enable_thinking=False · prompt V1（不变）

### 判卷（官方协议，不可变）
```bash
python src/evaluate.py \
  --questions-path data/public/questions.jsonl \
  --predictions-path predictions.jsonl \
  --metrics llm f1 bleu \
  --llm-judge refined \
  --evaluator-model qwen3-14b \
  --evaluator-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --concurrency 4
```
- 判卷哲学: "包含且不矛盾，完整且不越界"（官方 5 原则）
- judge 固定 Qwen3-14B refined · 温度 0.0 · thinking disabled

## 复现文件

| 文件 | 路径 |
|:--|:--|
| 评测主日志 | `LoCoMo_refined/results/archive_r6_0908/eval_master_0907_2200.log` |
| 预测 | `archive_r6_0908/predictions_all.jsonl`（1382 条）|
| 判卷结果 | `archive_r6_0908/scored_0907_2200.jsonl` |
| 评测库 | `eval_db_p2`（含 caption 数据面，blocks=393）|

## 结论

- 跨会话综合（cat4, 88.0%）与时间推理（cat2, 81.6%）最强
- 多证据聚合（cat1, 68.5%）最弱 — 结构性难点在"精确集合匹配"（漏项 43% + 追加 30%）
- 全部提升来自 SHM 技术范式（记忆表示/证据组织/检索/时间建模/多模态输入），评测方法全程未变
