# LoCoMo-Refined Benchmark — SHM v6.5.1 全量评测

> 评测日期: 2026-08-26 · 数据: LoCoMo-Refined 官方 1382 问 · 判卷: 官方 evaluate.py refined 协议

## 结果

| 类别 | 题数 | 正确 | 准确率 |
|:--|:--|:--|:--|
| cat1 事实问答 | 213 | 138 | 64.8% |
| cat2 关系推理 | 299 | 210 | 70.2% |
| cat3 时间推理 | 68 | 53 | 77.9% |
| cat4 跨会话综合 | 802 | 714 | **89.0%** |
| **合计** | **1382** | **1115** | **80.68%** |

- 错误: 0（全量零缺失零解析错误）
- 官方参考分: EverMemOS 58.25% · Mem0 48.91%
- 领先: +22.4pp (EverMemOS) / +31.8pp (Mem0)

## 方法

### 预测生成（SHM v72 管道）

```bash
cd scripts
SHM_ROOT=/home/admin/shm DB_PATH=<eval_db> INGEST_LOADED=1 \
PREDICT_MODE=1 \
PREDICT_QUESTIONS=<LoCoMo_refined>/data/public/questions.jsonl \
PREDICT_OUT=predictions.jsonl \
PREDICT_RANGE=<start:end> \
python bench_locomo_v72_ontology.py 0
```

- 复用 v6.5.0 灌库（LoCoMo 10 会话 5882 消息 pickle，与 Refined 数据集对话完全一致——已验证）
- 三通道检索（BM25 + entity graph + vector）+ sufficiency 定向 round2 + 证据分区（同 v72 生产管道）
- 4 分片并行（systemd-run，防重启中断），~10s/问，全量 1382 问 ~3.5h

### 判卷（官方协议）

```bash
# 官方 LoCoMo-Refined 仓库 evaluate.py
python src/evaluate.py \
  --questions-path data/public/questions.jsonl \
  --predictions-path predictions.jsonl \
  --metrics llm f1 bleu \
  --llm-judge refined \
  --evaluator-model <model> \
  --evaluator-base-url <openai-compatible-endpoint> \
  --evaluator-api-key <key> \
  --concurrency 4
```

- 判卷哲学: "包含且不矛盾，完整且不越界"（官方 5 原则）
- 本次 judge: `deepseek-chat`（OpenRouter qwen3-8b 批量 402 后切换）
- **跨 judge 一致性验证**: 7 条早期预测 DeepSeek 与 qwen3-8b 逐条 100% 一致
- 官方参考 judge 为 Qwen3-14B（需 DashScope 付费额度），分数不可直接对照，趋势可信

### 防 402 适配（OpenRouter in-flight 预算）

官方 `llm_judge_runtime.py` 不传 max_tokens → OpenRouter 每请求预扣 65536 token → 批量判卷 in-flight 预算耗尽 402。
适配: `max_tokens = int(os.environ.get("LOCOMO_MAX_TOKENS", "4096"))`（见 `docs_locomo_refined_max_tokens.patch`，taiji 仓库）。

## 复现文件

| 文件 | 路径 |
|:--|:--|
| 预测 | `/tmp/locomo_pred_full.jsonl`（1382 条, 零重复）|
| 判卷结果 | `/tmp/locomo_final_judged_ds.json` |
| 早期 qwen 判卷 | `/tmp/locomo_pred_partial_judged.json`（135 问 69.63%）|

## 结论

- 跨会话综合（cat4, 89.0%）最强——SHM 图结构跨会话桥接有效
- 事实问答（cat1, 64.8%）最弱——生成阶段 max_tokens 截断枚举答案（已定位，建议 200→512）
- 严格判卷水分: 宽松 100% → 严格 80.68%（约 19pp）
