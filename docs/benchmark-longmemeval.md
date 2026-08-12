# SHM × LongMemEval Benchmark (Phase 1)

> 实测结果记录 — 2026-08-10, SHM v5.21.12

## 环境

| 项 | 值 |
|---|---|
| SHM 版本 | v5.21.12 (GraphLite + FAISS + BM25) |
| Embedding | BAAI/bge-small-zh-v1.5, 512 维, device=auto (本机 RTX 2060 → cuda) |
| 服务 | systemd shm-server, 真实库 2579+ 节点 (含真实记忆, 非空库) |
| 数据 | LongMemEval oracle (xiaowu0162/longmemeval-cleaned), 500 实例全集 |
| 评测实例 | 10 (oracle 前 10) |

## 方法

- 脚本: `scripts/bench_longmemeval_shm.py`
- 写入: 逐会话串行灌入 (namespace=bench-longmemeval 隔离, force_promote=true,
  时间戳放 metadata), 避免并发写卡死 (已知坑)
- 检索: `POST /search/vector` (纯向量), top_k=10
- 判定: 答案关键词命中 → recall@k
- 清理: 评测后 `DELETE /memories/namespace/bench-longmemeval` (delete_namespace)

## 结果

```
实例数: 10 | 写入记忆: 243 条 | 耗时: 93s (9.31s/实例)
整体 recall@10: 1.0000 (10/10)
类型                命中率     样本
temporal-reasoning  1.0000     10
```

结果文件: `scripts/longmemeval_shm_results_1786359462.json`

## 对比 (2026-08-07 基线)

| 指标 | 2026-08-07 (bge-m3) | 2026-08-10 (v5.21.12, bge-small) |
|---|---|---|
| recall@10 | 0.90 | **1.00** |
| 耗时 (10 实例) | 1512s (25 分钟) | **93s (16× 提速)** |

提速来源: bge-small-zh-v1.5 CPU/GPU 编码 (8ms/条 vs bge-m3 OOM 降级) +
批量写 API + 写路径优化 (v5.21.9 hyperedge 批量边)。

## 复现

```bash
# 下载数据 (15MB)
mkdir -p /tmp/LongMemEval/data && cd /tmp/LongMemEval/data
curl -sL -C - "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json" -o longmemeval_oracle.json

# 跑评测 (10 实例, 向量检索, 跑完自动清理)
cd ~/self-evolving-hypergraph-memory
python3 scripts/bench_longmemeval_shm.py --limit 10 --endpoint vector --cleanup
```

## 已知限制

- 小样本 (10 实例), 且 oracle 前 10 全部为 temporal-reasoning 类型 —
  单类型、无类型间对比; 全量 500 实例未跑 (写路径历史超线性退化,
  需更大批量写优化后才有意义)
- 单次运行, 无多次均值 ± std
- vector 端点评测 (hybrid 端点在大库下实体匹配 O(n) 仍可能慢路径)
- 数据含英文对话, 中文查询场景另测 (中文检索 3/3 精准, 见测试套件)
