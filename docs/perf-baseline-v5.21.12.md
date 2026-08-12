# SHM 写路径性能基准 (优化前) — 2026-08-11 v5.21.12

实测环境: RTX 2060, bge-small-zh-v1.5 (512d, ONNX), 单机 127.0.0.1:8000
服务状态: 2901 nodes / 1701 hyperedges / faiss 44 / RSS 1446MB

## 写路径
| 场景 | avg | p50 | p95 | max |
|---|---|---|---|---|
| 单条写 (新 source, R2跳过) | 215ms | 212ms | - | 262ms |
| 单条写 (同 source, R2全跑) | 444ms | 411ms | - | 1096ms |
| 批量写 n=10 | 476ms/条 | - | - | - |
| 批量写 n=30 | 732ms/条 | - | - | - |
| 并发写 8线程×20 | 2694ms/条 | 2587ms | 3317ms | 3858ms |
| 并发吞吐 | 0.4 req/s | - | - | - |

## 读路径
| 场景 | avg | p50 | p95 |
|---|---|---|---|
| 向量检索 /search/vector | 14.4ms | 15.0ms | 16.8ms |
| 混合检索 /memories/retrieve | 12.2ms | 9.6ms | 29.5ms |

## 分解
- defense R2 语义漂移: +229ms/条 (同源-新源)
- embedding 热态: 4.2ms/条 | 冷态: 2821ms (首次加载)
- 规模曲线: 连续200条写入 410~450ms 稳定 (无超线性爆炸)

## 复现命令
- 单条/并发: python3 /tmp/perf_bench.py
- R2 分解: python3 /tmp/r2_bench.py
