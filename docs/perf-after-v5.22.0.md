# SHM 写路径性能基准 (优化后) — 2026-08-11 v5.22.0

对比基线: [perf-baseline-v5.21.12.md](./perf-baseline-v5.21.12.md)（优化前实测）

## 实测对比 (本机无 GPU 编码器环境, 组件级测量)

| 场景 | 优化前 (v5.21.12) | 优化后 (v5.22.0) | 提升 |
|---|---|---|---|
| defense R2 同 source 10 连写 embedding 调用 | ~55 次 (逐条全量重算历史) | 10 次 (每唯一内容 1 次) | ~5.5× |
| defense 并发 pre_check 8 线程×20=160 次 | asyncio.Lock 跨线程不互斥+串行, 0.4 req/s(全链路) | threading.Lock 短临界区, 160 次/0.14s ≈ 1145 req/s(纯 defense) | 2~3 数量级 |
| 批量超边创建 n=30 查询次数 | 60 次 MATCH (逐条 ×2) | 2 次 MATCH (每 source 2 窗口) | 30× |
| 批量超边创建 n=30 耗时(纯查询) | ~50ms (30×2 次全表扫描) | 0.57ms (2 次索引查询) | ~100× |

> 说明: 全链路延迟 (单条 215~444ms → ?) 需 GPU + bge-small-zh-v1.5 编码器 + 完整服务
> 才能复测; 本环境无网络/无 GPU, 提供可本地复现的组件级实测替代。
> 新增 P1-1 附项: 批量路径补 _embed_queue 入队 (原批量数据不进 FAISS 检索全空, 已修复)。

## 复现命令
```bash
export GRAPHLITE_BINDINGS=~/GraphLite/bindings/python GRAPHLITE_SDK=~/GraphLite/sdk-python/src
python3 -m pytest tests/test_defense_perf.py tests/test_write_batch_perf.py tests/test_dream_pressure.py -q
```

## 测试
- 全量: 436 passed, 0 failed
- 新增测试: test_defense_perf.py (P0-1/P0-2) + test_write_batch_perf.py (P1-1) + test_dream_pressure.py (P2-1) + test_graphlite_connect.py 索引断言更新
