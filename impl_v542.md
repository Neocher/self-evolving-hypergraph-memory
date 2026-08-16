# SHM v5.42.0 写入加速 Write-Throughput — 实施任务书 v2（CC 审查修正版）

## 背景
写入慢根因（CC 修正——任务书 v1 定位错层）：
1. **真瓶颈 = 同步 defense R2 embed**（defense.py:421-423 `encoder.embed`，每条写必跑，1.4s/条主源）——不在异步 embed 队列（write.py:844 是 poll loop 异步，不造成写延迟）
2. **ONNX 路径事实错误**：encoder.py:293-295 加载 `../data/all-MiniLM-L6-v2-int8`（384d MiniLM），不是 embedding/onnx/
3. **dimension 硬编码 bug**：encoder.py:373-374 `if self._onnx_model is not None: return 384`——bge 512d ONNX 会报 384 → FAISS 维度崩溃（P1 必须修）
4. embed 队列同步 CPU 调用没包 to_thread → flush 期间冻结整个服务（含读路径）

目标：单条写延迟 1.4s → ~0.2s（ONNX 加速同步路径）；批量吞吐 ↑（batch + to_thread）。

## 优先级：B（ONNX）> A（batch）> E（去重）——CC 修正

## 方案 B：ONNX int8（核心，降单条延迟）

### B1. 修 encoder.py 两处（P1）
- **dimension 属性**（L373-374）：`if self._onnx_model is not None: return 384` → **从模型输出读维度**（onnx 输出 shape[-1] 或 PyTorch 等价），否则 512d 崩 FAISS
- **ONNX 路径**（L293-295）：改为指向 `embedding/onnx/`（bge 专用目录），或确认现有 data/all-MiniLM-L6-v2-int8 的语义——**二选一统一**，产物放哪就指向哪

### B2. 导出 bge-small-zh-v1.5 → ONNX int8（512d）
- `optimum-cli export onnx --model BAAI/bge-small-zh-v1.5 --task feature-extraction embedding/onnx/`（或转译到项目现有导出脚本）
- 量化：onnxruntime quantize（dynamic int8，主要量化线性层）
- ⚠️ 模型下载走 hf-mirror（HF_ENDPOINT=https://hf-mirror.com）
- ⚠️ 维度必须 512 与 FAISS 一致
- ⚠️ pooling 一致性：ONNX mean(dim=1) 与 ST 路径 pooling 对比（bge 默认 mean，大概率一致但需实测确认）
- ⚠️ 归一化一致性：ST encode 若 normalize_embeddings=True，ONNX 也要对应（FAISS 用 L2）

### B3. 目录就位后 load() 自动走 ONNX
- defense R2 / ontology / embed 队列的单条 embed 一并加速（同一 TextEncoder 实例）

## 方案 A：batch + to_thread（吞吐补强）

### A1. _process_embed_queue 批量编码 + to_thread（write.py:838-849）
- 收集 batch contents → `await asyncio.to_thread(deps.encoder.embed_batch, contents)`（**必须 to_thread，否则 poll loop 冻结**）
- ONNX batch 分块 32（防 OOM，attention 矩阵峰值估算 >600MB/50 条）
- 循环内用返回矩阵切片，保持隔离跳过/hebbian/FAISS buffer 逻辑
- embed_batch 失败 → 逐条回退 embed()

### A2. hebbian 优先级修正（write.py:853）
- `qsubmit(deps, _run_hebbian_update, ...)` → 加 `priority="normal"`（50 条 flush 不全占 high，让 v5.40 低准入闸生效）

## 方案 E：hash 去重（收口到 embed_batch 内）
- `embed_batch` 内 consult + populate 现有 `_cache`（encoder.py:222-235，原文 key，LRU 512）
- 不要在 write.py 另起 hash 层（避免绕过缓存）

## 测试（~30 行）
- tests/test_embed_batch.py：
  - 批量 vs 逐条 cosine >0.999（**同 encoder 实例**，防混 ONNX/PyTorch）
  - 空/单条/混合长度
  - batch 失败回退逐条
  - **ONNX bge dimension==512**（防维度崩溃）
  - embedding/onnx/ 缺失 → 静默回退 PyTorch 零回归
  - **int8 vs float32 recall@10**（benchmark 集，降幅 <2%）；self-cosine >0.995 仅 sanity
  - 缓存命中不重复编码（spy 调用次数）
- **写入路径单条延迟对比**（v5.41 vs v5.42）——验证「1.4s→0.2s」正确口径
- 回归：写链路 + 检索中文召回；to_thread 后 poll loop 不阻塞（并发读不抖动）

## 关键约束
1. 单写线程铁律（embedding 纯计算，to_thread 不写库）
2. ONNX 失败/不存在 → 静默回退 Tier 3（零回归）
3. FAISS 维度 512 不变
4. 版本五处同步 v5.42.0 "Write-Throughput"
5. 不要 git commit
6. 测试：cd /home/admin/shm && GRAPHLITE_BINDINGS=/home/admin/GraphLite/bindings/python GRAPHLITE_SDK=/home/admin/GraphLite/sdk-python/src .venv/bin/python -m pytest tests/test_embed_batch.py -q 2>&1 | tail -3

## 输出
改了哪些文件/方法、代码摘要、单条延迟对比数据。中文。
