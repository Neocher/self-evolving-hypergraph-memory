# P3b HyDE 生产集成 — CC 设计任务书

## 背景
v5.51.0 P3a reranker 已发布（+0.8pp，收益有限，根因是召回层天花板：top-10 召回率仅 15.8%，排序优化不能创造缺失候选）。P3b = HyDE（Hypothetical Document Embeddings，假设文档增强检索）并入生产 retrieve()。评测脚本已证明 HyDE 有效（recall 7.5%→11.7%，+49%）。

## 目标
让生产 QueryRouter 具备 HyDE 能力：LLM 生成假设文档 → 与原始查询一起参与检索 → 合并去重。**默认行为零回归**（开关关闭路径逐字节等价现状）。

## 现状（已核实证据，请 read_file 确认后再基于证据设计，勿臆测行号）
1. `retrieval/query_router.py`：
   - `retrieve()` 入口 L1345：FUSION 走 `_fusion_retrieve`（L1479）→ `_finish` 统一出口（L1378，含 community/mesa/visual/property 补充 + `_deduplicate_and_sort` + P3a rerank L1405-1409）
   - `QueryRouterConfig` L200 dataclass；config/defaults.yaml `retrieval:` 段：`agentic_enabled: false`（L77 默认关）、`rerank_enabled: true`（L78 默认开）、`rerank_input_k: 40`（L79）
   - P3a `_rerank_results` L2949（排序层增强参考模式：enabled 开关 + `_rerank_failed` 永久降级标记 + 异常静默返回原列表）
   - **生产检索完全无 LLM 调用**（grep api.deepseek/llm_generate 在 retrieval/ 零命中）
2. 评测脚本 `/tmp/bench_locomo_prod.py` L252-273（HyDE 参考实现）：llm_generate(hyde_prompt) 生成假设段落 → embed(question)+embed(hypo) **双路** retrieve(FUSION) → 合并去重（seen_ids）。LLM 通道 = `/tmp/rag_v4_common.py` 的 `llm_generate`（urllib 同步直连 `api.deepseek.com/chat/completions`，无 /v1 前缀，key 从 .env 读取；prompt 模板在 bench_locomo_prod.py L252-256）
3. `core/llm_client.py` LLMClient（async httpx，梦境管道在用）：⚠️ **实测 httpx 直连 api.deepseek.com/chat/completions 返回 Vercel HTML 404（12 attempts 全失败），curl 直连同端点成功**——LLMClient 通道不可靠，P3b 不应依赖它（也不应改它，避免影响梦境管道）

## 设计决策点（CC 逐项拍板，给结论+理由）
1. **接入位置**：HyDE 生成放 retrieve() FUSION 分支前置（在 `_fusion_retrieve` 调用前生成 hypo 并复用现有 query_embedding 参数通道）还是 `_fusion_retrieve` 内部？考虑 `_agentic_retrieve`（L1419 条件分支）是否也应受益
2. **开关语义**：`hyde_enabled` 默认关（仿 agentic，生产高频低延迟、LLM 调用 1-2s 延迟不可接受）vs 默认开（评测直接受益）？给出推荐
3. **双路 vs 单路**：评测脚本是 question+hypo 双路检索合并（召回↑但 2 倍检索成本）；是否支持 `hyde_mode: dual|replace`（replace = 仅用 hypo 向量检索，省一半成本）？给出默认值
4. **LLM 通道**：独立同步 urllib 直连函数（仿 rag_v4_common，显式禁代理 `ProxyHandler({})`、超时 8s、失败返回 None→降级单路）vs 修复 LLMClient httpx（影响面大，不推荐）？给出推荐与放置位置
5. **缓存**：相同 query 的 HyDE 结果是否缓存（如 dict LRU + TTL）？防评测 200 问重复 LLM 调用；给出容量建议
6. **失败降级**：LLM 失败/超时 → 静默降级原始单路（零回归）；是否加永久失败标记（仿 `_rerank_failed`）防每查询都重试烧预算？
7. **评测集成**：bench_locomo_prod.py 改为调用生产 `retrieve(hyde=True)` 替代外部自建 HyDE，验证生产集成等效——评估改动点

## 约束
- 不碰 LLMClient（梦境管道在用，影响面扩散）
- 版本 bump v5.52.0（代号 HyDE-Query-Enhance）
- 新增测试必须走公共入口 retrieve()（禁直调内部方法，防假绿）
- 关键假设必须给出接线点证据（read_file/grep 确认调用链）

## 输出格式
设计决策表（决策点→结论→理由）+ 关键实现点清单（文件/位置/改动量级）+ 风险与降级路径 + 一句话总结
