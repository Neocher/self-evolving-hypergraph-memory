"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "5.27.3"
__version_info__ = (5, 27, 3)
__version_name__ = "Hermes-Integration-Check"
__release_date__ = "2026-08-12"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v5.27.2 (2026-08-12) Hermes-Integration-Check:
  • check_hermes_integration.py prefetch 标题兼容双格式 (官方 "SHM v5 记忆检索"
    + 旧版 "【SHM 记忆检索结果】") — 解决本机 shm 插件迁移 shm_v5 后检测误报
  • release 标题补齐 (v5.26.0/1/27.0 缺描述性标题, 历史风格 vX.Y.Z 特性摘要)

v5.27.0 (2026-08-12) Dream-Prune-Guard:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心变更 — 梦境 PRUNE 保护 (2026-08-12):
  • 方案①: force_promote=true 写入打顶层 protected 标记 → PRUNE 永不剪
    (2026-08-12 事故: 9 条全孤立旧节点 100% 被剪; 语义保护消除"显式永久保留"被误删)
  • 方案②: 批量剪枝比例护栏 — 单次剪枝 > 50% 活跃节点 → 中止本次剪枝全部保留
    (_MAX_PRUNE_RATIO=0.5 硬编码, 兜底任何未来剪枝逻辑缺陷)
  • 方案③: protected 节点永不参与合并 — 防合并击穿 (普通节点 τ 更高时
    protected 作 loser 被 DETACH DELETE 的语义漏洞)
  • 测试: 新增 8 用例 (protected 保留 / 9/9 中止 / 10% 正常剪 / 5/10 边界放行 /
    合并防护+回归 / 落库布尔闭环 / 无标记默认不保护)

v5.26.1 (2026-08-11) Hermes-Integration-Guide:
核心变更 — 写路径最终收敛 (2026-08-11):
  • dream API apply_candidate 整体闭包入队: PRUNE (DETACH DELETE) + MERGE 循环写
    在写线程执行, 事件循环不再被阻塞; 整体闭包保证 PRUNE→MERGE→_mark_applied
    顺序 (禁止拆开 submit), 队列满(503)时整体未执行 → deferred 语义, 幂等保持
  • dream_scheduler auto_apply 移除: 调度器 _run_dream 不再有 loop 线程同步写
    (原 _persist_community_nodes 几十次 execute_cypher 循环), 完全由
    _dream_poll_loop 经 qsubmit 整体闭包入队承接 (延迟 ≤1 poll interval, 可接受)
  • _persist_dream_state fire-and-forget task SDK 异常兜底: execute_cypher 抛
    ConnectionError/QueryError 不再泄漏 "Task exception was never retrieved"
    噪音, 记 ERROR 日志非致命
  • 语义确认: search 队列忙降级 → 首次 lazy ontology 同步写留在 loop 线程
    (一次性 + 超载瞬态, 可接受); hebbian 大 batch 保持逐条 submit
    (合并会改变 active_nodes 批内共激语义, 不可合并)

v5.24.0 (2026-08-11) Full-Write-Queue:
  • 请求路径写调用全部经 qsubmit 入队: gateway_api (write_sensory /
    store_episode / store_multimodal) + visual + communities (冲突 resolve /
    reconcile update_with_version 复合闭包) + ontology batch_relations
    (6 次 execute_cypher 组闭包) + hyperedges (三分支) — 事件循环不再被同步写阻塞
  • hebbian 持久化入队: _process_embed_queue 改 async, update 闭包经写队列
    (_persist_batch 的 SET/INSERT 在写线程), 5s poll loop 不卡 loop
  • app.py _persist_dream_state 入队: 调度器同步回调 → async 闭包 +
    create_task 桥接 (写线程执行 MATCH+SET/INSERT); 无队列/无 loop 降级同步直调
  • 随主写入队: 写路径 entity_resolver.process / extract_and_relate 闭包入队
    (ALIAS_OF / RELATES_TO 写在写线程); 检索路径首次 lazy 同步预同步入队
  • dream auto_apply 入队 + 队列深度检查 (积压 > max_pending/2 时延迟,
    避免梦境大块写占满单写者额度)
  • 后台/推理路径维持判定: dream 主路径 (asyncio.to_thread) 与
    /index/rebuild (显式运维) 不入队; 检索读验证留 loop (qsubmit 只收写)

v5.23.0 (2026-08-11) Write-Serialized:
  • 写串行化队列 core/write_queue.py — 所有 GraphLite 写调用收敛到专用写线程
    串行执行（queue.Queue + 单 worker executor 桥接 + concurrent Future），
    事件循环不再被同步写阻塞: 8 并发写 3.2s/条 → 排队 + 写, 读请求不受影响
  • 集成: write.py 全部 GraphLite 写调用 (create_episode / ensure_session /
    link_to_session / create_visual_node / execute_cypher / 超边创建) 经
    qsubmit 入队; MATCH 检查+INSERT 幂等对组闭包整体入队 (写线程内原子)
  • 背压: max_pending=100 满则拒新写 (503); 单写等待 30s 超时; 队列满/超时
    由 qsubmit 转 HTTPException 503; 迟到完成语义 (超时后任务仍落库, 不重试)
  • 生命周期: app startup 创建单例, shutdown 先 drain 在途写再关 GraphLite
  • 测试: FIFO 顺序 / 异常传播 / 超时迟到完成 / 队列满拒绝 / 写线程重入直连 /
    读不受写影响 / 8 并发写 80 条吞吐基准

v5.22.0 (2026-08-11) Write-Perf:
  • P0-1: R2 参考 embedding 缓存 (content_hash → embedding, FIFO LRU 512 条)
    —— 同 source 连续写入省 200ms+/条
  • P0-2: defense 锁拆细 — asyncio.Lock → threading.Lock 短临界区 + 专用小池,
    并发写吞吐 0.4 → 数十 req/s; fail-closed 两处 QUARANTINE 原样保留
  • P1-1: 批量接口真正批量化 — 超边创建按 source 合并, 每 source 2 次 MATCH
    (原逐条 2n 次), 批量写均摊 ≤100ms/条
  • P1-2: EpisodeNode (source, created_at) 复合索引 (尽力而为, 失败仅日志)
  • P2-1: 梦境调度写入压力感知 — 持续写入推迟梦境触发, 消除批量写偶发超时

v5.21.12 (2026-08-10) Dream-Fix:
  • 修正：EmbeddingConfig.model_name 默认值 bge-m3→bge-small-zh-v1.5
    （YAML 早已回退，代码默认未同步）；FAISS 无"动态适配"（dimension 恒 512）

v5.21.7 (2026-08-09) CB-Config:
  • 熔断配置注入 (2026-08-09):
  • GraphLiteStore 构造传 cb_config=cfg.circuit_breaker
    (YAML/默认配置不再死配置, 熔断阈值可调)
  • conftest fixture cb_config 透传 + 注入断言测试
  • 测试: 398 passed

v5.21.6 (2026-08-09) Dream-Integrity:
  • 梦境候选孤儿社区 + 共享成员湮灭 + 外部边误删 双路径闭环

  • _persist_community_nodes 重写: 删 DETACH DELETE 全删 + 增量 upsert
    + COMMUNITY_MEMBER 边 (双阶段, 只删自己边) + execute_cypher 写路径
  • dream_pipeline 同源湮灭修复: Phase 1 限定 cid + Phase 3 max_community_by_member
  • F1 共享成员湮灭 + F2 外部边误删 双路径闭环
  • 测试: 398 passed (dream 15/15 + 幂等 + 外部边存活)

v5.21.5 (2026-08-09) Fallback-Fix:
  • LLM fallback 轮转 9→12 (openrouter 不再永不触达)

  • fallback 循环 range 9→12 (4端点×3次), url_idx 覆盖全部端点
  • 最后一个端点 openrouter 不再永不触达 ([3,3,3,0]→[3,3,3,3])
  • 日志计数 3→12 修正 + 4 个轮转测试 (序列/401/403/主端点)
  • 测试: 390 passed 1 skipped

v5.21.4 (2026-08-09) Param-Cleanup:
  • gate_threshold 死参数删除 (LSP 静默断裂修复)

  • EvolvableParams.gate_threshold 纯死参数删除 (从不被 _evolve 调节,
    同步目标 QueryRouter 无此属性) — LSP 静默断裂修复
  • SSM gate 阈值演化不受影响 (dual_gate.adapt_threshold 独立)
  • 测试: 386 passed 1 skipped

v5.21.3 (2026-08-09) P0-Stability:
  • SSM gate learn 闭环 + 5 幽灵方法 + toLower 契约

  • SSM gate learn 闭环: 写路径接线 learn() 正负信号 (outcome-gate_value
    方向) + reward 连续非负 + alpha 上界 clamp + fail-open 容差 ≤1e-9 —
    修复 warmup 后约半数正常写入被静默过滤 (数据丢失)
  • 5 个幽灵方法实现 (graphlite_store): create/get_visual_node,
    get_visual_nodes, get_or_create_session, link_session_member —
    修复视觉记忆/会话关联静默失败 + /memories/visual 500
  • toLower(e.content) CONTAINS 死代码: 4 处删除 + b64 中文限制契约
    文档化 (GraphLite lexer 不支持 UTF-8, 中文 L4 兜底依赖向量/BM25)
  • 测试: 386 passed 1 skipped (真实 GraphLite 集成测试 8 条)

v5.21.2 (2026-08-05) BM25-Harden:
  • BM25 空语料日志降噪 + bm25_build_timeout 双语义拆分
  • Embedding 升级 BAAI/bge-m3 (1024维) + FAISS 维度动态适配

📌 验证:
  • 全量测试 386 passed 1 skipped (21s)
  • g_mlp 正样本学习 0.6164→0.6470 (修复前反降 0.5603)
  • GraphLite 集成: visual roundtrip / session 幂等 / text_search 契约

⚙️ 部署: deploy.sh 一键部署 | systemd shm-server | Docker 就绪"""
