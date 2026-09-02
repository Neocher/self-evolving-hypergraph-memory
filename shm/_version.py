"""SHM — 自演化超图记忆系统 版本信息"""

__version__ = "6.13.0"
__version_info__ = (6, 13, 0)
__version_name__ = "DataPlaneRestore"
__release_date__ = "2026-09-03"

VERSION_SUMMARY = f"""SHM v{__version__} ({__version_name__})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v6.13.0 (2026-09-03) DataPlaneRestore:
  • 达摩院 R1A 评测数据面重建 — REBUILD_SHADOW_ONLY=1 在已灌 episode 层
    的评测库 (eval_db_p1) 补跑 LLM 影子段 (压缩记忆块/实体抽取/ontology/
    schema) 落 eval_db*_index.pkl，恢复 0901 口径 blocks=393/entity/
    ontology 组织段（episode 层零写入，修复第0轮 65.05% 的 pkl 缺失消融）
  • bge-reranker CUDA OOM 静默降级显式化 — [RERANK] 标记 + torch.cuda.
    empty_cache（排序语义零改动）
  • CTX_DUMP=1 评测检索 ctx 落盘（每题 jsonl，事后归因用，默认关）

v6.11.0 (2026-09-02) FusionChannelPool:
  • 本机 embedding 强制 CUDA（config device: cuda）— ST bge-m3 ~2.2G 显存
  • device=cuda 时跳过 ONNX（ONNX 为 CPU 版）；auto 解析回 cpu 仍走 ONNX
  • 实测: rebuild 6020 节点 3 分 22 秒 (ONNX CPU 84 分钟+), 检索冒烟命中

v6.10.0 (2026-08-31) FullIndex:
  • 修复: 向量索引长期退化 — 梦境 CommunityNode 占 94.5% 从不参与重建，
    仅 37 个 EpisodeNode 有向量。_rebuild_index_overgraph 现同时索引
    EpisodeNode + CommunityNode（summary 为文本源），batch_upsert
    label-aware，主检索通道按 label 分次查询合并去重（OverGraph
    label_filter 为 AND 语义，多 label 恒空）
  • Hebbian 边仍仅 Episode-Episode（社区节点不建边，语义保留）
  • 回归测试: tests/test_vector_index_degradation.py（5 例）
  • 验证: 重启 rebuild 37 → 6020 节点全量入索引，检索冒烟命中

v6.9.1 (2026-08-31) DepFix:
  • 补 overgraph>=0.17.0 依赖声明 (requirements.txt + pyproject.toml)
  • 修复: overgraph 引擎此前未入依赖清单 → 新环境 clone 后装不上引擎
  • 升级路径: pip install --upgrade overgraph → pytest 全量 → 重启生效

v6.5.0 (2026-08-25) Accuracy-Suite:
  • 方案 D: valid_time 索引 + at_year 过滤 (cat=2 时间推理根治)
  • 方案 B: sufficiency 门控 × Agentic 定向 round2 (缺什么补什么)
  • 方案 A: AtomicFact × PropertyVerNode 三元组联合查询通道
  • 方案 C: RPE 惊奇度信号 → 检索轻重排 (默认关零回归)
  • LoCoMo 200 问 acc=100.0% (四类全对, 基线 87.0% → +13.0pp)
  • tests/test_memory_accuracy.py ×16 记忆体真实行为测试

v6.4.0 (2026-08-24) Fact-Gated-Retrieval:
  • AtomicFact 事实级中间层（P0-③，EverOS 93.05 参考）：AtomicFactNode
    (subject/predicate/object/valid_time/source_episode) + 梦境规则抽取
    (_persist_atomic_facts) + _fact_retrieve 检索通道（实体命中事实文本进
    上下文，独立段 0.85，默认关）；Phase A2 注入纪律（相关性过滤 + 同
    subject+predicate valid_time 最新仲裁）
  • sufficiency 门控（P0 ①）：证据充分跳过实体扩展，不足才全量——修复
    自适应频率门控 -2pp（84.5→86.5）
  • D-MEM RPE 写入门控（多巴胺奖励预测误差借鉴）：surprise=1-max_sim +
    utility 三分流（深度/缓存/忽略），默认关零回归
  • 评测：LoCoMo 200 问 87.0%（174/200，cat1 76.7/cat2 93.7/cat3 84.6/
    cat4 87.7）— v6.1.0 后新高；含 v6.3.3 审计 P0 批量修复
  • health 字段改名 faiss_index_size → vector_index_size（OverGraph HNSW）
  • 测试: 全量门禁 + test_atomic_fact(8) + test_write_gate_rpe(8)
v6.3.3 (2026-08-24) P0-Audit-Fix:
  • P0-1 命名空间隔离 fail-closed + visibility 过滤真实现 (search.py)
  • P0-2 空串哨兵收缩到 new_value 比较点，mutation 不再污染 (overgraph_store.py)
  • P0-3 移除 τ 恒真门卫死代码 (write.py + gateway_api.py)
  • P0-4 自适应衰减接线: register/update_importance/refresh(重置created_at)
  • H1 Hebbian 持久化注入 store + HebbianConfig 补 persist 字段
  • 回归测试 ×4 (namespace_failclosed/sentinel_no_value/tau_wiring/timeout_flag)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v6.3.2 (2026-08-23) Ontology-Governance:
  • 借鉴《本体论增强智能问数》P0：PropertyVerNode 补口径治理字段
    (formula/filter_/owner，可选 None 跳过，向后兼容零迁移)
  • 漂移检测：新模式被本体守卫拒绝 → WARNING 告警触发治理
    (防本体静默腐烂/与数据脱节)
  • Codex 批1审核闭环：P1 unknown_action 漂移日志死代码修复 +
    P2 日志路径文案 + _persist_entities 异常 WARNING/计数修正
  • 评测口径对齐 v6.1.0（关 entity/attr 扩展，LoCoMo 主角恒定噪音）
  • 测试: 全量 1150 passed 全绿
v6.3.1 (2026-08-23) Schema-Entity-Persist-Fix:
  • 修复 v6.2.0 P0-① 生产缺陷：候选模式（candidate_store 非 None）下
    PERSIST 直接模式（PRUNE/MERGE）不跑 → _persist_entities 永不落库 →
    EntityNode=0 → Schema 自演化（P0-②）生产无消费对象（dry_run 恒 0）
  • 修复：dream run() Step 7 候选分支同样经 write_queue 提交幂等的
    _persist_entities + _persist_schema_evolution（sha1 elementKey /
    blake3 证据键幂等只增写）；PRUNE/MERGE 破坏性操作仍经 apply 人工放行
  • 测试: +1（候选模式实体落库断言，全量 1149 passed 全绿）
v6.3.0 (2026-08-23) Schema-Self-Evolution:
  • Schema 自演化 P0-② 实体属性/关系自我进化闭环（CC 设计 → OC/Hermes 实现
    → Codex 两轮审核）— 属性/关系演化态存 EntityNode.props 的 attrs_json/
    rels_json 侧车，跨置信阈值（T_SOLIDIFY=0.60 + 0.15 迟滞带）固化
    PropertyVerNode 版本链 + REL_<谓词> 边
  • 新增 core/attribute_extractor.py — 纯规则属性提取（5 属性 × 中英模式，
    实体锚点 ±80 窗口定位，blake3 证据键幂等）
  • 新增 core/schema_evolver.py — 分区计票置信度（0.6*diversity+0.4*strength，
    单分区封顶 CAP=5，≥2 独立分区高分）+ 五态状态机（EMERGE/SOLIDIFY/
    STRENGTHEN/CORRECT/IGNORE）；证据键含 partition 防单规则刷票
  • relation_extractor +4 谓词（PARTNER_WITH/SUBSIDIARY_OF/MEMBER_OF/
    COMPETES_WITH，中英双列 8 例精确命中，旧谓词零回归）
  • overgraph_store +6 方法 — locked_update_entity_props（锁内 read-modify-
    write 防并发写丢失）、create_rel_edge（predicate label 校验 + 三元组
    幂等）、get_rel_neighbors、get_entity_attributes/relations（出边+入边
    direction=in 镜像）、get_entity_episodes_by_episode
  • 梦境 PERSIST 接入 _persist_schema_evolution（write_queue 顺序：entities
    → schema-evolution；失败聚合异常 → degraded 自愈重放）
  • 检索通道 P1 — _attribute_expansion 属性匹配扩召回（固化值 token 匹配，
    score = max(种子) × 0.85 降权，双路径接线）
  • API — POST /ontology/evolve（dry_run 支持）+ 3 个 GET（attributes /
    relations / neighbors）
  • Codex 审核闭环：批1 2×P1+8×P2+3×P3 全部修复（聚合异常→degraded、
    关系侧车改分区计票、label 校验、入边镜像、负权重回退等）
  • 测试: +20（test_attribute_extractor 4 / test_schema_evolver 6 /
    test_schema_store_ops 6 真实 OverGraph 集成 / test_schema_attribute_
    expansion 5），全量 1142 passed 全绿
v6.2.0 (2026-08-23) Schema-Entity-Persistence:
  • Schema 自演化 P0-① 实体落库闭环 — 梦境 _entity_linking_step 产出的
    entity_links 经 _persist_entities 幂等落库 EntityNode + MENTIONS 边
  • overgraph_store 新增 5 方法：create_entity（sha1 确定性 key 幂等）/
    get_entity / get_entity_by_id / link_entity_to_episode / get_entity_episodes
    （检索候选定位）；失败降级不阻塞（PERSIST degraded 自愈语义）
  • 门禁 1128 passed 全绿（含 write_queue 顺序断言更新）
v6.1.1 (2026-08-23) BGE-M3-Embedding:
  • ONNX SessionOptions 线程优化 — intra=8/inter=1（16 核机器实测
    最优点，单条 embed -28%、批量16 -22%；默认会话 16 线程多进程
    超订风险显式消除），读回 options 验证生效 + 门禁全绿
v6.1.0 (2026-08-23) BGE-M3-Embedding:
  • Embedding 升级 bge-small-zh-v1.5(512d) → BAAI/bge-m3 ONNX O2 CPU
    （EmbeddedLLM/bge-m3-onnx-o2-cpu，MRL 截断 512 保 HNSW 契约）
  • LoCoMo 200 问决定性验证 84.0% → 88.5%（+4.5pp，历史最高）；
    bge-small 全面移除（encoder/测试/embedding/onnx git rm）
v6.0.0 (2026-08-19) OverGraph-Engine:
  • 图引擎迁移阶段1 — 新增 graph/overgraph_store.py OverGraphStore
    （OverGraph Rust/PyO3 0.17.0，Apache-2.0）与 GraphLiteStore 同接口契约
    （39 公开方法 + 4 向量方法），config graph.backend 单开关切换，
    svc.graph_store 属性名保留（duck-typing 上层零改动）
  • 零 b64 — OverGraph 中文原生直写直查（CONTAINS 中文子串直用），
    GraphLite 的 {{b64}} 透明编解码整体移除（读侧 helper 保留兼容遗留库）
  • GQL 白名单翻译层 — SHM 101 处裸 GQL 收敛：INSERT 节点→typed
    upsert_node、INSERT 边→CREATE、逗号 MATCH→重复 MATCH、weight 移入
    SET、RETURN e.*→e、裸 RETURN→合成、LIKE→CONTAINS（PoC 8 轮实证）
  • CAS/超边/时间锚 — 版本号乐观锁（WHERE version SET + mutation_stats）、
    HyperedgeNode+HYPEREDGE_MEMBER 边+weight 属性、created_at 业务 float
    秒属性保真、begin_write_txn 补偿链原子化
  • FAISS 同期替换为 OverGraph HNSW — retrieval/vector_index.py
    VectorIndexAdapter（faiss.Index 鸭子类型）：vector_search(mode=dense)
    + EpisodeNode.dense_vector；uuid5 映射契约保留；R1 PoC 定标 score 恒为
    cosine → d=1/s-1 映射保 1/(1+d) [0,1] 下游；视觉 _visual_index 保留
    FAISS 独立空间（384d，只换主通道）
  • 配置 — GraphConfig.backend/hnsw + OverGraphConfig(database_path) +
    defaults.yaml graph/overgraph 块；backend: graphlite 默认 → 存量零感知
  • 测试: test_overgraph_store（与 GraphLiteStore 行为对拍：create/get/
    query/超边/CAS/中文/翻译层）+ test_overgraph_vector（公共入口 retrieve/
    search/vector；score 方向；faiss_id_map 回填）+ 全量回归
v5.53.0 (2026-08-19) Cross-Message-Expansion:
  • P3c 跨消息多跳增强（cat1）— 实体扩召回补充通道 _entity_expansion：
    查询专名实体（连续大写序列 + _PROPERTY_CANDIDATE_STOPWORDS 停用词过滤
    + normalize_entity_name 规范化，top-3）→ 单条 OR CONTAINS GQL（避免
    N+1）跨会话召回 EpisodeNode append（每实体 max-10、总 ≤20）；扩展分
    = max(种子分) × boost(0.9) 仅低于最高种子
  • 【CC P3c】max 锚 + boost 0.9 同版本修复 — 推翻 R1-P1 的 min 锚契约：
    cat1 聚合场景要求跨会话证据进 LLM 上下文（评测 docs[:40]→rerank
    top-12），min 锚使扩展分 ≈0.25 沉底进不了 top-40；max 锚 + boost 0.9
    使扩展分 ≈0.81 仅低于最高种子，稳进 top-40，由内部/外部 rerank 双兜底
    收敛语义相关性（2026-08-19）
  • CJK 处理 — 先 _extract_proper_nouns 提取 ASCII 专名，纯中文/无专名
    查询返回原 results（中文零回归）；中英混合（"Apple 最近做了什么"）
    提取 apple 走 CONTAINS（v5.31.4+ 中文原生直写，英文词 CONTAINS 可用；
    R1 P0 修复，2026-08-19）
  • 时间锚上界过滤 — now_ts（session 时间锚）非空且 time_filter 开启时
    AND e.created_at <= $at_ts（created_at 为时间戳秒数，int 转换）
  • 双路径接线 — retrieve() _finish（community → mesa → visual →
    property → entity_expansion）+ _agentic_round（防 MESA 历史教训：
    agentic 缺接线导致通道静默失效）
  •     配置 — EntityExpansionConfig（settings.py 注册 + defaults.yaml 块，
    enabled=true/boost=0.9/max_results=10/max_entities=3/time_filter=true）
    + QueryRouterConfig.entity_expansion 嵌套 + api/app.py getattr 条件透传
    （None 不传让默认 factory 生效，旧配置对象零回归）
  • 测试: test_entity_expansion（公共入口 retrieve(level=FUSION)：跨会话
    聚合召回 / 时间锚 GQL spy / CJK 跳过 / enabled=false 零回归 / boost
    钳制）+ 全量回归
v5.52.0 (2026-08-19) HyDE-Query-Enhance:
  • 生产管道集成 HyDE 假设文档增强检索（P3b）— LLM 生成假设段落与原始
    查询一起参与融合检索（评测 recall 7.5%→11.7% +49%，补齐 P3a 排序
    优化无法创造的缺失候选）
  • retrieval/hyde.py — urllib 同步直连 api.deepseek.com/chat/completions
    （无 /v1 前缀、显式禁代理 ProxyHandler({{}})，评测实测 httpx 直连 404、
    curl 直连通）；deepseek-chat（DEEPSEEK_MODEL 可覆盖），max_tokens=150
    temperature=0.3；模块级 OrderedDict LRU（256/TTL 3600s）+ 线程锁
  • 失败降级防每查询重试 — 确定性失败（缺 key/HTTP 401/403）永久跳过标记；
    瞬时失败（超时/5xx/网络错）60s 冷却窗口；任何异常 → None 零回归
  • QueryRouter.retrieve(hyde=...) — 镜像 rerank 参数模式（None → 读
    config.hyde_enabled 默认关，关闭路径逐字节等价）；hyde_mode dual
    （原始+假设双路合并默认）/ replace（仅假设向量单路省一半成本）
  • 配置 — QueryRouterConfig + RetrievalConfig（__post_init__ 校验 mode
    ∈ {{dual, replace}}）+ defaults.yaml hyde_enabled/hyde_mode/hyde_timeout
    （1.5s）+ api/app.py getattr 透传
  • 测试: test_hyde_production（关闭路径等价 / dual 双路 spy / LLM 失败
    单路 / replace 向量替换 / 失败降级单元）+ 全量回归
  • Codex R1 修复轮（同版本补丁）— P0-1 生产 auto→FUSION 接线：
    _level_from_strategy 加 config 参数（rerank/hyDE 任一开启时 auto 进
    FUSION，补齐 P3a 宣称但未落地的三路融合生产生效；REST+GatewayAPI
    双入口解 _qr 取 QueryRouter.config）；P0-2 检索超时预算 helper
    _retrieve_timeout（FUSION+HyDE→5.0s，否则 3.0s）+ hyde_timeout 2.0→1.5
    （QueryRouterConfig/RetrievalConfig/defaults.yaml/app.py 四处接线）；
    P2 QueryRouterConfig.__post_init__ 校验 hyde_mode + HyDE prompt 语言
    一致性约束（中文查询生成中文假设段落）+ 缓存 single-flight（并发同
    query 仅一次 LLM 调用）+ 测试增强（全链路快照对比/中文 prompt/配置
    感知映射/超时预算）
v5.51.0 (2026-08-18) Rerank-Fusion:
  • 生产管道集成 bge-reranker 重排（P3a）— 补齐 vs MindMemOS
    最大差距路径（v5.47 RAG v4 管道含 rerank 时 69.5%）
  • FUSION 统一出口懒加载 CrossEncoder，sigmoid 归一化覆盖 score，
    失败静默降级；配置 rerank_enabled/rerank_input_k 可调
  • 修复生产入口死代码（P0）：/memories/retrieve + A2A/ACP/CLI
    三条协议路径接入 strategy→level 映射（hybrid→FUSION）
  • 修复历史欠账：三路融合（vector+BM25+entity）首次在生产 HTTP
    路径生效；prewarm_reranker 兜冷启动；缓存 key 补 strategy/ns/shared
  • degraded 信号根治：thread-local 降级标志（熔断 open/重试耗尽置位），
    替换 level 前缀猜测；_tag_degraded 防御式守卫
  • 测试: test_query_router_rerank（29+）+ TestP3aR7EntityDegradation +
    全量回归 1114 passed；Codex 审核链 R1-R8 收敛
v5.50.1 (2026-08-18) NullTauFix:
  • 修复 GraphLite 真实引擎 NULL 序列化为字符串 'Null' 导致 float()
    崩溃 — BM25/entity 检索通道静默瘫痪（bm25=0 entity=0，400 次）
  • 新增 _safe_float_tau() 防御解析，替换 query_router.py 5 处
    （BM25 索引构建 / 实体匹配 / L4 fallback）
  • 真实引擎 200 问 fusion 56.0%（baseline 54.0%，+2.0pp），
    三路融合（vector+BM25+entity）恢复
  • 测试: test_null_tau_fix（13 passed，含真实引擎 'Null' 场景）+ 全量回归
v5.50.0 (2026-08-17) Schema-AttrOps:
  • P2 Schema 演化深化 — 属性别名合并 + 中文映射学习的零回归最小闭环
    （CC 设计 D1-D6 通过，P0 范围实施）
  • 存储只读查询 — graphlite_store.get_distinct_attr_names()：
    PropertyVerNode.attr_name distinct 清单（复用 query_cypher 永不抛
    契约，GraphLite 失败 → []），供 LLM 决策别名合并的 canonical 候选
  • ontology_evolution 别名写入 — _apply_attr_ops(parsed, current,
    distinct_attrs) 纯函数：attr_ops 数组（op=merge_alias, canonical,
    aliases）→ extended JSON 顶层 attr_aliases（canonical → alias 列表）；
    守卫 max 1 attr_op/轮 + canonical/alias 非泛词 + canonical ∈
    distinct_attrs（孤儿 alias 无消费方 → skip）；evolve_once 签名加
    distinct_attrs（None → skip attr_ops），_build_prompt 注入属性名清单；
    与类型决策正交可同轮发生，落盘复用 _extended_only + _atomic_write
  • 检索通道内消费 — QueryRouter._expand_attr_aliases(terms, aliases)
    纯函数（term 命中 alias → 追加 canonical，去重保序）；_property_
    temporal_retrieve 在 _extract_property_terms 后、_attr_name_matches
    过滤前插入归一（空表恒等短路）；QueryRouter __init__ 加 attr_aliases
    参数 + app.py 注入 ontology_extended.attr_aliases
  • 零回归 — attr_aliases 为空（默认）时检索逐字节等价
  • 测试: 新增 test_schema_attr_ops（distinct 查询 / _apply_attr_ops 守卫 /
    alias 扩展 / 通道内消费走公共入口）

v5.49.0 (2026-08-17) Mesa-Synthesis:
  • P1 MESA 记忆增强检索 — 检索利用 CommunityNode 摘要合成「梦境产物」节点：
    种子（前 5 结果）→ get_communities_by_seeds → BM25-on-summary 相关度 →
    阈值闸口（0.5）→ 合成节点 append（node_id=community_id 可回溯，
    content=summary）
  • 相对尾分缩放 — 合成分 = relevance × min(种子分) × mesa_boost(0.4)，严格
    低于种子（0.4<1）且低于 community_expansion 成员（0.4<0.6）；fact_track=
    "active" 不误吃 core ×1.1；无 archived 字段（_filter_archived 恒保留）
  • 默认关零回归 — mesa.enabled=False（与 community 默认开不同），关闭时主检索
    字节级等价；GraphLite 失败/开关关闭 → 静默返回原 results
  •     自演化接入 — EvolvableParams.mesa_boost + validate[0,0.59] + _sync_params 同步；
    RetrievalSnapshot.mesa_hit_count/mesa_avg_score 统计；DiagnosisEngine 规则
    （命中多且强→升；零命中→降；中间/弱命中→维持）
  • 配置 — MesaConfig + defaults.yaml mesa 段 + QueryRouterConfig mesa_* 字段 +
    api/app.py QRCfg 透传
  • 测试: 新增 test_mesa_synthesis（默认关零回归 / 合成节点 / score 数学保证 /
    阈值 / max_nodes / 异常降级）

v5.47.1 (2026-08-17) TauDecay-NonAdaptive-Fix:
  • 修复非自适应 τ 衰减钳制回归 — enable_adaptive=False 时 _get_effective_
    tau_decay 直接返回配置的 tau_decay_seconds（恢复 v5.31.3 语义，完全尊重
    配置），不再被 tau_decay_min=300 / tau_decay_max=7200 钳制；双轨特性
    fact_track=="core" 的 ×2.0 IMP_BOOST_FACTOR 保留，自适应分支
    （enable_adaptive=True）钳制逻辑不变
  • 回归来源 — v5.34.0 双轨事实特性把 min/max 钳制错误带入非自适应分支，
    导致 tau_decay_seconds=100 配置实际生效 300s，test_decay_threshold_candidate
    边界断言失败（基线 932 passed + 1 failed）
  • 验证: TestTauDecay 8 passed + 全量 933 passed 0 failed

v5.48.0 (2026-08-17) Agentic-Retrieval:
  • P0-2 Agentic 检索规划 — 多步锚点检索 + session 时间锚注入（CC 设计 →
    OpenCode 实施 → Codex R1-R5 五轮复核闭环）
  • session_ts 参数注入 — 相对时间词（yesterday/last year/today）按 session
    时间锚而非墙钟解析（None 回落 time.time() 向后兼容）；全链路透传
    （REST /memories/retrieve、GatewayAPI、self_evolving、MCP v1+v2、
    A2A/ACP）；_relative_time_at_ts/_property_time_mode/_apply_time_decay
    now 下沉为参数
  • _agentic_retrieve 多步编排（agentic_enabled 默认关，第 1 轮 = 现有
    FUSION 全路径字节级等价）— 意图分类（time/identity/attribute/event/
    multi_hop）→ 通道路由 → 证据不足才 refine → 锚点提取（实体+属性+时间）
    发起第二轮；三重防死循环（seen 节点去重 / max_steps=3 硬上限 /
    min_new 锚点枯竭提前停）
  • 规则原语 — _classify_intent/_route_channels/_sufficiency_check/
    _extract_anchors/_channels_from_anchors；英文属性词表 + 词边界匹配
    （\\b，market_cap ↔ market cap 归一）；撇号收缩还原表
    （don't/let's/ain't/o'clock/y'all → 停用词，防伪实体）
  • 测试: test_agentic_retrieve 30 + test_mcp_session_ts 3 + R2-R4 回归
    （全量 987 passed，+38 新增零回归）

v5.47.0 (2026-08-16) Entity-Property-Time:
  • P0-1 实体-属性-时间三维建模 — 补齐与 MindMemOS 差距三大结构性缺失之一：
    实体中心 + 属性时间版本链（消息级扁平存储 → 属性可追溯），最小正解
  • 存储 — 新增 PropertyVerNode 节点 {{id, entity_id, attr_name, value, valid_from,
    expired_at}}；更新建新版本，旧版本 expired_at 打标 + (old)-[:SUPERSEDES]->(new)
    血统边（复用 archive_node 双 MATCH + INSERT 范式；GraphLite 多语句只执行
    第一条 → SET expired_at 与 INSERT 边拆独立 execute）
  • 写入 — entity_resolver 编排 + graphlite_store 原语：RelationTriple.attributes
    （ACQUIRED 金额等，正则确定性零 LLM）→ attr_name 派生 {{relation}}_{{key}}；
    无 MERGE → 查存在→插 两段式幂等（同值 no-op）；valid_from 严格单调
    （同微秒/时钟回拨 → 旧版 + 1ms，排序稳定）
  • 版本约束 — 每 (entity_id, attr_name) 保留最近 N=8 版，写时惰性裁剪
    （超限 DETACH DELETE 最旧；不复用 tau 衰减）
  • schema — AttributeDef 末尾追加 temporal: bool = False（纯 dataclass 默认值，
    向后兼容零迁移脚本）
  • 检索 — 新增 _property_temporal_retrieve 通道（仿 _community_expansion append
    模式，_finish 去重/排序前追加）：候选实体（英文大写词序列 + 中文组织名 +
    停用词过滤）→ PropertyVerNode；时间意图复用 _time_keywords：最近/现在 →
    最新未过期版；含年份 → 该时点前最新版；无时间词 → 全部未过期版；扩展分 =
    min(种子分) × 0.6（相对尾分缩放严格低于种子）；无 archived 字段恒保留
  • 测试: test_property_temporal 14 用例（版本创建 / supersedes 链 / 时间检索
    最近与具体年份 / N=8 裁剪 / 向后兼容 / HTTP 全链路 / 静默降级）

v5.46.2 (2026-08-16) V-Mem-R2-Fixes:
  • P2-a V-Mem Codex R2 缺陷修复（2 🟡 P2 + 2 ⚪ P3，判定"需修改"）
  • 🟡 P2-1 视觉索引与 id_map/meta 非原子交换 — 新增 _visual_lock +
    _visual_snapshot()（读侧锁内一次性快照 (index, id_map, meta) 一致三元组）；
    add_visual_node「查重 → index.add → 三字典 swap」与 prewarm「合并 + 构建 +
    swap」均持锁原子完成 → 杜绝「新 index + 旧 map」fid 错配（重建期）、
    并发 add fid 碰撞与「新节点已入 index 未入 map」漏召回；_visual_recall
    全程用快照，空通道改以快照 id_map 判断（count 锁外读会随并发漂移）
  • 🟡 P2-2 写队列超时路径不更新视觉索引 — qsubmit_visual_index 统一收口
    （visual.py / write.py / gateway_api 三处 hook）：成功与超时路径都补索引
    （超时 = 任务已入队将迟到完成、DB 仍会落库，防「节点入库但不可检索」），
    队列满/关闭（未入队、DB 未落库）不补索引（防幽灵节点）；识别经
    HTTPException.__cause__ 区分 TimeoutError 与拒绝异常
  • ⚪ P3-1 存量 512d VisualNode 未迁移 — prewarm 跳过日志 + docstring 标注
    known-limitation（两空间 bge vs CLIP 本质不可比；生产 VisualNode=0 无存量）
  • ⚪ P3-2 VERSION_SUMMARY 测试计数修正 — 全量实跑 895 passed +
    1 pre-existing flaky（test_decay_threshold_candidate 基线同挂）+ 1 skipped
  • 测试: test_visual_recall 新增 5 用例（并发快照 2：并发 add 无 fid 碰撞 /
    snapshot 不暴露中间态；超时路径 3：超时 503 但补索引 / 队列满 / 关闭不产生
    幽灵索引），test_visual_recall + vector_search + retrieve_routes 53 passed，
    全量 895 passed 1 pre-existing flaky 1 skipped

v5.46.1 (2026-08-16) V-Mem-R1-Fixes:
  • P2-a V-Mem Codex R1 缺陷修复（2 🟠 P1 + 2 🟡 P2，判定"需修改"）
  • 🟠 P1-1 视觉索引只在启动 prewarm_visual 构建一次，写入永不入索引 —
    QueryRouter.add_visual_node() 写路径增量入索引（384d 同空间校验 +
    幂等查重 + 索引未构建时惰性引导）；visual.py / write.py 多模态 /
    gateway_api 三处 create_visual_node 后统一 hook（SelfEvolvingRetrieval
    包装层 _qr 解包透传）；prewarm 重建时合并 _visual_vecs 增量节点，
    防 DB 快照覆盖并发写入
  • 🟠 P1-2 /memories/visual 写路径落 512d bge 文本向量被 prewarm 跳过 —
    方案 Y（两空间 bge vs CLIP 本质不可比，按任务指令选语义一致路径）：
    写路径改 CLIP 文本 512d @ seed42 投影 → 384d 落库，与 multimodal
    图像路径/检索 query 同处 CLIP 投影空间，节点立即可检索
  • 🟡 P2-1 首次检索懒加载 CLIP 拖垮 3s 检索预算 — prewarm 构建成功后
    to_thread 预热 CLIP（30s 超时静默降级）；_visual_recall 冷启动守卫：
    真实 ClipEmbedder._model is None → 跳过视觉通道（绝不触发模型加载）
  • 🟡 P2-2 测试非真实 CLIP — 修正恒真断言（空通道不得挂载 CLIP）+
    新增增量索引/512d 写路径集成/CLIP 冷启动隔离用例；真实 ClipEmbedder
    冒烟（模型已缓存时启用，离线 skip）
  • 测试: test_visual_recall 21 用例 + write_queue_v524 适配，全量
    884 passed 1 pre-existing flaky（test_decay_threshold_candidate 基线同挂）

v5.46.0 (2026-08-16) V-Mem-Modal-Route:
  • P2-a V-Mem 模态路由检索 — 多模态写侧（ClipEmbedder/WhisperEmbedder/
    MediaStore/VisualNode）已完备但检索侧零消费，VisualNode 无法被
    /memories/retrieve 召回；本次补视觉检索通道（CC 设计审查通过，方案 A）
  • QueryRouter._visual_recall — 补充非替代：CLIP 512d 文本 query →
    共享写路径 512→384 投影（seed 42 列归一，与 write.py 逐元素一致）→
    384d VisualVectorStore 检索 VisualNode → 相对尾分缩放（1/(1+dist) ×
    min(种子分) × boost 0.6，严格低于文本种子）→ append modality="visual"
  • prewarm_visual — 启动异步建索引：GQL 拉取 VisualNode（LIMIT
    visual_limit）→ JSON 字符串 embedding 解析 → 非 384d 防御性跳过；
    全程 try/except 静默降级；空通道短路（无 GQL、无 CLIP）
  • 空库零开销 — _visual_index None/ntotal==0 → 直接返回，检索路径
    不碰 GraphLite 视觉表、不惰性创建 CLIP
  • 接线 — QueryRouter.services 参数（共享写路径 CLIP/投影）+
    app.py lifespan prewarm_visual + EpisodicResult.modality 字段
    （向后兼容，纯增量）
  • 测试: test_visual_recall 14 用例（端到端 HTTP modality=visual /
    文本零回归 / 空通道短路 / CLIP 降级 / 投影一致性 seed 42 /
    embedding JSON 解析），全量 884 passed 1 pre-existing flaky

v5.45.0 (2026-08-16) Poison-Guard:
  • P2-b MAPLE-Guard 内容级投毒检测 (R6) Codex 审核修复
    （2 🟠 P1 + 4 🟡 P2 + 2 ⚪ P3，判定"需修改"）
  • 🟠 P1-1 GatewayAPI 检索路径补 scan_content → MCP/A2A/ACP 外部
    agent 拿到真实 risk_level（修复前恒 None，与 search.py 同模式，
    去重后循环 fail-open 标记）
  • 🟠 P1-2 critical 正则误报收紧 — 裸 "system prompt" / "你是一个
    管理员" / "I am now an assistant" 不再误判 critical（默认
    silent=True 下误隔离=良记忆数据损失）；需命令/覆盖语境
    （ignore/override/现在/从此/不再是 等）才触发；新增 3 负例回归
  • 🟡 P2-1 R6 写端点级接线测试 — TestClient POST /memories/episodes
    注入内容 → quarantine 标记 / r6_enabled=False 正常写入
  • 🟡 P2-2 DefenseConfig 双定义消除 — settings.py 删除本地类 import
    复用 core.defense 版（同 v5.44.1 OntologyConfig 方案 A），
    app.py 逐字段手工映射删除
  • 🟡 P2-3 对抗输入性能用例 — 长 ignore 链 / 长 URL 无 TLD /
    长零宽字符串，断言 < 100ms 防灾难性回溯
  • ⚪ P3-1 控制字符阈值注释自洽（实现为严格大于 8，注释与实现一致）
  • ⚪ P3-2 defense.py pre_check docstring 5→6 条规则
  • 测试: test_content_guard + test_defense_perf + test_write_routes
    全绿 + py_compile

v5.44.1 (2026-08-16) Ontology-Config-Drift-Fix:
  • 修复 OntologyConfig 双定义漂移（生产冒烟发现）— config/settings.py
    本地 OntologyConfig 缺 conflict_penalty_factor（core 版有），cfg.ontology
    传给 OntologyValidator → write_validate L1186 AttributeError → except
    → passed=True 静默降级 → 冲突检测/撤销永不触发（历史遗留漂移，P1 暴露）
  • 修复：settings.py 删除本地类，import 复用 core.ontology_validator 版
    （消除双定义永久漂移；死字段 rules_path 零消费方随删）
  • 生产验证：P1 conflict revoked 日志确认撤销真实触发（user/direct
    覆盖 agent/inferred → SUPERSEDES 血统边建立）
  • 测试: test_ontology_validator + test_conflict_revocation 76 passed +
    回归 51 passed，全量 828 passed 1 pre-existing flaky

v5.44.0 (2026-08-16) Conflict-Revocation:
  • 显式冲突撤销（P1，对标 TEPA arXiv:2608.07429）— 原有冲突检测
    （write_validate + ConflictNode）无撤销动作，新旧记忆都 active。
    本次在写路径内做确定性立即归档：检测到 same_entity_diff_value
    等值精确冲突 → source_type 分级裁决（direct>tool>inferred）→
    新胜则 archive_node(old, replacement=new) 建 SUPERSEDES 血统边
    （qsubmit priority=low 不抢写额度，时序严格在新节点落库后）
  • 稳定事实防护 — protected 旧节点绝不自动归档（留 ConflictNode 待
    人工）；fact_track=core 同级/降级不归档（严格更高源才可）；
    弱匹配 contradictory_claim 只建 ConflictNode
  • restore 端点 — POST /episodes/{id}/restore 可翻转软删（archived=false）
  • 修复 GraphLite 'Null' 字符串 bug — 缺失属性返回 'Null' 非 None，
    float('Null') 抛 ValueError 致冲突检测静默降级；_as_num 归一化
  • 修复字段失配 — 裁决依赖的 tau_value/trust_score 生产不存在
    （实际 tau_initial）；trust_score 死列移除，信任比较改 source_type
    权重代理；or 0.5 回退改 is None 判断
  • 测试: test_conflict_revocation.py 新建 25 用例（真实 GraphLite +
    HTTP 全链路），全量 828 passed 1 pre-existing flaky

v5.43.0 (2026-08-16) Retrieval-Evolution:
  • 检索策略自演化真实化（P0，对标 EvolveMem/ERSkill/SAGE）— 原
    SelfEvolvingRetrieval 骨架生产空转（82 次检索零触发）：信号错位
    （quality() 衡量"结果多"恰非失败）+ 旋钮错位（规则调 weight_fusion_*
    非生效旋钮）。CC 设计审查确认根因 → 五轮 Codex 复核闭环
  • 触发真实化 — 即时硬失败（degraded/延迟>500ms）∪ 周期强制
    （probe_every=40 + 6h 时间兜底）；触发快照加权防历史稀释
  • 旋钮真实化 — 规则调生产默认路径真实消费的 top_k_l1/top_k_vector/
    top_k_keyword（HYPERGRAPH/_hypergraph_retrieve 链路）；删死旋钮演化
    （tau_weight/vector_weight 无消费者）；同旋钮 delta 增量合并防互覆
  • 梦境自评探针 — DreamPipeline.retrieval_health_probe()：抽核心节点 →
    直调内层 _qr → recall@10 → 低召回喂 guard（不污染 _total_calls）
  • 持久化 — data/retrieval_evolved.json 原子写（tempfile.mkstemp+
    os.replace），启动 restore_state + validate；apply 空转保护
  • 修复潜伏 bug — check_revert 的 min_samples=3 从未生效（单样本误回滚）
  • 测试: test_self_evolving.py 21→56 用例（含真实 FAISS 行为断言 +
    mutation 验证），全量 803 passed 1 pre-existing flaky

v5.42.1 (2026-08-16) Lazy-Load-Fix:
  • 懒加载 AttributeError 修复（R3 终审遗留 🟡 P2）— _do_embed/embed_batch
    懒加载守卫上移到 ONNX 分支前 + 条件合并为 `_onnx_model is None and
    _model is None`：load() 优先加载 ONNX（只设 _onnx_model 不设 _model），
    旧守卫位于 ONNX 分支之后 → 未先 load() 的 encoder 首次 embed 即
    AttributeError: 'NoneType' object has no attribute 'encode'（潜伏隐患，
    生产因 app.py 启动即 load() 不触发）
  • 测试: test_embed_lazy_load_onnx + test_embed_batch_lazy_load_onnx_cold
    （冷启动守卫独立覆盖，TDD 复现→修复），test_encoder + test_embed_batch
    22 passed；全量 768 passed 1 pre-existing flaky（test_decay_threshold_candidate
    基线同挂，与本改动无关）

v5.42.0 (2026-08-16) Write-Throughput:
  • 写入加速 — CC 修正真瓶颈=同步 defense R2 embed（每条写必跑，非异步队列）：
    encoder.load() ONNX 优先（embedding/onnx/，bge-small-zh-v1.5 导出 512d），
    ~2ms/条 vs PyTorch ~7ms/条（3.4x），batch32 ~1.1ms/条
  • ONNX 路径事实修正 — 原指向 data/all-MiniLM-L6-v2-int8（384d MiniLM 实际
    不存在）→ 统一 embedding/onnx/；dimension 从模型输出读（修复 L373-374
    `if onnx: return 384` 硬编码，bge 512d 不再崩 FAISS）
  • pooling 实测修正 — bge v1.5 实为 CLS pooling + Normalize（1_Pooling/
    config.json pooling_mode_cls_token=true，非任务书假设 mean）：ONNX 路径
    CLS+L2 归一化与 ST encode 一致（cosine=1.000000，FAISS L2=cosine）
  • int8 质量实测 — 动态 int8（per-tensor 0.967 cosine / per-channel 0.845）
    与 QDQ 静态量化（0.64，onnxruntime 1.25 校准 bug）均不达 recall 降幅<2%
    约束（实测降幅 8-23%）→ 落地 fp32 优化图（onnxruntime transformer
    optimizer 融合 Attention/SkipLN，零损失）；model_int8.onnx 保留待评估
  • 队列批量编码 — _process_embed_queue 批量 embed_batch + asyncio.to_thread
    （poll loop 不再被 embed 冻结），失败回退逐条；ONNX 分块 32 防 OOM
  • 缓存去重 — embed_batch 内 consult/populate 原文缓存（LRU 512，与
    _cached_embed 同 key），队列 flush 重复内容零重算（不另起 hash 层）
  • hebbian 优先级 — 队列 flush 的 hebbian qsubmit 加 priority="normal"
    （50 条 flush 不全占 high 额度，v5.40 低准入闸生效）
  • 测试: 新增 test_embed_batch 8 用例（batch vs 逐条 cosine>0.999 同实例 /
    空 / 单条 / 批量失败回退逐条 / ONNX dimension==512 / onnx 缺失静默回退
    PyTorch 零回归 / fp32 ONNX vs ST recall@10 降幅<2% / 缓存命中零重算）

v5.41.0 (2026-08-15) Community-Expansion:
  • 社区扩召回 — 检索利用 CommunityNode（生产 2849 个）修复 LongMemEval-S
    多会话 Recall@10=0.859 短板：种子（前 5 结果）→ 社区（BM25-on-summary
    相关度，CC 修正：keywords 未落库，summary ≤800 字散文含 Keywords 行
    词法足够）→ 成员 append，补充非替代
  • 边方向 — (c:CommunityNode)-[:COMMUNITY_MEMBER]->(e:EpisodeNode)
    （社区→成员，CC 修正：v1 写反）；get_communities_by_seeds +
    get_community_members 批量查询原语（复用 query_cypher 永不抛契约，
    失败返回 [] 静默降级）
  • 相对尾分缩放 — 扩展分 = relevance × min(种子分) × boost(0.6)，严格低于
    种子；relevance < threshold(0.5) 丢弃（CC 修正：非绝对 0.6）
  • 插入点 — retrieve() _finish 去重/排序前 append（闭包捕获 query/
    query_embedding/raw_query；候选经 _deduplicate_and_sort 单点去重 +
    core/画像 boost + score 钳制，不双重放大）
  • 配置 — defaults.yaml + settings.py RetrievalConfig.community_expansion
    {{enabled, boost, threshold, max_members}}，关闭时行为 = 现状（bit 级一致）
  • 评测 — 真实 GraphLite 写 2-3 个多会话问题 + _persist_one_community 造
    真实社区边，扩召回开/关 multi-session Recall@10 对比
  • 测试: 新增 test_community_expansion 10 用例（配置 YAML 接线 / 边方向真实
    GraphLite / 不相关不加分 / 成员排除种子 / 假阳性护栏 / GraphLite 失败
    静默降级 / 开关关闭 bit 级一致 / 画像不双重放大 / 3s 超时预算），
    test_retrieve_routes 回归无破坏

v5.40.0 (2026-08-15) Write-Priority:
  • 写队列优先级 — 单 queue.Queue → queue.PriorityQueue（天然规避双队列 notify
    死睡 bug）：入队元组 (0 if high else 1, seq, task)，seq 用 itertools.count
    （同优先级 FIFO + 元组永不比较 _WriteTask）；SENTINEL 包装为最低优先元组
  • 外部写优先 — qsubmit 默认 priority="high"（kwargs 提取不污染 fn 参数）；
    低准入闸 low_max（默认 max_pending - 10%，为 high 预留）：low/normal 入队
    qsize()>=low_max 即拒，high 仅在队列真满时 503（与现状一致）
  • 梦境 PERSIST 切块（决定性）— _persist_communities 30-60s 单体长任务拆为
    逐社区 low 任务（_persist_one_community，~3s/块）+ 阶段 3 同源湮灭最后
    一块（_persist_communities_prune_edges，全局成员集，湮灭语义保持）；
    块间写线程排空 high → 外部写不再被梦境长任务饿到 30s 超时 503
  • auto-apply/_persist_dream_state 显式 priority="normal"（梦境低价值写走闸）
  • 测试: 新增 test_write_queue_priority 7 用例（高优先插队/低不饿死/背压+
    high 预留/重入/切块端到端），test_write_queue + v524/v525/v528/v529 回归全绿

v5.39.0 (2026-08-15) User-Profile:
  • 显式用户画像层 — 对标 Profile-Graph Memory（2606.06036）：从用户直述
    （source_type=direct）节点提取稳定偏好/身份/工作，检索画像值命中
    score ×1.2 加分 + search_profile 旁路上下文，不做重排
  • 新建 core/user_profile.py — scan_preference_candidates（复用
    fact_track.CORE_KEYWORDS 值提取子集 + 补「工作/工作于」+ 英文低误报词
    I live in/I work at/my favorite；source_type==direct 硬门控 + 句首
    第一人称 ^(我|我的|i\\b) 防误识别）、aggregate（按值去重归一 +
    source_type 权重累积 direct 1.0/tool 0.7/inferred 0.5 + 多源计数加分）、
    build_profile（preferences/identity/work 分组，空候选 → 空 dict）、
    load_profile/save_profile（JSON 原子读写 temp+rename，缺失/损坏降级）
  • 检索加分 — query_router _deduplicate_and_sort 画像命中 ×1.2（复用
    core-boost 模式，乘正因子单调）；set_user_profile 模块级注入内存常驻；
    search_profile 旁路返回画像上下文（prepend 到 prompt 不参与主排序）
  • app.py 接线 — _startup_rebuild 后台扫描 EpisodeNode → 构建画像 →
    落盘 data/user_profile.json + 注入内存常驻（全量重扫覆盖写，幂等）
  • 存储 — JSON + 内存常驻，不用 GraphLite Hyperedge（≥2 成员硬约束 +
    检索不可见 + N+1 反查）；data/ 整体 gitignore
  • 测试: 新增 test_user_profile 10 用例（scan 3 含误识别用例 / aggregate 2 /
    build_profile 2 / search_profile 2 / 检索加分 1），core_boost 无回归

v5.38.0 (2026-08-15) Ontology-Evolution:
  • Schema 自演化 — 对标 MindMemOS：ontology 随数据生长。梦境 SYNTHESIZE 后
    聚合全部社区 topics/report 做 1 次 LLM 调用（复用 llm_client.chat，
    temperature=0.1, response_format=json_object）→ new_type /
    merge_existing / skip 三选一
  • 新建 core/ontology_evolution.py — load_extended（缺失/损坏 → 空 dict
    降级）、merged_types（原生优先不覆盖）、evolve_once（守卫：max 1 新类型/轮；
    ≥2 非泛 conflict_keys；跨类型 key 冲突 → skip first-match-wins；merge 仅
    合并 conflict_keys 去重不覆盖 description；LLM 失败 → skip 不阻塞；
    原子写 temp+rename + asyncio.to_thread 落盘）、classify_with_extended
  • OntologyValidator 惰性合并 — __init__ 加 extended_types 参数 + 实例属性
    _ontology_types（ONTOLOGY_TYPES 与 extended 合并，原生优先不覆盖），3 处
    ONTOLOGY_TYPES → self._merged_ontology_types()（:234/:1111/:1232），
    不硬改模块全局（防污染 test_ontology_validator.py:23 模块级 import）
  • 梦境接入 — run() SYNTHESIZE 后插入 _ontology_evolution_step(communities)；
    llm_client 空/失败 → 直接返回不阻塞梦境
  • app.py 接线 — 启动加载 data/ontology_extended.json（data/ 整体 gitignore）
    供 OntologyValidator 合并；构造 OntologyEvolution 注入 DreamPipeline
  • 测试: test_ontology_evolution 8 用例（merge 3 / 加载 2 / 集成 1 /
    CC 修正 3：跨类型冲突 + 全泛词 + max1），ontology_validator /
    fact_track / skill_bridge 无回归

v5.37.1 (2026-08-15) Skill-Bridge:
  • 判重硬话（Dedup-Harden，R9 终审修复）— _STEM_WHITELIST 补齐 12 对高频变形：
    analyses→analysis、processes→process、systems→system、documents→document、
    tools→tool、files→file、notes→note、tables→table、findings→finding、
    errors→error、comparisons→comparison、evaluations→evaluation（与原有
    9 对合并共 21 对；data/dream_candidates 实测 systems×39/documents×34/
    tools×32/analyses×6 等高频复数未归一 → 近重复漏放行）
    ——research-workflow-analysis vs research workflow analyses 归一后
    3 token 重叠判重（旧逻辑仅 2 重叠漏判）
  • 框架停用词补全（R9 终审修复）— consists/various/diverse/includes/
    include 进 _STOPWORDS：两个不同主题长报告不再因共享 consists+various+
    report（3 token）被误判重复（旧逻辑实测误杀）
  • _stem_word 词干盲剥 P1 修复 — 通用尾缀剥离把 cases→cas、speed→spe、
    summary→summar、status→statu、houses→hous、news→new 剥坏（与 docstring
    声称守卫矛盾），case/cases、summary/summaries、status/statuses 无法归一
    → 改显式白名单映射，白名单外单词原样返回（news/analysis/process/status
    天然不动），零误剥风险
  • 判重链路增强 — 停用词过滤 → 词干归一 → 排除判重泛词（collection/
    operational/technical/set/log 进 _DEDUP_GENERIC，仅从重叠计数排除，
    skill 名/description 保留原文）；重叠阈值 2→3（单/双 token 弱信号不
    阻断，仅泛词重叠不误杀）
  • 命名增强 — 长句截断到 _MAX_NAME_WORDS 而非整体作废（保留 ≥2 词下限）；
    patterns 回退遍历全部（原只试第一个）；纯中文/无拉丁 token 回退 patterns
    仍失败记 warning 不再静默跳过
  • 测试: 白名单归一 21 例 + 反例守卫 26 词（speed/sing/thing/red/feed 等
    不剥）+ analyses 判重公共入口回归 + 不同主题长报告框架词不误杀回归；
    手动验收脚本补 assert 3/3

v5.37.0 (2026-08-15) Skill-Bridge:
  • 记忆→Skill 一体化 — 梦境巩固产出自动固化为 Hermes skill
    （Memory-Skill-Bridge），SHM 自演化闭环从「记忆内演化」扩展到
    「记忆→行为」：检索到的新模式沉淀为可执行 skill
  • 新建 core/skill_bridge.py — extract_reusable_patterns 质量门三合一
    （patterns 非空 + report 非 raw JSON + 非元话术）→ generate_skill_name
    动作短语 kebab-case 命名（不用 hash 主名，宁缺毋滥）→ should_create_skill
    同名/描述关键词重叠判重 → generate_skill_md（frontmatter + 场景/步骤/来源）
    → sync_from_dream 主入口（每轮最多 3 个，不覆盖已有 skill）
  • 时序修复: auto_apply_candidates 返回 community_summaries（apply 前从
    内存收集，候选文件删除后不再依赖读回）；api/app.py _dream_poll_loop
    auto_apply 后用返回摘要调 sync_from_dream（纯文件 IO，无 LLM 调用，
    poll loop 60s 不卡）
  • 消费端: Hermes skills_dir.rglob("SKILL.md") 递归扫描，直接写
    ~/.hermes/skills/<name>/SKILL.md 即被加载
  • 测试: 新增 test_skill_bridge 6 用例（质量门 4 坏例 + 命名/判重/
    frontmatter/集成 7 社区 4 坏 3 好 → 3 skill）
  • 梦境聚类分区合并 bug 修复 — _cluster_step 合并连通分量分区时丢弃
    _detect_communities 返回的 cid，逐节点分配独立社区 ID（生产 896
    节点 = 896 社区）→ Leiden/Louvain 24 社区结果全废 → SYNTHESIZE
    永远只处理单节点社区 → summarize_community（需 ≥2 节点）永不触发
    → patterns 恒空 → 社区摘要恒为模板废话
  • 修复: partition[nid] = next_comm + cid 保留社区归属（同分量同 cid
    的节点归同一社区），跨分量用 next_comm 偏移防 cid 冲突（每个分量
    内 cid 独立从 0 编号）；singleton 节点沿用 next_comm 逐节点编号
  • 测试: 新增 test_dream_cluster_fix 3 用例（mock 分区 {{a:0,b:0,c:1}}
    → 2 社区 / 跨分量偏移 {{a:0,b:0}}+{{c:0,d:0}} → 2 社区 / singleton
    混合 → 2 社区），旧代码三用例全挂

v5.35.0 (2026-08-15) Core-Boost-MultiSource:
  • 检索 core 优先 — fact_track 数据透传（BM25/实体匹配/回退/超图/向量/
    关键词 六通道组装补齐 + graph_expansion 邻居回查补全）+ core 轨 ×1.1
    温和 boost（retrieve() _finish 统一出口，覆盖 L1/L2/L3/L4/FUSION/
    graph_expansion 全部路径）
  • 多源支持度 — EvidenceTracker.is_multi_source（≥2 个不同来源确认同一
    事实）；write.py 写多源 → tau_initial=0.85 提升持久性（零新增查询、
    零 embedding，复用写路径现成多源计数）
  • BM25 旧索引兼容 — 读取 .get("fact_track") 缺省 active，不抛错
  • score 非 [0,1] 但乘正因子单调，不破坏排序语义

v5.34.0 (2026-08-15) Dual-Track-Facts:
  • 双轨事实记忆 (Dual-Track Facts) — MemSIF 双轨启发：稳定事实 (core)
    不应因 τ 衰减被误归档 (DUM)，事件/临时内容 (active) 随 τ 正常衰减
  • 新建 core/fact_track.py — classify_fact_track 按 (本体类型 → 关键词 →
    默认 active) 判定：core 本体类型 (person_birth/person_death/
    organization_founded/relationship/location_fact/scientific_claim)
    直判 core；内容含中文持久化关键词 (喜欢/我是/我住/一直/偏好/经常/
    住在 等) 判 core；其余默认 active (保守)
  • 写时分类 — create_episode 落库 fact_track 字段 (write.py 经
    write_validate 的 ontology_type + 内容分类)；graphlite_store 一处
    setdefault("fact_track", "active") 覆盖全部写入点
  • τ 衰减差异化 — compute_tau/compute_strength 加 fact_track 参数，
    _get_effective_tau_decay 对 core 轨 ×2.0 boost (等效 importance=1.0)
    再走 min/max 钳制；dream_pipeline 用 created_at 重算 τ 时透传 fact_track
  • 顺手修 tau_decay docstring 过期残留 (1 - I·m → 1 + I·m·IMP_BOOST_FACTOR)
  • 检索 core 优先 + 多源提升：已落地 (v5.35.0)

v5.33.0 (2026-08-15) Source-Trust:
  • 来源信任分级接通 — 写时 source_type 分级落库（direct/tool/inferred，
    默认 direct 向后兼容，一处 setdefault 覆盖全部写入点）
  • 防洗白校验 — agent 来源（hermes/codex/claude/opencode 等非 user）
    声明 direct 强制降级 inferred；只有 source=="user" 才允许 direct；
    gateway_api 3 处直调 create_episode 同步 source_type（堵 A2A 洗白漏洞）
  • promote_to_episode 写死 source_type="inferred"（系统提升非用户直述）
  • ConfidenceCalibrator 注入 DreamPipeline（app.py 构造点接通，梦境
    CALIBRATE 步骤非空转，Manufactured Confidence 落生产链路）
  • 检索排序加权缓做（v5.34，需改 4 处 SELECT 收益有限）

v5.32.1 (2026-08-15) Gateway-Archive-Align:
  • GatewayAPI.retrieve 加 include_archived: bool = False 参数并透传给
    query_router（SelfEvolvingRetrieval / 裸 QueryRouter 均已支持）
  • Cypher 兜底按 include_archived 条件化 — archived_clause 变量，
    include_archived=True 时不加过滤（主通道返回归档节点，兜底同步）；
    OR 组加括号 WHERE ({{conditions}}){{archived_clause}} 防 AND 优先级错乱

v5.32.0 (2026-08-15) Archive-Supersedes:
  • 结构化归档 + supersedes 血统链 — 梦境「遗忘」从物理删除改为归档
    （archived=true）+ 建 (old)-[:SUPERSEDES]->(new) 血统边，对标 MindMemOS
  • PRUNE 改归档: _persist_prune DETACH DELETE → graphlite_store.archive_node
    （τ 衰减节点归档而非删除；pruned_ids 仍供 FAISS remove_ids）
  • RESOLVE 改归档: _persist_merge loser DETACH DELETE → archive_node(loser,
    winner)，winner 拼接摘要后建 SUPERSEDES 边，历史可追溯
  • 检索层默认排除 archived: retrieve() 加 include_archived 参数 + _filter_archived
    后置过滤；Cypher 兜底追加 (archived IS NULL OR archived = false)；向量路由
    Python 侧过滤；写时基线 create_episode 默认 archived=false
  • 旧数据兼容: 过滤写 IS NULL OR = false（不误排除无 archived 字段旧节点）

v5.31.6 (2026-08-14) Graceful-Shutdown:
  • SIGTERM/SIGINT 优雅退出（三体协奏 Phase1 CC 设计 → Phase2 OpenCode
    实施 → Phase3 Codex 审核）— lifespan 注册信号处理器转发 uvicorn
    handle_exit → should_exit → Server.shutdown → drain 写队列 → close
    → Sled 落盘释放锁（堵住 kill -9 之外的隐性终止路径；实测纠正：
    loop.stop() 会跳过 lifespan shutdown 段）
  • open 失败自动备份损坏库 — connect 包 try/except：GraphLite.open
    失败 → copytree 到 .corrupt.<ts> 保留崩溃现场 → 裸 re-raise
  • Codex 审核 ✅（uvicorn 0.51 源码链实测）+ 修 2 🟡（信号处理器内
    日志→置标志位防死锁；备份时间戳加 %f 防同秒碰撞）
  • 测试: 新增 test_signal_shutdown + connect 备份用例，全量 631+ passed

v5.31.5 (2026-08-14) CAS-Optimistic-Lock:
  • 乐观锁 CAS 化 — update_with_version 改单条 GQL（MATCH WHERE version
    = X SET version = X+1, ...），rows_affected 检测版本冲突（>0 成功 /
    0 冲突）——消除两步法读后写前竞态窗口 + 免一次读查询
  • 依赖引擎 rows_affected（fork e53e3df）——多语句返回最后一条 COMMIT
    的 rows_affected=0，故 CAS 不用 BEGIN/COMMIT 包裹（单查询本身原子：
    SHM 单写线程 + Sled 单写锁串行）
  • 测试: 乐观锁用例适配新接口（execute 返回 rows_affected）+ 全量
    631 passed

v5.31.4 (2026-08-14) Native-UTF8-NoB64:
  • 引擎级 UTF-8 修复落地 — GraphLite fork (Neocher/GraphLite 4452a96)
    lexer 字节边界 panic 根治（145 处关键字切片 + is_keyword_match +
    _timewindow_literal），中文 INSERT/CONTAINS/LIKE 原生支持 7/7 验证
  • 去 b64 透明编码 — _gql_value/_interpolate 原生直写（读端 _decode_b64
    保留兼容旧数据）；生产迁移 3144 节点（content 416 + summary 2612 +
    entity_name 101 + name 15；OntologyType 内部 id node_xxx 用 name
    等值匹配修复 SET 静默失败）
  • ⚠️ 依赖更新 — requirements.txt/README 注明必须使用修复版 fork
    （Neocher/GraphLite），勿用上游（lexer UTF-8 bug 未修）
  • 测试: 全量 631 passed（interpolate 断言更新为新语义）

v5.31.3 (2026-08-14) Write-Path-Breaker-Neutral:
  • 写路径熔断中立补漏 (瑕疵1) — _flush_hebbian_batch 改用 execute_cypher
    (写路径, P2-2: 不 record_success/failure)，不再走 query_cypher 读路径
    (带重试 + 熔断计数)。v5.31.1 的 Hebbian 分号修复只是降低概率，未根除：
    若 GraphLite 对某批 20 对 MATCH+INSERT 报 QueryError，读路径会重试 2 次
    + record_failure → 写失败仍污染读熔断窗口。execute_cypher 不吞异常契约
    → rebuild_index 两处调用点补 try/except 兜底 (失败记日志不中断重建)。
  • ontology 幂等短路精确化 (瑕疵2) — sync_entity_types_to_graphlite 短路
    条件从 count>0 改为「ENTITY_TYPE_MAP 全部类型已存在 + 实体数齐全」：
    v5.31.2 的短路在同步中途中断 (库里只有部分类型) 时，剩余类型/实体/
    IS_A 边永不补齐；写入路径只提取实体共现不建 OntologyType，无法自愈。
    仍保留短路目的 (避免 180 实体 × 5-6 次 execute_cypher 全量重跑)。
  • H1 风格 id 转义 (瑕疵3) — _flush_hebbian_batch 内 f-string 直接插值
    src/dst 重蹈 H1 教训，改为经 _gql_value 转义 (含 ' / \\ 的 id 不再
    裸插注入 GQL，与 graphlite_store 的 21 处 H1 修复保持一致)。
  • 测试: 新增 10 用例 (写路径熔断中立 3 / 特殊字符 id 建边 4 / 短路
    精确化 3) — 全量 614 passed
  • 写队列永久卡死修复（生产 8/14 全线写超时根因）— sync_entity_types_
    to_graphlite 全量 180 实体 × 5-6 次 execute_cypher 在写线程执行，
    生产库 INSERT 新 OntologyType 触发 GraphLite 引擎挂起 → 写线程卡死
    → 所有写请求 30s 超时。熔断 closed（v5.31.1 Hebbian 修复后）首次
    暴露：此前熔断 open 时 sync 全部快速失败，掩盖了此路径。
  • 修复: sync 加幂等短路 — OntologyType 已有数据（count>0）直接标记
    _ontology_synced 返回，不再全量重跑；增量实体由写入路径覆盖。
  • 附带: 清库重建流程（备份在线复制不完整= DATABASE_OPEN_ERROR，
    损坏库 INSERT 全报 QUERY_ERROR → mv 保留 + 重启自动建新库，
    数据由 Hermes 同步管道重建）
  • 验证: 新库首次写入 2.32s（全量 sync）→ 二次写入 0.02s（短路）；
    OntologyType 23 / OntologyEntity 180 / 检索 0.02s hits=1

v5.31.1 (2026-08-14) Hebbian-Batch-Fix:
  • Hebbian 批量建边分号拼接修复 — 启动熔断风暴根因：system.py 用
    "; ".join(pending) 拼接多条 MATCH+INSERT，GraphLite 对分号多语句
    静默截断只执行第一条 / QUERY_ERROR；episode 少时失败数不达熔断阈值
    (1/10=10%<50%) 未暴露，episode 多时 (374 连接=8 批) 8 次失败打满
    窗口 (80%≥50%) → 熔断 open → BM25 prewarm/ontology 同步全被拒 →
    启动后 1-2 分钟假死 (health nodes=0, 检索空)
  • 修复: _flush_hebbian_batch 单条多模式 MATCH + 多边 INSERT（逗号分隔），
    HEBBIAN_BATCH 50→20（GraphLite 单查询上限实测 20 对 OK / 25 对静默丢弃）
  • 验证: 临时库 20 对全建 / 25 对正确检测失败 / 全量测试通过

v5.31.0 (2026-08-13) Starlink-Audit-Fix:
  • 星链批量审计修复 19 项（H1/H2/M1/M2 + P2-P9 + M3-M5 + L1-L4）— 599 passed
  • H1 GQL 注入面转义 21 处：id/metadata/session_id 等外部可达字段全部经
    _gql_value 转义（' / \ 不再裸插注入 GQL）
  • H2 梦境深度守卫：PERSIST 写队列过半满 → 跳过本次写回（degraded，下次
    梦境 upsert 自愈）；M1 引擎死锁探测：独立只读 ping 探针 + diagnose()
    快照（心跳/在途任务/线程栈），自动重建仍由 systemd/人工决策
  • M2 乐观锁原子化：update_with_version 读 version + SET 并入同一 session
    锁临界区（消除"读后、SET 前"另一写线程抢先更新的漏检窗口）
  • P2-P9 性能主线：P2 embed 批扫 conns 提出批循环 / P3 RESOLVE 嵌入记忆化 +
    Jaccard 预筛（sim ≤ 0.66 < 0.8 决策等价）/ P4 BM25 构建异步化（锁只护
    swap 短临界区，zombie 构建不钉死锁）/ P6 fusion 三通道 ThreadPoolExecutor
    并行 / P8 merge content 截断 2000 / P9 ontology types 预筛缓存（N=2000
    时 4M 次正则 → 2000 次）
  • M3 shutdown 哨兵重放 + 积压任务 set_exception 立即失败（uvicorn 关闭
    不再挂起）；M4 CJK 查询跳过 entity/L4 通道（GraphLite CONTAINS 不支持
    UTF-8，通道恒空，省全表扫描）；M5 EpisodeCache（OrderedDict LRU+TTL
    4096/600s，flush 预填检索零回查）
  • P7 时态+情节窗口查询合并单条（3600s 窗 LIMIT 20，to_thread 不冻 loop）；
    L1-L4 机械修复：裸 except→ValueError / home 路径降级回退 / 慢写非死锁
    计数归零 / create+ensure+link 合成单闭包（失败短路消除半写）

v5.30.0 (2026-08-13) Interpolate-Prefix-Fix:
  • _interpolate 前缀碰撞修复 (P0 静默数据丢失) — 旧实现按 dict 序逐键
    str.replace，$t1 会误替换 $t10 内的前缀（'$t1' in '$t10'）→
    query_router 实体匹配对 ≥10 候选（t0..t12）生成多段 CONTAINS 时
    $t10..$t12 被污染，检索静默漏检
  • 改单次 re.sub(r"\\$([A-Za-z_]\\w*)", callback)：按捕获完整键名查
    params，键序无关、无前缀碰撞；未知键返回原 match（未命中不替换语义
    保持不变）；str/ascii/b64/int-float/numpy/None/空串哨兵类型分支全保留
  • 测试: 新增 11 用例（前缀碰撞 / 键序无关 / ≥10 段 CONTAINS / 类型分支
    保真 / 未知占位符 / 空参数）

v5.29.0 (2026-08-13) Dream-Write-Lock:
  • 梦境与写库并发卡死修复 (2026-08-12 生产 8/12 两次 SIGKILL) —
    梦境拉取 (F1) 与 hyperedge_sweep (F3) 改 to_thread, 事件循环不再被
    大查询阻塞; 梦境 PERSIST (F2) 收敛到 write_queue 单写线程串行
    (不再裸 to_thread 绕开单写线程约束 → 引擎级挂起根因消除)
  • GraphLiteStore session 访问锁 (F5): _locked_query/_locked_execute 包装
    全部 34 处 _session.query/execute, RLock 可重入兜底跨线程并发访问
  • write_queue 看门狗增强 (F6): 连续 3 次超时 + 疑似卡死 → logger.critical
    告警建议人工重启 (不自动重启, 由 systemd/人工决策)
  • 测试: 新增 6 用例 (F1 拉取不阻塞 loop / F2 persist 经 write_queue 串行 /
    F5 并发读写 session 有锁保护)

v5.28.0 (2026-08-13) Write-Queue-Watchdog:
  • 写队列永久卡死修复 — submit() 改用 asyncio.shield(wrap_future) 不占 executor
    线程（旧 run_in_executor 单 worker 被占死 → 全 503 永不恢复）
  • 看门狗仅线程死亡时重启（alive+慢写不重启避免双写并发；空闲不误判）
  • shutdown 满队列 put_nowait 跳过 sentinel（uvicorn 关闭不挂起）

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
