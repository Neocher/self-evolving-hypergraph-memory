# R8 引擎不变量设计定稿 v2 (P1-1, 达摩院深度评审修订版)

> v1 经达摩院 deep 评审 (693s, 5 必读输入只读实证核验, 报告 ~/outputs/damoyuan/shm-r8-engine-design-review.md)
> v2 修订依据: 评审 Top5 + Step0 spike 证据 (见 §0.1)。用户裁决: D1=③ D2=接口+注入 D3=采纳 E1' (不变)
> 仓库基线 v6.19.0, 全部改动 env 开关默认 off; 版本统一 bump v6.20.0

## 0.1 Step0 spike 证据 (2026-09-08, 决定事实流源)

评审实证: pkl triples×全局 dia_to_ep 归因坏 (1916 行仅 21 行唯一可归会话, conv-30/42 为 0 行, conv-50 独吞 1547) + triple 语料对靶区缺员 (T2 3/4、T4 15/28 题有 gold 成员全缺)。
**Spike 补证**: 官方 questions.jsonl `evidence_messages` 与 conversations.jsonl 原文 9/9 命中; 三靶题实体-值证据 (q0035 Lab/Chihuahua mixes → D19:12+D26:13; q0001_c50 mansion/luxury car → D1:3+D2:1; q0139_c50 purple shiny guitar → D16:19) **原文全部可达**。

→ **事实流源 = EpisodeNode 原文** (评测 OverGraph 5882 全量, 官方 (sample_id, 会话内 message_index) 结构天然归因, created_at 时序完整), **不依赖 pkl triples/dia_to_ep**。triples(1916)/entity_eps(22) 降级为槽词表/实体词典提示。数据面零写入零依赖成立。

## 0.2 范围出清 (v2 新增, 评审 §2.1 要求)

- **评测验证面语义** = "会话末当前态" (session_ts=conv 末锚, harness 现实): E1' 评测侧只对当前态题 (q0035 形状) 声称可验证
- **时间窗题 (q0036/q0061 形状) 与极性题 (q0045 形状)**: 评测 harness 无 per-question 时点 + resolve 无窗/否定维 → **显式不在本轮评测验证语义**, 由 G2 构造集 (构造含真实会话时间锚的窗/极性场景) + 真实系统侧覆盖; 防止 G2 误验收 (任务书 G2 场景集须含旧状态/窗/极性题时, 判据对齐此出清)
- 任务书勘误注: conv-48#q0017、conv-47#q0003 任务书列 T1, cat1_t_labels.json 实为 T4; conv-26#q0028 书 T2 实 FLOOR — 对拍以 JSON (r8-ownership.json) 为准

## 1. 四模块设计 (E1'~E4, 本体论验收)

### E1' 状态时点裁决 — `core/state_projection.py` (纯函数)
**本体论**: 时间建模 (valid-time) | **验收问**: 题问时点该 (实体,槽) 的值由什么决定? → 由 (值,生效时间) 集合的确定性 latest-wins → 过。
- `resolve(facts, entity, attr, session_ts) -> State{value,ts,ep_id}`: ts≤session_ts 内 latest-wins; tie-break=ep 序; 否定证据行 (negation=True) 参与裁决 (v2 加维, 服务 q0045 形状的真实系统侧, 评测面不开)
- `timeline(facts, entity, attr) -> [State]`: 全版本链
- **复用边界 (评审 §2.1.3, v2 补)**: 裁决语义与 QueryRouter 既有 `_property_temporal_retrieve` (P0-1, at_time: valid_from≤ts 未过期) / `_fact_retrieve` (subject+predicate latest 仲裁) **同构** → 本模块 = 共享裁决器 (事实流注入式), 评测侧喂 EpisodeNode 时间流 (无 PropertyVerNode → 前两者 no-op), 真实系统侧喂 PropertyVerNode 投影; **禁止第三条平行实现** — 实现时优先抽取既有两个通道的裁决逻辑为公共函数, 新模块调用之
- G1: ① latest-wins ② 历史插入不改 latest ③ session_ts 全早 → N/A ④ tie-break 同秒 ⑤ ts==session_ts 含/不含边界 ⑥ 否定行裁决 ⑦ 真日历跨日

### E4 槽位原文检索 — `QueryRouter.retrieve_slot` + 槽值规则
**本体论**: 关系显式化+属性分层 | **验收问**: (entity×slot) 成员全集如何确定性取得? → 集合查询+规则抽取, 不做 top-N → 过。
- 签名: `retrieve_slot(entity, slot, session_ts=None, scope=None) -> [SlotHit{ep_id, ts, value, snippet}]`; 检索对象 = **EpisodeNode 原文** (scope 会话内, created_at 排序), 槽值由确定性规则从命中消息上下文窗抽取
- **契约 (v2 补)**: 成员行=去重值 (同一成员多提及并 1 行, 附提及 ep 列表); 排序=时间升序; slot 词形归一小写/复形; 跨会话同名实体按 scope 隔离
- 槽词表: entity_eps 实体词典 + 1066 attribute 归一槽表; Q→(entity,slot) 由题面名词短语→槽表匹配 (规则, 装配层, 有 oracle)
- **真实系统**: 图实体边分支, 与评测分支共用返回契约
- G1: ① 40 注入删 5 → 全量 35 ② 无槽 → 空 ③ 时点过滤边界 ④ 跨会话同名反例 ⑤ 去重 (3 提及并 1) ⑥ 词形归一 (John/john)
- **槽语义门 (v2, 承接 T3 谓词义/义项 4 题 q0001_c49/q0010_c50/q0050_c42/q0033_c44)**: 槽值域断言 — 题面谓词事件类型 (事故/保养/参赛/陪伴) × 值语义 (车坏/保养/在玩/忙) 匹配, 违例剔除或降级提示; 规则词表 (事故类: crash/broke/accident…)

### E2 模态/归属一等化 — `core/modality_rules.py` + 写路径字段
**本体论**: 公理约束+作用域隔离 | **验收问**: 模态与主体写时确定性标注? → 规则模块+字段 → 过。
- 字段: 抽取器输出附加 `modality ∈ {actual, planned, wish, inferred, in_talks}` (v2 细化, 评审 q0004 判别轴=洽谈态非 actual/planned/wish) + `subject` (双人会话需 speaker 字段, v2 补: ExtractedAttribute 追加 speaker)
- **评测近似面加谓词语义门 (v2, 评审 §2.3.1, 最高冲突项)**: 减法过滤**只在 did/own/visit/play 类题面启用**; plan/aspire/want/potential 类**禁用** (反例 oracle: q0013_c44 gold=三次计划, q0026_c47 gold 含 aspire, q0004_c43 gold 含 potential sponsorship in talks)
- G1: ① wish/aspire/would/hope → 标 planned/wish ② in_talks 与 wish 可分 (q0004 形状) ③ 否定+过去式 ("had planned to" 已执行→actual) ④ 修辞 want ("if you want" 非愿望) ⑤ **不该过滤的绝不过滤** (q0013/q0026/q0004 反例) ⑥ subject 双人 he/she 追踪

### E3 实例身份 key — `core/instance_key.py`
**本体论**: 实体类型化 (同一性) | **验收问**: 两提及同实例? → key 判定 → 过。
- `key = sha1(entity|type|iso_week|obj_desc)` — **obj 含修饰语描述维 (v2, q0139 形状: 同 obj=guitar 但 yellow-octopus vs shiny-purple 必须异 key)**; 跨年 iso_week 边界
- **范围扩展 (v2, 承接 T3 实体解析 4 题)**: 别名归一词典 (colleague→Rob, q0014_c41) + 指代解析基本规则 (代词→会话最近实体, q0047_c42 they→turtles; 双人说话人归属用 speaker)
- 写路径挂点: entity_resolver 事件级下沉 (与 _update_property_version 链共存, v2 补说明)
- G1: ① q0139 形状 (同 obj 异描述 → 异 key) ② 跨年邻接周 ③ obj 缺失回退 ④ 同一事件两提及 → 同 key ⑤ 别名归一

## 2. 67 题归属表 (v2 补, 评审 Top5 #2)

| T 类 | 题数 | 承接 | 说明 |
|---|---|---|---|
| T1 | 1 | E3 | q0139 (desc 维) |
| T2 | 4 | E1' (评测=当前态子集) | 窗/极性题出清 G2+真实系统 (见 §0.2) |
| T4 | 28 | E4 | 原文槽检索 (证据在场; "在场被截断"与"绑定/未具名"可达, "triple 未抽"在原文路线下消失) |
| T3 | 13 | E2×5 / E3×4 / E4槽语义门×4 | E2: q0004/q0009/q0012/q0014_c42 部分/q0076; E3: q0014_c41/q0016_c43/q0023_c44/q0047_c42; E4slot: q0001_c49/q0010_c50/q0033_c44/q0050_c42; q0014_c42=E2∩E3 |
| PSEUDO | 14 | 剔除 | 纯列举形态 (判卷人为惩罚) |
| FLOOR | 7 | 剔除 | judge/gold 噪声地板 |
**合计可修 46 = E1'4 + E4 28(+4slot) + E2 5 + E3 5**, 零无主。完整表: `~/shm-evolve/studies/r8-ownership.json`

## 3. 验证体系 (v2 补全, 评审 Top5 #3/#4)

- **G1 oracle**: 每模块测试文件 + **映射/装配层 oracle (v2 补, 最大假绿源)**: 真实 pkl 切片 + 官方 evidence_messages fixture (q0035/q0001_c50/q0139_c50 各 2-3 事实行, 断言归因会话/时点/值正确) — 固化 fixtures/tests/fixtures/r8_*.json; **只读守卫**: INGEST_LOADED 下评测进程对 pkl/OverGraph 零写入断言
- **G2 构造集** (v2 补): 每不变量 ≥3 场景 + 判据 — E1': 旧状态/窗/极性构造场景 (含真实时间锚, 校验评测出清语义外实现正确) / E4: 多提及去重/成员全回/跨会话隔离 / E2: plan 题不禁/其他题禁 / E3: q0139 双吉他/别名/指代; 判据=确定性输出断言
- **G3 多探针**: cat4 子集 + LongMemEval-S + 真实查询 ≥2 一致才采信
- **G4 全量评测**: 不回归协议 (v6.19.0 基线 83.14 口径) + 各开关回滚判据; A/B 300 样本快速失败条款 (继承 brief §三)
- **开关**: `R8_CTX=1` 总闸 + 谓词语义门子开关 (v2, 替代四条独立 env, 评审 §2.5); off 时与 v6.19.0 逐字节等价 (G1 哨兵标注)

## 4. 实施顺序 (v2 重排, 评审 Top5 #5)

1. **Step0 事实流 spike** — ✅ 已完成 (2026-09-08): 原文路线证据入档 (§0.1), 注入源缺陷从"评审发现"变"实现方正式证据"
2. **Step1 纯函数组** (E1' resolve / E3 key / E2 rules, 并行, 各自 G1 + 全量 pytest 1212)
3. **Step2 原文事实流加载器 + retrieve_slot 接口** (同提交, 附映射层 oracle) — E1'/E4 评测面自此时可验证
4. **Step3 评测装配接线**: R8_CTX 总闸 + 谓词语义门; 槽表/语义门词表; 单变量 A/B (300 样本快速失败)
5. **Step4 E2/E3 写路径字段一等化** (真实系统, 独立提交, pytest + G3 真实侧; 不动评测)
6. 每步: 版本 bump 规范化, 全量 pytest, G4 不回归 (第 3 步起跑评测)

## 5. 否决清单 (防复发, v1 继承)

任何 judge flip / cat1+pp 验收门 / 判卷协议 / prompt V1 / pkl·OverGraph 写入 / top-N 截断式槽查询; E1' 第三套平行时间实现; E2 无谓词语义门的全局减法。
