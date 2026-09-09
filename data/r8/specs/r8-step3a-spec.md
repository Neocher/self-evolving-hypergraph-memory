# R8 Step3a 规格: 确定性槽证据抽取管线 (原文→SlotFact)

> 权威: r8-engine-design.md v2 §1-E4/§4-Step3。前置: Step1 纯函数组 (core/modality_rules.py) + Step2 槽引擎 (retrieval/slot_facts.py) 已验收勿动。
> 词表 (只读): .r8-spec/q2slot-v1.json (46题→entity/slot/gate) · .r8-spec/slot-triggers-v1.json (45槽→触发词) · .r8-spec/fixtures-r8-oracle.json (真实证据行)

## 目标
评测面原文 (EpisodeNode/会话消息) → (entity×slot) 确定性槽证据的抽取管线 (纯规则, 零 LLM 零 IO 零时间依赖)。Step3b 装配用它构建 SlotIndex 注入 retrieve_slot (Step3b 不做)。

## 定位 (诚实边界, v1 语义)
v1 = **确定性槽证据句检索 + 轻量值切分 (尽力而为)**: 完整 NP 值解析不在 v1 (若 oracle 显示不足, 值级文法留 v2)。因此:
- query/members 输出的事实 value 有两级: 切分成功 → 值片段; 切分失败 → 规范句文本 (回退整句)
- 装配层消费 = 证据句全量 (时间序) + members 去重 (供统计) — 集合闭包由"证据句全集给 reader"承担, 非 NP 级闭包

## 交付白名单
1. 新文件 `retrieval/slot_extract.py` (纯, 禁 IO/LLM/系统钟; import core.modality_rules + retrieval.slot_facts):
   - `extract_slot_facts(msgs, entity, slot, triggers, gate="did") -> list[SlotFact]`
     * msgs: [{dia, session_id, speaker, text, ts(可比较)}] 会话内时间序
     * 归属过滤: 句 speaker==entity (自述) 或 句含 entity 整词 (他述, 如 "John's favorite...") → 候选; 其余剔除
     * trigger 过滤: 句小写含 slot 触发词任一词边界命中 → 候选
     * 模态门: gate="did" → classify()∈{planned,wish,inferred} 剔除; gate="offer" → wish 剔除 (in_talks/actual 保留); gate="plan" → 不过滤
     * 值切分: 触发词命中位置后的宾语窗 (≤8 tokens, 修剪 a/the/my/our/new 等冠词前缀), 按 and/,/;/ but/then 切多值, 每片段=value; 窗空/纯触发词 → 回退整句文本 (strip 后)
     * SlotFact: session_id=msgs 会话, ep_id=句在会话内序号, ts=消息 ts, entity, slot, value=片段或整句, msg_ref=dia
   - `build_slot_index(msgs_all, lexicon, triggers) -> SlotIndex` (多 entity/slot 批量, 供 Step3b)
2. 新测试 `tests/retrieval/test_slot_extract.py` (TDD 先行)

## G1 硬性用例
构造句 (每类 ≥1):
① 自述句入: speaker=Audrey "Pepper and Panda are Lab mixes" (slot dog_breed, trig mix) → 命中, 值含 lab mixes
② 他述含名句入: speaker=James "John's favorite game is CS:GO" (entity John, slot favorite_game) → 命中
③ 他述无名句剔: speaker=James "my favorite game is CS:GO" 配 entity John → 不命中
④ 无触发词剔: "I went to the park" 配 purchase_item → 不命中
⑤ did 门滤 wish: "I wish I had a new car" → 剔除; "I would love a mansion" → 剔除
⑥ did 门保留 actual: "I bought a mansion" → 保留
⑦ offer 门: "in talks with Gatorade" 保留; "Nike would be really cool" (wish) 剔除
⑧ plan 门: "we plan to hike" → 保留 (不过滤)
⑨ 值切分 and/，: "I visited Oregon and Florida" → values ["Oregon","Florida"]
⑩ 冠词修剪: "I bought a new mansion" → "mansion"
⑪ 窗空回退整句
⑫ 去重/排序: 同值两提及 members 合并 mentions
真实片段 oracle (读 .r8-spec/fixtures-r8-oracle.json 的 snippet/text, 断言句级可达 — 值级仅构造句断言):
⑬ conv-44 q0035 dog_breed: D19:12/D26:13 文本各抽到 ≥1 事实且句含 "mixes"
⑭ conv-50 q0001 purchase_item: D1:3/D2:1 各 ≥1 事实 (句含 mansion / car)
⑮ conv-50 q0139 guitar_style: D16:19 ≥1 (句含 guitar/purple)

## 纪律
- 禁改既有文件 (含 retrieval/slot_facts.py、query_router.py、core/*、评测/数据面); .r8-spec 只读
- 纯函数禁 IO (词表由调用方传入, 模块不读文件); 不 commit 不 push 不跑评测; 禁委派子 agent; 受阻立即写明原因不空转

## 完成标准 (交付摘要附证据)
1. 红→绿 (测试先落盘)
2. pytest tests/retrieval/test_slot_extract.py 全过贴真实输出
3. 全量 pytest (1291+新增) 零回归贴计数
4. git status 仅白名单 2 新文件 + 既有遗留; 无既有文件改动
5. 边界与设计差异说明 (含值切分失败率观察/回退率 — 可在构造集上自评)
