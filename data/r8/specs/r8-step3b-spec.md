# R8 Step3b 规格: bench 评测装配接线 (R8_SLOT env 门, 默认 off 零回归)

> 权威: r8-engine-design.md v2 §1-E4/§4-Step3。前置已验收: Step1 core 三纯函数 · Step2 slot_facts/SlotIndex + retrieve_slot · Step3a slot_extract (extract_slot_facts/build_slot_index) · 词表 .r8-spec/q2slot-v1.json + slot-triggers-v1.json。
> 装配先例 (同模式): scripts/bench_locomo_v72_ontology.py L1212 `_r6_evidence_forms_ctx` — FACT_CLUSTERS=1/NEG_CLEAN=1 env 门, 默认全 off → v6.17.0 逐字节等价。Step3b 照此模式。

## 目标
评测组织段 ctx 装配处追加 R8_SLOT 分支: 对题面 (qa_id→lexicon 得 entity/slot/gate) 用 Step3a 抽取管线从**该会话原文消息**构建 SlotIndex, 取槽位确定性证据 (members/句行, 时间序全量不截断), 以 `[SLOT EVIDENCE]` 段追加到 ctx 尾部 (reader 前), 供 reader 完整枚举。原文消息行逐字不变 (红线), reader prompt V1 不动 (红线)。

## 交付白名单
1. 新文件 `scripts/slot_assembly.py` (bench 侧装配, 禁 LLM; 可读文件=仅词表两 JSON):
   - `append_slot_evidence(ctx, question, qa_id, msgs, lexicon, triggers) -> ctx`
     * qa_id 形如 conv-50#q0001 (lexicon key); 无词条 → 原样返回 ctx (不报错)
     * msgs: [{dia, session_id, speaker, text, ts}] 会话内时间序 (调用方给本 conv)
     * build_slot_index(msgs, {qa_id: entry}, triggers) → 对每 entity: SlotIndex.query(entity, slot) → 行全量; 渲染段:
       ```
       [SLOT EVIDENCE] <entity> | <slot> (确定性槽证据, 时间序):
       - value: <value>  (msg <dia>, <ts 日期串>)
       ...
       ```
     * 追加位置 ctx 尾部; 行引用消息号 不复制原文逐字 (防重复原文膨胀; value=切分值或句)
   - `build_msgs_from_cache(msg_by_id, episode_cache, session_date_by_conv) -> list`: bench 内存 → msgs (dia=key; text=msg text; speaker=解析行首 "Name:" 前缀, 无则 ""; session_id/ts 从 episode_cache/会话日期映射, 顺序=会话内出现序)
2. 修改 `scripts/bench_locomo_v72_ontology.py` (唯一既有文件改动, 追加式):
   - import slot_assembly + 常量 R8_SLOT = os.environ.get("R8_SLOT")=="1" (与 FACT_CLUSTERS 同区)
   - `_r6_evidence_forms_ctx` 签名扩展传参 (ctx, question, qa_id=None, conv_msgs=None, q2slot=None, slot_triggers=None), 体内 `if R8_SLOT and qa_id and conv_msgs: ctx = slot_assembly.append_slot_evidence(...)` (R8_SLOT off → 完全原路径零回归)
   - L1654 调用点传 qa_id/conv_msgs/词表 (词表在 R8_SLOT 时加载 .r8-spec/*.json; off 不加载)
3. 新测试 `tests/scripts/test_slot_assembly.py` (TDD 先行):
   - 纯函数可测面: append_slot_evidence 渲染正确性 (构造 ctx+msgs+词条 → 段存在/值行全/无词条原样/空槽位 → 不追加), build_msgs_from_cache 映射 (speaker 前缀/顺序/session)
   - bench 侧不做集成测试 (评测脚本), 以 `python3 -m py_compile` + 语法检查 + 全量 pytest 保证

## G1 硬性用例
append_slot_evidence: ① 有词条有命中 → 段含全部 value 行 ② 无词条 (陌生 qa_id) → ctx 原样 ③ 槽位零命中 → ctx 原样 (不追加空段) ④ 多 value 行时间序 (ts 升序) ⑤ 渲染含 dia 引用且不含全文逐字长句 (截断 ≤120 字符/行) ⑥ ctx 既有内容不被改动 (前缀保留)
build_msgs_from_cache: ⑦ 文本行 "Calvin: text" → speaker=Calvin ⑧ 无前缀 → speaker="" ⑨ 顺序=输入序 ⑩ session_id/ts 映射正确
构造 msgs 用真实文本 (fixtures-r8-oracle.json 的 5 行 snippet 语义) — 断言段值含 mixes/mansion/guitar 相关

## 纪律
- 禁改 slot_facts/slot_extract/core/query_router 及任何其他文件; .r8-spec 只读 (运行时仅读两词表)
- bench 改动仅限上面点位的追加式修改 (diff --stat 应极小, 附证明); 不 commit 不 push 不跑评测 (A/B 归精卫显式启动)
- 禁委派子 agent; 受阻立即写明原因不空转

## 完成标准 (交付摘要附证据)
1. 红→绿 (测试先落盘)
2. pytest tests/scripts/test_slot_assembly.py 全过贴输出
3. 全量 pytest (1306+新增) 零回归贴计数 (bench 不在 pytest 集内但须 py_compile 通过)
4. git status: 新文件 2 (slot_assembly.py + 测试) + bench 单文件修改; diff --stat 附 bench 改动量
5. R8_SLOT=1 干跑验证 (只构造不判卷): 单 conv 单题 append 段输出样例贴出 (可选, 若可行); R8_SLOT 未设 → 行为与基线一致 (哨兵说明)
6. 边界与设计差异说明
