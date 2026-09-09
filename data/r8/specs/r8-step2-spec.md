# R8 Step2 规格: 槽位确定性检索引擎 + QueryRouter.retrieve_slot 接线 (env 门)

> 权威: r8-engine-design.md v2 §1-E4 / §3 / §4-Step2。实现前必读。基线 v6.19.0。
> 数据: 映射 oracle 夹具 .r8-spec/fixtures-r8-oracle.json (5 行真实官方证据, 只读)。

## 目标
评测面可验证性的第一步: (entity×slot) 成员全集确定性检索引擎 (纯) + QueryRouter 挂 retrieve_slot 方法 (R8_CTX 总闸默认 off, off 时逐字节等价 v6.19.0)。slot 值抽取词表与原文装配属 Step3, 本步不实现。

## 交付白名单
1. 新文件 `retrieval/slot_facts.py` (纯, 无 IO 无 LLM 无时间依赖):
   - `SlotFact` dataclass: session_id, ep_id(可比较序), ts(可比较, 入参注入), entity, slot, value, msg_ref(可选)
   - `SlotIndex.from_facts(facts) -> SlotIndex` (去重/归并同实体槽行; 每行保留)
   - `SlotIndex.query(entity, slot, session_ts=None, scope=None) -> list[SlotFact]`:
     * 全量返回不截断; 排序 ts 升序 (同 ts 按 ep_id 升序)
     * scope 非 None → 仅返回 session_id==scope 行 (跨会话同名隔离)
     * session_ts 非 None → 仅 ts<=session_ts (含边界, 与 state_projection resolve 一致)
     * 无命中 → []
   - `SlotIndex.members(entity, slot, session_ts=None, scope=None) -> list[dict]`: 成员行=按 value 归一(小写/strip)去重, 每 value 首次出现行 + mentions=[全部命中 ep_id]; 排序 ts 升序 (接口契约: 成员行去重语义, 供 cat1 LIST/SET 判卷形态)
   - `parse_session_datetime("1:56 pm on 8 May, 2023") -> str|float`: 确定性解析为可比较时间戳 (自定 epoch 基准如 "%Y-%m-%dT%H:%M" 字符串亦可, 须可比较; 不取系统时钟)
2. 修改 `retrieval/query_router.py` (唯一既有文件改动, 追加式, 不动既有逻辑):
   - 新增方法 `retrieve_slot(self, entity, slot, session_ts=None, scope=None)`:
     * 读取 env R8_CTX (os.environ.get("R8_CTX")=="1"); off/缺省 → 返回 [] (日志注明关闭)
     * on 且已注入索引 (`self._r8_slot_index` 由 `set_slot_index(idx)` 设置, 默认 None) → 调 SlotIndex.query 返回
     * on 但索引 None → 返回 [] + 日志
   - 新增方法 `set_slot_index(self, idx)` (置 None 可清除)
3. 新测试 (TDD 先行, 禁 mock 逻辑):
   - `tests/retrieval/test_slot_facts.py`: 引擎 G1
   - `tests/retrieval/test_retrieve_slot.py`: 接线 G1 (env 门 + 索引注入 + off 等价)
   - 测试加载夹具用仓库内路径 .r8-spec/fixtures-r8-oracle.json (只读引用, 勿复制勿改)

## G1 硬性用例
slot_facts: ① 注入 40 行删 5 (同槽不同 value) → query 仍全量返回 35 ② (entity,slot) 无行 → [] ③ session_ts 过滤边界 (== 含) ④ scope 隔离: 同 entity+slot 跨两会话 → 无 scope 全回/带 scope 只回本会话 ⑤ 去重: 同 value 3 提及 → members 1 行 + mentions 含 3 ep_id ⑥ 排序 ts 升序 ⑦ 词形归一: "John" vs "john " 同 member
retrieve_slot 接线: ① env 未设 → [] (off 等价) ② R8_CTX=1 + set_slot_index(引擎实例) → 走引擎返回 ③ R8_CTX=1 无索引 → [] ④ set_slot_index(None) 清除后 → []
oracle (test_slot_facts 内, 读夹具): 断言 5 行夹具每行 (session_id 归因正确=夹具 sample_id; 消息序 dia→ep_id 由 D{s}:{n} 确定性映射 session_index=n? 定义: dia "D{s}:{m}" → session_id=f"{sample_id}-s{s}", ep_id=m; ts 由 parse_session_datetime(session_dt) 解析) → query/members 能回 gold 值 (Jack Russell mixes/Lab mixes/mansion/luxury car/shiny purple guitar 各可达)

## 纪律
- 禁改 retrieval/query_router.py 既有方法体 (只追加新方法, import 增 SlotFact/SlotIndex); 禁改其他任何既有文件 (引擎 core/评测/pkl/图库/灌库/上步 6 白名单文件); .r8-spec 只读
- 纯函数禁 IO (env 读取仅限 retrieve_slot 方法内, 引擎模块本体零 IO)
- 不 commit 不 push 不跑评测; 禁委派子 agent; 受阻立即写明原因不空转
- 上步 3 模块 (core/state_projection.py 等) 已验收, 不得改动

## 完成标准 (交付摘要附证据)
1. 红→绿 (测试先行落盘再实现)
2. pytest tests/retrieval/test_slot_facts.py tests/retrieval/test_retrieve_slot.py 全过贴真实输出
3. 全量 pytest (1277+新增) 零回归贴计数
4. git status: 新增 4 文件 (slot_facts.py + 2 测试 + query_router 修改 tracked) + 既有未跟踪遗留, 无其他改动; 附 git diff --stat 证明 query_router 仅追加
5. env off 等价哨兵: R8_CTX 未设时 retrieve_slot 返回 [] 且 query_router 既有行为不变 (测试 ① 覆盖)
6. 边界与设计差异说明
