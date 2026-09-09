"""达摩院 R6 证据形态三改造 — AC1-AC5 验收测试 (v6.18.0).

三改造 (scripts/*.py 确定性模块 + bench 装配层):
- R6-A fact_clusters: [FACT CLUSTERS] 实体-谓词聚合完整列表 + 事件窗口去重;
- R6-B time_anchors: TIME ANCHORS 行 = 原文短语在前 + 粒度标签 + 污染防御;
- R6-C neg_clean: 摘要否定句过滤 + 组织段题面实体优先。
AC1/AC2/AC3 断言模块行为; AC4 断言原文逐字不变 (三改造附加/过滤);
AC5 断言 reader prompt V1 diff=空 (bench 源码快照对照)。纯文本断言 +
纯函数单测, 不 import bench (其顶层加载检索/灌库/embedding, 非单测面)。
"""
import datetime
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fact_clusters as fc  # noqa: E402
import neg_clean as nc  # noqa: E402
import time_anchors as ta  # noqa: E402

BENCH = SCRIPTS / "bench_locomo_v72_ontology.py"
TEXT = BENCH.read_text(encoding="utf-8")


def _d(y, m, dd):
    return datetime.date(y, m, dd)


# ── AC1: R6-A [FACT CLUSTERS] 聚合 ─────────────────────────────────────────

_AC1_DOCS = [
    (1, "[date: 1:56 pm on 8 May, 2023] [Caroline] I have a dog named Toby."),
    (2, "[date: 2:10 pm on 20 May, 2023] [Caroline] We adopted a kitten called Sara."),
    (3, "[date: 5 June, 2023] [Caroline] And a third pet: a parrot named Mango."),
]


def test_ac1_pet_full_list_with_message_refs():
    """3 宠物消息 → 聚合完整列表带消息号引用 (计数/枚举题服务): 不漏 3/4 只。"""
    block = fc.render_cluster_block(_AC1_DOCS)
    assert "[FACT CLUSTERS]" in block
    assert "3 pets: " in block                      # 完整计数 (不漏项)
    for name, n in (("Toby", 1), ("Sara", 2), ("Mango", 3)):
        assert f"{name} (msg {n})" in block          # 值 + 消息号引用
    # 同值跨消息多次拥有 → 消息号列表并集, 值仍只计一个 (不重复计数宠物数)
    docs = _AC1_DOCS + [(4, "[date: 6 June, 2023] [Caroline] my dog Toby is still with us.")]
    own, _ = fc.aggregate(docs)
    assert own["owns_pet"]["Toby"] == [1, 4]
    block4 = fc.render_cluster_block(docs)
    assert block4.count("Toby") == 1                # 值不因跨消息重复出现
    # 纯提及 (无拥有谓词) 不并入 owns_pet 消息号 (谓词-值聚合, 非实体提及)
    docs5 = _AC1_DOCS + [(5, "[date: 7 June, 2023] [Caroline] Toby loves the garden.")]
    own5, _ = fc.aggregate(docs5)
    assert own5["owns_pet"]["Toby"] == [1]


def test_ac1_event_dedup_same_entity_same_window():
    """事件实例去重: 同实体 (Acme) + 同谓词 (rejection) + 同日期窗口 (周) 只计一次;
    回指表述 ('the Acme rejection' / 'that rejection from Acme') 并入既有实例。"""
    docs = [
        (4, "[date: 9 June, 2023] [Caroline] I got a rejection letter from Acme last week."),
        (5, "[date: 15 June, 2023] [Caroline] the Acme rejection still stings."),
        (6, "[date: 16 June, 2023] [Caroline] also that rejection from Acme came with feedback."),
    ]
    block = fc.render_cluster_block(docs)
    assert "rejection: 1 instance from Acme (msg 4, 5, 6)" in block
    _, events = fc.aggregate(docs)
    assert len(events) == 1                          # 三消息归并为一个实例


def test_ac1_harness_wiring_fact_clusters():
    """bench 装配层存在: FACT_CLUSTERS env 开关 (默认 off) + _r6_evidence_forms_ctx
    接线 + append_fact_clusters 调用 (R6-A 在 ctx 装配层, 同 P0-b TIME ANCHORS 路径)。"""
    assert 'FACT_CLUSTERS = os.environ.get("FACT_CLUSTERS", "0") == "1"' in TEXT
    # R8 Step3b (规格 data/r8/r8-step3b-spec.md): _r6_evidence_forms_ctx 签名扩展 —
    # 追加 qa_id/conv_msgs/q2slot/slot_triggers 供 R8_SLOT 分支装配 [SLOT EVIDENCE] 段。
    assert "def _r6_evidence_forms_ctx(ctx, question, qa_id=None, conv_msgs=None," in TEXT
    assert "ctx = _r6_evidence_forms_ctx(ctx, question)" in TEXT
    assert "fact_clusters.append_fact_clusters(ctx, max_docs=30)" in TEXT
    assert "import fact_clusters" in TEXT


# ── AC2: R6-B TIME ANCHORS 行形态 (原文短语在前 + 粒度标签) ────────────────

def test_ac2_time_anchor_line_phrase_before_absolute():
    """TIME ANCHORS 行 = 原文相对短语置前 + 解析区间置后 (reader 可回显 REL-gold,
    判卷禁换算); 断言短语在行中先于解析区间出现。"""
    msg = "[date: 9 June, 2023] [Caroline] my school event was the week before 9 June 2023."
    ann = ta.inline_annotation(msg)
    assert ann is not None
    phrase_idx = ann.find("the week before 9 June 2023")
    abs_idx = ann.find("29 May 2023 to 4 June 2023")
    assert phrase_idx >= 0 and abs_idx >= 0
    assert phrase_idx < abs_idx                     # 原文短语在前

    org = ("[DIRECT EVIDENCE]\n"
           "[1] [date: 9 June, 2023] [Caroline] my school event was the week before 9 June 2023.\n")
    out = ta.append_anchor_block(org)
    line = next(l for l in out.splitlines() if l.startswith("[1]") and " -> " in l)
    assert line.index("the week before 9 June 2023") < line.index("29 May 2023 to 4 June 2023")


def test_ac2_granularity_label_present():
    """解析结果带粒度标签 [granularity: day|week|month|year|range], 让 reader 对齐
    gold 粒度 (粒度越界 RANGE 17/MONTH 9/YEAR 4 是 cat2 主错因)。"""
    # week 粒度
    a = ta.find_anchors("[date: 9 June, 2023] [X] my event was the week before 9 June 2023.")
    assert a and a[0].granularity == "week"
    # day 粒度 (yesterday)
    b = ta.find_anchors("[date: 8 May, 2023] [X] I went yesterday.")
    assert b and b[0].granularity == "day"
    # range 粒度 (weekend 两天块)
    c = ta.find_anchors("[date: 17 July, 2023] [X] we went camping last weekend.")
    assert c and c[0].granularity == "range"
    # 行内渲染含 [granularity: ...]
    ann = ta.inline_annotation("[date: 9 June, 2023] [X] my event was the week before 9 June 2023.")
    assert ann is not None and "[granularity: week]" in ann


def test_ac2_predates_msg_date_annotation():
    """消息日期污染防御: 事件相对词解析区间整体早于消息 [date:] → 标注
    [predates msg date] (防 reader 把消息日期当日当事件日期 — cat2 污染)。"""
    ann = ta.inline_annotation("[date: 1:56 pm on 8 May, 2023] [X] I went to a support group yesterday.")
    assert ann is not None and "[predates msg date]" in ann
    # 事件不早于消息日期 (today/this week 类) → 不标注
    ann2 = ta.inline_annotation("[date: 8 May, 2023] [X] I will travel next week.")
    assert ann2 is not None and "[predates msg date]" not in ann2


def test_ac2_original_p0b_calendar_tests_stay_green():
    """原 P0-b 真日历语义不回退 (行格式/粒度改动不破坏解析值): 相对词 → 绝对区间
    仍精确; 渲染两形态仍同现。"""
    msg = "[date: 17 July, 2023] [Melanie] we went camping with my fam two weekends ago."
    ann = ta.inline_annotation(msg)
    assert ann is not None
    assert "two weekends ago" in ann and "8 July 2023 to 9 July 2023" in ann
    # 组织段 [TIME ANCHORS] 仍引用 DIRECT EVIDENCE 编号
    org = ("[DIRECT EVIDENCE]\n"
           "[1] [date: 17 July, 2023] [Melanie] we went camping two weekends ago.\n")
    out = ta.append_anchor_block(org)
    assert "[TIME ANCHORS" in out and "[1] two weekends ago -> " in out


# ── AC3: R6-C 摘要清洁 ─────────────────────────────────────────────────────

def test_ac3_negative_sentence_filtered():
    """装配层过滤摘要中的否定式表述整句: 含 'cannot be determined' 的块摘要输出被
    滤/改写为事实行或空 (拒答修复; 不改 LLM prompt)。"""
    line = "[MEMORY BLOCK 1] Caroline adopted Toby. The exact date cannot be determined."
    cleaned = nc.clean_summary_line(line)
    assert "cannot be determined" not in cleaned
    assert "Caroline adopted Toby." in cleaned      # 事实句保留
    # 全否定摘要 → 空 (输出空段而非否定句)
    assert nc.clean_summary_line("[MEMORY BLOCK 2] The pet's name is unspecified.") == ""
    assert nc.filter_negative_sentences("No evidence about the trip date.") == ""


def test_ac3_entity_first_ordering():
    """组织段按题面实体优先排列: 题面含实体名 → 该实体 [ENTITY: ...] 段前置。"""
    ctx = ("[ENTITY: Zeta]\n- zeta facts\n\n"
           "[ENTITY: Alpha]\n- alpha facts\n\n"
           "[RELATIONS]\n- X\n\n"
           "[DIRECT EVIDENCE]\n[1] [date: 8 May, 2023] [X] alpha text")
    out = nc.entity_first_ctx(ctx, "Which facts about Alpha are in the log?")
    assert out.index("[ENTITY: Alpha]") < out.index("[ENTITY: Zeta]")
    # 无题面实体命中 → 原 ctx (零回归)
    assert nc.entity_first_ctx(ctx, "How many trips did we plan?") == ctx


def test_ac3_neg_clean_harness_wiring():
    """bench 装配层存在: NEG_CLEAN env 开关 (默认 off) + clean_ctx/entity_first_ctx 接线。"""
    assert 'NEG_CLEAN = os.environ.get("NEG_CLEAN", "0") == "1"' in TEXT
    assert "ctx = neg_clean.clean_ctx(ctx)" in TEXT
    assert "ctx = neg_clean.entity_first_ctx(ctx, question)" in TEXT
    assert "import neg_clean" in TEXT


# ── AC4: 原文消息逐字不变 ──────────────────────────────────────────────────

def test_ac4_raw_messages_verbatim_under_all_forms():
    """三改造均为附加/过滤: R6-A 聚合 / R6-B 时间锚行 / R6-C 否定过滤 都不改 raw
    编号消息行原文 (判卷近重复敏感, F5 + R2c 教训)。"""
    raw = ("[DIRECT EVIDENCE]\n"
           "[1] [date: 8 May, 2023] [Caroline] I went to a support group yesterday and it was so powerful.\n"
           "[2] [date: 9 June, 2023] [Caroline] my school event was last week, the date cannot be determined.\n")
    # R6-A: append_fact_clusters 只追加段, 不改行
    out_fc = fc.append_fact_clusters(raw)
    for line in raw.splitlines():
        assert line in out_fc, f"R6-A 改动原文: {line!r}"
    # R6-B: append_anchor_block 只追加 [TIME ANCHORS] 段
    out_ta = ta.append_anchor_block(raw)
    for line in raw.splitlines():
        assert line in out_ta, f"R6-B 改动原文: {line!r}"
    # R6-C: clean_ctx 只过滤非 raw 摘要行; raw 编号消息行原样 (含否定词也不动)
    ctx_with_summary = raw + "[MEMORY BLOCK 1] A trip happened. The date is unspecified.\n"
    out_neg = nc.clean_ctx(ctx_with_summary)
    for line in raw.splitlines():
        assert line in out_neg, f"R6-C 改动原文: {line!r}"
    assert "The date is unspecified." not in out_neg


def test_ac4_aggregate_does_not_mutate_input():
    """fact_clusters.aggregate 纯函数: 输入文档逐字节不变。"""
    docs = list(_AC1_DOCS)
    before = list(docs)
    fc.aggregate(docs)
    assert docs == before


# ── AC5: reader prompt V1 diff=空 ─────────────────────────────────────────

_READER_PROMPT_V1_SNAPSHOT = """Answer the question based on the conversation snippets below. Reason across snippets if needed (e.g., infer dates from session timestamps).

Conversation snippets:
{ctx}

Question: {question}
Answer:"""


def test_ac5_reader_prompt_v1_unchanged():
    """reader prompt V1 原文逐字节不变 (AC5 diff=空) — R6 三改造全在 ctx 装配层
    (证据结构形态), prompt 侧零改动; 禁任何 '请换算/请聚合' 式指令 (R2c 覆辙)。"""
    m = re.search(r"_READER_PROMPT_V1 = (\"\"\".*?\"\"\")", TEXT, re.S)
    assert m is not None
    assert m.group(1) == f'"""{_READER_PROMPT_V1_SNAPSHOT}"""'
    # 新开关只出现在装配层, 不进入 prompt 常量
    assert "FACT_CLUSTERS" not in _READER_PROMPT_V1_SNAPSHOT
    assert "NEG_CLEAN" not in _READER_PROMPT_V1_SNAPSHOT
    for banned in ("please aggregate", "please clean", "请聚合", "请过滤"):
        assert banned not in TEXT


# ── R7-1 极性守卫 (达摩院 R6 FAIL 研究 §7, 2026-09-05) ──
# NEG_CLEAN 重写: 只剔"认识论退却"帧 (cannot be determined/无法判断/unspecified),
# 保留事实性否定与缺席断言 ("no evidence that X moved" / "X didn't mention Y" —
# 这些是 Yes/No 与缺席类 gold 的正确依据, conv-44#q0045 反面教材)。

_EPISTEMIC_DROP = [
    "The date cannot be determined from the conversation.",
    "It is unclear whether Caroline attended.",
    "Insufficient evidence to answer this question.",
    "The pet's name is unspecified.",
    "The date is unspecified.",
    "无法确定具体时间。",
    "信息不足, 无从得知。",
    "Jon's location is not clear from the context.",
    "She could not be determined to have moved.",
]

_FACTUAL_KEEP = [
    "There is no evidence that Alex moved to Seattle.",
    "No evidence that she adopted a pet.",
    "Sara did not mention any pets.",
    "John never said he liked basketball.",
    "She didn't mention Y at all.",
    "未提及任何宠物。",
    "The dog's name was not mentioned by Toby.",
]

_EPISTEMIC_DROP_EXTRA = [
    "No evidence about the trip date.",  # 泛化缺席(无事实宾语) → 剔
    "There is no evidence in the conversation.",  # 泛化 → 剔
]


def test_r71_polarity_guard_drops_epistemic_retreat():
    """认识论退却帧 → 剔 (is_negative=True): 诱导拒答的元认知退却。"""
    for s in _EPISTEMIC_DROP + _EPISTEMIC_DROP_EXTRA:
        assert nc.is_negative(s), f"应剔认识论退却: {s}"


def test_r71_polarity_guard_keeps_factual_negation():
    """事实性否定/缺席断言 → 保留 (is_negative=False): Yes/No 题的正确答案依据。"""
    for s in _FACTUAL_KEEP:
        assert not nc.is_negative(s), f"应保留事实否定: {s}"

