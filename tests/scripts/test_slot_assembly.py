"""R8 Step3b 确定性槽证据装配 — scripts/slot_assembly.py G1 oracle 单测 (TDD 先行).

本体论验收句: 评测组织段 ctx 如何确定性获得槽位 (entity×slot) 证据闭包?
    → Step3a 抽取管线 (build_slot_index) + SlotIndex.query 行全量, 以 [SLOT
      EVIDENCE] 段追加 ctx 尾部 (reader 前), 行=value+dia 引用+ts, 不复制原文
      逐字 (超 120 字符截断) → 过。

G1 硬性覆盖 (spec §G1):
append_slot_evidence ① 有词条有命中 → 段含全部 value 行 ② 无词条 (陌生 qa_id)
→ ctx 原样 ③ 槽位零命中 → ctx 原样 (不追加空段) ④ 多 value 行时间序 (ts 升序)
⑤ 渲染含 dia 引用且不含全文逐字长句 (截断 ≤120 字符/行) ⑥ ctx 既有内容不被改动
(前缀保留); build_msgs_from_cache ⑦ 文本行 "Calvin: text" → speaker=Calvin
⑧ 无前缀 → speaker="" ⑨ 顺序=输入序 ⑩ session_id/ts 映射正确。
oracle (真实文本, 读 data/r8/fixtures-r8-oracle.json 5 行 snippet 语义): 段值含
mixes / mansion / guitar(shiny) 相关。
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO))

import slot_assembly  # noqa: E402
from retrieval.slot_facts import parse_session_datetime  # noqa: E402

_FIXTURE = REPO / "data/r8" / "fixtures-r8-oracle.json"
_Q2SLOT = REPO / "data/r8" / "q2slot-v1.json"
_TRIGGERS = REPO / "data/r8" / "slot-triggers-v1.json"


def _load(name):
    return json.loads(name.read_text(encoding="utf-8"))


ORACLE = _load(_FIXTURE)["facts"]
LEXICON = _load(_Q2SLOT)
TRIGGERS = _load(_TRIGGERS)

# 会话内时间锚 (确定性可比较, 入参注入; 断言只关心相对序 / 展示串)。
_T1 = "2023-03-23T11:53"
_T2 = "2023-03-26T16:45"
_T3 = "2023-08-31T14:55"


def _msg(dia, session_id, speaker, text, ts):
    """append_slot_evidence 入参消息行 (会话内时间序)。"""
    return {"dia": dia, "session_id": session_id, "speaker": speaker,
            "text": text, "ts": ts}


def _oracle_snippets(sample_id, slot):
    """夹具行 (只读): sample_id+slot → [(dia_id, speaker, session_dt, snippet)]。"""
    return [
        (r["dia_id"], r["speaker"], r["session_dt"], r["snippet"])
        for r in ORACLE
        if r["sample_id"] == sample_id and r["slot"] == slot
    ]


def _ctx_rows(section):
    """[SLOT EVIDENCE] 段文本 → [(dia, value, ts)] (value=行 '- value:' 后片段)。"""
    out = []
    for ln in section.splitlines():
        if not ln.startswith("- value:"):
            continue
        body = ln[len("- value:"):].strip()
        m = re.search(r"\s{2,}\(msg ([^,]+), ([^)]*)\)\s*$", body)
        if not m:
            out.append((None, body, ""))
            continue
        val = body[:m.start()].strip()
        out.append((m.group(1).strip(), val, m.group(2).strip()))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# append_slot_evidence — G1 ①–⑥
# ═══════════════════════════════════════════════════════════════════════════

def test_01_has_entry_with_hits_renders_all_value_rows():
    """① 有词条有命中 (conv-50#q0001 Calvin purchase_item) → 段含全部 value 行。"""
    ctx = "[DIRECT EVIDENCE]\n[1] some retrieved line"
    rows = _oracle_snippets("conv-50", "purchase_item")  # D1:3 mansion / D2:1 car
    msgs = []
    for dia, speaker, sdt, snippet in rows:
        s, _m = dia.lstrip("D").split(":")
        msgs.append(_msg(dia, f"conv-50-s{s}", speaker, snippet,
                         parse_session_datetime(sdt)))
    out = slot_assembly.append_slot_evidence(ctx, "What did Calvin buy?",
                                             "conv-50#q0001", msgs, LEXICON, TRIGGERS)
    assert out.startswith(ctx), "既有 ctx 前缀保留"
    assert "[SLOT EVIDENCE] Calvin | purchase_item (确定性槽证据, 时间序):" in out
    rows_out = _ctx_rows(out)
    values = [v.lower() for _d, v, _t in rows_out]
    assert any("mansion" in v for v in values), rows_out
    assert any("car" in v for v in values), rows_out
    dias = {d for d, _v, _t in rows_out}
    assert {"D1:3", "D2:1"} <= dias, dias


def test_02_unknown_qa_id_returns_ctx_unchanged():
    """② 无词条 (陌生 qa_id) → ctx 原样 (不报错, 不追加)。"""
    ctx = "[DIRECT EVIDENCE]\n[1] some line"
    msgs = [_msg("D1:3", "conv-50-s1", "Calvin",
                 "I just bought a new mansion", _T1)]
    out = slot_assembly.append_slot_evidence(ctx, "q", "conv-999#q0000",
                                             msgs, LEXICON, TRIGGERS)
    assert out == ctx


def test_03_zero_slot_hits_returns_ctx_unchanged():
    """③ 槽位零命中 (词条存在但 msgs 无触发证据) → ctx 原样 (不追加空段)。"""
    ctx = "some existing context"
    msgs = [_msg("D9:1", "conv-50-s9", "Calvin",
                 "I went to the park for a walk", _T1)]
    out = slot_assembly.append_slot_evidence(ctx, "q", "conv-50#q0001",
                                             msgs, LEXICON, TRIGGERS)
    assert out == ctx
    assert "[SLOT EVIDENCE]" not in out


def test_04_multi_value_rows_time_ascending():
    """④ 多 value 行时间序 (ts 升序): 早 ts 的 value 行出现在晚 ts 之前。"""
    ctx = "prefix"
    msgs = [
        _msg("D2:1", "conv-50-s2", "Calvin", "I just bought a new car", _T2),
        _msg("D1:3", "conv-50-s1", "Calvin", "I bought a mansion", _T1),
    ]
    out = slot_assembly.append_slot_evidence(ctx, "q", "conv-50#q0001",
                                             msgs, LEXICON, TRIGGERS)
    rows_out = _ctx_rows(out)
    ts_list = [t for _d, _v, t in rows_out]
    assert ts_list == sorted(ts_list), rows_out
    # mansion (D1:3, 早) 行先于 car (D2:1, 晚) 行。
    i_m = out.index("- value: mansion")
    i_c = out.index("- value: car")
    assert i_m < i_c, out


def test_05_line_has_dia_ref_and_truncates_long_verbatim():
    """⑤ 渲染含 dia 引用且不含全文逐字长句 (value 截断 ≤120 字符)。"""
    ctx = "prefix"
    long_text = ("After weeks of saving and careful consideration and driving all "
                 "over the city to compare every single option at dozens of "
                 "different dealerships across three states, I finally bought "
                 "a silver sports sedan with leather seats, a premium sound "
                 "system and every safety package the dealer offered")
    assert len(long_text) > 200
    msgs = [_msg("D1:7", "conv-50-s1", "Calvin", long_text, _T1)]
    out = slot_assembly.append_slot_evidence(ctx, "q", "conv-50#q0001",
                                             msgs, LEXICON, TRIGGERS)
    assert out.startswith(ctx)
    rows_out = _ctx_rows(out)
    assert rows_out, out
    dia, val, ts = rows_out[0]
    assert dia == "D1:7"
    assert long_text not in val, "不复制原文逐字长句"
    assert len(val) <= 120, (len(val), val)


def test_06_existing_ctx_prefix_untouched():
    """⑥ ctx 既有内容不被改动 (前缀逐字保留, 段只追加在尾部)。"""
    ctx = "[DIRECT EVIDENCE]\n[1] first line\n[2] second line"
    msgs = [_msg("D1:3", "conv-50-s1", "Calvin",
                 "I just bought a new mansion", _T1)]
    out = slot_assembly.append_slot_evidence(ctx, "q", "conv-50#q0001",
                                             msgs, LEXICON, TRIGGERS)
    assert out[:len(ctx)] == ctx
    assert out[len(ctx):].lstrip("\n").startswith("[SLOT EVIDENCE]")


# ═══════════════════════════════════════════════════════════════════════════
# build_msgs_from_cache — G1 ⑦–⑩
# ═══════════════════════════════════════════════════════════════════════════

def test_07_colon_prefix_text_yields_speaker():
    """⑦ 文本行 "Calvin: text" → speaker=Calvin, text 去前缀。"""
    by_id = {"ep_0": "Calvin: That event sounds great! I just bought a mansion"}
    cache = {"ep_0": {"id": "ep_0", "session_id": "conv-50",
                      "created_at": 1000.0, "content": by_id["ep_0"]}}
    msgs = slot_assembly.build_msgs_from_cache(by_id, cache, {"conv-50": "11:53 am on 23 March, 2023"})
    assert msgs[0]["speaker"] == "Calvin"
    assert msgs[0]["text"] == "That event sounds great! I just bought a mansion"
    assert msgs[0]["dia"] == "ep_0"


def test_08_no_prefix_yields_empty_speaker():
    """⑧ 无说话人前缀 → speaker="", text 原样。"""
    by_id = {"ep_1": "Just a plain message without any speaker prefix."}
    cache = {"ep_1": {"id": "ep_1", "session_id": "conv-50", "created_at": 1.0}}
    msgs = slot_assembly.build_msgs_from_cache(by_id, cache, {})
    assert msgs[0]["speaker"] == ""
    assert msgs[0]["text"] == "Just a plain message without any speaker prefix."


def test_09_output_order_matches_input_order():
    """⑨ 顺序=msg_by_id 输入序 (即便 key 数值乱序)。"""
    by_id = {
        "ep_9": "Calvin: first in dict",
        "ep_2": "Calvin: second in dict",
        "ep_5": "Calvin: third in dict",
    }
    cache = {k: {"id": k, "session_id": "conv-50", "created_at": float(i)}
             for i, k in enumerate(by_id)}
    msgs = slot_assembly.build_msgs_from_cache(by_id, cache, {})
    assert [m["dia"] for m in msgs] == ["ep_9", "ep_2", "ep_5"]


def test_10_session_id_and_ts_mapping():
    """⑩ session_id/ts 从 episode_cache/会话日期映射正确 (日期→规范时间串)。"""
    by_id = {"ep_0": "Calvin: new mansion"}
    cache = {"ep_0": {"id": "ep_0", "session_id": 5, "created_at": 999.0}}
    sess_date = {5: "11:53 am on 23 March, 2023"}
    msgs = slot_assembly.build_msgs_from_cache(by_id, cache, sess_date)
    assert msgs[0]["session_id"] == 5
    assert msgs[0]["ts"] == "2023-03-23T11:53"


def test_10b_bracket_db_form_cache_parsing():
    """bench 灌库 DB 同形 content ('[date: ...] [Speaker] text') → speaker/text/ts。"""
    content = "[date: 11:53 am on 23 March, 2023] [Calvin] That event sounds great!"
    by_id = {"ep_0": content}
    cache = {"ep_0": {"id": "ep_0", "session_id": "conv-50", "content": content}}
    msgs = slot_assembly.build_msgs_from_cache(by_id, cache,
                                               {"conv-50": "11:53 am on 23 March, 2023"})
    assert msgs[0]["speaker"] == "Calvin"
    assert msgs[0]["text"] == "That event sounds great!"
    assert msgs[0]["ts"] == "2023-03-23T11:53"


# ═══════════════════════════════════════════════════════════════════════════
# oracle — 真实文本 (fixtures 5 行 snippet 语义): 段值含 mixes / mansion / guitar
# ═══════════════════════════════════════════════════════════════════════════

def test_oracle_dog_breed_mixes():
    """conv-44 q0035 dog_breed (Audrey): D19:12/D26:13 证据 → 段值含 'mixes'。"""
    ctx = "existing"
    msgs = []
    for dia, speaker, sdt, snippet in _oracle_snippets("conv-44", "dog_breed"):
        s, _m = dia.lstrip("D").split(":")
        msgs.append(_msg(dia, f"conv-44-s{s}", speaker, snippet,
                         parse_session_datetime(sdt)))
    qa_id = "conv-44#q0035"
    assert qa_id in LEXICON
    out = slot_assembly.append_slot_evidence(ctx, "q", qa_id, msgs, LEXICON, TRIGGERS)
    values = " ".join(v for _d, v, _t in _ctx_rows(out)).lower()
    assert "mixes" in values, out


def test_oracle_purchase_mansion():
    """conv-50 q0001 purchase_item (Calvin): D1:3/D2:1 证据 → 段值含 mansion/car。"""
    ctx = "existing"
    msgs = []
    for dia, speaker, sdt, snippet in _oracle_snippets("conv-50", "purchase_item"):
        s, _m = dia.lstrip("D").split(":")
        msgs.append(_msg(dia, f"conv-50-s{s}", speaker, snippet,
                         parse_session_datetime(sdt)))
    qa_id = "conv-50#q0001"
    out = slot_assembly.append_slot_evidence(ctx, "q", qa_id, msgs, LEXICON, TRIGGERS)
    values = " ".join(v for _d, v, _t in _ctx_rows(out)).lower()
    assert "mansion" in values, out
    assert "car" in values, out


def test_oracle_guitar_style_shiny():
    """conv-50 q0139 guitar_style (Calvin, Dave 他述直呼): D16:19 → 段值含 shiny。"""
    ctx = "existing"
    msgs = []
    for dia, speaker, sdt, snippet in _oracle_snippets("conv-50", "guitar_style"):
        s, _m = dia.lstrip("D").split(":")
        msgs.append(_msg(dia, f"conv-50-s{s}", speaker, snippet,
                         parse_session_datetime(sdt)))
    qa_id = "conv-50#q0139"
    assert qa_id in LEXICON
    out = slot_assembly.append_slot_evidence(ctx, "q", qa_id, msgs, LEXICON, TRIGGERS)
    rows_out = _ctx_rows(out)
    values = " ".join(v for _d, v, _t in rows_out).lower()
    assert "shiny" in values, out
    assert any(d == "D16:19" for d, _v, _t in rows_out), rows_out
