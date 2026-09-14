# -*- coding: utf-8 -*-
"""P2 槽位闭包 v2 测试 — retrieval/slot_closure.py + scripts/slot_assembly.render_slot_block_v2.

依据: /home/user/shm-evolve/studies/r8-p2-taskbook.md
覆盖: 问题类型路由 / 值规范化与去噪 / 枚举切分 / 成员去重排序 / 预算 / 替换式装配
      / 非集合题与零命中不注入 / 原文消息行逐字不变 (红线) / 无 qa_id 查表 (红线)
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import slot_assembly  # noqa: E402
from retrieval.slot_closure import (  # noqa: E402
    _np_after_trigger, _trigger_span, clean_members, detect_value_type, extract_objects_v2,
    is_count_question, is_noise_value, is_set_question, load_value_types, normalize_value,
    render_closure, route_question, score_member, split_enumeration, type_match,
)
from retrieval.slot_facts import SlotFact  # noqa: E402

_TRIG = {"book_read": ["read", "reading", "book", "novel"],
         "vacation_state": ["vacation", "visited", "went to", "trip", "travel"]}


# ── 问题类型路由 ─────────────────────────────────────────────────────────
def test_set_question_positives():
    assert is_set_question("What books has Melanie read?")
    assert is_set_question("How many times did Jon visit?")
    assert is_set_question("Which events has Jon participated in?")
    assert is_set_question("List the places Maria visited")


def test_set_question_negatives():
    assert not is_set_question("When did Maria move to Oregon?")
    assert not is_set_question("Who supports Caroline?")
    assert not is_set_question("")
    assert is_count_question("How many times did Jon visit?")
    assert not is_count_question("What books has Melanie read?")


# ── 值规范化与去噪 ───────────────────────────────────────────────────────
def test_normalize_value_strips_fillers_and_punct():
    assert normalize_value("to Oregon.") == "Oregon"
    assert normalize_value('"Charlotte\'s Web"') == "Charlotte's Web"
    assert normalize_value("[img: a photo] mansion") == "mansion"
    assert normalize_value("  in   Florida ") == "Florida"


def test_is_noise_value():
    for junk in ["I", "me", "so", "it", "my", "you", "?", "-", "  ",
                 "you got in your library?", "and then we went"]:
        assert is_noise_value(junk), junk
    for good in ["Oregon", "Charlotte's Web", "networking events", "yoga"]:
        assert not is_noise_value(good), good


def test_split_enumeration():
    assert split_enumeration("fair, networking events") == ["fair", "networking events"]
    assert split_enumeration("Oregon, Florida") == ["Oregon", "Florida"]
    assert split_enumeration("mansion") == ["mansion"]


def test_score_member_prefers_question_overlap_and_proper_nouns():
    assert score_member("Oregon", "What states has Maria vacationed at?") >= 1
    assert score_member("Oregon trip", "What states has Maria vacationed at?") > score_member("we went somewhere", "What states has Maria vacationed at?")


# ── 成员清理与排序 ───────────────────────────────────────────────────────
def _fact(v, ts="2023-05-01", ref="ep_1"):
    return SlotFact(session_id=0, ep_id=1, ts=ts, entity="Maria", slot="vacation_state",
                    value=v, msg_ref=ref)


def test_clean_members_dedup_and_filter():
    pairs = [("to Oregon.", _fact("to Oregon.")),
             ("Oregon", _fact("Oregon", ts="2023-06-01", ref="ep_9")),
             ("I", _fact("I")),
             ("you got in your library?", _fact("you got in your library?")),
             ("Florida", _fact("Florida", ts="2023-07-01", ref="ep_12"))]
    members = clean_members(pairs, question="What states has Maria vacationed at?", entity="Maria")
    vals = [m["value"] for m in members]
    assert vals.count("Oregon") == 1, vals          # 去重 (规范化后同键)
    assert "Florida" in vals
    assert "I" not in vals and not any("?" in v for v in vals), vals


# ── 渲染与预算 ───────────────────────────────────────────────────────────
def test_render_closure_budget_and_provenance():
    members = [{"value": f"place-{i}", "ts": "2023-05-01", "msg_ref": f"ep_{i}", "score": 1}
               for i in range(50)]
    seg = render_closure(members, "Maria", "vacation_state", budget_chars=200, max_members=80)
    assert seg.startswith("[SLOT EVIDENCE] Maria | vacation_state")
    assert len(seg) <= 260
    assert "(msg ep_0)" in seg
    assert render_closure([], "Maria", "vacation_state") == ""


# ── 题面现场路由 (无 qa_id 查表, 红线) ───────────────────────────────────
def test_route_question_uses_text_not_qa_id():
    hits = route_question("What books has Melanie read?",
                          speakers=["Melanie", "Caroline"], triggers=_TRIG)
    assert hits, "应能在无 qa_id 情况下从题面路由"
    assert hits[0]["entity"] == "Melanie"
    assert hits[0]["slot"] == "book_read"
    assert "via" in hits[0] and "question" in hits[0]["via"]


def test_route_question_empty_when_no_signal():
    assert route_question("", speakers=["Melanie"], triggers=_TRIG) == []


# ── 替换式装配 (含红线断言) ──────────────────────────────────────────────
_MSGS = [
    {"dia": "ep_1", "session_id": 0, "speaker": "Maria", "ts": "2023-05-01",
     "text": "I went on a road trip to Oregon."},
    {"dia": "ep_2", "session_id": 0, "speaker": "Maria", "ts": "2023-06-01",
     "text": "We visited Florida last year."},
]
_CTX = ("[ENTITY: conv-41/Maria]\n"
        "- conv-41/Maria learned_value: different perspectives\n"
        "- conv-41/Maria asked_about: John's inspiration\n"
        "\n"
        "[DIRECT EVIDENCE]\n"
        "[1] [date: 1 Jan, 2023] [Maria] I went on a road trip to Oregon.\n"
        "[2] [date: 2 Feb, 2023] [Maria] We visited Florida last year.\n")
_LEX = {"conv-41#q0029": {"entities": ["Maria"], "slot": "vacation_state", "gate": "did"}}


def test_v2_replaces_entity_segment_keeps_raw_lines():
    q = "What states has Maria vacationed at?"
    out = slot_assembly.render_slot_block_v2(_CTX, q, "conv-41#q0029", _MSGS, _LEX, _TRIG)
    assert out != _CTX
    assert "[SLOT EVIDENCE]" in out
    assert "[ENTITY: conv-41/Maria]" not in out, "组织段应被替换"
    # 红线: 原文消息行逐字不变
    for line in _CTX.splitlines():
        if line.startswith("[1] ") or line.startswith("[2] "):
            assert line in out, f"原文行被改动: {line}"


def test_v2_non_set_question_no_injection():
    q = "When did Maria move to Oregon?"
    assert slot_assembly.render_slot_block_v2(_CTX, q, "conv-41#q0029", _MSGS, _LEX, _TRIG) == _CTX


def test_v2_zero_members_no_injection():
    """零命中不注入 (类型无关题 + 词表实体无命中 → 不注入, 隔离路由影响)。

    2026-09-11 语义更新: 类型题 (us_state 等) 走**类型锚定通道**, 同会话内
    该类型的成员不再要求说话人归属 → 类型题总能注入 (实测 r@10 0.191→0.207);
    因此"零命中不注入"的用例改用类型无关题来断言。
    """
    q = "What did Nobody say about the project?"
    empty_lex = {"conv-41#q0029": {"entities": ["Nobody"], "slot": "vacation_state", "gate": "did"}}
    out = slot_assembly.render_slot_block_v2(_CTX, q, "conv-41#q0029", _MSGS, empty_lex, _TRIG,
                                             route_first=False)
    assert "[SLOT EVIDENCE]" not in out


def test_v2_route_first_bypasses_lexicon():
    """v2 主路径: 题面现场路由 (不依赖 qa_id 词表) —— 空词表也能注入。"""
    q = "What states has Maria vacationed at?"
    out = slot_assembly.render_slot_block_v2(_CTX, q, "conv-41#q0029", _MSGS, {}, _TRIG)
    assert "[SLOT EVIDENCE]" in out and "Maria" in out


def test_v2_append_when_no_entity_segment():
    q = "What states has Maria vacationed at?"
    out = slot_assembly.render_slot_block_v2("BASE-CTX", q, "conv-41#q0029", _MSGS, _LEX, _TRIG)
    assert out.startswith("BASE-CTX")
    assert "[SLOT EVIDENCE]" in out


# ── v1 路径不受影响 (off 等价) ───────────────────────────────────────────
def test_v1_append_unchanged_for_empty_lexicon():
    assert slot_assembly.append_slot_evidence("X", "q", "unknown#q0", _MSGS, {}, _TRIG) == "X"


# ── P1 类型化值校验 ──────────────────────────────────────────────────────
def test_detect_value_type():
    assert detect_value_type("What states has Maria vacationed at?") == "us_state"
    assert detect_value_type("Which countries did he visit?") == "country"
    assert detect_value_type("What books has Melanie read?") == "work_book"
    assert detect_value_type("What activities has John done?") == "activity"
    assert detect_value_type("How many times did Jon visit?") == "count"
    assert detect_value_type("What did Caroline buy?") == "generic"


def test_type_match_closed_vocab():
    res = load_value_types()
    assert type_match("Oregon", "us_state", res)
    assert not type_match("photo album", "us_state", res)
    assert type_match("Florida", "us_state", res)
    assert not type_match("mansion", "us_state", res)


def test_type_match_places_and_activities():
    res = load_value_types()
    assert type_match("France", "country", res)
    assert type_match("London", "place", res)
    assert not type_match("Charlotte's Web", "place", res)
    assert type_match("hiking", "activity", res)
    assert type_match("surfing", "activity", res)
    assert not type_match("cherishing the time we", "activity", res), "长从句不得算活动"


def test_type_match_generic_passthrough():
    assert type_match("anything at all", "generic", {})


def test_normalize_extracts_quoted_span():
    assert normalize_value('He read "Charlotte\'s Web" as a kid') == "Charlotte's Web"
    assert normalize_value('"Becoming Nicole"') == "Becoming Nicole"
    assert normalize_value("mansion") == "mansion"


def test_clean_members_typed_filter_keeps_only_typed():
    from retrieval.slot_facts import SlotFact
    pairs = [("to Oregon.", SlotFact(session_id=0, ep_id=1, ts="2023-05-01", entity="Maria",
                                     slot="vacation_state", value="to Oregon.", msg_ref="ep_1")),
             ("group has made me feel accepted",
              SlotFact(session_id=0, ep_id=2, ts="2023-05-02", entity="Maria",
                       slot="vacation_state", value="group has made me feel accepted", msg_ref="ep_2"))]
    members = clean_members(pairs, question="What states has Maria vacationed at?", entity="Maria")
    vals = [m["value"] for m in members]
    assert vals == ["Oregon"], vals   # 非州名的从句被类型门剔除


# ── A) 谓词论元抽取 (宾语 NP) ────────────────────────────────────────────
def test_trigger_span_word_boundary():
    """回归: 触发词必须词边界匹配 (曾用 str.find 命中单词内部 → 产出 "ed Italy" 碎片)。"""
    sp = _trigger_span("We visited Italy last year.", "visit")
    assert sp is not None
    assert "We visited Italy last year."[sp[0]:sp[1]].lower() == "visited"
    assert _trigger_span("He is visiting Rome.", "visit") is not None
    assert _trigger_span("She has a visitor badge.", "visit") is not None   # visitor 也算屈折
    assert _trigger_span("Nothing here.", "visit") is None


def test_np_after_trigger_skips_determiners_and_stops():
    assert _np_after_trigger("I just bought a mansion last week.", "bought") == "mansion"
    assert _np_after_trigger("We visited Florida last year.", "visited") == "Florida"
    assert _np_after_trigger("I am playing basketball with friends.", "playing") == "basketball"
    assert _np_after_trigger("She bought a mansion and a car.", "bought") == "mansion"
    assert _np_after_trigger("I bought it yesterday.", "bought") in (None, "yesterday")


def test_extract_objects_v2_end_to_end():
    msgs = [{"dia": "ep_1", "session_id": 0, "speaker": "Maria", "ts": "2023-05-01",
             "text": "I went on a road trip to Oregon."},
             {"dia": "ep_2", "session_id": 0, "speaker": "Maria", "ts": "2023-06-01",
              "text": "We visited Florida last year."},
             {"dia": "ep_3", "session_id": 0, "speaker": "John", "ts": "2023-07-01",
              "text": "Maria visited Canada."}]
    rows = extract_objects_v2(msgs, "Maria", "vacation_state",
                              ["visited", "went to", "trip", "vacation"], gate="did",
                              question="What states has Maria vacationed at?")
    vals = {getattr(r, "value", "") for r in rows}
    assert "Florida" in vals, vals
    assert any("Oregon" in v for v in vals), vals
    assert not any(v.lower().startswith(("ed ", "ing ")) for v in vals), f"不得出现屈折碎片: {vals}"


def test_typed_scan_ignores_speaker_attribution():
    """类型通道不裁剪说话人归属 —— 集合题证据常由他人提供。

    依据 (2026-09-11 实测): "I saw that you had \"The Alchemist\"…" 这类句子
    由对话伙伴说出, 却是 gold 证据; 严格归属门会丢掉真 gold。
    """
    q = "What states has Maria vacationed at?"
    # 词表实体为 Nobody (触发词通道零命中), 类型通道仍应扫出州名
    lex = {"conv-41#q0029": {"entities": ["Nobody"], "slot": "vacation_state", "gate": "did"}}
    out = slot_assembly.render_slot_block_v2(_CTX, q, "conv-41#q0029", _MSGS, lex, _TRIG,
                                             route_first=False)
    assert "[SLOT EVIDENCE]" in out and "Oregon" in out
