# -*- coding: utf-8 -*-
"""计数题路径 v1 测试 — 锚词伪槽 / 锚词提及抽取 / 实例计数 / [SLOT COUNT] 符号读出.

依据: .tri-spec/count-path-spec.md (Hermes 交付, 三体空转后由 Hermes 直改)
现场缺口: 20 道 how many 题里 6 道连槽位候选都没有; 计数的输入是"值"不是"事件实例"。
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import slot_assembly  # noqa: E402
from retrieval.slot_closure import (  # noqa: E402
    count_anchor_terms, count_instances, extract_anchor_mentions, is_count_question,
    route_count_question, route_question,
)


# ── 锚词提取 ─────────────────────────────────────────────────────────────
def test_count_anchor_terms_filters_stopwords_and_speakers():
    a = count_anchor_terms("How many times has Melanie gone to the beach in 2023?", speakers=["Melanie"])
    assert "beach" in a
    assert "times" not in a and "melanie" not in a


def test_count_anchor_terms_object_family():
    assert "children" in count_anchor_terms("How many children does Melanie have?", speakers=["Melanie"])
    a = count_anchor_terms("How many turtles does Nate have?", speakers=["Nate"])
    assert "turtles" in a or "turtle" in a


def test_count_anchor_terms_empty_question():
    assert count_anchor_terms("", speakers=[]) == []


# ── 锚词提及抽取 ─────────────────────────────────────────────────────────
_MSGS = [
    {"dia": "ep_1", "session_id": 0, "speaker": "Melanie", "ts": "2023-05-01",
     "text": "We went to the beach last weekend, it was great."},
    {"dia": "ep_2", "session_id": 0, "speaker": "Melanie", "ts": "2023-05-02",
     "text": "I love my new garden."},
    {"dia": "ep_3", "session_id": 0, "speaker": "Caroline", "ts": "2023-05-03",
     "text": "The beach was crowded yesterday."},
]


def test_extract_anchor_mentions_matches_and_attributes():
    facts = extract_anchor_mentions(_MSGS, ["beach"], entity="Melanie")
    refs = {f.msg_ref for f in facts}
    assert "ep_1" in refs, refs
    assert "ep_2" not in refs


def test_extract_anchor_mentions_no_anchor():
    assert extract_anchor_mentions(_MSGS, [], entity="Melanie") == []


# ── 实例计数 ─────────────────────────────────────────────────────────────
class _F:
    def __init__(self, v, ts):
        self.value, self.ts = v, ts


def test_count_instances_merges_within_window():
    facts = [_F("beach", "2023-05-01"), _F("beach", "2023-05-04")]
    assert count_instances(facts, window_days=7) == 1
    far = [_F("beach", "2023-05-01"), _F("beach", "2023-06-20")]
    assert count_instances(far, window_days=7) == 2
    assert count_instances([], window_days=7) == 0


# ── 路由 (硬门: how many 题不得零候选) ───────────────────────────────────
def test_route_count_question_anchor_fallback():
    q = "How many times has Melanie gone to the beach in 2023?"
    hits = route_count_question(q, speakers=["Melanie"], triggers={})
    assert hits, "无触发词命中时必须给出锚词伪槽"
    assert hits[0]["slot"].startswith("anchor:")
    assert hits[0]["entity"] == "Melanie"
    assert hits[0].get("terms")


def test_route_count_question_prefers_real_slot():
    trig = {"vacation_state": ["beach", "gone to", "trip"]}
    hits = route_count_question("How many times has Melanie gone to the beach?", speakers=["Melanie"],
                                triggers=trig)
    assert hits and hits[0]["slot"] == "vacation_state"


# ── 符号读出: [SLOT COUNT] ───────────────────────────────────────────────
_CTX = ("[ENTITY: conv-26/Melanie]\n- conv-26/Melanie likes: beach\n\n"
        "[DIRECT EVIDENCE]\n"
        "[1] [date: 1 May, 2023] [Melanie] We went to the beach last weekend.\n")


def test_v2_count_question_emits_slot_count_line():
    q = "How many times has Melanie gone to the beach in 2023?"
    out = slot_assembly.render_slot_block_v2(_CTX, q, "conv-26#q0030", _MSGS, {}, {})
    assert "[SLOT COUNT]" in out, out
    assert "n=" in out


def test_v2_non_count_question_has_no_count_line():
    q = "What states has Maria vacationed at?"
    lex = {"conv-41#q0029": {"entities": ["Maria"], "slot": "vacation_state", "gate": "did"}}
    msgs = [{"dia": "ep_1", "session_id": 0, "speaker": "Maria", "ts": "2023-05-01",
             "text": "We visited Florida last year."}]
    out = slot_assembly.render_slot_block_v2(_CTX, q, "conv-41#q0029", msgs, lex,
                                             {"vacation_state": ["visited"]})
    assert "[SLOT COUNT]" not in out


def test_is_count_question_family():
    assert is_count_question("How many times has Melanie gone to the beach?")
    assert is_count_question("How many children does Melanie have?")
    assert not is_count_question("What books has Melanie read?")
