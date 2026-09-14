# -*- coding: utf-8 -*-
"""类型锚定抽取测试 — 不依赖触发词, 按类型扫会话 (A 方案落地件).

实测依据 (2026-09-11, 真实库): 触发词通道 gold 覆盖 0.249 → 类型锚定通道 0.494。
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from retrieval.slot_closure import (  # noqa: E402
    _quoted_spans, _title_runs, load_value_types, typed_scan_candidates,
)


def _m(dia, speaker, text, ts="2023-05-01"):
    return {"dia": dia, "session_id": 0, "speaker": speaker, "text": text, "ts": ts}


def test_us_state_scan_without_triggers():
    msgs = [_m("ep_1", "Maria", "My family and I went on a road trip to Oregon last year."),
            _m("ep_2", "Maria", "I have a picture from a vacation in Florida."),
            _m("ep_3", "Maria", "Nothing relevant here.")]
    got = {f.value for f in typed_scan_candidates(msgs, "us_state", entity="Maria")}
    assert "Oregon" in got and "Florida" in got, got
    assert "Mansion" not in got


def test_country_scan():
    msgs = [_m("ep_1", "Maria", "I visited Japan and France in 2019.")]
    got = {f.value for f in typed_scan_candidates(msgs, "country", entity="Maria")}
    assert {"Japan", "France"} <= got, got


def test_activity_scan_uses_lexicon():
    msgs = [_m("ep_1", "John", "I love hiking and yoga on weekends.")]
    got = {f.value for f in typed_scan_candidates(msgs, "activity", entity="John")}
    assert "hiking" in got and "yoga" in got, got


def test_work_book_quoted_titles():
    msgs = [_m("ep_1", "Melanie", 'I read "Charlotte\'s Web" as a kid.')]
    got = {f.value for f in typed_scan_candidates(msgs, "work_book", entity="Melanie")}
    assert any("Charlotte" in v for v in got), got


def test_title_runs_requires_media_context():
    assert "Silent Patient" in " ".join(_title_runs("I read The Silent Patient last week"))
    assert _title_runs("We went to New York City") == []      # 无书籍/影视语境 → 不取


def test_quoted_spans_extracts_inner_text():
    assert _quoted_spans('She said "Becoming Nicole" was great') == ["Becoming Nicole"]


def test_generic_type_returns_empty():
    msgs = [_m("ep_1", "Maria", "I bought a mansion.")]
    assert typed_scan_candidates(msgs, "generic", entity="Maria") == []
    assert typed_scan_candidates(msgs, "count", entity="Maria") == []


def test_typed_scan_dedups_and_keeps_msg_ref():
    msgs = [_m("ep_1", "Maria", "Oregon is nice."), _m("ep_2", "Maria", "I love Oregon.")]
    facts = typed_scan_candidates(msgs, "us_state", entity="Maria")
    vals = [f.value for f in facts]
    assert vals.count("Oregon") == 1, vals
    assert facts[0].msg_ref == "ep_1"
