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


# ── 扩类型 (2026-09-11): family_member / event / music_genre / game_title ──────────
from retrieval.slot_closure import (  # noqa: E402
    _looks_like_title, detect_value_type, type_match, load_value_types,
)


def test_type_detection_new_types():
    assert detect_value_type("Which of James's family members have visited him?") == "family_member"
    assert detect_value_type("Which events has Jon participated in?") == "event"
    assert detect_value_type("What kind of music does Dave listen to?") == "music_genre"
    assert detect_value_type("What are John and James' favorite games?") == "game_title"
    assert detect_value_type("What video games does Nate play?") == "game_title"
    # 既有类型不被抢走
    assert detect_value_type("What states has Maria vacationed at?") == "us_state"
    assert detect_value_type("What books has Melanie read?") == "work_book"


def test_scan_family_and_event_and_music():
    msgs = [_m("ep_1", "James", "My mother and my sister visited me last month."),
            _m("ep_2", "Jon", "I attended a fair and a networking event for my business."),
            _m("ep_3", "Dave", "I mostly listen to classic rock and some jazz.")]
    fam = {f.value for f in typed_scan_candidates(msgs, "family_member", entity="James")}
    ev = {f.value for f in typed_scan_candidates(msgs, "event", entity="Jon")}
    mu = {f.value for f in typed_scan_candidates(msgs, "music_genre", entity="Dave")}
    assert {"mother", "sister"} <= fam, fam
    assert {"fair", "networking event"} <= ev, ev
    assert {"classic rock", "jazz"} <= mu, mu


def test_game_title_uses_dedicated_triggers_and_rejects_np_overflow():
    msgs = [_m("ep_1", "John", "My favorite game is CS:GO and I also started playing AC Valhalla.")]
    got = {f.value for f in typed_scan_candidates(msgs, "game_title", entity="John")}
    assert any("CS:GO" in v for v in got), got
    assert any("AC Valhalla" in v for v in got), got


def test_looks_like_title_strict_rules():
    assert _looks_like_title("AC Valhalla") and _looks_like_title("Witcher 3")
    assert _looks_like_title("CS:GO") and _looks_like_title("FIFA 23")
    assert not _looks_like_title("Catan - it's a")       # 分隔破折号 → NP 越界
    assert not _looks_like_title("Chess afterward just")  # 小词/副词
    assert not _looks_like_title("playing the game")      # 小词
    assert not _looks_like_title("a")                     # 过短


def test_type_match_new_types():
    res = load_value_types()
    assert type_match("mother", "family_member", res)
    assert not type_match("Mansion", "family_member", res)
    assert type_match("networking events", "event", res)   # 复数形态
    assert type_match("classic rock", "music_genre", res)
    assert type_match("Witcher 3", "game_title", res)
    assert not type_match("Catan - it's a", "game_title", res)
