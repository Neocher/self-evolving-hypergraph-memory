"""R8 Step3a 确定性槽证据抽取 — retrieval/slot_extract.py G1 oracle 单测 (TDD 先行).

本体论验收句: 评测面原文如何确定性到达 (entity×slot) 证据?
    → 归属过滤 + 触发词边界 + 模态门 + 宾语窗轻量值切分 (v1 尽力而为) → 过。

G1 硬性覆盖 (spec §G1 ①–⑫):
① 自述句入 (Audrey / Lab mixes) ② 他述含名句入 (James→John's favorite game)
③ 他述无名句剔 ("my favorite..." 配 John) ④ 无触发词剔 (park / purchase_item)
⑤ did 门滤 wish ⑥ did 门保留 actual ⑦ offer 门保留 in_talks / 剔 wish
⑧ plan 门不过滤 ⑨ 值切分 and ⑩ 冠词前缀修剪 ⑪ 窗空回退整句
⑫ 同值两提及 → members 1 行 + mentions 归并
+ oracle (读 data/r8/fixtures-r8-oracle.json 真实文本 + slot-triggers-v1.json 词表):
⑬ conv-44 q0035 dog_breed D19:12/D26:13 ⑭ conv-50 q0001 purchase_item D1:3/D2:1
⑮ conv-50 q0139 guitar_style D16:19
"""
import json
from pathlib import Path

from retrieval.slot_extract import build_slot_index, extract_slot_facts
from retrieval.slot_facts import parse_session_datetime

_FIXTURE = Path(__file__).resolve().parents[2] / "data/r8" / "fixtures-r8-oracle.json"
_TRIGGERS = Path(__file__).resolve().parents[2] / "data/r8" / "slot-triggers-v1.json"

# 会话内时间锚 (可比较, 入参注入; 断言只关心相对序)。
_T1, _T2 = "2023-09-24T17:53", "2023-10-28T14:36"

# 词表字面量 (G1 构造句, 对齐 data/r8/slot-triggers-v1.json)。
_TRIG_DOG_BREED = ["breed", "dog", "mix", "lab", "chihuahua", "jack russell", "puppy"]
_TRIG_FAVORITE_GAME = ["favorite", "game", "csgo", "apex", "play"]
_TRIG_PURCHASE = ["bought", "buy", "purchase", "got", "new", "mansion", "car"]
_TRIG_VACATION = ["vacation", "visited", "went to", "trip", "travel"]
_TRIG_ENDORSE = ["endorse", "offer", "sponsor", "nike", "gatorade", "deal", "brand"]
_TRIG_PLAN_HIKE = ["plan", "hike", "hiking"]


def _msg(dia, session_id, speaker, text, ts):
    """构造 extract_slot_facts 入参消息行 (会话内时间序)。"""
    return {"dia": dia, "session_id": session_id, "speaker": speaker,
            "text": text, "ts": ts}


# ── ① 自述句入: 触发词命中 + 宾语窗/回退整句可达 "lab mixes" ──────────────

def test_self_statement_trigger_hit_reaches_lab_mixes():
    """speaker==entity (自述) + 触发词 mix (命中 mixes) → ≥1 事实, 值含 'lab mixes'。"""
    msgs = [_msg("D26:13", "conv-44-s26", "Audrey",
                 "Pepper and Panda are Lab mixes", _T1)]
    facts = extract_slot_facts(msgs, "Audrey", "dog_breed",
                               ["mix"], gate="did")
    assert len(facts) >= 1
    assert all(f.entity == "Audrey" and f.slot == "dog_breed" for f in facts)
    assert any("lab mixes" in f.value.lower() for f in facts), facts


# ── ② 他述含名句入 ─────────────────────────────────────────────────────────

def test_other_speaker_named_mention_hit():
    """他述句含 entity 整词 ('John's favorite...') → 归属 entity, 命中。"""
    msgs = [_msg("D12:4", "conv-47-s12", "James",
                 "John's favorite game is CS:GO", _T1)]
    facts = extract_slot_facts(msgs, "John", "favorite_game",
                               _TRIG_FAVORITE_GAME, gate="did")
    assert len(facts) >= 1
    assert all(f.entity == "John" and f.slot == "favorite_game" for f in facts)
    assert all(f.msg_ref == "D12:4" for f in facts)


# ── ③ 他述无名句剔 ─────────────────────────────────────────────────────────

def test_other_speaker_without_name_or_address_excluded():
    """他述句无 entity 整词亦无直呼 ('my favorite...' 配 John) → 不命中。"""
    msgs = [_msg("D12:5", "conv-47-s12", "James",
                 "my favorite game is CS:GO", _T1)]
    facts = extract_slot_facts(msgs, "John", "favorite_game",
                               _TRIG_FAVORITE_GAME, gate="did")
    assert facts == []


# ── ④ 无触发词剔 ──────────────────────────────────────────────────────────

def test_no_trigger_word_excluded():
    """句不命中任何触发词 (purchase_item 词表) → 剔除。"""
    msgs = [_msg("D3:1", "conv-50-s3", "Calvin", "I went to the park", _T1)]
    assert extract_slot_facts(msgs, "Calvin", "purchase_item",
                              _TRIG_PURCHASE, gate="did") == []


# ── ⑤ did 门滤 wish ───────────────────────────────────────────────────────

def test_did_gate_filters_wish_statements():
    """gate=did: classify∈{planned,wish,inferred} 剔除 (wish/aspire/would)。"""
    for text in ("I wish I had a new car", "I would love a mansion"):
        msgs = [_msg("D1:1", "conv-50-s1", "Calvin", text, _T1)]
        facts = extract_slot_facts(msgs, "Calvin", "purchase_item",
                                   _TRIG_PURCHASE, gate="did")
        assert facts == [], (text, facts)


# ── ⑥ did 门保留 actual ───────────────────────────────────────────────────

def test_did_gate_keeps_actual_purchase():
    """gate=did: 实际购买陈述保留 (值切分为 mansion)。"""
    msgs = [_msg("D1:3", "conv-50-s1", "Calvin", "I bought a mansion", _T1)]
    facts = extract_slot_facts(msgs, "Calvin", "purchase_item",
                               _TRIG_PURCHASE, gate="did")
    assert len(facts) >= 1
    assert any(f.value.lower() == "mansion" for f in facts), facts


# ── ⑦ offer 门: 保留 in_talks / 剔 wish ───────────────────────────────────

def test_offer_gate_keeps_in_talks_excludes_wish():
    """gate=offer: in_talks 保留, wish (would) 剔除。"""
    keep = [_msg("D5:2", "conv-43-s5", "John",
                 "We are in talks with Gatorade", _T1)]
    keep_facts = extract_slot_facts(keep, "John", "endorsement_offer",
                                    _TRIG_ENDORSE, gate="offer")
    assert len(keep_facts) >= 1, keep_facts

    drop = [_msg("D5:3", "conv-43-s5", "John",
                 "Nike would be really cool", _T1)]
    assert extract_slot_facts(drop, "John", "endorsement_offer",
                              _TRIG_ENDORSE, gate="offer") == []


# ── ⑧ plan 门不过滤 ───────────────────────────────────────────────────────

def test_plan_gate_does_not_filter_planned():
    """gate=plan: classify=planned 仍保留 (不过滤)。"""
    msgs = [_msg("D1:1", "conv-44-s1", "Audrey", "we plan to hike", _T1)]
    facts = extract_slot_facts(msgs, "Audrey", "plan_hike",
                               _TRIG_PLAN_HIKE, gate="plan")
    assert len(facts) >= 1, facts


# ── ⑨ 值切分 and ──────────────────────────────────────────────────────────

def test_value_split_on_and_yields_multi_values():
    """宾语窗按 and 切多值: Oregon / Florida 各为独立 value。"""
    msgs = [_msg("D9:2", "conv-41-s9", "Jon", "I visited Oregon and Florida", _T1)]
    facts = extract_slot_facts(msgs, "Jon", "vacation_state",
                               _TRIG_VACATION, gate="did")
    values = {f.value for f in facts}
    assert {"Oregon", "Florida"} <= values, (facts, values)


# ── ⑩ 冠词前缀修剪 ────────────────────────────────────────────────────────

def test_article_prefix_trimmed_to_head_noun():
    """'a new mansion' → 修剪冠词/修饰前缀 → 'mansion'。"""
    msgs = [_msg("D1:4", "conv-50-s1", "Calvin", "I bought a new mansion", _T1)]
    facts = extract_slot_facts(msgs, "Calvin", "purchase_item",
                               _TRIG_PURCHASE, gate="did")
    assert any(f.value.lower() == "mansion" for f in facts), facts


# ── ⑪ 窗空回退整句 ────────────────────────────────────────────────────────

def test_empty_window_falls_back_to_whole_sentence():
    """触发词位于句尾、宾语窗为空 → 回退整句文本 (strip 后)。"""
    msgs = [_msg("D1:5", "conv-50-s1", "Calvin", "I bought a car.", _T1)]
    facts = extract_slot_facts(msgs, "Calvin", "purchase_item",
                               ["car"], gate="did")
    assert len(facts) >= 1
    assert any(f.value.strip().lower() == "i bought a car." for f in facts), facts


# ── ⑫ 同值两提及 → members 1 行 + mentions 归并 ────────────────────────────

def test_build_slot_index_members_dedup_same_value_merges_mentions():
    """build_slot_index: 同值两提及 (两会话内消息) → members 1 行, mentions 含两 ep。"""
    msgs_all = [
        _msg("D1:3", "conv-50-s1", "Calvin", "I bought a mansion", _T1),
        _msg("D1:9", "conv-50-s1", "Calvin", "I bought a mansion", _T2),
    ]
    lexicon = {"conv-50#q0001": {"entities": ["Calvin"],
                                 "slot": "purchase_item", "gate": "did"}}
    triggers = {"purchase_item": _TRIG_PURCHASE}
    idx = build_slot_index(msgs_all, lexicon, triggers)
    # 两提及各成一行 (ep 不同), 成员去重后并一行。
    assert len(idx.query("Calvin", "purchase_item", scope="conv-50-s1")) == 2
    ms = idx.members("Calvin", "purchase_item", scope="conv-50-s1")
    assert len(ms) == 1
    assert ms[0]["value"].lower() == "mansion"
    assert ms[0]["mentions"] == [1, 2]  # 时间升序, 同 ts 按 ep 升序


# ── oracle: 读 data/r8/fixtures-r8-oracle.json 真实文本 (句级可达) ────────

def _load_oracle_rows():
    """读夹具行 (只读引用), 映射为 (row, msgs单消息, entity, slot)。"""
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    out = []
    for r in raw["facts"]:
        s, m = r["dia_id"].lstrip("D").split(":")
        msg = _msg(
            r["dia_id"], f"{r['sample_id']}-s{s}", r["speaker"],
            r["snippet"], parse_session_datetime(r["session_dt"]),
        )
        out.append((r, [msg], r["entity"], r["slot"]))
    return out


def _oracle_triggers(slot):
    """读 slot-triggers-v1.json 词表 (只读)。"""
    return json.loads(_TRIGGERS.read_text(encoding="utf-8"))[slot]


def test_oracle_dog_breed_mixes_sentences_reachable():
    """⑬ conv-44 q0035 dog_breed: D19:12/D26:13 各抽到 ≥1 事实且句含 mixes。"""
    rows = [x for x in _load_oracle_rows() if x[2] == "Audrey" and x[3] == "dog_breed"]
    assert {r[0]["dia_id"] for r in rows} == {"D19:12", "D26:13"}
    for raw, msgs, entity, slot in rows:
        facts = extract_slot_facts(msgs, entity, slot,
                                   _oracle_triggers(slot), gate="did")
        assert len(facts) >= 1, (raw["dia_id"], facts)
        assert "mixes" in raw["snippet"].lower()
        assert any("mixes" in f.value.lower() for f in facts), (raw["dia_id"], facts)


def test_oracle_purchase_item_mansion_car_sentences_reachable():
    """⑭ conv-50 q0001 purchase_item: D1:3/D2:1 各 ≥1 (句含 mansion / car)。"""
    rows = [x for x in _load_oracle_rows() if x[2] == "Calvin" and x[3] == "purchase_item"]
    assert {r[0]["dia_id"] for r in rows} == {"D1:3", "D2:1"}
    expect = {"D1:3": "mansion", "D2:1": "car"}
    for raw, msgs, entity, slot in rows:
        facts = extract_slot_facts(msgs, entity, slot,
                                   _oracle_triggers(slot), gate="did")
        assert len(facts) >= 1, (raw["dia_id"], facts)
        kw = expect[raw["dia_id"]]
        assert kw in raw["snippet"].lower()
        assert any(kw in f.value.lower() for f in facts), (raw["dia_id"], facts)


def test_oracle_guitar_style_shiny_purple_sentence_reachable():
    """⑮ conv-50 q0139 guitar_style: D16:19 ≥1 (Dave 他述直呼→Calvin, 句含 guitar/purple)。"""
    rows = [x for x in _load_oracle_rows() if x[0]["dia_id"] == "D16:19"]
    assert len(rows) == 1
    raw, msgs, entity, slot = rows[0]
    facts = extract_slot_facts(msgs, entity, slot,
                               _oracle_triggers(slot), gate="did")
    assert len(facts) >= 1, facts
    # 证据消息 (Dave 对 Calvin 吉他) 句含 guitar/purple; 抽取归属经第二人称直呼。
    snip = raw["snippet"].lower()
    assert "guitar" in snip and "purple" in snip
    assert all(f.msg_ref == "D16:19" for f in facts)
