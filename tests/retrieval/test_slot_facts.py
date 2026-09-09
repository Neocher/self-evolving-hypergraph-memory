"""R8 E4 槽位原文检索 — retrieval/slot_facts.py G1 oracle 单测 (TDD 先行).

本体论验收句: (entity×slot) 成员全集如何确定性取得?
    → SlotIndex 确定性集合查询/排序/去重/词形归一 (不做 top-N) → 过。

G1 硬性覆盖: ① 注入 40 行删 5 (同槽不同 value) → query 全量返回 35
② (entity,slot) 无行 → [] ③ session_ts 过滤含边界 (== 可见)
④ scope 隔离: 同 entity+slot 跨两会话 → 无 scope 全回/带 scope 只回本会话
⑤ 去重: 同 value 3 提及 → members 1 行 + mentions 含 3 ep_id
⑥ 排序 ts 升序 (同 ts 按 ep_id 升序) ⑦ 词形归一: "John" vs "john " 同 member
+ oracle: data/r8/fixtures-r8-oracle.json (只读), 5 行归因/时点/原文可达。
"""
import json
from pathlib import Path

from retrieval.slot_facts import SlotFact, SlotIndex, parse_session_datetime

_FIXTURE = Path(__file__).resolve().parents[2] / "data/r8" / "fixtures-r8-oracle.json"


def _sf(session_id, ep_id, ts, entity, slot, value, msg_ref=None):
    """构造 SlotFact 测试行的简写。"""
    return SlotFact(session_id=session_id, ep_id=ep_id, ts=ts,
                    entity=entity, slot=slot, value=value, msg_ref=msg_ref)


def _dia_parts(dia_id):
    """dia "D{s}:{m}" → (s, m): 会话号 + 会话内消息 (episode) 序。"""
    s, m = dia_id.lstrip("D").split(":")
    return int(s), int(m)


# ── ① 注入 40 行删 5 → query 全量返回 35 ───────────────────────────────────

def test_from_facts_merges_duplicates_query_returns_all_rows():
    """注入 40 行 (35 唯一 + 5 完全相同重复, 同槽含不同 value) → 归并后 query 全量 35。"""
    unique = [
        _sf(f"conv-44-s{1 + i // 20}", i + 1, i + 1, "Audrey", "dog_breed",
            f"dog_value_{i:02d}")
        for i in range(35)
    ]
    injected = unique + list(unique[-5:])  # 40 行 = 35 唯一 + 5 完全重复
    assert len(injected) == 40
    idx = SlotIndex.from_facts(injected)
    out = idx.query("Audrey", "dog_breed")
    assert len(out) == 35, "完全重复行归并, 其余每行保留, 全量返回不截断"
    assert {f.value for f in out} == {f"dog_value_{i:02d}" for i in range(35)}


# ── ② (entity, slot) 无行 → [] ─────────────────────────────────────────────

def test_no_matching_entity_slot_returns_empty():
    """无 (entity, slot) 命中 → query/members 均返回 []。"""
    idx = SlotIndex.from_facts([
        _sf("conv-44-s1", 1, 10, "Audrey", "dog_breed", "jack"),
    ])
    assert idx.query("nobody", "nothing") == []
    assert idx.members("nobody", "nothing") == []


# ── ③ session_ts 过滤含边界 (== 可见) ──────────────────────────────────────

def test_session_ts_inclusive_boundary_filter():
    """session_ts 非 None → 仅 ts<=session_ts; ts==session_ts 属可见 (含边界)。"""
    idx = SlotIndex.from_facts([
        _sf("conv-44-s1", 1, 10, "Audrey", "dog_breed", "a"),
        _sf("conv-44-s1", 2, 20, "Audrey", "dog_breed", "b"),
        _sf("conv-44-s1", 3, 30, "Audrey", "dog_breed", "c"),
    ])
    assert [f.ts for f in idx.query("Audrey", "dog_breed", session_ts=20)] == [10, 20]
    assert [f.ts for f in idx.query("Audrey", "dog_breed", session_ts=10)] == [10]
    assert idx.query("Audrey", "dog_breed", session_ts=5) == []


# ── ④ scope 隔离: 同 entity+slot 跨两会话 ──────────────────────────────────

def test_scope_isolates_cross_session_same_entity_slot():
    """同 entity+slot 跨两会话: 无 scope 全回, 带 scope 只回本会话行。"""
    idx = SlotIndex.from_facts([
        _sf("conv-44-s1", 1, 1, "Audrey", "dog_breed", "jack"),
        _sf("conv-44-s1", 2, 2, "Audrey", "dog_breed", "lab"),
        _sf("conv-44-s2", 1, 1, "Audrey", "dog_breed", "chihuahua"),
    ])
    assert len(idx.query("Audrey", "dog_breed")) == 3  # 无 scope 全回
    assert {f.value for f in idx.query("Audrey", "dog_breed", scope="conv-44-s1")} == {
        "jack", "lab",
    }
    assert {f.value for f in idx.query("Audrey", "dog_breed", scope="conv-44-s2")} == {
        "chihuahua",
    }


# ── ⑤ 去重: 同 value 3 提及 → members 1 行 + mentions 含 3 ep_id ──────────

def test_members_dedupes_value_with_all_mention_eps():
    """同 value 3 提及 (3 个 ep_id) → members 1 行, mentions 含全部 3 ep_id。"""
    idx = SlotIndex.from_facts([
        _sf("conv-44-s1", 2, 5, "Audrey", "activity", "hiking"),
        _sf("conv-44-s1", 7, 10, "Audrey", "activity", "hiking"),
        _sf("conv-44-s1", 3, 15, "Audrey", "activity", "hiking"),
    ])
    ms = idx.members("Audrey", "activity")
    assert len(ms) == 1
    assert ms[0]["value"] == "hiking"
    assert set(ms[0]["mentions"]) == {2, 7, 3}
    assert len(ms[0]["mentions"]) == 3
    # 成员行 = 每 value 首次出现行 (ts 最小者)
    assert ms[0]["ep_id"] == 2
    assert ms[0]["ts"] == 5


# ── ⑥ 排序 ts 升序 (同 ts 按 ep_id 升序) ───────────────────────────────────

def test_query_sorted_ts_asc_then_ep_asc():
    """query 排序 ts 升序; 同 ts 按 ep_id 升序 (确定性, 与插入序无关)。"""
    idx = SlotIndex.from_facts([
        _sf("conv-44-s1", 9, 30, "Audrey", "dog_breed", "a"),
        _sf("conv-44-s1", 1, 10, "Audrey", "dog_breed", "b"),
        _sf("conv-44-s1", 5, 20, "Audrey", "dog_breed", "c"),
        _sf("conv-44-s1", 2, 20, "Audrey", "dog_breed", "d"),
    ])
    out = idx.query("Audrey", "dog_breed")
    assert [f.value for f in out] == ["b", "d", "c", "a"]


# ── ⑦ 词形归一: "John" vs "john " 同 member ───────────────────────────────

def test_members_value_case_strip_normalization():
    """value 归一 (小写/strip): "John" 与 "john " 归并为一个 member。"""
    idx = SlotIndex.from_facts([
        _sf("conv-44-s1", 1, 5, "John", "person", "John"),
        _sf("conv-44-s1", 2, 9, "John", "person", "john "),
    ])
    ms = idx.members("John", "person")
    assert len(ms) == 1, "'John' 与 'john ' 词形归一后为同一 member"
    assert ms[0]["value"] == "John"  # 每 value 首次出现行 (首行原词)
    assert set(ms[0]["mentions"]) == {1, 2}
    # query 为全量行 (查询层不去重): 两行均返回
    assert len(idx.query("John", "person")) == 2


# ── parse_session_datetime: 确定性 + 可比较 ────────────────────────────────

def test_parse_session_datetime_deterministic_and_comparable():
    """会话时间锚解析为定长可比较字符串; 同输入同输出, 12h 时钟正确折叠。"""
    t1 = parse_session_datetime("1:56 pm on 8 May, 2023")
    t2 = parse_session_datetime("5:53 pm on 24 September, 2023")
    assert t1 == "2023-05-08T13:56"
    assert t1 < t2  # 字典序即时间序
    assert parse_session_datetime("1:56 pm on 8 May, 2023") == t1  # 确定性
    assert parse_session_datetime("12:00 am on 1 January, 2024") == "2024-01-01T00:00"
    assert parse_session_datetime("12:30 pm on 31 December, 2023") == "2023-12-31T12:30"


# ── oracle: 映射夹具 5 行归因 / 时点 / 原文可达 ────────────────────────────

def _load_fixture_rows():
    """读 data/r8/fixtures-r8-oracle.json (只读引用), 归因映射为 SlotFact 行。"""
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    rows = []
    for r in raw["facts"]:
        s, m = _dia_parts(r["dia_id"])
        rows.append((r, SlotFact(
            session_id=f"{r['sample_id']}-s{s}",
            ep_id=m,
            ts=parse_session_datetime(r["session_dt"]),
            entity=r["entity"],
            slot=r["slot"],
            value=r["value"],
            msg_ref=r.get("dia_id"),
        )))
    return rows


def test_oracle_attribution_session_time_and_gold_reachable():
    """5 行夹具: 会话/时点归因正确 + (entity,slot) query/members 回 gold 值。"""
    rows = _load_fixture_rows()
    assert len(rows) == 5
    idx = SlotIndex.from_facts([f for _, f in rows])

    # 每行归因: session_id={sample_id}-s{s}, ep_id=m, ts=parse(session_dt)
    for raw, f in rows:
        s, m = _dia_parts(raw["dia_id"])
        assert f.session_id == f"{raw['sample_id']}-s{s}", "会话归因须含夹具 sample_id"
        assert f.ep_id == m
        assert f.ts == parse_session_datetime(raw["session_dt"])

    # 会话时点边界含入 (ts==session_ts 可见) + scope 隔离到本会话
    for raw, f in rows:
        hits = idx.query(f.entity, f.slot, session_ts=f.ts, scope=f.session_id)
        assert hits, f"{f.entity}/{f.slot} @{f.session_id} 应回本会话行"
        assert [h.value for h in hits] == [f.value]

    # 真日历时间窗边界: conv-44 dog_breed 10 月末锚点含 9 月末旧行 (含边界)
    audrey = idx.query("Audrey", "dog_breed", session_ts="2023-10-28T14:36")
    assert [f.value for f in audrey] == ["Jack Russell mixes", "Lab mixes"]

    # gold 值各可达 (query 行级 + members 成员级)
    gold = {
        ("Audrey", "dog_breed"): {"Jack Russell mixes", "Lab mixes"},
        ("Calvin", "purchase_item"): {"mansion", "luxury car"},
        ("Calvin", "guitar_style"): {"shiny purple guitar"},
    }
    for (entity, slot), values in gold.items():
        qvals = [f.value for f in idx.query(entity, slot)]
        mvals = [m["value"] for m in idx.members(entity, slot)]
        for v in values:
            assert v in qvals, f"{entity}/{slot} gold {v!r} query 不可达"
            assert v in mvals, f"{entity}/{slot} gold {v!r} members 不可达"

    # 行级时序: conv-44 dog_breed 两行按真实日历 ts 升序
    conv44 = idx.query("Audrey", "dog_breed")
    assert [f.value for f in conv44] == ["Jack Russell mixes", "Lab mixes"]
