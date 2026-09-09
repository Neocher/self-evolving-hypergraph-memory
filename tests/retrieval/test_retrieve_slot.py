"""R8 E4 QueryRouter.retrieve_slot 接线 — tests/retrieval/test_retrieve_slot.py (G1).

env 门: R8_CTX=1 总闸默认 off; off/缺省 → [] (v6.19.0 逐字节等价哨兵)。

G1 硬性覆盖:
① env 未设 → [] (off 等价, 既有 query_router 行为不变)
② R8_CTX=1 + set_slot_index(真实 SlotIndex 引擎) → 走引擎确定性返回
③ R8_CTX=1 无索引 (未注入) → []
④ set_slot_index(None) 清除注入后 → []
禁 mock 检索逻辑: 只经 monkeypatch 读 env 开关, 引擎为真实 retrieval.slot_facts.SlotIndex。
"""
from retrieval.query_router import QueryRouter
from retrieval.slot_facts import SlotFact, SlotIndex

_TS1 = "2023-09-24T17:53"
_TS2 = "2023-10-28T14:36"


def _make_router():
    """零依赖 QueryRouter (graphlite/faiss/tfidf 均 None, 不触检索链路)。"""
    return QueryRouter(graphlite_store=None, faiss_index=None, tfidf_index=None)


def _make_index():
    """两行真实槽位事实 (Audrey/dog_breed 两会话), 供注入引擎。"""
    return SlotIndex.from_facts([
        SlotFact(session_id="conv-44-s19", ep_id=12, ts=_TS1,
                 entity="Audrey", slot="dog_breed",
                 value="Jack Russell mixes", msg_ref="D19:12"),
        SlotFact(session_id="conv-44-s26", ep_id=13, ts=_TS2,
                 entity="Audrey", slot="dog_breed",
                 value="Lab mixes", msg_ref="D26:13"),
    ])


# ── ① env 未设 → [] (off 等价) ─────────────────────────────────────────────

def test_env_unset_returns_empty_off_equivalent(monkeypatch):
    """R8_CTX 未设 (off) → retrieve_slot 返回 [], 与 v6.19.0 逐字节等价。"""
    monkeypatch.delenv("R8_CTX", raising=False)
    qr = _make_router()
    qr.set_slot_index(_make_index())  # 即使注入了索引, off 时也不走引擎
    assert qr.retrieve_slot("Audrey", "dog_breed") == []
    assert qr.retrieve_slot("Audrey", "dog_breed",
                            session_ts=_TS2, scope="conv-44-s26") == []


def test_env_zero_is_off_equivalent(monkeypatch):
    """R8_CTX 显式 '0' → off, 同样返回 []。"""
    monkeypatch.setenv("R8_CTX", "0")
    qr = _make_router()
    qr.set_slot_index(_make_index())
    assert qr.retrieve_slot("Audrey", "dog_breed") == []


# ── ② R8_CTX=1 + 注入索引 → 走引擎返回 ─────────────────────────────────────

def test_env_on_with_injected_index_routes_to_engine(monkeypatch):
    """R8_CTX=1 + set_slot_index(引擎) → retrieve_slot 委托 SlotIndex.query 返回。"""
    monkeypatch.setenv("R8_CTX", "1")
    qr = _make_router()
    idx = _make_index()
    qr.set_slot_index(idx)
    out = qr.retrieve_slot("Audrey", "dog_breed")
    assert out == idx.query("Audrey", "dog_breed")
    assert [f.value for f in out] == ["Jack Russell mixes", "Lab mixes"]
    # 接线透传 scope / session_ts 过滤
    scoped = qr.retrieve_slot("Audrey", "dog_breed", scope="conv-44-s19")
    assert [f.value for f in scoped] == ["Jack Russell mixes"]


# ── ③ R8_CTX=1 但索引未注入 → [] ──────────────────────────────────────────

def test_env_on_without_index_returns_empty(monkeypatch):
    """R8_CTX=1 但未 set_slot_index (索引缺省 None) → [] (不上探针不崩溃)。"""
    monkeypatch.setenv("R8_CTX", "1")
    qr = _make_router()
    assert qr.retrieve_slot("Audrey", "dog_breed") == []


# ── ④ set_slot_index(None) 清除 → [] ──────────────────────────────────────

def test_set_slot_index_none_clears_injection(monkeypatch):
    """set_slot_index(None) 清除注入 → on 态亦回退 []。"""
    monkeypatch.setenv("R8_CTX", "1")
    qr = _make_router()
    qr.set_slot_index(_make_index())
    assert len(qr.retrieve_slot("Audrey", "dog_breed")) == 2
    qr.set_slot_index(None)
    assert qr.retrieve_slot("Audrey", "dog_breed") == []
