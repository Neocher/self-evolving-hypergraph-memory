"""R8-SLOT-SCOPE-1 回归测试 — build_slot_index 的 conv 裁剪与 session_id 真实形态.

缺陷 (2026-09-11 精卫轮亲验): build_slot_index 按 `<conv>-` 前缀裁剪会话消息, 假设
session_id 形如 "conv-26-s1"; 但评测链路 (bench _r8_conv_msgs ← 评测库
e.session_id) 传 int conversation_idx (0..9) → 前缀裁剪恒空 → 抽取 0 条 →
append_slot_evidence 静默返回原 ctx (R8_SLOT=1 的 A/B ON 臂与 OFF 逐字节等价)。

覆盖:
  T1 (本缺陷回归, int 形态): session_id=int + 调用方已裁剪 msgs
      → extract_slot_facts 非空, 且 append_slot_evidence 产出含 [SLOT EVIDENCE] 的段
  T2 (原意图不破): session_id="conv-50-s1" → 只取 conv-50 的消息
  T3 (防跨会话): 混杂 "conv-50-s1"/"conv-99-s1" → 只取 conv-50
  T4 (无 # 前缀 qkey): 全量 msgs (原行为)
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))   # slot_assembly (bench 侧装配模块)
sys.path.insert(0, str(REPO))

import slot_assembly  # noqa: E402
from retrieval.slot_extract import build_slot_index, extract_slot_facts  # noqa: E402

_TRIG_PURCHASE = ["bought", "buy", "purchase", "got", "new", "mansion", "car"]


def _msg(dia, session_id, speaker, text, ts="2023-09-24T17:53"):
    return {"dia": dia, "session_id": session_id, "speaker": speaker, "text": text, "ts": ts}


def _lex(qkey, entity="Calvin", slot="purchase_item", gate="did"):
    return {qkey: {"entities": [entity], "slot": slot, "gate": gate}}


_TRIGGERS = {"purchase_item": _TRIG_PURCHASE}


# ── T1: int session_id 形态 (评测链路真实形态) ────────────────────────────
def test_t1_int_session_id_extracts_and_appends():
    """评测链路形态: session_id=int (conversation_idx), msgs 已由调用方裁剪到本题会话。"""
    msgs = [
        _msg("D1:3", 0, "Calvin", "I just bought a mansion last week"),
        _msg("D1:4", 0, "Caroline", "Calvin got a new car too"),
    ]
    lex = _lex("conv-26#q0036")
    idx = build_slot_index(msgs, lex, _TRIGGERS)
    rows = idx.query("Calvin", "purchase_item")
    assert rows, "int session_id 形态下必须抽到事实 (回归: 曾恒空)"
    ctx = slot_assembly.append_slot_evidence("[BASE]", "What did Calvin buy?",
                                             "conv-26#q0036", msgs, lex, _TRIGGERS)
    assert ctx != "[BASE]", "int session_id 形态下必须追加段 (回归: 曾静默零注入)"
    assert "[SLOT EVIDENCE]" in ctx
    assert "mansion" in ctx or "car" in ctx


def test_t1b_direct_extract_non_empty():
    msgs = [_msg("D2:1", 3, "Calvin", "I bought a new car")]
    facts = extract_slot_facts(msgs, "Calvin", "purchase_item", _TRIG_PURCHASE, gate="did")
    assert facts and facts[0].value


# ── T2: 标签串形态原行为不破 ──────────────────────────────────────────────
def test_t2_label_session_id_scopes_to_own_conv():
    msgs = [
        _msg("D1:3", "conv-50-s1", "Calvin", "I bought a mansion"),
        _msg("D9:1", "conv-99-s1", "Calvin", "I bought a car"),
    ]
    idx = build_slot_index(msgs, _lex("conv-50#q0001"), _TRIGGERS)
    rows = idx.query("Calvin", "purchase_item")
    refs = {r.msg_ref for r in rows}
    assert refs == {"D1:3"}, f"标签串形态必须只取本 conv, 实际 {refs}"


# ── T3: 跨会话混杂 (标签串) 仍不越界 ──────────────────────────────────────
def test_t3_cross_conv_labels_not_mixed():
    msgs = [
        _msg("D1:1", "conv-50-s1", "Calvin", "I bought a new car"),
        _msg("D2:1", "conv-99-s2", "Calvin", "I bought a mansion"),
    ]
    idx = build_slot_index(msgs, _lex("conv-50#q0001"), _TRIGGERS)
    refs = {r.msg_ref for r in idx.query("Calvin", "purchase_item")}
    assert refs == {"D1:1"}, f"不得混入异会话消息, 实际 {refs}"


# ── T4: 无 # 前缀 qkey → 全量 msgs (原行为) ────────────────────────────────
def test_t4_qkey_without_conv_prefix_uses_all_msgs():
    msgs = [
        _msg("D1:1", 7, "Calvin", "I bought a mansion"),
        _msg("D2:1", 8, "Calvin", "I bought a new car"),
    ]
    idx = build_slot_index(msgs, _lex("q0001"), _TRIGGERS)
    refs = {r.msg_ref for r in idx.query("Calvin", "purchase_item")}
    assert refs == {"D1:1", "D2:1"}, f"无 # 前缀应使用全量, 实际 {refs}"


# ── T5: int 形态下标签串不匹配的 conv 也不会被误裁零 ──────────────────────
def test_t5_int_ids_with_multiple_convs_caller_scoped():
    """调用方已裁剪, 但 session_id 为 int: 回落全量 (= 调用方范围), 不得静默为空。"""
    msgs = [_msg("D5:1", 9, "Calvin", "I bought a mansion")]
    idx = build_slot_index(msgs, _lex("conv-42#q0050"), _TRIGGERS)
    assert idx.query("Calvin", "purchase_item"), "int 形态 + 已裁剪 msgs 必须抽到事实"
