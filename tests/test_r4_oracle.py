"""达摩院 R4A oracle DB 同形注入 — 源码断言 (AC1-AC3 静态侧).

断言 scripts/bench_locomo_v72_ontology.py 的 ORACLE_INJECT 分支:
- ORACLE_MODE=db (默认, R4 用): evidence 按 dia_id → 灌库 DB 同形 content
  (带 '[date: <会话日期>] [speaker] text' 前缀的原文整条) 映射置顶 channels A;
  无 dia_id/映射失败走 content 去前缀反查回落, 仍找不到整题跳过并计数。
- ORACLE_MODE=legacy: 保留 v6.14.1 裸 evidence text 注入语句 (逐字节等价回归)。
纯文本断言, 不 import bench (其顶层会加载检索/灌库/embedding, 非单测面)。
"""
import re
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "bench_locomo_v72_ontology.py"
TEXT = SRC.read_text(encoding="utf-8")

# v6.14.1 (HEAD) ORACLE_INJECT 分支核心语句 (git show HEAD 提取, 逐字节)
_OLD_LEGACY_STMTS = [
    '_ev = [e.get("text", "").strip() for e in (q.get("evidence_messages") or []) if e.get("text")]',
    '_seen = {c[:200] for c in channels["A"]}',
    'channels["A"] = [e for e in _ev if e[:200] not in _seen] + channels["A"]',
]

def _legacy_stmts():
    """当前文件 ORACLE_MODE=legacy else 分支的三条核心语句 (去缩进)。"""
    m = re.search(r"else:  # ORACLE_MODE=legacy[^\n]*\n(.*?)(?=\n    ctx, docs)", TEXT, re.S)
    assert m is not None, "legacy else 分支缺失"
    out = []
    for ln in m.group(1).splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def test_oracle_mode_env_defaults_to_db():
    """ORACLE_MODE env 存在且缺省 db (R4 用); 非法值在 ORACLE_INJECT=1 下 fail loud。"""
    assert 'ORACLE_MODE = os.environ.get("ORACLE_MODE", "db")' in TEXT
    assert 'ORACLE_MODE not in ("db", "legacy")' in TEXT
    assert "sys.exit(2)" in TEXT


def test_db_branch_maps_evidence_via_dia_to_ep_to_db_content():
    """db 分支按 dia_id → _dia_to_ep → episode_cache/msg_by_id 取 DB 同形 content。"""
    assert '_dia_to_ep.get(_did, "")' in TEXT
    assert 'episode_cache.get(_ep, {}).get("content")' in TEXT
    assert "(msg_by_id.get(_ep) or \"\").strip()" in TEXT
    assert 'def _oracle_db_content(evidence_msg):' in TEXT


def test_db_branch_fuzzy_fallback_and_skip():
    """无 dia_id/映射失败: content 去前缀反查回落; 仍无 → 整题跳过注入并计数。"""
    assert "def _oracle_ep_by_text(text):" in TEXT
    assert r"^\[date:[^\]]*\]\s*" in TEXT  # 去 [date: ...] 前缀正则
    assert "_oracle_db_skip += 1" in TEXT
    assert "整题跳过注入" in TEXT


def test_db_branch_injects_prefixed_content_to_channel_a():
    """注入 = DB 同形 content 整条去重后前插 channels['A'] (与现逻辑一致的置顶)。"""
    assert 'channels["A"] = _inj + channels["A"]' in TEXT
    assert "_inj.append(_c)" in TEXT
    assert "_oracle_db_ok += 1" in TEXT


def test_legacy_branch_byte_equivalent_to_v6141():
    """AC3: legacy 分支语句与 v6.14.1 (HEAD) 逐字节等价 (仅外包裹 else/缩进)。"""
    assert _legacy_stmts() == _OLD_LEGACY_STMTS


def test_db_summary_printed_after_loop():
    """注入后打印汇总: 'ORACLE db 同形注入: N/M 题成功 (M 题 evidence 无映射跳过)'。"""
    assert "ORACLE db 同形注入:" in TEXT
    assert "题 evidence 无映射跳过" in TEXT


def test_comment_cites_cat2_deanchor_invalidation():
    """注释记录动机: 旧裸文本注入 = cat2 去锚失效形态 (r0: 26.5% < 生产 43.1%)。"""
    assert "26.5%" in TEXT and "43.1%" in TEXT
    assert "DB 同形" in TEXT
