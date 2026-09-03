"""达摩院 R2c 输出协议 prompt v2 — 源码断言 (AC3).

断言 scripts/bench_locomo_v72_ontology.py 内 _READER_PROMPT_V2 常量满足:
日期换算规则 + 粒度纪律 (禁时刻/禁弱化词/范围保范围) + 计数/枚举报数 + 范围纪律
+ ≥3 组 few-shot; 且无 concise/short/brief 长度压缩指令 (研究 §6: 实测伤分)。
纯文本断言, 不 import bench (其顶层会加载检索/灌库, 非单测面)。
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "bench_locomo_v72_ontology.py"
TEXT = SRC.read_text(encoding="utf-8")

_V2_BLOCK = re.search(r'_READER_PROMPT_V2 = """(.*?)"""\n', TEXT, re.S)
assert _V2_BLOCK is not None, "_READER_PROMPT_V2 常量缺失"
BODY = _V2_BLOCK.group(1)


def test_v2_has_date_anchor_conversion_rules():
    """日期锚换算: 相对时间词须对 [date:] 前缀做显式换算, 禁止把消息日期当事件日期。"""
    assert "[date:" in BODY
    for rel in ("yesterday", "last week", "two weekends", "last Monday"):
        assert rel in BODY, f"相对时间词规则缺 {rel!r}"
    assert "NOT" in BODY  # 显式否定消息日期=事件日期


def test_v2_has_granularity_discipline():
    """粒度纪律: 与证据可推粒度一致; 禁时刻 (hh:mm); 禁弱化词; 相对范围保留范围。"""
    assert "week before" in BODY  # 范围保留表达
    assert "hour/minute" in BODY   # 禁时刻指令
    for hedge in ("likely", "around", "approximately", "probably"):
        assert hedge in BODY, f"禁弱化词指令缺 {hedge!r}"


def test_v2_has_enumeration_and_count():
    """计数/枚举纪律: 全覆盖后显式报数 (N items / Two:)。"""
    assert "items" in BODY
    assert "count" in BODY.lower() or "Count" in BODY


def test_v2_has_scope_rule():
    """范围纪律 (cat4 overreach): 只答题面所问, 不追加未问细节。"""
    assert "not asked for" in BODY or "did not ask for" in BODY


def test_v2_has_three_few_shots():
    """2-3 个 few-shot: 相对日期换算 / 粒度匹配 / 枚举报数各一 (模板 {question} 另有 1 处)。"""
    assert BODY.count("\nQuestion: ") >= 4  # 3 few-shot + 1 模板占位


def test_v2_forbids_length_compression():
    """禁长度压缩指令 (research §6: 'short answer' 式指令 oracle 实测伤分)。"""
    lower = BODY.lower()
    for banned in ("concise", "short", "brief"):
        assert banned not in lower, f"V2 prompt 含禁词 {banned!r}"


def test_v2_format_placeholders_intact():
    """模板占位符唯一且无游离花括号 (可直接 .format(ctx=..., question=...))。"""
    assert BODY.count("{ctx}") == 1 and BODY.count("{question}") == 1
    assert BODY.count("{") == 2 and BODY.count("}") == 2


def test_v1_prompt_unchanged_v2_toggle_present():
    """PROMPT_V2 env 切换存在且默认开; v1 原文保留 (A/B 零回归臂)。"""
    assert 'os.environ.get("PROMPT_V2", "1") != "0"' in TEXT
    assert "_READER_PROMPT_V1" in TEXT
    assert "_READER_PROMPT_V2" in TEXT
