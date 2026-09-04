"""达摩院 P0-b 确定性时间层 — bench harness 源码断言 (AC2/AC4 + 开关语义).

断言 scripts/bench_locomo_v72_ontology.py (v6.17.0) 内 P0-b 装配:
- AC2: 双路径注解呈现都在 harness — (a) 组织路径 [TIME ANCHORS] 段引用 DIRECT
  EVIDENCE 消息编号 (append_anchor_block / "[DIRECT EVIDENCE]" 判定), (b) round2
  平铺路径每条 raw 消息后内联 '[time: ...]' (inline_numbered_lines);
- AC4: reader prompt V1 原文不变 (_READER_PROMPT_V1 常量逐字节 = 备份 v6.16.0 原文);
- 开关: TIME_ANCHORS env 缺省 '0' (off, 与 SESSION_SCOPE 同源关闭 → v6.16.0 逐字节
  等价基线), 独立于 SESSION_SCOPE 可 A/B; scope=on 时注解落在会话内消息 (引擎限定);
- 注释: 组织段文本只陈述事实 (无祈使)。
纯文本断言, 不 import bench (其顶层会加载检索/灌库/embedding, 非单测面)。
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "bench_locomo_v72_ontology.py"
TEXT = SRC.read_text(encoding="utf-8")

# reader prompt V1 原文快照 (v6.16.0 / .bak.p0b.20260904 逐字节): P0-b 注解在 ctx 装配层,
# prompt 侧零改动 — AC4 diff=空。禁任何'请换算/请用绝对日期'式指令 (R2c 覆辙 -8.3pp)。
_READER_PROMPT_V1_SNAPSHOT = """Answer the question based on the conversation snippets below. Reason across snippets if needed (e.g., infer dates from session timestamps).

Conversation snippets:
{ctx}

Question: {question}
Answer:"""


def _v1_from(src: str) -> str:
    m = re.search(r"_READER_PROMPT_V1 = (\"\"\".*?\"\"\")", src, re.S)
    assert m is not None, "_READER_PROMPT_V1 缺失"
    return m.group(1)


def test_time_anchors_env_defaults_off_independent():
    """TIME_ANCHORS env 存在且缺省 off ('0') — 与 SESSION_SCOPE 默认 off 同源 →
    双 off 与 v6.16.0 逐字节等价 (A/B 基线); 独立 env 可四象限 A/B。"""
    assert 'TIME_ANCHORS = os.environ.get("TIME_ANCHORS", "0") == "1"' in TEXT
    assert 'SESSION_SCOPE = os.environ.get("SESSION_SCOPE", "0") == "1"' in TEXT


def test_time_anchors_import_present():
    """确定性模块 import 进 harness (scripts/time_anchors.py, 零 LLM)。"""
    assert "import time_anchors" in TEXT
    assert (Path(__file__).resolve().parents[1] / "scripts" / "time_anchors.py").exists()


def test_org_path_time_anchors_section_annotated():
    """AC2 路径 (a): 组织 ctx 装配点判定 [DIRECT EVIDENCE] → append_anchor_block
    (尾部 [TIME ANCHORS] 段引用 DIRECT EVIDENCE 消息编号)。"""
    assert "def _time_anchor_final_ctx(ctx):" in TEXT
    assert 'if "[DIRECT EVIDENCE]" in ctx:' in TEXT
    assert "time_anchors.append_anchor_block(ctx, max_docs=30)" in TEXT
    # 装配点在 round2 之后、CTX_DUMP/prompt 之前 (ctx 最终形态上注解, 两路径同效)
    assert "ctx = _time_anchor_final_ctx(ctx)" in TEXT
    assert "if TIME_ANCHORS and not ctx" not in TEXT  # 不吞空 ctx


def test_flat_path_inline_annotation_annotated():
    """AC2 路径 (b): round2 平铺 ctx → 每条 raw 消息后内联 [time: ...]
    (inline_numbered_lines; 无组织段判定走内联)。"""
    assert "time_anchors.inline_numbered_lines(ctx.splitlines(), max_docs=20)" in TEXT
    assert "[time: " in TEXT  # 内联注解行形态出现在 harness


def test_config_log_prints_time_anchors_state():
    """启动配置行打印 TIME_ANCHORS 状态 (A/B 臂可溯)。"""
    assert "TIME_ANCHORS={'on' if TIME_ANCHORS else 'off'}" in TEXT


def test_reader_prompt_v1_unchanged_vs_backup():
    """AC4: reader prompt V1 原文逐字节不变 (diff=空, 对照 v6.16.0 快照) — P0-b 只
    注解 ctx 结构, 不向 prompt 加任何'请换算/请用绝对日期'式指令 (R2c 覆辙 -8.3pp)。"""
    assert _v1_from(TEXT) == f'"""{_READER_PROMPT_V1_SNAPSHOT}"""'
    v1 = _v1_from(TEXT)
    assert "Answer the question based on the conversation snippets below" in v1
    assert "{ctx}" in v1 and "{question}" in v1


def test_scope_on_annotations_stay_in_session_messages():
    """scope=on 时注解只落在会话内消息: 注解装配在 scoped ctx (DIRECT EVIDENCE 已
    引擎会话限定) 上做; 无 harness 层跨会话回捞逻辑 (污染池上注解是次优的)。"""
    # 装配点紧邻 round2 分支 (scoped 臂 ctx 只含会话内), 无附加跨会话检索
    assert "_time_anchor_final_ctx(ctx)" in TEXT
    # scoped 组织段直用引擎限定结果 (P0-a 既有), P0-b 不新增数据面
    assert "retrieve_channels(question, hitk=HITK_MODE, session_ts=session_ts, scope=_scope)" in TEXT


def test_org_section_text_is_factual_no_imperative():
    """[TIME ANCHORS] 段文本只陈述事实 (确定性解析结果), 无祈使指令行。"""
    mod = (Path(__file__).resolve().parents[1] / "scripts" / "time_anchors.py").read_text(encoding="utf-8")
    assert '"[TIME ANCHORS (deterministic calendar resolution)]' in mod
    for banned in ("please convert", "you must", "请换算", "请用绝对日期"):
        assert banned not in mod, f"确定性模块含祈使/说教词 {banned!r}"
