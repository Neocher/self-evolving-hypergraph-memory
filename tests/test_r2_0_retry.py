"""达摩院 R2-0 工程纪律回归 (round2 转化, 研究 §7 R2-0 行).

覆盖: llm_generate 瞬态失败自动重试 (2×, 总 3 次, 超时预算递增 300/600/900s 或等价)
+ 重试耗尽不静默 (append_predict_error 错误行落盘含 qa_id) + ctx_composition 组成字段。
只依赖 scripts/rag_v4_common (纯 stdlib), 不触 LLM/评测链路。
"""
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rag_v4_common  # noqa: E402


@pytest.fixture
def no_delay(monkeypatch):
    """测试不等待真实指数退避 (0.5s/1.0s)。"""
    monkeypatch.setattr(rag_v4_common, "_LLM_RETRY_DELAYS", (0.0, 0.0))


def test_llm_generate_retries_transient_then_succeeds(monkeypatch, no_delay):
    """AC1: 单次超时 (瞬态) → 自动重试 2 次后成功 (总 3 次尝试)。"""
    monkeypatch.delenv("MAAS_API_KEY", raising=False)
    calls = []

    def fake_chat(prompt, max_tokens=500, temperature=0.0, timeout=None):
        calls.append((max_tokens, temperature, timeout))
        if len(calls) < 3:
            raise TimeoutError(f"simulated timeout (attempt {len(calls)})")
        return "ok"

    monkeypatch.setattr(rag_v4_common, "_chat", fake_chat)
    out = rag_v4_common.llm_generate("q")
    assert out == "ok"
    assert len(calls) == 3  # 1 原发 + 2 重试
    # DeepSeek 基线 120s → 超时预算递增 120/240/360 (等价递增)
    assert [c[2] for c in calls] == [120, 240, 360]


def test_llm_generate_maas_timeout_budget_escalation(monkeypatch, no_delay):
    """AC1: MAAS 基线 300s → 超时预算 300/600/900 (研究 §7 '300s/600s/900s 或等价')。"""
    monkeypatch.setenv("MAAS_API_KEY", "test-key")
    timeouts = []

    def fake_chat(prompt, max_tokens=500, temperature=0.0, timeout=None):
        timeouts.append(timeout)
        raise urllib.error.HTTPError("http://x", 503, "busy", {}, None)

    monkeypatch.setattr(rag_v4_common, "_chat", fake_chat)
    with pytest.raises(urllib.error.HTTPError):
        rag_v4_common.llm_generate("q")
    assert timeouts == [300, 600, 900]


def test_llm_generate_all_fail_raises_after_3_attempts(monkeypatch, no_delay):
    """AC1: 3 次全败 → 抛异常 (由调用方落错误行), 不静默返回。"""
    monkeypatch.delenv("MAAS_API_KEY", raising=False)
    calls = []

    def fake_chat(prompt, max_tokens=500, temperature=0.0, timeout=None):
        calls.append(1)
        raise TimeoutError("always timeout")

    monkeypatch.setattr(rag_v4_common, "_chat", fake_chat)
    with pytest.raises(TimeoutError):
        rag_v4_common.llm_generate("q")
    assert len(calls) == 3


def test_llm_generate_non_retryable_raises_immediately(monkeypatch, no_delay):
    """确定性失败 (缺 key) 不重试: 1 次即抛。"""
    calls = []

    def fake_chat(prompt, max_tokens=500, temperature=0.0, timeout=None):
        calls.append(1)
        raise RuntimeError("DEEPSEEK_API_KEY not set (且未设 MAAS_API_KEY)")

    monkeypatch.setattr(rag_v4_common, "_chat", fake_chat)
    with pytest.raises(RuntimeError):
        rag_v4_common.llm_generate("q")
    assert len(calls) == 1


def test_retryable_classification():
    """瞬态 (429/5xx/超时/网络) 可重试; 4xx/缺 key 确定性失败不可重试。"""
    assert rag_v4_common._retryable(TimeoutError("t"))
    assert rag_v4_common._retryable(urllib.error.URLError("net"))
    assert rag_v4_common._retryable(urllib.error.HTTPError("http://x", 429, "rl", {}, None))
    assert rag_v4_common._retryable(urllib.error.HTTPError("http://x", 500, "e", {}, None))
    assert not rag_v4_common._retryable(urllib.error.HTTPError("http://x", 401, "auth", {}, None))
    assert not rag_v4_common._retryable(RuntimeError("DEEPSEEK_API_KEY not set"))
    assert not rag_v4_common._retryable(ValueError("bad request"))


def test_append_predict_error_writes_qa_id_line(tmp_path):
    """AC1: 3 次全败后错误行落盘 — 文件非空且含 qa_id + 错误原因, 不静默 skip。"""
    out = tmp_path / "predict_errors.jsonl"
    rag_v4_common.append_predict_error(str(out), "conv-1#q0007", TimeoutError("timeout after 900s"))
    rag_v4_common.append_predict_error(str(out), "conv-1#q0008", RuntimeError("boom"))
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0]["qa_id"] == "conv-1#q0007"
    assert "timeout" in lines[0]["error"]
    assert lines[1]["qa_id"] == "conv-1#q0008"
    assert "boom" in lines[1]["error"]


def test_ctx_composition_counts_sources():
    """AC2: CTX_DUMP 组成字段 — raw 证据条数/blocks 摘要条数/entity 段/来源占比。"""
    ctx = (
        "[ENTITY: Caroline]\n- Caroline attended_event_date: yesterday\n\n"
        "[GLOBAL CONTEXT (LLM-compressed memory blocks)]\n"
        "[MEMORY BLOCK 1] some summary one\n"
        "[MEMORY BLOCK 2] some summary two\n\n"
        "[DIRECT EVIDENCE]\n"
        "[1] [date: 4:33 pm on 12 July, 2023] [Caroline] hey\n"
        "[2] [date: 6:10 pm on 17 July, 2023] [Melanie] hi\n"
        "[10] [date: 9:12 am on 3 March, 2023] [Alex] yo\n"
    )
    comp = rag_v4_common.ctx_composition(ctx)
    assert comp["n_direct"] == 3
    assert comp["n_mem_block"] == 2
    assert comp["n_entity_seg"] == 1
    assert comp["n_rel_seg"] == 0
    assert comp["n_fact_seg"] == 0
    assert comp["chars_total"] == len(ctx)
    assert comp["chars_direct"] > 0
    assert 0.0 < comp["share_direct"] < 1.0
    assert abs(comp["share_direct"] + comp["share_org"] - 1.0) < 1e-9
    empty = rag_v4_common.ctx_composition("")
    assert empty["n_direct"] == 0 and empty["chars_total"] == 0
