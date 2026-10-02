"""评测臂 LLM 粘性缓存 (STICKY) 单元测试.

覆盖:
  - hyde 侧: HYDE_STICKY=1 时过期缓存条目仍命中 (不触网, mock build_opener 计数 0);
    HYDE_STICKY=0 (缺省) 时过期条目 TTL miss → 触网 (TTL 行为不变)。
  - bench 侧 (scripts/sticky_cache.py): cached_suff_check/cached_followup_query
    enabled=True 时同指纹二次不触 LLM (mock llm_generate 计数); enabled=False 时
    行为与现实现一致 (每调必触 LLM, 无缓存); 异常路径维持 return True,""/question
    兜底且不写缓存; 指纹确定性 (同输入同指纹 / 不同输入不同指纹)。
  - bench 接线源码断言: STICKY_SUFF 常量 + sticky_cache 薄委托接线存在。
纯 stdlib, 不触 LLM/评测链路; import 方式沿用 test_r2_0_retry (sys.path 插 scripts)。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sticky_cache  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
BENCH_SRC = SCRIPTS / "bench_locomo_v72_ontology.py"
BENCH_TEXT = BENCH_SRC.read_text(encoding="utf-8")


# ─── hyde 侧: HYDE_STICKY 粘性缓存 ──────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_hyde_state(monkeypatch):
    from retrieval import hyde
    hyde._PERM_FAILED = False
    hyde._last_fail_ts = 0.0
    hyde._cache.clear()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("HYDE_STICKY", raising=False)


def test_hyde_sticky_on_expired_entry_hits_no_network(monkeypatch):
    """HYDE_STICKY=1: 过期缓存条目仍命中, 不触网 (build_opener 计数 0)。"""
    from retrieval import hyde
    monkeypatch.setenv("HYDE_STICKY", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    hyde._cache["q"] = (time.time() - 7200, "stale hypo")  # 已过期 2 小时
    with patch.object(hyde.urllib.request, "build_opener") as bo:
        assert hyde.generate_hypothesis("q") == "stale hypo"
    bo.assert_not_called()


def test_hyde_sticky_off_expired_entry_ttl_miss(monkeypatch):
    """HYDE_STICKY=0 (缺省): 过期条目 TTL miss → 触网 (TTL 行为不变)。"""
    from retrieval import hyde
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    hyde._cache["q"] = (time.time() - 7200, "stale hypo")  # 已过期 2 小时

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"fresh hypo"}}]}'

    opener = MagicMock()
    opener.open.return_value = FakeResp()
    with patch.object(hyde.urllib.request, "build_opener", return_value=opener) as bo:
        assert hyde.generate_hypothesis("q") == "fresh hypo"
    bo.assert_called_once()


def test_hyde_sticky_off_fresh_entry_still_hits():
    """HYDE_STICKY 缺省且条目未过期: 原缓存命中路径不变 (零回归)。"""
    from retrieval import hyde
    hyde._cache["q"] = (time.time(), "cached hypo")
    with patch.object(hyde.urllib.request, "build_opener") as bo:
        assert hyde.generate_hypothesis("q") == "cached hypo"
    bo.assert_not_called()


# ─── bench 侧: cached_suff_check / cached_followup_query ───────────────────

@pytest.fixture(autouse=True)
def _clear_sticky_caches():
    sticky_cache._suff_cache.clear()
    sticky_cache._followup_cache.clear()


def test_suff_check_enabled_hits_cache_once():
    calls = []

    def fake_llm(prompt, max_tokens=500, temperature=0.0):
        calls.append(1)
        return '{"sufficient": false, "missing_info": "the date of the meeting"}'

    docs = ["doc a", "doc b"]
    r1 = sticky_cache.cached_suff_check(fake_llm, "when was the meeting", docs, True)
    r2 = sticky_cache.cached_suff_check(fake_llm, "when was the meeting", docs, True)
    assert r1 == (False, "the date of the meeting")
    assert r2 == r1
    assert len(calls) == 1  # 同指纹二次命中缓存, 不触 LLM


def test_suff_check_enabled_different_input_touches_llm():
    calls = []

    def fake_llm(prompt, max_tokens=500, temperature=0.0):
        calls.append(prompt)
        return '{"sufficient": true, "missing_info": ""}'

    docs_a = ["doc a"]
    docs_b = ["doc b"]
    sticky_cache.cached_suff_check(fake_llm, "q", docs_a, True)
    sticky_cache.cached_suff_check(fake_llm, "q", docs_b, True)
    assert len(calls) == 2  # 不同指纹分别触 LLM


def test_suff_check_disabled_behaves_like_current():
    """enabled=False: 每调必触 LLM (无缓存), 输出与现实现一致。"""
    calls = []

    def fake_llm(prompt, max_tokens=500, temperature=0.0):
        calls.append(1)
        return '{"sufficient": false, "missing_info": "x"}'

    docs = ["doc a"]
    r1 = sticky_cache.cached_suff_check(fake_llm, "q", docs, False)
    r2 = sticky_cache.cached_suff_check(fake_llm, "q", docs, False)
    assert r1 == r2 == (False, "x")
    assert len(calls) == 2  # off 无缓存, 每次触 LLM
    assert sticky_cache._suff_cache == {}


def test_suff_check_exception_returns_default_and_no_cache():
    """异常路径: return (True, "") 兜底且不写缓存。"""
    calls = []

    def fake_llm(prompt, max_tokens=500, temperature=0.0):
        calls.append(1)
        raise RuntimeError("boom")

    docs = ["doc a"]
    r1 = sticky_cache.cached_suff_check(fake_llm, "q", docs, True)
    assert r1 == (True, "")
    assert len(calls) == 1
    # 不写缓存: 再调仍触 LLM
    r2 = sticky_cache.cached_suff_check(fake_llm, "q", docs, True)
    assert r2 == (True, "")
    assert len(calls) == 2
    assert sticky_cache._suff_cache == {}


def test_followup_enabled_hits_cache_once():
    calls = []

    def fake_llm(prompt, max_tokens=500, temperature=0.0):
        calls.append(1)
        return "what time was the meeting"

    r1 = sticky_cache.cached_followup_query(fake_llm, "q", "the time", True)
    r2 = sticky_cache.cached_followup_query(fake_llm, "q", "the time", True)
    assert r1 == r2 == "what time was the meeting"
    assert len(calls) == 1


def test_followup_disabled_behaves_like_current():
    calls = []

    def fake_llm(prompt, max_tokens=500, temperature=0.0):
        calls.append(1)
        return "  query with spaces  "

    r1 = sticky_cache.cached_followup_query(fake_llm, "q", "m", False)
    r2 = sticky_cache.cached_followup_query(fake_llm, "q", "m", False)
    assert r1 == r2 == "query with spaces"
    assert len(calls) == 2
    assert sticky_cache._followup_cache == {}


def test_followup_exception_returns_question_and_no_cache():
    calls = []

    def fake_llm(prompt, max_tokens=500, temperature=0.0):
        calls.append(1)
        raise RuntimeError("boom")

    r1 = sticky_cache.cached_followup_query(fake_llm, "orig question", "m", True)
    assert r1 == "orig question"
    assert len(calls) == 1
    r2 = sticky_cache.cached_followup_query(fake_llm, "orig question", "m", True)
    assert r2 == "orig question"
    assert len(calls) == 2
    assert sticky_cache._followup_cache == {}


def test_fingerprints_deterministic_and_distinct():
    """指纹确定性: 同输入同指纹; 不同输入不同指纹 (含跨块边界歧义消除)。"""
    assert sticky_cache.suff_fingerprint("q", ["ab", "c"]) != \
        sticky_cache.suff_fingerprint("q", ["a", "bc"])
    assert sticky_cache.suff_fingerprint("q", ["a", "b"]) == \
        sticky_cache.suff_fingerprint("q", ["a", "b"])
    assert sticky_cache.followup_fingerprint("q", "x") == \
        sticky_cache.followup_fingerprint("q", "x")
    assert sticky_cache.followup_fingerprint("q", "x") != \
        sticky_cache.followup_fingerprint("q", "y")


# ─── bench 接线源码断言 (不 import bench, 其顶层会跑全量评测) ──────────────

def test_bench_sticky_suff_env_defaults_off():
    assert 'STICKY_SUFF = os.environ.get("STICKY_SUFF", "0") == "1"' in BENCH_TEXT


def test_bench_imports_sticky_cache_module():
    assert "import sticky_cache" in BENCH_TEXT


def test_bench_delegates_to_sticky_cache():
    assert "def suff_check(question, docs_top):" in BENCH_TEXT
    assert "return sticky_cache.cached_suff_check(llm_generate, question, docs_top, STICKY_SUFF)" in BENCH_TEXT
    assert "def followup_query(question, missing):" in BENCH_TEXT
    assert "return sticky_cache.cached_followup_query(llm_generate, question, missing, STICKY_SUFF)" in BENCH_TEXT
