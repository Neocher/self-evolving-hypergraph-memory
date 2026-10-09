"""本体暂存台账测试（v6.26.0）—— 未匹配概念的显式通道

覆盖：
- 开关语义（SHM_ONTOLOGY_STAGING）
- 记录 / 去重计数 / 实体与证据链
- 晋升与驳回裁决 + extended patch 产出
- 漂移代理指标
- fail-open（坏路径不抛）
- 落盘与重载
- **公共入口集成**：OntologyValidator._classify_ontology_type 未匹配时进台账
"""

from __future__ import annotations

import json
import os
import time

import pytest

from core import ontology_staging as stg
from core.ontology_staging import (
    OntologyStaging,
    StagedCandidate,
    candidate_key,
    get_staging,
    record_unmatched,
    reset_staging,
)


def _rec(s: OntologyStaging, text: str, **kw) -> str:
    """记录并断言拿到了键（供以 key 为参数的用例使用）。"""
    k = s.record(text, **kw)
    assert k is not None, f"record 未返回键: {text!r}"
    return k


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """每个用例独立台账文件 + 清洁单例 + 开关默认开。"""
    monkeypatch.delenv("SHM_ONTOLOGY_STAGING", raising=False)
    reset_staging(str(tmp_path / "staging.json"))
    yield
    reset_staging(str(tmp_path / "staging.json"))


# ─────────────────────── 基础：键与开关 ───────────────────────

def test_candidate_key_is_stable_and_normalized():
    """同句不同空白/大小写 → 同一键（去重的前提）。"""
    assert candidate_key("Hello   World") == candidate_key(" hello world ")
    assert candidate_key("A") != candidate_key("B")


def test_disabled_switch_records_nothing(tmp_path, monkeypatch):
    """SHM_ONTOLOGY_STAGING=0 → 不记录任何候选（一键关）。"""
    monkeypatch.setenv("SHM_ONTOLOGY_STAGING", "0")
    s = OntologyStaging(str(tmp_path / "off.json"))
    assert stg.enabled() is False
    assert s.record("任意未匹配文本") is None
    assert s.list_candidates() == []
    assert not os.path.exists(str(tmp_path / "off.json"))


@pytest.mark.parametrize("val", ["1", "true", "yes"])
def test_enabled_by_default_and_truthy(val, monkeypatch):
    monkeypatch.setenv("SHM_ONTOLOGY_STAGING", val)
    assert stg.enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "FALSE", ""])
def test_disabled_values_match_repo_convention(val, monkeypatch):
    """与 SHM_FACT_EXTRACT 同口径：\"0\"/\"false\"/\"\" → 关。"""
    monkeypatch.setenv("SHM_ONTOLOGY_STAGING", val)
    assert stg.enabled() is False


# ─────────────────────── 记录与去重 ───────────────────────

def test_record_creates_candidate():
    s = get_staging()
    key = s.record("用户喜欢在周末爬山", entities=["用户"], episode_id="ep-1")
    assert key and key == candidate_key("用户喜欢在周末爬山")
    items = s.list_candidates()
    assert len(items) == 1
    assert items[0]["count"] == 1
    assert items[0]["status"] == "pending"
    assert items[0]["entities"] == ["用户"]
    assert items[0]["evidence"] == ["ep-1"]


def test_repeat_increments_count_without_new_candidate():
    """同句被反复降级只累加计数 —— 不灌爆台账。"""
    s = get_staging()
    for i in range(5):
        s.record("同一句话被反复分类", episode_id=f"ep-{i}")
    items = s.list_candidates()
    assert len(items) == 1
    assert items[0]["count"] == 5
    assert items[0]["first_seen"] <= items[0]["last_seen"]
    assert len(items[0]["evidence"]) == 5


def test_evidence_and_entities_are_capped():
    """证据链与实体列表有上限，防单条候选无限膨胀。"""
    s = get_staging()
    for i in range(40):
        s.record("上限测试文本", entities=[f"e{i}"], episode_id=f"ep-{i}")
    c = s.list_candidates()[0]
    assert len(c["evidence"]) <= stg._MAX_EVIDENCE
    assert len(c["entities"]) <= 12


def test_empty_text_is_ignored():
    s = get_staging()
    assert s.record("") is None
    assert s.record("   ") is None
    assert s.list_candidates() == []


def test_list_sorted_by_count_desc():
    """推荐排序：最常被降级的概念 = 最该补的本体缺口。"""
    s = get_staging()
    for _ in range(3):
        s.record("高频缺口")
    s.record("低频缺口")
    items = s.list_candidates()
    assert [i["sample_text"] for i in items] == ["高频缺口", "低频缺口"]


# ─────────────────────── 裁决与晋升 ───────────────────────

def test_promote_and_extended_patch():
    """晋升 → 产出可合入 ontology_extended.json 的片段。"""
    s = get_staging()
    k = _rec(s, "预算是运维成本口径")
    assert s.promote(k, "financial_metric") is True

    assert s.list_candidates(status="pending") == []
    promoted = s.list_candidates(status="promoted")
    assert len(promoted) == 1 and promoted[0]["promoted_to"] == "financial_metric"

    patch = s.build_extended_patch()
    assert "financial_metric" in patch
    assert "预算是运维成本口径" in patch["financial_metric"]["conflict_keys"]


def test_patch_merges_same_type_and_not_polluted_by_pending():
    s = get_staging()
    k1 = _rec(s, "毛利率口径 A")
    k2 = _rec(s, "毛利率口径 B")
    s.record("尚未裁决的候选")
    s.promote(k1, "financial_metric")
    s.promote(k2, "financial_metric")
    patch = s.build_extended_patch()
    assert len(patch["financial_metric"]["conflict_keys"]) == 2
    assert "尚未裁决的候选" not in json.dumps(patch, ensure_ascii=False)


def test_reject_excludes_from_pending():
    s = get_staging()
    k = _rec(s, "噪声文本")
    assert s.reject(k) is True
    assert s.list_candidates(status="pending") == []
    assert len(s.list_candidates(status="rejected")) == 1


def test_promote_unknown_key_returns_false():
    assert get_staging().promote("deadbeefdeadbeef", "generic_fact") is False


def test_promote_does_not_touch_live_ontology(tmp_path):
    """本模块不直接改活本体 —— 只记裁决 + 产片段。"""
    live = tmp_path / "ontology_extended.json"
    live.write_text('{"keep": {"conflict_keys": ["x"]}}', encoding="utf-8")
    s = OntologyStaging(str(tmp_path / "s.json"))
    s.promote(_rec(s, "新候选"), "some_type")
    assert json.loads(live.read_text(encoding="utf-8")) == {"keep": {"conflict_keys": ["x"]}}


# ─────────────────────── 漂移代理指标 ───────────────────────

def test_stats_unmatched_rate():
    s = get_staging()
    for _ in range(3):
        s.note_classification(True)
    s.note_classification(False)
    st = s.stats()
    assert st["counters"] == {"matched": 3, "unmatched": 1}
    assert st["unmatched_rate"] == 0.25


def test_stats_empty_is_zero_not_crash():
    st = get_staging().stats()
    assert st["unmatched_rate"] == 0.0
    assert st["total_candidates"] == 0


def test_stats_top_pending_ranks_highest():
    s = get_staging()
    for _ in range(4):
        s.record("最热缺口")
    s.record("冷缺口")
    top = s.stats()["top_pending"]
    assert top[0]["sample_text"] == "最热缺口" and top[0]["count"] == 4


# ─────────────────────── fail-open 与持久化 ───────────────────────

def test_fail_open_on_bad_path(tmp_path):
    """坏路径（目录不可写）→ 不抛异常，主写入流程不受影响。"""
    bad = tmp_path / "nope" / "deeper" / "s.json"
    bad.parent.mkdir(parents=True)
    os.chmod(bad.parent, 0o500)
    s = OntologyStaging(str(bad))
    try:
        assert s.record("fail-open 测试") is not None   # 内存记账仍成功
        assert s.list_candidates()[0]["count"] == 1
    finally:
        os.chmod(bad.parent, 0o700)


def test_hot_path_hooks_never_raise(monkeypatch):
    """模块级热路径钩子在内部异常时静默降级。"""
    class Boom:
        def note_classification(self, *_a, **_k): raise RuntimeError("boom")
        def record(self, *_a, **_k): raise RuntimeError("boom")
    monkeypatch.setattr(stg, "get_staging", lambda *a, **k: Boom())
    stg.note_classification(True)            # 不抛
    assert stg.record_unmatched("x") is None  # 返回 None 而非异常


def test_flush_and_reload_roundtrip(tmp_path):
    p = str(tmp_path / "round.json")
    s = OntologyStaging(p)
    s.record("持久化文本", episode_id="ep-x")
    s.note_classification(True)
    assert s.flush() is True

    s2 = OntologyStaging(p)
    items = s2.list_candidates()
    assert len(items) == 1 and items[0]["sample_text"] == "持久化文本"
    assert s2.stats()["counters"]["matched"] == 1


def test_time_based_flush_persists_low_volume(tmp_path, monkeypatch):
    """低写入量下也必须在时间阈值内落盘（否则 SIGKILL 会丢整段记账）。"""
    p = str(tmp_path / "timed.json")
    s = OntologyStaging(p)
    monkeypatch.setattr(stg, "_FLUSH_INTERVAL_S", 0.0)
    s.record("低写入量记账")
    for _ in range(40):
        if os.path.exists(p):
            break
        time.sleep(0.05)
    assert os.path.exists(p), "时间阈值应触发落盘"
    assert json.loads(open(p, encoding="utf-8").read())["candidates"]


def test_corrupt_file_degrades_to_empty(tmp_path):
    """文件损坏 → 空台账降级，不抛。"""
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    s = OntologyStaging(str(p))
    assert s.list_candidates() == []
    assert s.record("损坏后仍可写") is not None


def test_from_dict_tolerates_unknown_fields():
    c = StagedCandidate.from_dict({"key": "k", "sample_text": "t", "future_field": 1})
    assert c.key == "k" and c.status == "pending"


# ─────────────────────── 公共入口集成（AGENTS.md 要求） ───────────────────────

def test_validator_unmatched_goes_to_staging():
    """走真实分类入口：未匹配文本必须进台账，而非静默丢失。"""
    from core.ontology_validator import OntologyValidator

    v = OntologyValidator()
    text = "这是一句绝对不可能命中任何冲突键的独特文本_zzy_9911"
    v._classify_ontology_type(text, ["实体A"])

    items = get_staging().list_candidates()
    assert any(i["sample_text"].startswith("这是一句绝对不可能命中") for i in items), \
        "未匹配文本必须被台账记录（修复前是静默丢弃）"


def test_validator_matched_does_not_pollute_staging():
    """命中类型时不应产生候选，但应计入 matched 计数。"""
    from core.ontology_validator import OntologyValidator

    v = OntologyValidator()
    merged = v._merged_ontology_types()
    # 取任一原生类型的首个冲突键构造必中文本
    otype, info = next(iter(merged.items()))
    key = (info.get("conflict_keys") or ["x"])[0]
    v._classify_ontology_type(f"文本包含 {key}", [])

    assert get_staging().list_candidates(status="pending") == []
    assert get_staging().stats()["counters"]["matched"] >= 1


def test_evolution_classify_uses_same_channel():
    """ontology_evolution.classify_with_extended 与 validator 同口径。"""
    from core.ontology_evolution import classify_with_extended

    classify_with_extended("另一句不可能命中的独特文本_qq_8842", [])
    assert any("_qq_8842" in i["sample_text"]
               for i in get_staging().list_candidates())
