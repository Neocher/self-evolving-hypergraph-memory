"""用户画像层 User-Profile (v5.39.0) 测试。

覆盖任务书测试清单：
1. scan：direct + 句首第一人称 + 模式词 → 命中；agent 报告 inferred → 不命中（误识别）；
   英文低误报词 → 命中
2. aggregate：同值去重归一 + source_type 权重累积 + 多源计数加分
3. build_profile：空候选 → 空 dict 不 crash；分组结构正确
4. search_profile：画像命中 → 返回上下文；空画像 → 空结果
5. _deduplicate_and_sort：画像命中 score ×1.2 且钳制 ≤1.0，非命中不变（防主链路回归 + P1 越界防回归）
"""
from __future__ import annotations

import pytest

from core.user_profile import (
    aggregate,
    build_profile,
    rebuild_or_keep,
    scan_preference_candidates,
    scan_rows,
)
from retrieval.query_router import QueryRouter, set_user_profile


@pytest.fixture(autouse=True)
def _clean_profile():
    """隔离模块级画像注入，避免跨用例/跨文件污染。"""
    set_user_profile({})
    yield
    set_user_profile({})


# ─── scan ──────────────────────────────────────────────────

def test_scan_direct_first_person_hit():
    """「我喜欢咖啡」direct + 句首我 → 命中 preferences/咖啡。"""
    cands = scan_preference_candidates([
        {"content": "我喜欢咖啡", "source_type": "direct"},
    ])
    assert cands == [{"value": "咖啡", "group": "preferences", "source_type": "direct"}]


def test_scan_agent_report_inferred_no_hit():
    """hermes 报告含「我是」但 source_type=inferred → 不命中（误识别用例）。"""
    cands = scan_preference_candidates([
        {"content": "我是资深工程师，擅长架构设计", "source_type": "inferred"},
    ])
    assert cands == []


def test_scan_english_low_false_positive_hit():
    """英文 I live in SF（direct + 句首 I）→ 命中 identity/SF。"""
    cands = scan_preference_candidates([
        {"content": "I live in SF", "source_type": "direct"},
    ])
    assert cands == [{"value": "SF", "group": "identity", "source_type": "direct"}]


def test_scan_english_my_favorite_hit():
    """英文 My favorite ...（direct + 句首 My，大写 M）→ 命中 preferences（声明可达性回归）。"""
    cands = scan_preference_candidates([
        {"content": "My favorite color is blue", "source_type": "direct"},
    ])
    assert len(cands) == 1
    assert cands[0]["group"] == "preferences"
    assert cands[0]["value"] == "colorisblue"


# ─── scan_rows（query_cypher 行兼容 + b64 + 来源门控） ─────────

def test_scan_rows_alias_flat_b64_chinese():
    """别名扁平行 {b64} 中文内容 → 解码后扫描命中（同 system.py:323 语义）。"""
    import base64
    raw = "{b64}" + base64.b64encode("我喜欢咖啡".encode("utf-8")).decode("ascii")
    nodes = scan_rows([{"e.content": raw, "e.source_type": "direct"}])
    assert nodes == [{"content": "我喜欢咖啡", "source_type": "direct"}]
    cands = scan_preference_candidates(nodes)
    assert cands == [{"value": "咖啡", "group": "preferences", "source_type": "direct"}]


def test_scan_rows_alias_flat_missing_source_defaults_inferred():
    """别名扁平行缺失 source_type → 默认 inferred（防绕过 direct 门控）。"""
    nodes = scan_rows([{"e.content": "我是资深工程师"}])
    assert nodes == [{"content": "我是资深工程师", "source_type": "inferred"}]
    assert scan_preference_candidates(nodes) == []


# ─── rebuild_or_keep（空扫描不覆盖已有画像） ─────────────────

def test_rebuild_or_keep_empty_scan_keeps_existing():
    """空扫描/查询失败（rows=[]）→ 保留已有画像，防空覆盖。"""
    existing = {"preferences": {"咖啡": {"weight": 1.0, "sources": 1}}}
    assert rebuild_or_keep(existing, []) == existing
    assert rebuild_or_keep(existing, None) == existing


def test_rebuild_or_keep_nonempty_rebuild_overwrites():
    """非空重建结果 → 覆盖旧画像。"""
    nodes = [{"content": "我喜欢喝茶", "source_type": "direct"}]
    out = rebuild_or_keep({"preferences": {"咖啡": {"weight": 1.0, "sources": 1}}}, nodes)
    assert out == {"preferences": {"喝茶": {"weight": 1.0, "sources": 1}}}


# ─── aggregate ─────────────────────────────────────────────

def test_aggregate_dedup_weight():
    """同值去重：direct 1.0 + tool 0.7 累积 + 多源计数加分 0.1。"""
    grouped = aggregate([
        {"value": "咖啡", "group": "preferences", "source_type": "direct"},
        {"value": "咖啡", "group": "preferences", "source_type": "tool"},
    ])
    entry = grouped["preferences"]["咖啡"]
    assert entry["weight"] == pytest.approx(1.0 + 0.7 + 0.1)
    assert entry["sources"] == 2


def test_aggregate_normalize_grouping():
    """归一：标点/空格/「的」去除；分组正确。"""
    grouped = aggregate([
        {"value": "咖啡，", "group": "preferences", "source_type": "direct"},
        {"value": "软件工程师", "group": "work", "source_type": "direct"},
    ])
    assert "咖啡" in grouped["preferences"]
    assert "软件工程师" in grouped["work"]
    assert grouped["identity"] == {}


# ─── build_profile ─────────────────────────────────────────

def test_build_profile_empty_candidates():
    """空候选 → 空 dict 不 crash。"""
    assert build_profile([]) == {}
    assert build_profile(None) == {}


def test_build_profile_structure():
    """分组结构：preferences/identity/work 只含有值组。"""
    profile = build_profile([
        {"value": "咖啡", "group": "preferences", "source_type": "direct"},
        {"value": "北京", "group": "identity", "source_type": "direct"},
    ])
    assert set(profile.keys()) == {"preferences", "identity"}
    assert "咖啡" in profile["preferences"]
    assert profile["preferences"]["咖啡"]["weight"] == pytest.approx(1.0)


# ─── search_profile（旁路） ────────────────────────────────

def test_search_profile_hit():
    """画像命中 → 返回上下文块（matched + context + hits）。"""
    set_user_profile(build_profile([
        {"value": "咖啡", "group": "preferences", "source_type": "direct"},
    ]))
    router = QueryRouter.__new__(QueryRouter)
    out = router.search_profile("用户喜欢什么咖啡")
    assert out["matched"] is True
    assert "咖啡" in out["context"]
    assert out["hits"][0]["group"] == "preferences"


def test_search_profile_empty_profile():
    """空画像 → 空结果（matched=False，context 空）。"""
    router = QueryRouter.__new__(QueryRouter)
    out = router.search_profile("随便问问")
    assert out == {"matched": False, "context": ""}


# ─── 检索加分（防主链路回归） ──────────────────────────────

def test_deduplicate_profile_boost():
    """画像值命中 → score ×1.2；命中 1.0 钳制到 1.0（EpisodicResult 契约 le=1.0）。"""
    set_user_profile({"preferences": {"咖啡": {"weight": 1.0, "sources": 1}}})
    results = [
        {"node_id": "a", "content": "今天喝咖啡", "score": 1.0, "fact_track": "active"},
        {"node_id": "b", "content": "开会讨论排期", "score": 0.9, "fact_track": "active"},
    ]
    out = QueryRouter._deduplicate_and_sort(results)
    # score=1.0 × 1.2 = 1.2 → 钳制 1.0（越界会让 /memories/retrieve 500）
    assert out[0]["node_id"] == "a" and out[0]["score"] == 1.0
    assert out[1]["node_id"] == "b" and out[1]["score"] == 0.9


def test_deduplicate_profile_boost_math_below_cap():
    """非边界：0.8 × 1.2 = 0.96（boost 数学未被钳制吞掉，排序语义保留）。"""
    set_user_profile({"preferences": {"咖啡": {"weight": 1.0, "sources": 1}}})
    results = [
        {"node_id": "a", "content": "今天喝咖啡", "score": 0.8, "fact_track": "active"},
        {"node_id": "b", "content": "开会讨论排期", "score": 0.85, "fact_track": "active"},
    ]
    out = QueryRouter._deduplicate_and_sort(results)
    # a: 0.8×1.2=0.96 > b: 0.85（boost 反超 + 未越界）
    assert out[0]["node_id"] == "a" and out[0]["score"] == pytest.approx(0.8 * 1.2)
    assert out[1]["node_id"] == "b" and out[1]["score"] == 0.85


def test_deduplicate_profile_boost_namespace_agnostic():
    """P2-单租户语义固化：加分不感知 namespace（模块级 _USER_PROFILE 全局）。

    文档化声明（query_router.py / user_profile.py / app.py）：当前单租户部署，
    跨 namespace 画像共享为已知接受语义；未来多租户需 {namespace: profile} 键控，
    本测试在重构时须随之更新（断言加分仍对任意 namespace 生效）。
    """
    set_user_profile({"preferences": {"咖啡": {"weight": 1.0, "sources": 1}}})
    results = [
        {"node_id": "a", "content": "今天喝咖啡", "score": 0.8, "fact_track": "active"},
        {"node_id": "b", "content": "开会讨论排期", "score": 0.8, "fact_track": "active"},
    ]
    out = QueryRouter._deduplicate_and_sort(results)
    # 加分不区分 namespace：命中画像的 a 即使属于任意 namespace 仍 ×1.2
    assert out[0]["node_id"] == "a" and out[0]["score"] == pytest.approx(0.8 * 1.2)
    assert out[1]["node_id"] == "b" and out[1]["score"] == 0.8
