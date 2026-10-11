# -*- coding: utf-8 -*-
"""社区成员 sidecar 索引覆盖率与门控修复（v6.33）验收测试。

对齐任务书 AC1..AC6：
  AC1 索引覆盖率 100%：构建后 sidecar 键数 == 全部 CommunityNode 数，
      空社区存空列表（而非缺失）。
  AC2 门控不再误判：批次中含 1 个不在索引的社区 → 不回落到边遍历、该社区
      返回空集、其余社区与 sidecar 一致（默认按社区门控）。
  AC3 严格模式仍可用：strict_fallback=True → 任一缺失即整批回落原路径。
  AC4 关闭 sidecar 零回归：enabled=False → 完全走边遍历（与改动前一致）。
  AC5 后台刷新与冷却：缺失占比超阈值触发刷新、冷却期不重复、失败不影响检索。
  AC6 提速保持：sidecar 就绪后 114 个社区取成员 < 100ms（小样本，不跑全量构建）。
  另：community_expansion.budget_s 在代码默认值 / settings / yaml 三处均为 2.0。

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_community_index_coverage.py -q
"""
from __future__ import annotations

import logging
import time

import pytest

from core.dream_pipeline import DreamPipeline

LABEL_COMMUNITY = "CommunityNode"


def _create_episode(store, eid: str, content: str, tau: float = 1.0) -> None:
    """创建 EpisodeNode（archived=False 显式落库）。"""
    store.create_episode({
        "id": eid, "content": content, "tau_initial": tau,
        "fact_track": "active", "archived": False,
    })


def _create_community(store, cid: str) -> None:
    """直接 upsert 一个 CommunityNode（无成员边，供覆盖率/空列表断言）。"""
    store._locked_upsert_node(LABEL_COMMUNITY, cid, {"id": cid})


def _persist_community(store, cid: str, members: list[str], report: str) -> None:
    """用生产 _persist_one_community 造真实社区（含 COMMUNITY_MEMBER 边）。"""
    DreamPipeline()._persist_one_community(
        store, {"id": cid, "members": members, "report": report}, "eval", 0
    )


def _community_count(store) -> int:
    """直接按构建同款 GQL 数全部 CommunityNode（绕过翻译层，最贴近 AC1 措辞）。"""
    result = store._locked_execute_gql(
        "MATCH (c:CommunityNode) RETURN count(*) AS cnt"
    )
    rows = list((result or {}).get("rows") or [])
    if not rows:
        return 0
    row = rows[0]
    if isinstance(row, dict):
        return int(row.get("cnt", 0) or 0)
    return int(row[0] if isinstance(row, (list, tuple)) else 0)


class TestIndexCoverage:
    """AC1 — 索引覆盖 100% 社区（空社区存空列表）。"""

    def test_build_covers_all_communities_with_empty_lists(self, overgraph_store):
        _create_episode(overgraph_store, "ep0", "c0", tau=1.0)
        _persist_community(overgraph_store, "comm_a", ["ep0"], "ra")
        _persist_community(overgraph_store, "comm_empty", [], "empty report")

        edge_total, index = overgraph_store.build_community_members_index(page_size=4)

        assert edge_total == 1, f"edge_total={edge_total}"
        assert len(index) == _community_count(overgraph_store), (
            f"sidecar 键数 {len(index)} != 全部社区数 {_community_count(overgraph_store)}"
        )
        assert "comm_empty" in index, "空社区未被收录"
        assert index["comm_empty"] == [], "空社区应存空列表而非缺失"
        assert set(index.get("comm_a", [])) == {"ep0"}

    def test_loaded_index_keeps_empty_lists(self, overgraph_store):
        """空列表条目须经 refresh→load 后仍保留（否则「索引缺失=不存在」前提失效）。"""
        _create_episode(overgraph_store, "ep0", "c0", tau=1.0)
        _persist_community(overgraph_store, "comm_a", ["ep0"], "ra")
        _persist_community(overgraph_store, "comm_empty", [], "empty report")
        assert overgraph_store.refresh_community_members_index() is True

        loaded = overgraph_store._load_community_members_index()
        assert loaded is not None
        assert "comm_empty" in loaded
        assert loaded["comm_empty"] == []


class TestPerCommunityGating:
    """AC2 — 默认按社区门控：缺失社区 → 空集，不回落到边遍历。"""

    def test_missing_community_treated_as_empty_not_fallback(
        self, overgraph_store, monkeypatch, caplog,
    ):
        _create_episode(overgraph_store, "ep_a", "A", tau=1.0)
        _persist_community(overgraph_store, "comm_in", ["ep_a"], "report")
        assert overgraph_store.refresh_community_members_index() is True

        # 构建快照之后新建的社区（不在索引里）
        _create_episode(overgraph_store, "ep_b", "B", tau=1.0)
        _persist_community(overgraph_store, "comm_out", ["ep_b"], "report")

        edge_calls: list[list[str]] = []
        real_edges = overgraph_store._community_members_from_edges

        def spy_edges(community_ids, limit):
            edge_calls.append(list(community_ids))
            return real_edges(community_ids, limit)

        refresh_calls: list[tuple[int, int]] = []
        monkeypatch.setattr(overgraph_store, "_community_members_from_edges", spy_edges)
        monkeypatch.setattr(
            overgraph_store,
            "_maybe_refresh_community_members_index",
            lambda missing, total: refresh_calls.append((missing, total)) or True,
        )

        with caplog.at_level(logging.WARNING):
            result = overgraph_store.get_community_members_batch(
                ["comm_in", "comm_out"], 10
            )

        assert edge_calls == [], "缺失社区不应回落原边遍历"
        assert {m["member_id"] for m in result["comm_in"]} == {"ep_a"}
        assert result["comm_out"] == [], "缺失社区应视为空集"
        assert refresh_calls == [(1, 2)], "缺失占比信息未送达刷新调度"
        assert any("missing 1/2 communities (50.0%)" in r.message
                   for r in caplog.records), "缺失 WARNING（含缺失数与占比）未记录"


class TestStrictFallback:
    """AC3 — strict_fallback=True 保留旧「任一缺失即整批回落」。"""

    def test_strict_fallback_whole_batch(self, overgraph_store, monkeypatch):
        _create_episode(overgraph_store, "ep_a", "A", tau=1.0)
        _persist_community(overgraph_store, "comm_in", ["ep_a"], "report")
        assert overgraph_store.refresh_community_members_index() is True

        overgraph_store.config.community_members_index_strict_fallback = True
        overgraph_store.reload_community_members_index()

        _create_episode(overgraph_store, "ep_b", "B", tau=1.0)
        _persist_community(overgraph_store, "comm_out", ["ep_b"], "report")

        edge_calls: list[list[str]] = []
        real_edges = overgraph_store._community_members_from_edges

        def spy_edges(community_ids, limit):
            edge_calls.append(list(community_ids))
            return real_edges(community_ids, limit)

        monkeypatch.setattr(overgraph_store, "_community_members_from_edges", spy_edges)
        result = overgraph_store.get_community_members_batch(
            ["comm_in", "comm_out"], 10
        )

        assert edge_calls == [["comm_in", "comm_out"]], "严格模式下应整批回落原路径"
        assert {m["member_id"] for m in result["comm_in"]} == {"ep_a"}
        assert {m["member_id"] for m in result["comm_out"]} == {"ep_b"}


class TestDisabledZeroRegression:
    """AC4 — 关闭 sidecar（enabled=False）→ 完全走边遍历，零回归。"""

    def test_disabled_sidecar_still_edge_traversal(self, overgraph_store, monkeypatch):
        _create_episode(overgraph_store, "ep_a", "A", tau=1.0)
        _persist_community(overgraph_store, "comm_a", ["ep_a"], "report")
        _create_episode(overgraph_store, "ep_b", "B", tau=1.0)
        _persist_community(overgraph_store, "comm_b", ["ep_b"], "report")

        overgraph_store.config.community_members_index_enabled = False
        overgraph_store.reload_community_members_index()

        edge_calls: list[list[str]] = []
        real_edges = overgraph_store._community_members_from_edges

        def spy_edges(community_ids, limit):
            edge_calls.append(list(community_ids))
            return real_edges(community_ids, limit)

        monkeypatch.setattr(overgraph_store, "_community_members_from_edges", spy_edges)
        result = overgraph_store.get_community_members_batch(
            ["comm_a", "comm_b"], 10
        )

        assert edge_calls == [["comm_a", "comm_b"]], "关闭 sidecar 应整批走边遍历"
        assert {m["member_id"] for m in result["comm_a"]} == {"ep_a"}
        assert {m["member_id"] for m in result["comm_b"]} == {"ep_b"}


class TestBackgroundRefresh:
    """AC5 — 缺失占比超阈值触发刷新、冷却期不重复、失败不影响检索。"""

    def test_refresh_triggered_and_cooldown(self, overgraph_store, monkeypatch):
        spawns: list[int] = []
        monkeypatch.setattr(
            overgraph_store,
            "_spawn_community_members_index_refresh",
            lambda: spawns.append(1),
        )
        overgraph_store._community_members_index_last_refresh_ts = None

        # 1/2 = 50% > 5% 阈值 → 触发；冷却期内第二次不触发
        assert overgraph_store._maybe_refresh_community_members_index(1, 2) is True
        assert overgraph_store._maybe_refresh_community_members_index(1, 2) is False
        assert len(spawns) == 1, "冷却期内不应重复触发后台刷新"

    def test_refresh_below_threshold_not_triggered(self, overgraph_store, monkeypatch):
        spawns: list[int] = []
        monkeypatch.setattr(
            overgraph_store,
            "_spawn_community_members_index_refresh",
            lambda: spawns.append(1),
        )
        overgraph_store._community_members_index_last_refresh_ts = None

        # 1/20 = 5%，不超阈值（>5% 才触发）
        assert overgraph_store._maybe_refresh_community_members_index(1, 20) is False
        assert spawns == []

    def test_refresh_failure_does_not_affect_retrieval(
        self, overgraph_store, monkeypatch,
    ):
        def boom() -> bool:
            raise RuntimeError("boom")

        monkeypatch.setattr(overgraph_store, "refresh_community_members_index", boom)
        # worker 失败 fail-soft：不向上抛异常
        overgraph_store._community_members_index_refresh_worker()

        # 检索不受影响（无 sidecar 时走边遍历仍正确）
        _create_episode(overgraph_store, "ep_a", "A", tau=1.0)
        _persist_community(overgraph_store, "comm_a", ["ep_a"], "report")
        result = overgraph_store.get_community_members_batch(["comm_a"], 10)
        assert {m["member_id"] for m in result["comm_a"]} == {"ep_a"}


class TestSpeed:
    """AC6 — sidecar 就绪后批量取成员提速保持（小样本 114 社区，不跑全量构建）。"""

    def test_sidecar_batch_114_communities_fast(self, overgraph_store):
        for i in range(1, 114):
            _create_community(overgraph_store, f"comm_{i}")
        _create_episode(overgraph_store, "ep0", "c0", tau=1.0)
        _persist_community(overgraph_store, "comm_0", ["ep0"], "report")
        assert overgraph_store.refresh_community_members_index() is True

        # 显式预加载 sidecar（对齐「sidecar 就绪后」前提），只计时批量取成员
        overgraph_store._load_community_members_index()
        ids = [f"comm_{i}" for i in range(114)]

        t0 = time.monotonic()
        result = overgraph_store.get_community_members_batch(ids, 10)
        elapsed = time.monotonic() - t0

        assert len(result) == 114
        assert {m["member_id"] for m in result["comm_0"]} == {"ep0"}
        assert result["comm_1"] == []
        assert elapsed < 0.1, f"sidecar 路径 114 社区耗时 {elapsed * 1000:.1f}ms ≥ 100ms"


class TestBudgetS:
    """community_expansion.budget_s 三处（代码默认 / settings / yaml）均为 2.0。"""

    def test_budget_s_is_2_0(self):
        import yaml

        from config.settings import (
            CommunityExpansionConfig,
            _get_defaults_path,
            get_settings,
        )

        assert CommunityExpansionConfig().budget_s == 2.0
        assert get_settings().retrieval.community_expansion.budget_s == 2.0

        raw = yaml.safe_load(_get_defaults_path().read_text(encoding="utf-8"))
        assert raw["retrieval"]["community_expansion"]["budget_s"] == 2.0
