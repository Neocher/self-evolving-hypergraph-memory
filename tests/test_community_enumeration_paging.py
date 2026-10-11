# -*- coding: utf-8 -*-
"""社区 id 枚举 keyset 游标分页（v6.34）验收测试。

对齐任务书 AC1..AC4：
  AC1 枚举完整：构建后 sidecar 键数 == 全部 CommunityNode 数（含空社区条目），
      与 `MATCH (c:CommunityNode) RETURN count(*)` 一致。
  AC2 空社区被收录：值为空列表的条目数 == 社区总数 − 有成员社区数。
  AC3 分页正确性：mock 构造 >10,000 个社区 → keyset 分页无遗漏、无重复（集合
      去重后计数相等），游标严格 `>` 推进，满页末页仍多查一次确认扫尽。
  AC4 fail loud 保留：游标无法推进 / 页数超安全上限 → 抛 OverGraphError 且
      不写盘（输出文件未生成 / 未被覆盖）。

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_community_enumeration_paging.py -q
"""
from __future__ import annotations

import json

import pytest

from core.dream_pipeline import DreamPipeline
from graph.overgraph_store import (
    _MAX_QUERY_ROWS,
    OverGraphError,
    _atomic_write_json,
)

LABEL_COMMUNITY = "CommunityNode"


def _create_episode(store, eid: str, content: str, tau: float = 1.0) -> None:
    """创建 EpisodeNode（archived=False 显式落库）。"""
    store.create_episode({
        "id": eid, "content": content, "tau_initial": tau,
        "fact_track": "active", "archived": False,
    })


def _create_community(store, cid: str) -> None:
    """直接 upsert 一个 CommunityNode（无成员边，空社区）。"""
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


class TestEnumerationCoverage:
    """AC1/AC2 — keyset 分页枚举完整、空社区被收录为空列表。"""

    def test_build_index_keys_equal_community_count(self, overgraph_store):
        _create_episode(overgraph_store, "ep0", "c0", tau=1.0)
        _persist_community(overgraph_store, "comm_a", ["ep0"], "ra")
        _persist_community(overgraph_store, "comm_b", ["ep0"], "rb")
        _create_community(overgraph_store, "comm_empty_1")
        _create_community(overgraph_store, "comm_empty_2")

        edge_total, index = overgraph_store.build_community_members_index(page_size=2)

        assert edge_total == 2, f"edge_total={edge_total}"
        assert len(index) == _community_count(overgraph_store), (
            f"sidecar 键数 {len(index)} != 全部社区数 {_community_count(overgraph_store)}"
        )
        assert set(index.keys()) == {
            "comm_a", "comm_b", "comm_empty_1", "comm_empty_2",
        }

    def test_empty_communities_recorded_as_empty_lists(self, overgraph_store):
        _create_episode(overgraph_store, "ep0", "c0", tau=1.0)
        _persist_community(overgraph_store, "comm_with_members", ["ep0"], "ra")
        _create_community(overgraph_store, "comm_empty_1")
        _create_community(overgraph_store, "comm_empty_2")
        _create_community(overgraph_store, "comm_empty_3")

        _, index = overgraph_store.build_community_members_index(page_size=4)

        empty = {k for k, v in index.items() if v == []}
        with_members = {k for k, v in index.items() if v}
        assert empty == {"comm_empty_1", "comm_empty_2", "comm_empty_3"}
        assert with_members == {"comm_with_members"}
        assert len(empty) == len(index) - len(with_members)


class TestKeysetPagination:
    """AC3 — mock 构造 >10,000 社区，分页无遗漏、无重复、游标推进。"""

    def test_collects_all_pages_without_miss_or_dup(
        self, overgraph_store, monkeypatch,
    ):
        total = 25_001
        all_ids = sorted(f"comm_{i:06d}" for i in range(total))
        calls: list[tuple[str, dict | None]] = []

        def fake(gql, params=None):
            calls.append((gql, params))
            cur = (params or {}).get("cur", "")
            subset = [i for i in all_ids if not cur or i > cur]
            return {"rows": [{"id": i} for i in subset[:_MAX_QUERY_ROWS]]}

        monkeypatch.setattr(overgraph_store, "_locked_execute_gql", fake)

        got = overgraph_store._enumerate_community_ids()

        assert got == all_ids, "分页必须无遗漏、无重复、顺序稳定"
        assert len(got) == len(set(got)) == total
        assert len(calls) == 3, f"应 3 页（ceil(25001/10000)），实际 {len(calls)} 页"
        # 游标严格 `>` 推进：第 2/3 页携带上一页最后一个 id
        assert calls[1][1] == {"cur": all_ids[_MAX_QUERY_ROWS - 1]}
        assert calls[2][1] == {"cur": all_ids[2 * _MAX_QUERY_ROWS - 1]}

    def test_full_last_page_still_probes_next(self, overgraph_store, monkeypatch):
        """恰好 10,000 个社区：满页也是末页 → 仍多查一次确认扫尽（不重不漏）。"""
        all_ids = sorted(f"comm_{i:05d}" for i in range(_MAX_QUERY_ROWS))
        calls: list[tuple[str, dict | None]] = []

        def fake(gql, params=None):
            calls.append((gql, params))
            cur = (params or {}).get("cur", "")
            subset = [i for i in all_ids if not cur or i > cur]
            return {"rows": [{"id": i} for i in subset[:_MAX_QUERY_ROWS]]}

        monkeypatch.setattr(overgraph_store, "_locked_execute_gql", fake)

        got = overgraph_store._enumerate_community_ids()

        assert got == all_ids
        assert len(calls) == 2, "满页末页后须再查一次空页确认扫尽"
        assert calls[1][1] == {"cur": all_ids[-1]}


class TestFailLoud:
    """AC4 — 游标无法推进 / 页数超上限 → 抛错且不写盘。"""

    def test_cursor_stuck_raises(self, overgraph_store, monkeypatch):
        def fake(gql, params=None):
            # 引擎异常：每页都返回满页且最后一个 id 恒为 "same"（游标不推进）
            return {"rows": [{"id": "same"}] * _MAX_QUERY_ROWS}

        monkeypatch.setattr(overgraph_store, "_locked_execute_gql", fake)
        with pytest.raises(OverGraphError, match="did not advance"):
            overgraph_store._enumerate_community_ids()

    def test_page_cap_exceeded_raises(self, overgraph_store, monkeypatch):
        monkeypatch.setattr(
            "graph.overgraph_store._COMMUNITY_ENUM_MAX_PAGES", 3
        )
        counter = iter(range(100))

        def fake(gql, params=None):
            return {"rows": [{"id": f"comm_{next(counter):06d}"}]}

        monkeypatch.setattr(overgraph_store, "_locked_execute_gql", fake)
        with pytest.raises(OverGraphError, match="exceeded"):
            overgraph_store._enumerate_community_ids(limit=1)

    def test_enumeration_rows_over_limit_raises(self, overgraph_store, monkeypatch):
        def fake(gql, params=None):
            return {"rows": [{"id": "x"}] * (_MAX_QUERY_ROWS + 1)}

        monkeypatch.setattr(overgraph_store, "_locked_execute_gql", fake)
        with pytest.raises(OverGraphError, match="page limit"):
            overgraph_store._enumerate_community_ids()

    def test_failed_refresh_does_not_overwrite(self, overgraph_store, monkeypatch):
        """构建抛错 → refresh 返回 False 且不覆盖已有 sidecar 文件。"""

        def boom(page_size=15):
            raise OverGraphError("enumeration boom")

        monkeypatch.setattr(overgraph_store, "build_community_members_index", boom)
        out = overgraph_store._community_members_index_path()
        _atomic_write_json(out, {"sentinel": True})

        assert overgraph_store.refresh_community_members_index() is False
        assert json.load(open(out, encoding="utf-8")) == {"sentinel": True}, (
            "构建失败却覆盖了已有 sidecar"
        )
