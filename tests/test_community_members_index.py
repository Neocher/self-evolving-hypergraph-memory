# -*- coding: utf-8 -*-
"""社区成员 sidecar 索引（v6.31 检索超时修复 Part 4）验收测试。

覆盖（对齐任务书 AC1..AC7）：
  AC1 构建可用：build_community_members_index 目标侧分批产出正确 edge_total /
       community_total；单批触达 10,000 行上限自动折半重试（降批在日志可见）。
  AC3 语义等价：sidecar 路径与边遍历路径逐社区、逐成员、逐字段、逐顺序完全一致，
      含 archived 为 None 的真实样本（缺陷A）。
  AC4 顺序等价：tau 相同的成员按 member_id 升序的确定性 tie-break（缺陷B）。
  AC5 未命中回落：sidecar 中不存在的社区 id → 整次回落原边遍历且结果正确。
  AC6 缺失零回归：无 sidecar 文件 / 删除 sidecar 文件 / 开关关闭 → 与无该功能
      完全一致，不抛异常。
  AC7 提速（代理断言）：sidecar 命中时不再触发边遍历 GQL（结构性慢路径），改走
      get_episodes_batch（真实耗时提速在真实 store 上端到端验证）。
  AC10 RED→GREEN：本文件即验证目标。

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_community_members_index.py -q
"""
from __future__ import annotations

import json
import logging
import os

import pytest

from core.dream_pipeline import DreamPipeline
from graph.overgraph_store import OverGraphError, _atomic_write_json

LABEL_EPISODE = "EpisodeNode"


def _create_episode(store, eid: str, content: str, tau: float = 1.0) -> None:
    """创建 EpisodeNode（archived=False 显式落库）。"""
    store.create_episode({
        "id": eid, "content": content, "tau_initial": tau,
        "fact_track": "active", "archived": False,
    })


def _create_episode_no_archived(store, eid: str, content: str, tau: float = 1.0) -> None:
    """创建无 archived 属性的 EpisodeNode（模拟旧数据 → 边遍历 archived=None）。"""
    store._locked_upsert_node(LABEL_EPISODE, eid, {
        "id": eid, "content": content, "tau_initial": tau,
        "fact_track": "active",
    })


def _persist_community(store, cid: str, members: list[str], report: str) -> None:
    """用生产 _persist_one_community 造真实社区边（社区→成员）。"""
    DreamPipeline()._persist_one_community(
        store, {"id": cid, "members": members, "report": report}, "eval", 0
    )


class TestBuildIndex:
    """AC1 — 目标侧分批构建产出正确索引。"""

    def test_build_produces_edge_and_community_totals(self, overgraph_store):
        _create_episode(overgraph_store, "ep0", "c0", tau=1.0)
        _create_episode(overgraph_store, "ep1", "c1", tau=1.0)
        _create_episode(overgraph_store, "ep2", "c2", tau=1.0)
        _persist_community(overgraph_store, "comm_a", ["ep0", "ep1"], "ra")
        _persist_community(overgraph_store, "comm_b", ["ep1", "ep2"], "rb")

        edge_total, index = overgraph_store.build_community_members_index(page_size=2)

        assert edge_total == 4, f"edge_total={edge_total}"
        assert set(index.keys()) == {"comm_a", "comm_b"}
        assert set(index["comm_a"]) == {"ep0", "ep1"}
        assert set(index["comm_b"]) == {"ep1", "ep2"}

    def test_build_shrinks_on_truncation(self, overgraph_store, monkeypatch, caplog):
        """单批触达 10,000 行 → 自动折半重试（最小粒度 1 个 id），日志可见。"""
        for i in range(4):
            _create_episode(overgraph_store, f"ep{i}", f"c{i}", tau=1.0)

        real = overgraph_store._locked_execute_gql
        scan_batches: list[list[str]] = []

        def fake(gql, params=None):
            if "COMMUNITY_MEMBER" in gql:
                ids = list((params or {}).get("batch_ids", []))
                scan_batches.append(ids)
                if len(ids) >= 2:
                    # 模拟引擎静默截断：恰好回满 10,000 行
                    return {"rows": [{"cid": "comm_a", "eid": ids[0]}] * 10000}
                return {"rows": [{"cid": "comm_a", "eid": ids[0]}]}
            return real(gql, params)

        monkeypatch.setattr(overgraph_store, "_locked_execute_gql", fake)
        with caplog.at_level(logging.WARNING):
            edge_total, index = overgraph_store.build_community_members_index(
                page_size=4
            )

        assert edge_total == 4, "降批后仍须扫尽 4 条边"
        assert set(index.get("comm_a", [])) == {"ep0", "ep1", "ep2", "ep3"}
        assert sum(len(b) for b in scan_batches) >= 4
        assert any("shrinking" in r.message for r in caplog.records), (
            "降批行为未在日志可见"
        )

    def test_build_fails_loud_when_single_id_truncated(
        self, overgraph_store, monkeypatch,
    ):
        """单 id 仍触顶 → 抛 OverGraphError（绝不接受截断写出残缺索引）。"""
        _create_episode(overgraph_store, "ep0", "c0", tau=1.0)

        real = overgraph_store._locked_execute_gql

        def fake(gql, params=None):
            if "COMMUNITY_MEMBER" in gql:
                return {"rows": [{"cid": "comm_a", "eid": "ep0"}] * 10000}
            return real(gql, params)

        monkeypatch.setattr(overgraph_store, "_locked_execute_gql", fake)
        with pytest.raises(OverGraphError):
            overgraph_store.build_community_members_index(page_size=1)


class TestCommunityMembersSidecar:
    """AC3/AC4/AC5/AC6 — sidecar 读取层语义等价与回落。"""

    def test_sidecar_matches_edges_field_by_field(self, overgraph_store):
        """逐社区、逐成员、逐字段、逐顺序一致，含 archived=None 与 tau tie-break。"""
        _create_episode(overgraph_store, "ep_a", "A 内容", tau=0.5)          # archived False
        _create_episode(overgraph_store, "ep_b", "B 内容", tau=0.9)
        _create_episode_no_archived(overgraph_store, "ep_c", "C 内容", tau=0.9)  # archived None
        _persist_community(overgraph_store, "comm_x", ["ep_a", "ep_b", "ep_c"],
                           "K8s 集群网络排障")

        edge_result = overgraph_store._community_members_from_edges(["comm_x"], 10)
        assert overgraph_store.refresh_community_members_index() is True
        index = overgraph_store._load_community_members_index()
        sidecar_result = overgraph_store._community_members_from_sidecar(
            ["comm_x"], index, 10
        )

        assert list(edge_result.keys()) == list(sidecar_result.keys())
        a, b = edge_result["comm_x"], sidecar_result["comm_x"]
        assert [m["member_id"] for m in a] == [m["member_id"] for m in b]
        for m1, m2 in zip(a, b):
            assert m1 == m2  # 逐字段
            assert list(m1.keys()) == list(m2.keys())

        # 顺序：tau DESC + tau 相同按 member_id 升序（0.9 的 ep_b/ep_c 在前）
        assert [m["member_id"] for m in a] == ["ep_b", "ep_c", "ep_a"]
        by_id = {m["member_id"]: m for m in a}
        assert by_id["ep_a"]["archived"] is False
        assert by_id["ep_c"]["archived"] is None, "缺陷A：archived 缺失应为 None"
        assert by_id["ep_b"]["content"] == "B 内容"
        assert by_id["ep_b"]["fact_track"] == "active"
        assert by_id["ep_b"]["tau_value"] == 0.9

    def test_sidecar_path_avoids_edge_scan(self, overgraph_store, monkeypatch):
        """AC7 代理：sidecar 命中时不触发边遍历 GQL（结构性慢路径）。"""
        _create_episode(overgraph_store, "ep_a", "A", tau=1.0)
        _persist_community(overgraph_store, "comm_in", ["ep_a"], "report")
        assert overgraph_store.refresh_community_members_index() is True

        edge_calls: list[list[str]] = []
        real_edges = overgraph_store._community_members_from_edges

        def spy(community_ids, limit):
            edge_calls.append(list(community_ids))
            return real_edges(community_ids, limit)

        monkeypatch.setattr(overgraph_store, "_community_members_from_edges", spy)
        result = overgraph_store.get_community_members_batch(["comm_in"], 10)
        assert {m["member_id"] for m in result["comm_in"]} == {"ep_a"}
        assert edge_calls == [], "sidecar 命中却仍走了边遍历路径"

    def test_missing_community_falls_back_to_edges(self, overgraph_store):
        """AC5：sidecar 中不存在的社区 id → 整次回落原边遍历且结果正确。"""
        _create_episode(overgraph_store, "ep_a", "A", tau=1.0)
        _persist_community(overgraph_store, "comm_in", ["ep_a"], "report")
        assert overgraph_store.refresh_community_members_index() is True

        _create_episode(overgraph_store, "ep_b", "B", tau=1.0)
        _persist_community(overgraph_store, "comm_out", ["ep_b"], "report")

        result = overgraph_store.get_community_members_batch(
            ["comm_in", "comm_out"], 10
        )
        assert {m["member_id"] for m in result["comm_in"]} == {"ep_a"}
        assert {m["member_id"] for m in result["comm_out"]} == {"ep_b"}

    def test_missing_sidecar_file_is_equivalent_to_disabled(self, overgraph_store):
        """AC6：无 sidecar 文件 → 走原边遍历，结果正确、不抛异常。"""
        _create_episode(overgraph_store, "ep_a", "A", tau=1.0)
        _persist_community(overgraph_store, "comm_a", ["ep_a"], "report")

        result = overgraph_store.get_community_members_batch(["comm_a"], 10)
        assert {m["member_id"] for m in result["comm_a"]} == {"ep_a"}
        assert overgraph_store._load_community_members_index() is None

    def test_deleted_sidecar_file_zero_regression(self, overgraph_store):
        """AC6：构建后删除 sidecar 文件 → 行为与无该功能一致，不抛异常。"""
        _create_episode(overgraph_store, "ep_a", "A", tau=1.0)
        _persist_community(overgraph_store, "comm_a", ["ep_a"], "report")
        assert overgraph_store.refresh_community_members_index() is True

        os.remove(overgraph_store._community_members_index_path())
        overgraph_store.reload_community_members_index()

        result = overgraph_store.get_community_members_batch(["comm_a"], 10)
        assert {m["member_id"] for m in result["comm_a"]} == {"ep_a"}

    def test_disabled_switch_ignores_sidecar(self, overgraph_store):
        """开关关闭 → sidecar 视为未加载，完全走原边遍历。"""
        _create_episode(overgraph_store, "ep_a", "A", tau=1.0)
        _persist_community(overgraph_store, "comm_a", ["ep_a"], "report")
        assert overgraph_store.refresh_community_members_index() is True

        overgraph_store.config.community_members_index_enabled = False
        overgraph_store.reload_community_members_index()
        assert overgraph_store._load_community_members_index() is None


class TestAtomicWrite:
    """_atomic_write_json 原子写与替换。"""

    def test_write_and_replace(self, tmp_path):
        path = str(tmp_path / "idx.json")
        _atomic_write_json(path, {"a": ["1"], "_meta": {"edge_total": 1}})
        assert json.load(open(path, encoding="utf-8"))["a"] == ["1"]

        _atomic_write_json(path, {"b": ["2"]})
        assert json.load(open(path, encoding="utf-8")) == {"b": ["2"]}
