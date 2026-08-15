"""
梦境聚类分区合并 bug 回归测试（v5.36.0 Cluster-Fix）
=====================================================
根因: _cluster_step 合并连通分量分区时忽略 _detect_communities 返回的 cid，
每节点独立成社区（生产 896 节点 = 896 社区）→ 社区摘要永不触发。

修复: partition[nid] = next_comm + cid 保留社区归属，跨分量偏移防冲突。

覆盖:
  · mock 分区 {a:0,b:0,c:1} → 2 社区（a+b 一组，c 单独）— 旧代码 3 社区必挂
  · 跨分量偏移: {a:0,b:0} + {c:0,d:0} → 仍 2 社区（cid 不冲突）
  · singleton 混合: {a:0,b:0} + 孤立 c → 2 社区（a+b, c）
"""
from __future__ import annotations

from unittest.mock import patch

from core.dream_pipeline import DreamPipeline


def _node(nid: str) -> dict:
    return {"id": nid, "content": f"content of {nid}"}


def _community_member_sets(communities: list[dict]) -> set[frozenset]:
    return {frozenset(c["members"]) for c in communities}


class TestClusterPartitionMerge:
    def test_mock_partition_two_communities(self):
        """mock _detect_communities 返回 {a:0,b:0,c:1} → 2 社区（a+b, c）。

        旧代码忽略 cid 逐节点分配 → 3 社区，本用例必挂。
        """
        pipe = DreamPipeline()
        nodes = [_node("a"), _node("b"), _node("c")]
        connections = {"a": {"b": 1.0}, "b": {"c": 1.0}}  # 单连通分量 a-b-c

        with patch.object(
            pipe, "_detect_communities", return_value={"a": 0, "b": 0, "c": 1}
        ) as mock_detect:
            communities = pipe._cluster_step(nodes, connections)

        mock_detect.assert_called_once()
        assert _community_member_sets(communities) == {
            frozenset({"a", "b"}),
            frozenset({"c"}),
        }

    def test_cross_component_cid_offset(self):
        """两个连通分量各返回 {x:0,y:0} → 仍 2 社区，cid 不冲突。

        _detect_communities 的 cid 在各分量内独立从 0 编号，
        合并时必须用 next_comm 偏移（next_comm + cid）。
        """

        def fake_detect(sub):
            nodes = set(sub.nodes)
            if nodes == {"a", "b"}:
                return {"a": 0, "b": 0}
            if nodes == {"c", "d"}:
                return {"c": 0, "d": 0}
            raise AssertionError(f"unexpected subgraph nodes: {nodes}")

        pipe = DreamPipeline()
        nodes = [_node("a"), _node("b"), _node("c"), _node("d")]
        connections = {"a": {"b": 1.0}, "c": {"d": 1.0}}  # 两个独立分量

        with patch.object(pipe, "_detect_communities", side_effect=fake_detect):
            communities = pipe._cluster_step(nodes, connections)

        assert _community_member_sets(communities) == {
            frozenset({"a", "b"}),
            frozenset({"c", "d"}),
        }

    def test_singleton_mixed_with_component(self):
        """{a:0,b:0} 分量 + 孤立节点 c → 2 社区（a+b, c）。"""

        pipe = DreamPipeline()
        nodes = [_node("a"), _node("b"), _node("c")]
        connections = {"a": {"b": 1.0}}  # c 无连接 → 单例分量

        with patch.object(
            pipe, "_detect_communities", return_value={"a": 0, "b": 0}
        ) as mock_detect:
            communities = pipe._cluster_step(nodes, connections)

        mock_detect.assert_called_once()
        assert _community_member_sets(communities) == {
            frozenset({"a", "b"}),
            frozenset({"c"}),
        }
