"""
梦境管道 persist 同源湮灭修复测试
=================================
验证 _persist_communities Phase 3 修复：
  1. 共享成员只保留最大社区边（不再互删成孤儿）
  2. Phase 3 仅作用于指定社区，不波及外部社区
"""
from __future__ import annotations

import time
import uuid

from core.dream_pipeline import DreamPipeline
from core.tau_decay import TauDecayEngine


def _create_episode(overgraph_store, ep_id: str, content: str = "test content") -> None:
    """在 GraphLite 中创建一个 EpisodeNode。"""
    overgraph_store.create_episode({
        "id": ep_id,
        "content": content,
        "created_at": time.time(),
        "tau_initial": 1.0,
        "tau_value": 0.6,
        "source": "test",
        "trust_score": 0.8,
    })


def _make_community(
    cid: str, members: list[str], report: str = "test report"
) -> dict:
    """构造 _persist_communities 期望的 community dict 格式。"""
    return {
        "id": cid,
        "members": members,
        "report": report,
    }


def _insert_community(overgraph_store, cid: str, name: str = "test_comm") -> None:
    """在 GraphLite 中创建一个 CommunityNode。"""
    overgraph_store.execute_cypher(
        "INSERT (c:CommunityNode {id: $id, name: $name, "
        "summary: $summary, leiden_score: $score, "
        "created_at: $created_at})",
        {
            "id": cid,
            "name": name,
            "summary": "test community",
            "score": 0.5,
            "created_at": time.time(),
        },
    )


def _insert_member_edge(overgraph_store, cid: str, eid: str) -> None:
    """在 GraphLite 中创建 COMMUNITY_MEMBER 边。"""
    overgraph_store.execute_cypher(
        "MATCH (c:CommunityNode {id: $cid}), "
        "(e:EpisodeNode {id: $eid}) "
        "INSERT (c)-[:COMMUNITY_MEMBER]->(e)",
        {"cid": cid, "eid": eid},
    )


def _edge_exists(overgraph_store, cid: str, eid: str) -> bool:
    """检查 COMMUNITY_MEMBER 边是否存在。"""
    result = overgraph_store.execute_cypher(
        "MATCH (c:CommunityNode {id: $cid})"
        "-[:COMMUNITY_MEMBER]->"
        "(e:EpisodeNode {id: $eid}) RETURN e",
        {"cid": cid, "eid": eid},
    )
    return bool(result)


class TestPipelinePersistSharedMember:
    """_persist_communities Phase 3 同源湮灭修复验证。"""

    def test_shared_member_keeps_largest_community(self, overgraph_store):
        """双社区共享成员 → persist 后成员只属最大社区（不湮灭）。

        构造 C1(3 members, [epX, epA, epB]) + C2(2 members, [epX, epC])。
        epX 共享 → persist 后只属于 C1（member_count 更大）。
        修复前：Phase 3 WHERE c.id <> $cid 使 C1/C2 互删 epX 边 → 孤儿。
        """
        pipe = DreamPipeline()
        c1_id = f"c1-{uuid.uuid4().hex[:8]}"
        c2_id = f"c2-{uuid.uuid4().hex[:8]}"
        epX = f"epX-{uuid.uuid4().hex[:8]}"
        epA = f"epA-{uuid.uuid4().hex[:8]}"
        epB = f"epB-{uuid.uuid4().hex[:8]}"
        epC = f"epC-{uuid.uuid4().hex[:8]}"

        for ep_id in [epX, epA, epB, epC]:
            _create_episode(overgraph_store, ep_id, f"content of {ep_id}")

        communities = [
            _make_community(c1_id, [epX, epA, epB], "C1: larger (3 members)"),
            _make_community(c2_id, [epX, epC], "C2: smaller (2 members)"),
        ]

        created = pipe._persist_communities(
            overgraph_store, communities, f"dream-{uuid.uuid4().hex[:8]}"
        )
        assert created == 2

        # 回归断言：epX 只属 C1（最大社区），不属 C2
        assert _edge_exists(overgraph_store, c1_id, epX), (
            f"epX should belong to larger C1, but edge missing → orphaned"
        )
        assert not _edge_exists(overgraph_store, c2_id, epX), (
            f"epX should NOT belong to smaller C2, but stale edge exists"
        )

        # 独有成员边不受影响
        assert _edge_exists(overgraph_store, c1_id, epA), (
            f"C1 exclusive member epA edge missing"
        )
        assert _edge_exists(overgraph_store, c1_id, epB), (
            f"C1 exclusive member epB edge missing"
        )
        assert _edge_exists(overgraph_store, c2_id, epC), (
            f"C2 exclusive member epC edge missing"
        )

    def test_preserves_external_edges(self, overgraph_store):
        """外部社区边不被 Phase 1 波及（共享成员场景）。

        构造：
        - 外部社区 EXT 与 dream 社区共享成员 epX（EXT→epX 边）
        - dream 社区 C1(=[epX, epY]) + C2(=[epX, epZ])

        验证：
        - EXT→epX 边存活（Phase 1 修复前：MATCH (c:CommunityNode) 无社区过滤
          → 删除所有社区到 epX 的边，含 EXT→epX；修复后按 {id: $cid} 限定）
        - C1（最大）保留 epX 边，C2 失去 epX 边（Phase 3 正确清理）
        """
        pipe = DreamPipeline()
        ext_cid = f"ext-{uuid.uuid4().hex[:8]}"
        c1_id = f"c1-{uuid.uuid4().hex[:8]}"
        c2_id = f"c2-{uuid.uuid4().hex[:8]}"
        epX = f"epX-{uuid.uuid4().hex[:8]}"
        epY = f"epY-{uuid.uuid4().hex[:8]}"
        epZ = f"epZ-{uuid.uuid4().hex[:8]}"

        # 创建所有 EpisodeNode
        for ep_id in [epX, epY, epZ]:
            _create_episode(overgraph_store, ep_id, f"content of {ep_id}")

        # 预置外部社区 + 边到共享成员 epX（epX 同时属于 EXT + dream C1/C2）
        _insert_community(overgraph_store, ext_cid, "external_community")
        _insert_member_edge(overgraph_store, ext_cid, epX)

        # dream 社区：C1 含 epX+epY，C2 含 epX+epZ（epX 共享）
        communities = [
            _make_community(c1_id, [epX, epY], "C1: 2 members → larger"),
            _make_community(c2_id, [epX, epZ], "C2: 2 members → equal size"),
        ]

        created = pipe._persist_communities(
            overgraph_store, communities, f"dream-{uuid.uuid4().hex[:8]}"
        )
        assert created == 2

        # F2 核心断言：外部社区到共享成员 epX 的边存活
        # （修复前 Phase 1 MATCH (c:CommunityNode) 无社区过滤 → 必挂）
        assert _edge_exists(overgraph_store, ext_cid, epX), (
            f"F2 FAIL: EXT→epX edge was deleted — "
            f"Phase 1 MATCH lacks community filter ({ext_cid}→{epX})"
        )

        # epX 属于先出现的 C1（等大社区取先出现——sorted by member_count，
        # 相同则按 new_member_sets 插入序；C1 先加入 communities 列表）
        assert _edge_exists(overgraph_store, c1_id, epX), (
            f"epX should belong to C1 (first in communities list), but edge missing"
        )
        assert not _edge_exists(overgraph_store, c2_id, epX), (
            f"epX should NOT belong to C2, but stale edge exists"
        )

        # 独有成员边正常
        assert _edge_exists(overgraph_store, c1_id, epY), (
            f"C1 exclusive member epY edge missing"
        )
        assert _edge_exists(overgraph_store, c2_id, epZ), (
            f"C2 exclusive member epZ edge missing"
        )


# ══════════════════════════════════════════════════════════════════
# v5.27.0 梦境 PRUNE 保护修复（2026-08-12 事故：9 条全孤立旧节点 100% 被剪）
#  ① force_promote 节点打 protected 标记 → PRUNE 永不剪
#  ② 单次剪枝比例 > 50% → 中止本次剪枝（全部保留）
#  ③ protected 节点永不参与合并（防合并击穿）
# ══════════════════════════════════════════════════════════════════


def _make_old_node(eid, protected=False, tau=0.05, created_at=None):
    """构造命中全部剪枝候选条件的旧节点（除 protected 外）。

    - created_at 默认 7201s 前（≥ 7200s 低龄保护线）
    - tau 默认 0.05（< decay_threshold 0.1，且 ≤ 0.3 高 τ 保护线）
    - connections={} → degree=0 ≤ 1
    注意：_prune_step 直接读 tau_value，因此 tau_initial 与 tau_value 都置为 tau。
    """
    if created_at is None:
        created_at = time.time() - 7201
    node = {
        "id": eid,
        "content": f"content-{eid}",
        "created_at": created_at,
        "tau_initial": tau,
        "tau_value": tau,
    }
    if protected:
        node["protected"] = True  # 仅 protected=true 时携带标记（兼容旧数据无标记）
    return node


class TestPruneProtection:
    """v5.27.0 PRUNE 保护：protected 标记 + 批量比例护栏 + 合并防护。"""

    def _pipe(self):
        return DreamPipeline(tau_engine=TauDecayEngine())

    def test_prune_keeps_force_promote_node(self):
        """方案①：protected 节点 τ 衰减后仍不被剪，普通节点同条件照常被剪。

        且比例 1/2 = 50% 不触发中止 → 验证两方案正交。
        """
        pipe = self._pipe()
        nodes = [
            _make_old_node("prot", protected=True),
            _make_old_node("normal"),
        ]
        keep, _, pruned, ops = pipe._prune_step(nodes, {})
        keep_ids = {n["id"] for n in keep}
        assert "normal" not in keep_ids      # 普通节点照常被剪（方案①不改变普通剪枝）
        assert "prot" in keep_ids            # protected 保留（方案①）
        assert pruned == 1
        assert len(ops) == 1

    def test_prune_aborts_when_ratio_over_50pct(self):
        """方案②：事故复现——9 条全孤立旧节点，9/9 > 50% → 中止，全部保留。"""
        pipe = self._pipe()
        nodes = [_make_old_node(f"n{i}") for i in range(9)]
        keep, conns, pruned, ops = pipe._prune_step(nodes, {})
        assert pruned == 0
        assert {n["id"] for n in keep} == {f"n{i}" for i in range(9)}  # 原 nodes 原样返回
        assert ops == []

    def test_prune_normal_ratio_unchanged(self):
        """方案②：正常比例（1/10 = 10% ≤ 50%）剪枝不受影响。"""
        pipe = self._pipe()
        now = time.time()
        nodes = [_make_old_node("old")] + [
            {"id": f"fresh{i}", "content": f"c{i}",
             "created_at": now, "tau_initial": 1.0, "tau_value": 1.0}
            for i in range(9)
        ]
        keep, _, pruned, ops = pipe._prune_step(nodes, {})
        assert pruned == 1
        assert "old" not in {n["id"] for n in keep}
        assert len(ops) == 1

    def test_prune_mixed_protected_and_old_nodes(self):
        """方案①+② 联动：事故场景混合。

        - 8 候选 + 1 protected → 8/9 > 50% → 中止，全保留（含 protected）
        - 5 候选 + 5 protected → 5/10 = 50% 恰好放行（> 严格大于）→ 正常剪 5 候选
        """
        pipe = self._pipe()
        nodes = [_make_old_node(f"old{i}") for i in range(8)] + \
                [_make_old_node("prot", protected=True)]
        keep, _, pruned, ops = pipe._prune_step(nodes, {})
        assert pruned == 0
        assert "prot" in {n["id"] for n in keep}

        nodes2 = [_make_old_node(f"old{i}") for i in range(5)] + \
                 [_make_old_node(f"prot{i}", protected=True) for i in range(5)]
        keep2, _, pruned2, _ = pipe._prune_step(nodes2, {})
        assert pruned2 == 5
        assert {n["id"] for n in keep2} == {f"prot{i}" for i in range(5)}

    def test_protected_node_never_merged_away(self):
        """方案③：protected 节点永不作为合并方/被合并方（防合并击穿）。

        protected 节点 τ 更低（0.05）、普通节点 τ 更高（1.0）、内容相同
        （Jaccard=1.0 ≥ 0.8）→ 修复前 protected 作 loser 被 DETACH DELETE；
        修复后两节点均保留、无合并操作。
        """
        pipe = self._pipe()
        nodes = [
            {"id": "protected-low-tau", "content": "相同记忆内容",
             "created_at": time.time() - 7201, "tau_value": 0.05, "protected": True},
            {"id": "normal-high-tau", "content": "相同记忆内容",
             "created_at": time.time() - 7201, "tau_value": 1.0},
        ]
        remaining, ops = pipe._find_and_merge_conflicts(nodes)
        assert ops == []                                      # 不合并
        assert {n["id"] for n in remaining} == \
            {"protected-low-tau", "normal-high-tau"}          # 两节点都保留
        assert "protected-low-tau" in {n["id"] for n in remaining}

    def test_normal_nodes_still_merge(self):
        """方案③：普通节点对不受影响，仍按 τ 高低正常合并（回归）。"""
        pipe = self._pipe()
        nodes = [
            {"id": "loser", "content": "相同记忆内容",
             "created_at": time.time() - 7201, "tau_value": 0.05},
            {"id": "winner", "content": "相同记忆内容",
             "created_at": time.time() - 7201, "tau_value": 1.0},
        ]
        remaining, ops = pipe._find_and_merge_conflicts(nodes)
        assert len(ops) == 1
        assert {n["id"] for n in remaining} == {"winner"}
