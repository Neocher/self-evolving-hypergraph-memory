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


def _create_episode(graphlite_store, ep_id: str, content: str = "test content") -> None:
    """在 GraphLite 中创建一个 EpisodeNode。"""
    graphlite_store.create_episode({
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


def _insert_community(graphlite_store, cid: str, name: str = "test_comm") -> None:
    """在 GraphLite 中创建一个 CommunityNode。"""
    graphlite_store.execute_cypher(
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


def _insert_member_edge(graphlite_store, cid: str, eid: str) -> None:
    """在 GraphLite 中创建 COMMUNITY_MEMBER 边。"""
    graphlite_store.execute_cypher(
        "MATCH (c:CommunityNode {id: $cid}), "
        "(e:EpisodeNode {id: $eid}) "
        "INSERT (c)-[:COMMUNITY_MEMBER]->(e)",
        {"cid": cid, "eid": eid},
    )


def _edge_exists(graphlite_store, cid: str, eid: str) -> bool:
    """检查 COMMUNITY_MEMBER 边是否存在。"""
    result = graphlite_store.execute_cypher(
        "MATCH (c:CommunityNode {id: $cid})"
        "-[:COMMUNITY_MEMBER]->"
        "(e:EpisodeNode {id: $eid}) RETURN e",
        {"cid": cid, "eid": eid},
    )
    return bool(result)


class TestPipelinePersistSharedMember:
    """_persist_communities Phase 3 同源湮灭修复验证。"""

    def test_shared_member_keeps_largest_community(self, graphlite_store):
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
            _create_episode(graphlite_store, ep_id, f"content of {ep_id}")

        communities = [
            _make_community(c1_id, [epX, epA, epB], "C1: larger (3 members)"),
            _make_community(c2_id, [epX, epC], "C2: smaller (2 members)"),
        ]

        created = pipe._persist_communities(
            graphlite_store, communities, f"dream-{uuid.uuid4().hex[:8]}"
        )
        assert created == 2

        # 回归断言：epX 只属 C1（最大社区），不属 C2
        assert _edge_exists(graphlite_store, c1_id, epX), (
            f"epX should belong to larger C1, but edge missing → orphaned"
        )
        assert not _edge_exists(graphlite_store, c2_id, epX), (
            f"epX should NOT belong to smaller C2, but stale edge exists"
        )

        # 独有成员边不受影响
        assert _edge_exists(graphlite_store, c1_id, epA), (
            f"C1 exclusive member epA edge missing"
        )
        assert _edge_exists(graphlite_store, c1_id, epB), (
            f"C1 exclusive member epB edge missing"
        )
        assert _edge_exists(graphlite_store, c2_id, epC), (
            f"C2 exclusive member epC edge missing"
        )

    def test_preserves_external_edges(self, graphlite_store):
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
            _create_episode(graphlite_store, ep_id, f"content of {ep_id}")

        # 预置外部社区 + 边到共享成员 epX（epX 同时属于 EXT + dream C1/C2）
        _insert_community(graphlite_store, ext_cid, "external_community")
        _insert_member_edge(graphlite_store, ext_cid, epX)

        # dream 社区：C1 含 epX+epY，C2 含 epX+epZ（epX 共享）
        communities = [
            _make_community(c1_id, [epX, epY], "C1: 2 members → larger"),
            _make_community(c2_id, [epX, epZ], "C2: 2 members → equal size"),
        ]

        created = pipe._persist_communities(
            graphlite_store, communities, f"dream-{uuid.uuid4().hex[:8]}"
        )
        assert created == 2

        # F2 核心断言：外部社区到共享成员 epX 的边存活
        # （修复前 Phase 1 MATCH (c:CommunityNode) 无社区过滤 → 必挂）
        assert _edge_exists(graphlite_store, ext_cid, epX), (
            f"F2 FAIL: EXT→epX edge was deleted — "
            f"Phase 1 MATCH lacks community filter ({ext_cid}→{epX})"
        )

        # epX 属于先出现的 C1（等大社区取先出现——sorted by member_count，
        # 相同则按 new_member_sets 插入序；C1 先加入 communities 列表）
        assert _edge_exists(graphlite_store, c1_id, epX), (
            f"epX should belong to C1 (first in communities list), but edge missing"
        )
        assert not _edge_exists(graphlite_store, c2_id, epX), (
            f"epX should NOT belong to C2, but stale edge exists"
        )

        # 独有成员边正常
        assert _edge_exists(graphlite_store, c1_id, epY), (
            f"C1 exclusive member epY edge missing"
        )
        assert _edge_exists(graphlite_store, c2_id, epZ), (
            f"C2 exclusive member epZ edge missing"
        )
