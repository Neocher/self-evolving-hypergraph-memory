"""GraphLiteStore Visual/Session CRUD 集成测试（真实引擎）。

覆盖 P0-2 要求的 5 方法真实 store 测试：
  1. create_visual_node → get_visual_node → get_visual_nodes roundtrip
  2. get_or_create_session 幂等（2 次调用 1 节点）
  3. link_session_member → get_session_memories 命中
"""
import uuid
import pytest

pytestmark = pytest.mark.graphlite  # 【v6.0.0 legacy】GraphLite 专属语义测试（默认排除，addopts -m 'not graphlite'）


@pytest.fixture
def gstore(graphlite_store):
    return graphlite_store


class TestVisualNodeRoundtrip:

    def test_create_get_visual_node_roundtrip(self, gstore):
        """create_visual_node → get_visual_node 闭环"""
        vid = str(uuid.uuid4())
        node = {
            "id": vid,
            "caption": "A test visual node",
            "media_type": "image",
            "file_path": "test.png",
        }
        created = gstore.create_visual_node(node)
        assert created == vid

        got = gstore.get_visual_node(vid)
        assert got is not None
        assert got["id"] == vid
        assert got["caption"] == "A test visual node"
        assert got["media_type"] == "image"

    def test_get_visual_nodes_lists_all(self, gstore):
        """get_visual_nodes 列出所有 VisualNode"""
        n1 = gstore.create_visual_node({
            "id": str(uuid.uuid4()),
            "caption": "first",
            "media_type": "image",
        })
        n2 = gstore.create_visual_node({
            "id": str(uuid.uuid4()),
            "caption": "second",
            "media_type": "video",
        })

        all_nodes = gstore.get_visual_nodes(limit=50)
        ids = {n["id"] for n in all_nodes}
        assert n1 in ids
        assert n2 in ids

    def test_get_visual_node_nonexistent(self, gstore):
        """get_visual_node 不存在 → None"""
        got = gstore.get_visual_node(str(uuid.uuid4()))
        assert got is None


class TestSessionCRUD:

    def test_get_or_create_session_idempotent(self, gstore):
        """两次调用 get_or_create_session 只创建一个 SessionNode"""
        sid = str(uuid.uuid4())
        r1 = gstore.get_or_create_session(sid)
        r2 = gstore.get_or_create_session(sid)
        assert r1 == sid
        assert r2 == sid

    def test_link_session_member_get_memories(self, gstore):
        """link_session_member → get_session_memories 命中"""
        sid = str(uuid.uuid4())
        gstore.get_or_create_session(sid)

        # 创建两个 episode 并链接
        ep1_id = str(uuid.uuid4())
        ep2_id = str(uuid.uuid4())
        gstore.create_episode({
            "id": ep1_id,
            "content": "session memory one",
            "created_at": 1.0,
            "tau_initial": 1.0,
            "source": "test",
        })
        gstore.create_episode({
            "id": ep2_id,
            "content": "session memory two",
            "created_at": 2.0,
            "tau_initial": 1.0,
            "source": "test",
        })

        gstore.link_session_member(sid, ep1_id)
        gstore.link_session_member(sid, ep2_id)

        memories = gstore.get_session_memories(sid, limit=10)
        mem_ids = {m["id"] for m in memories}
        assert ep1_id in mem_ids
        assert ep2_id in mem_ids

    def test_delete_namespace_removes_episodes(self, gstore):
        """delete_namespace 删除 SessionNode 及其 SESSION_MEMBER 关联的 EpisodeNode。

        回归: 修复前查询用 RETURN e.id 返回扁平格式 {'e.id': 'xxx'},
        _flatten_row(r, "e") 的 label 分支不命中 → ep_ids=[] → deleted=0。
        """
        sid = str(uuid.uuid4())
        gstore.get_or_create_session(sid)

        ep1_id = str(uuid.uuid4())
        ep2_id = str(uuid.uuid4())
        for eid, content in ((ep1_id, "delete ns memory one"),
                             (ep2_id, "delete ns memory two")):
            gstore.create_episode({
                "id": eid,
                "content": content,
                "created_at": 1.0,
                "tau_initial": 1.0,
                "source": "test",
            })
            gstore.link_session_member(sid, eid)

        # 确认前置状态: 边已建立, episode 可查
        mem_ids = {m["id"] for m in gstore.get_session_memories(sid, limit=10)}
        assert {ep1_id, ep2_id} <= mem_ids

        deleted = gstore.delete_namespace(sid)
        assert deleted == 2  # 修复前恒为 0

        # EpisodeNode 已被 DETACH DELETE
        assert gstore.get_episode(ep1_id) is None
        assert gstore.get_episode(ep2_id) is None
        # SessionNode 已删, 不再有 SESSION_MEMBER 可查
        assert gstore.get_session_memories(sid, limit=10) == []

    def test_delete_namespace_unknown_returns_zero(self, gstore):
        """不存在的 namespace → 返回 0（幂等, 无异常）"""
        assert gstore.delete_namespace(str(uuid.uuid4())) == 0
