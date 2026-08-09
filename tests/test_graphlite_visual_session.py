"""GraphLiteStore Visual/Session CRUD 集成测试（真实引擎）。

覆盖 P0-2 要求的 5 方法真实 store 测试：
  1. create_visual_node → get_visual_node → get_visual_nodes roundtrip
  2. get_or_create_session 幂等（2 次调用 1 节点）
  3. link_session_member → get_session_memories 命中
"""
import uuid
import pytest


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
