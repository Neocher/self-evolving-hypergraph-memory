"""GraphLiteStore 写路径测试（真实引擎）。

覆盖: create_episode → get_episode → update_with_version → get_episode 闭环。
之前 update_with_version 因 SET 语法错误静默失败 (返回 True 但值不变):
  - Bug1: SET e.content: 'x' (冒号, INSERT 格式) → GQL 语法错
  - Bug2: SET content = 'x' (无 e. 前缀) → 静默不生效
  - Fix: SET e.content = 'x' (前缀 + 等号)
"""
import uuid
import pytest


@pytest.fixture
def gstore(graphlite_store):
    return graphlite_store


def _make_episode(content: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "content": content,
        "created_at": 1.0,
        "tau_initial": 1.0,
        "tau_value": 0.6,
        "source": "test",
        "trust_score": 0.8,
    }


class TestEpisodeWritePath:

    def test_create_get_roundtrip(self, gstore):
        """写入 → 读回 闭环。"""
        ep = _make_episode("闭环测试")
        eid = gstore.create_episode(ep)
        assert eid == ep["id"]
        got = gstore.get_episode(eid)
        assert got is not None
        assert got["content"] == "闭环测试"

    def test_update_with_version_ascii(self, gstore):
        """英文内容更新生效。"""
        ep = _make_episode("original")
        eid = gstore.create_episode(ep)
        ok = gstore.update_with_version(eid, {"content": "updated"}, expected_version=1)
        assert ok is True
        got = gstore.get_episode(eid)
        assert got["content"] == "updated"

    def test_update_with_version_chinese(self, gstore):
        """中文内容更新生效（b64 编解码路径）。"""
        ep = _make_episode("原始内容")
        eid = gstore.create_episode(ep)
        ok = gstore.update_with_version(eid, {"content": "更新后的中文内容"}, expected_version=1)
        assert ok is True
        got = gstore.get_episode(eid)
        assert got["content"] == "更新后的中文内容"

    def test_update_multiple_fields(self, gstore):
        """多字段更新: content + trust_score 同时更新。"""
        ep = _make_episode("原内容")
        eid = gstore.create_episode(ep)
        ok = gstore.update_with_version(
            eid,
            {"content": "新内容", "trust_score": 0.95},
            expected_version=1,
        )
        assert ok is True
        got = gstore.get_episode(eid)
        assert got["content"] == "新内容"
        assert abs(float(got["trust_score"]) - 0.95) < 1e-6

    def test_update_nonexistent_returns_true(self, gstore):
        """GraphLite 对不存在的节点 SET 静默成功 (返回 status 行, 无副作用)。

        GraphLite 语义: MATCH 无匹配时 SET 不报错, 返回 status 行。
        乐观锁版本检查 (WHERE version = expected) 在此引擎下不可靠
        (WHERE 不匹配时 SET 静默跳过但返回 status), 故不承诺 False。
        """
        ok = gstore.update_with_version(
            str(uuid.uuid4()), {"content": "x"}, expected_version=1
        )
        assert ok is True
