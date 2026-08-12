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

    def test_create_with_chinese_list(self, gstore):
        """中文 list 字段写入/读回 (b64 防 PANIC)。"""
        ep = _make_episode("带标签的内容")
        ep["tags"] = ["张三", "李四"]
        eid = gstore.create_episode(ep)
        got = gstore.get_episode(eid)
        assert got is not None
        assert got.get("tags") == '["张三", "李四"]'

    def test_update_nonexistent_returns_true(self, gstore):
        """乐观锁: 不存在节点 → False (两步法可区分不存在与成功)。

        修复前: MATCH 无匹配时 SET 静默返回 status 行, 无法区分不存在。
        修复后: 先 MATCH 查 version, 无 rows 直接返回 False。
        """
        ok = gstore.update_with_version(
            str(uuid.uuid4()), {"content": "x"}, expected_version=1
        )
        assert ok is False

    def test_create_episode_sets_version_1(self, gstore):
        """create_episode 后 version 应为 1 (乐观锁基线)。"""
        ep = _make_episode("基线测试")
        eid = gstore.create_episode(ep)
        got = gstore.get_episode(eid)
        assert int(got["version"]) == 1

    def test_optimistic_lock_success(self, gstore):
        """version 匹配 → 更新成功, version +1。"""
        ep = _make_episode("v1 内容")
        eid = gstore.create_episode(ep)
        ok = gstore.update_with_version(eid, {"content": "v2 内容"}, expected_version=1)
        assert ok is True
        got = gstore.get_episode(eid)
        assert got["content"] == "v2 内容"
        assert int(got["version"]) == 2

    def test_optimistic_lock_stale(self, gstore):
        """version 不匹配 (stale) → 返回 False, 内容不变。"""
        ep = _make_episode("原始")
        eid = gstore.create_episode(ep)
        ok = gstore.update_with_version(eid, {"content": "不应写入"}, expected_version=99)
        assert ok is False
        got = gstore.get_episode(eid)
        assert got["content"] == "原始"
        assert int(got["version"]) == 1

    def test_optimistic_lock_nonexistent(self, gstore):
        """不存在节点 → 返回 False。"""
        ok = gstore.update_with_version(
            str(uuid.uuid4()), {"content": "x"}, expected_version=1
        )
        assert ok is False

    def test_optimistic_lock_chinese(self, gstore):
        """中文更新带 version 递增 (b64 编解码路径)。"""
        ep = _make_episode("中文原始")
        eid = gstore.create_episode(ep)
        ok = gstore.update_with_version(
            eid, {"content": "中文更新后的内容"}, expected_version=1
        )
        assert ok is True
        got = gstore.get_episode(eid)
        assert got["content"] == "中文更新后的内容"
        assert int(got["version"]) == 2

    def test_optimistic_lock_force_skip_check(self, gstore):
        """expected_version=None → 跳过版本检查直接写入 (force 路径)。"""
        ep = _make_episode("force 原始")
        eid = gstore.create_episode(ep)
        ok = gstore.update_with_version(eid, {"content": "force 覆盖"}, expected_version=None)
        assert ok is True
        got = gstore.get_episode(eid)
        assert got["content"] == "force 覆盖"

    def test_update_with_version_comma_value(self, gstore):
        """值含 ', ' (ASCII 直存) 不拆坏 SET 子句 (修复前 split(', ') 会损坏 SQL)。"""
        ep = _make_episode("original")
        eid = gstore.create_episode(ep)
        ok = gstore.update_with_version(eid, {"content": "a, b"}, expected_version=1)
        assert ok is True
        got = gstore.get_episode(eid)
        assert got["content"] == "a, b"
        assert int(got["version"]) == 2

    def test_ontology_fields_roundtrip(self, gstore):
        """本体独立字段 (ontology_type/entity_name/entity_value) 写入 → 读回 闭环 (b64 根治基础)。"""
        ep = _make_episode("张三出生于1990年")
        ep["ontology_type"] = "person_birth"
        ep["entity_name"] = "张三"
        ep["entity_value"] = "1990"
        eid = gstore.create_episode(ep)
        got = gstore.get_episode(eid)
        assert got["ontology_type"] == "person_birth"
        assert got["entity_name"] == "张三"
        assert got["entity_value"] == "1990"

    def test_force_promote_protected_flag_storage_layer_roundtrip(self, gstore):
        """【存储层闭环】protected 布尔标记写入 → 读回 闭环。

        本测试直调 gstore.create_episode 并手动预置 protected，验证的是
        存储层本身的读写能力（中文 content 走 b64 + 布尔 protected 非 b64
        同写不冲突；读取经 _flatten_row 还原为 Python True）。
        force_promote=true → protected 标记的**生产链路**（路由层加标记）由
        tests/test_write_routes.py 的 TestForcePromoteProtectedFlagRoute 覆盖。
        """
        ep = _make_episode("重要记忆")
        ep["protected"] = True
        eid = gstore.create_episode(ep)
        got = gstore.get_episode(eid)
        assert got is not None
        assert got["content"] == "重要记忆"          # b64 中文解码正常
        assert got.get("protected") in (True, "true", 1)  # 布尔标记还原

    def test_episode_without_protected_flag_defaults_unprotected(self, gstore):
        """【v5.27.0】未打 protected 标记的节点读回不携带标记（兼容旧数据，默认不保护）。"""
        ep = _make_episode("普通记忆")
        eid = gstore.create_episode(ep)
        got = gstore.get_episode(eid)
        assert got.get("protected") in (None, False)
