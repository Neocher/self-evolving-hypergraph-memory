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


class TestFlattenBadB64:
    """L1：_flatten_row 对坏 b64 内容不抛异常、保留原文。

    修复前裸 `except:` 吞掉一切异常（含 KeyboardInterrupt）；
    修复后限定 `except ValueError:`——b64decode 的 binascii.Error 与
    decode 的 UnicodeDecodeError 都是 ValueError 子类，语义不变。
    """

    def test_invalid_b64_keeps_original(self):
        """坏 b64 块（非 base64 字符）→ 解码失败 → 保留原文不抛。"""
        from graph.graphlite_store import GraphLiteStore
        row = {"n": {"Node": {"properties": {"content": "{b64}!!!not-base64!!!"}}}}
        out = GraphLiteStore._flatten_row(row)
        assert out["n"]["content"] == "{b64}!!!not-base64!!!"

    def test_b64_invalid_chars_keeps_original(self):
        """含非法字符的 b64 串 → 不抛、保留原文。"""
        from graph.graphlite_store import GraphLiteStore
        row = {"n": {"Node": {"properties": {"content": "{b64}%%%invalid"}}}}
        out = GraphLiteStore._flatten_row(row)
        assert out["n"]["content"] == "{b64}%%%invalid"

    def test_valid_b64_decodes(self):
        """合法 b64 仍正常解码（修复不破坏正常路径）。"""
        from base64 import b64encode
        from graph.graphlite_store import GraphLiteStore
        payload = b64encode("中文内容".encode("utf-8")).decode("ascii")
        row = {"n": {"Node": {"properties": {"content": f"{{b64}}{payload}"}}}}
        out = GraphLiteStore._flatten_row(row)
        assert out["n"]["content"] == "中文内容"

    def test_garbage_utf8_bytes_keeps_original(self):
        """b64 合法但 decode('utf-8') 失败（非法 UTF-8 字节）→ 不抛、保留原文。"""
        from graph.graphlite_store import GraphLiteStore
        # b64encode(b'\xff\xfe\xfd') = /v79，是合法 b64 但非法 UTF-8
        row = {"n": {"Node": {"properties": {"content": "{b64}/v79"}}}}
        out = GraphLiteStore._flatten_row(row)
        assert out["n"]["content"] == "{b64}/v79"

    def test_unicode_decode_error_is_valueerror(self):
        """判别性：UnicodeDecodeError 确实是 ValueError 子类（防修错类型）。"""
        assert issubclass(UnicodeDecodeError, ValueError)
        import binascii
        assert issubclass(binascii.Error, ValueError)

    def test_keyboard_interrupt_not_swallowed(self):
        """判别性：修复后裸 except 已消失——源码中 _flatten_row 内不再有裸 except。"""
        import inspect
        from graph.graphlite_store import GraphLiteStore
        src = inspect.getsource(GraphLiteStore._flatten_row)
        assert "except:" not in src, "L1 修复后 _flatten_row 不应再有裸 except"


class TestH1GqlEscaping:
    """H1：外部可达的 id/session 插值全部经 _gql_value 转义。

    注入面：GET /memories/episodes/{id}（get_episode）、X-Session-Id 头
    （ensure_session/get_or_create_session/link_session_member）、
    batch ids、hyperedge/visual/namespace 家族。含 ' 或 \ 的 id 不应以
    裸引号形式出现在发给引擎的 GQL 中（修复前被 except 吞，仅表现为 404）。
    """

    def _escaped_id(self, store, method: str, *args):
        """记录发给 session 的 GQL，断言其中无裸引号 id（对每个参数都检查）。"""
        from graph.graphlite_store import _gql_value
        sent: list[str] = []
        real_query = store._locked_query
        real_execute = store._locked_execute
        store._locked_query = lambda gql: (sent.append(gql), real_query(gql))[1]
        store._locked_execute = lambda gql: (sent.append(gql), real_execute(gql))[1]
        try:
            getattr(store, method)(*args)
        finally:
            store._locked_query = real_query
            store._locked_execute = real_execute
        assert sent, f"{method} 应至少发出一条 GQL"
        for evil in args:
            expected = _gql_value(str(evil))
            # 每条 GQL 必须包含转义后的字面量，且不含裸 evil 串
            for gql in sent:
                assert expected in gql, f"GQL 未含转义字面量 {expected!r}: {gql!r}"
                assert f"'{evil}'" not in gql, f"GQL 含裸 id 插值: {gql!r}"

    def test_get_episode_escapes_quote(self, gstore):
        """id 含单引号 → GQL 无裸引号（修复前: MATCH ... {id: 'a'b'} 语法错/注入）。"""
        self._escaped_id(gstore, "get_episode", "a'b")

    def test_get_episode_escapes_backslash(self, gstore):
        """id 含反斜杠 → GQL 无裸反斜杠。"""
        self._escaped_id(gstore, "get_episode", "a\\b")

    def test_get_episode_escapes_combined(self, gstore):
        """id 含 ' 与 \\ 组合 → GQL 无裸引号。"""
        self._escaped_id(gstore, "get_episode", "x'y\\z")

    def test_ensure_session_escapes(self, gstore):
        """X-Session-Id 头（ensure_session）→ GQL 无裸引号。"""
        self._escaped_id(gstore, "ensure_session", "sess'one")

    def test_get_or_create_session_escapes(self, gstore):
        """get_or_create_session（外部可达，X-Session-Id）→ GQL 无裸引号。"""
        self._escaped_id(gstore, "get_or_create_session", "sess\\two")

    def test_link_session_member_escapes(self, gstore):
        """link_session_member（X-Session-Id）→ GQL 无裸引号。"""
        # 两个参数都是注入面：session_node_id + episode_id
        self._escaped_id(gstore, "link_session_member", "sess'x", "ep'y")

    def test_link_to_session_escapes(self, gstore):
        """link_to_session（namespace 路径）→ GQL 无裸引号。"""
        self._escaped_id(gstore, "link_to_session", "ns'a", "ep\\b")

    def test_get_session_memories_escapes(self, gstore):
        """get_session_memories（namespace 路径）→ GQL 无裸引号。"""
        self._escaped_id(gstore, "get_session_memories", "ns'q")

    def test_hyperedge_family_escapes(self, gstore):
        """hyperedge id / member / visual / namespace 家族 → GQL 无裸引号。"""
        self._escaped_id(gstore, "get_hyperedge_members", "he'1")
        self._escaped_id(gstore, "get_hyperedges_by_node", "ep'1")
        self._escaped_id(gstore, "get_visual_node", "vis\\1")

    def test_delete_namespace_escapes(self, gstore):
        """delete_namespace（外部可达 DELETE /namespaces/{ns}）→ GQL 无裸引号。"""
        self._escaped_id(gstore, "delete_namespace", "ns'z")

    def test_create_episode_with_evil_id(self, gstore):
        """create_episode 带含 ' 的 id → 写入后能按原 id 读回（转义不破坏功能）。"""
        evil = "evil'id"
        eid = gstore.create_episode({"id": evil, "content": "x", "created_at": 1.0})
        assert eid == evil
        got = gstore.get_episode(evil)
        assert got is not None, "转义后的 id 应可正常读回"
        assert got.get("content") == "x"

    def test_create_episode_evil_id_roundtrip_ascii_content(self, gstore):
        """含反斜杠 id 写入 → 读回（功能不破坏）。"""
        evil = "evil\\id"
        gstore.create_episode({"id": evil, "content": "y", "created_at": 1.0})
        got = gstore.get_episode(evil)
        assert got is not None and got.get("content") == "y"


class TestM2AtomicUpdateWithVersion:
    """M2：update_with_version 读 version + SET 包进同一个 _session_lock。

    修复前：Step1 读 version 与 Step2 SET 是两次独立的 _locked_query/_locked_execute，
    中间存在"读后、SET 前"窗口——另一写线程抢先更新会让本次乐观锁漏检
    （读了旧 version，SET 却落在新状态上）。修复后单次持锁完成读+写
    （直接调 self._session.query/execute，不再分两次 _locked_*）。
    """

    def test_single_lock_span_read_and_set(self):
        """读+SET 在同一持锁周期：query 与 execute 均在锁内执行（无解锁窗口）。"""
        import threading
        from graph.graphlite_store import GraphLiteStore

        class TrackLock:
            """委托包装 _session_lock：记录 acquire/release，暴露 _held。"""
            def __init__(self):
                self._inner = threading.Lock()
                self._held = False
            def acquire(self, *a, **k):
                r = self._inner.acquire(*a, **k)
                if r:
                    self._held = True
                return r
            def release(self):
                self._held = False
                self._inner.release()
            def __enter__(self):
                self.acquire()
                return self
            def __exit__(self, *exc):
                self.release()
                return False

        lock = TrackLock()
        held_at_call: list[bool] = []

        class FakeSession:
            def query(self, gql):
                held_at_call.append(lock._held)
                return type("R", (), {"rows": [{"v": 1}]})()
            def execute(self, gql):
                held_at_call.append(lock._held)
                return 1  # v5.31.4+: execute 返回 rows_affected

        store = GraphLiteStore.__new__(GraphLiteStore)
        store._session = FakeSession()
        store._session_lock = lock
        store._db = None
        store.config = type("cfg", (), {"database_path": ""})()

        ok = store.update_with_version("n1", {"content": "x"}, expected_version=1)
        assert ok is True
        assert held_at_call == [True], \
            f"execute 应在持锁状态下执行（CAS 单条无 query）: {held_at_call}"

    def test_double_update_exactly_one_wins(self):
        """并发双写同 expected_version → 恰好一个 True（原子化后无漏检）。"""
        import threading
        from graph.graphlite_store import GraphLiteStore

        class FakeSession:
            """同一 session 上双线程并发 update_with_version。

            store._session_lock 是 RLock——整个 CAS 在锁内，两线程严格串行：
            线程 A WHERE version=1 匹配 → SET version=2 → 返回 1 → True；
            线程 B WHERE version=1 不匹配（version 已 2）→ 返回 0 → False。
            """
            def __init__(self):
                self.version = 1
            def query(self, gql):
                return type("R", (), {"rows": [{"v": self.version}]})()
            def execute(self, gql):
                # v5.31.4+: 模拟真实引擎 CAS——WHERE version 匹配则递增并返回 1
                import re
                m = re.search(r"WHERE e\.version = (\d+)", gql)
                if m and int(m.group(1)) == self.version:
                    self.version += 1
                    return 1
                return 0

        session = FakeSession()
        store = GraphLiteStore.__new__(GraphLiteStore)
        store._session = session
        store._session_lock = threading.RLock()
        store._db = None
        store.config = type("cfg", (), {"database_path": ""})()

        results = []
        barrier = threading.Barrier(2)
        def worker():
            barrier.wait()
            results.append(store.update_with_version("n1", {"content": "w"}, expected_version=1))
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(1 for r in results if r) == 1, \
            f"双写同 expected_version 应恰好一个成功: {results}"

    def test_version_mismatch_false(self):
        """version 不匹配 → False（原子化不破坏原有语义）。"""
        from graph.graphlite_store import GraphLiteStore
        class FakeSession:
            def query(self, gql):
                return type("R", (), {"rows": [{"v": 5}]})()
            def execute(self, gql):
                raise AssertionError("version 不匹配不应执行 SET")
        store = GraphLiteStore.__new__(GraphLiteStore)
        store._session = FakeSession()
        store._session_lock = __import__("threading").RLock()
        store._db = None
        store.config = type("cfg", (), {"database_path": ""})()
        assert store.update_with_version("n1", {"content": "x"}, expected_version=1) is False

    def test_missing_node_false(self):
        """节点不存在 → False（无 rows）。"""
        from graph.graphlite_store import GraphLiteStore
        class FakeSession:
            def query(self, gql):
                return type("R", (), {"rows": []})()
            def execute(self, gql):
                raise AssertionError("节点不存在不应执行 SET")
        store = GraphLiteStore.__new__(GraphLiteStore)
        store._session = FakeSession()
        store._session_lock = __import__("threading").RLock()
        store._db = None
        store.config = type("cfg", (), {"database_path": ""})()
        assert store.update_with_version("n1", {"content": "x"}, expected_version=1) is False


class TestArchiveNode:
    """Archive-Supersedes：archive_node 归档节点 + 建 SUPERSEDES 血统边。"""

    @staticmethod
    def _store_with_session(rows_affected):
        from graph.graphlite_store import GraphLiteStore

        class RecordingSession:
            def __init__(self, affected):
                self.affected = affected
                self.executed: list[str] = []

            def execute(self, gql):
                self.executed.append(gql)
                return self.affected

        session = RecordingSession(rows_affected)
        store = GraphLiteStore.__new__(GraphLiteStore)
        store._session = session
        store._session_lock = __import__("threading").RLock()
        store._db = None
        store.config = type("cfg", (), {"database_path": ""})()
        return store, session

    def test_archive_sets_archived_true(self):
        """归档存在节点 → SET archived=true 并返回 True。"""
        store, session = self._store_with_session(rows_affected=1)
        assert store.archive_node("n1") is True
        assert len(session.executed) == 1
        assert "SET e.archived = true" in session.executed[0]
        assert "id: 'n1'" in session.executed[0]

    def test_archive_missing_returns_false(self):
        """节点不存在（rows_affected=0）→ False，不建边。"""
        store, session = self._store_with_session(rows_affected=0)
        assert store.archive_node("n1") is False
        assert len(session.executed) == 1

    def test_archive_with_replacement_builds_supersedes(self):
        """replacement 非空 → 建 SUPERSEDES 边（第二条 execute）。"""
        store, session = self._store_with_session(rows_affected=1)
        assert store.archive_node("n1", "n2") is True
        assert len(session.executed) == 2
        assert "SUPERSEDES" in session.executed[1]
        assert "id: 'n1'" in session.executed[1]
        assert "id: 'n2'" in session.executed[1]
