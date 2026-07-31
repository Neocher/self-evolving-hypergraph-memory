"""GraphLiteStore.connect() 双名兼容修复测试（mock 层，不依赖真实引擎）。

背景：GraphLite 本版要求 graph 名带 / 前缀（如 /shm），但旧生产库用 default。
修复后 connect() 依次尝试 default → /shm，都不行则按序创建 schema/graph。
本文件用 __new__ 构造 + 状态机 FakeSession 模拟 session，验证三条路径与格式转换。
"""
import pytest
from unittest.mock import MagicMock, patch


class FakeGraphLiteSession:
    """模拟 GraphLite Session 的最小状态机（跟踪 schema/graph 是否存在）。

    - schema/graph 不存在时 SESSION SET * 抛错（模拟真实引擎行为）
    - CREATE SCHEMA / CREATE GRAPH 成功后相应对象变为可用
    """

    def __init__(self, schema_exists: bool = False, graphs: set | None = None):
        self.schema_exists = schema_exists
        self.graphs: set = set(graphs or [])
        self.calls: list[str] = []

    def execute(self, sql: str) -> None:
        sql = sql.strip()
        self.calls.append(sql)
        if sql.startswith("SESSION SET SCHEMA"):
            if not self.schema_exists:
                raise RuntimeError("schema /shm 不存在")
        elif sql.startswith("SESSION SET GRAPH"):
            graph = sql.rsplit(" ", 1)[-1]
            if graph not in self.graphs:
                raise RuntimeError(f"graph {graph} 不存在")
        elif sql.startswith("CREATE SCHEMA"):
            self.schema_exists = True
        elif sql.startswith("CREATE GRAPH"):
            self.graphs.add(sql.rsplit(" ", 1)[-1])


def _make_store(tmp_path: pytest.TempPathFactory, session) -> "GraphLiteStore":
    """__new__ 构造 GraphLiteStore，mock GraphLite.open 返回带 session 的 db。"""
    from graph.graphlite_store import GraphLiteStore

    store = GraphLiteStore.__new__(GraphLiteStore)
    store.config = type("cfg", (), {"database_path": str(tmp_path / "test_graphlite_db")})()
    db = MagicMock()
    db.session.return_value = session
    with patch("graph.graphlite_store.GraphLite") as mock_gl:
        mock_gl.open.return_value = db
        store.connect()
    return store


def test_connect_new_db_creates_graph(tmp_path):
    """全新库：default 与 /shm 都连不上 → 按序创建 schema/graph → _graph_name == '/shm'。"""
    session = FakeGraphLiteSession()  # 无 schema、无 graph（全新库）
    store = _make_store(tmp_path, session)

    assert store._graph_name == "/shm"
    # 全新库无 schema：2 个候选的探测 + 创建流程里的 SET SCHEMA，共 3 次
    assert session.calls.count("SESSION SET SCHEMA /shm") == 3
    # 创建顺序：CREATE SCHEMA → SESSION SET SCHEMA → CREATE GRAPH → SESSION SET GRAPH
    creates = [c for c in session.calls if c.startswith("CREATE")]
    assert creates == ["CREATE SCHEMA /shm", "CREATE GRAPH /shm"]
    assert session.calls[-1] == "SESSION SET GRAPH /shm"
    assert store._db_path.endswith("test_graphlite_db")


def test_connect_existing_default_graph(tmp_path):
    """旧生产库：已有 default graph → 直接连上，不执行任何 CREATE。"""
    session = FakeGraphLiteSession(schema_exists=True, graphs={"default"})
    store = _make_store(tmp_path, session)

    assert store._graph_name == "default"
    assert not [c for c in session.calls if c.startswith("CREATE")]
    assert session.calls[-1] == "SESSION SET GRAPH default"


def test_connect_existing_slash_shm_graph(tmp_path):
    """新格式库：default 不存在但 /shm 存在 → 连 /shm，不执行任何 CREATE。"""
    session = FakeGraphLiteSession(schema_exists=True, graphs={"/shm"})
    store = _make_store(tmp_path, session)

    assert store._graph_name == "/shm"
    assert not [c for c in session.calls if c.startswith("CREATE")]
    assert session.calls[-1] == "SESSION SET GRAPH /shm"


def test_connect_schema_exists_no_graph(tmp_path):
    """边界：schema 存在但 graph 都不存在 → 只创建 graph，不重复 CREATE SCHEMA。"""
    session = FakeGraphLiteSession(schema_exists=True, graphs=set())
    store = _make_store(tmp_path, session)

    assert store._graph_name == "/shm"
    creates = [c for c in session.calls if c.startswith("CREATE")]
    # CREATE SCHEMA 可能被尝试但失败（已存在被 pass 吞掉），不得出现第二次
    assert creates.count("CREATE SCHEMA /shm") <= 1
    # 最终连上新 graph
    assert session.calls[-1] == "SESSION SET GRAPH /shm"


def test_get_all_connections_format():
    """get_all_connections 应把 list[dict] 转为 {src: {dst: weight}}（含嵌套键兜底）。"""
    from graph.graphlite_store import GraphLiteStore

    store = GraphLiteStore.__new__(GraphLiteStore)
    store._db = None
    store._session = None
    # 两种行格式：AS 别名(src/dst/weight) 与 嵌套键(a.id/b.id/r.weight)；缺失行应跳过
    store.get_all_hebbian_connections = MagicMock(return_value=[
        {"src": "n1", "dst": "n2", "weight": 0.42},     # AS 别名格式
        {"a.id": "n2", "b.id": "n3", "r.weight": 0.7},  # 嵌套键兜底格式
        {"src": "n1", "dst": None},                    # dst 缺失 → 跳过
        {},                                            # 空行 → 跳过
    ])
    result = store.get_all_connections()
    assert result == {
        "n1": {"n2": 0.42},
        "n2": {"n3": 0.7},
    }
