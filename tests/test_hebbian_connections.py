"""get_all_connections 兼容性测试（mock 层，不依赖真实引擎）"""
import pytest
from unittest.mock import MagicMock


def _make_store() -> "OverGraphStore":
    """构造无 _db 的 OverGraphStore 实例（避免 __del__ 报错）。"""
    from graph.overgraph_store import OverGraphStore
    store = OverGraphStore.__new__(OverGraphStore)
    store._db = None
    return store


def test_get_all_connections_converts_format():
    """get_all_connections 应把 list[dict] 转为 {src: {dst: weight}}。"""
    store = _make_store()
    # mock 底层查询返回旧格式行
    store.get_all_hebbian_connections = MagicMock(return_value=[
        {"src": "a1", "dst": "b1", "weight": 0.5},
        {"src": "a1", "dst": "b2", "weight": 0.3},
        {"src": "b1", "dst": "a1", "weight": 0.9},
    ])
    result = store.get_all_connections()
    assert result == {
        "a1": {"b1": 0.5, "b2": 0.3},
        "b1": {"a1": 0.9},
    }


def test_get_all_connections_empty():
    """无连接时应返回空字典。"""
    store = _make_store()
    store.get_all_hebbian_connections = MagicMock(return_value=[])
    assert store.get_all_connections() == {}


def test_get_all_connections_exception_safe():
    """底层查询异常时应返回空字典而非抛错。"""
    store = _make_store()
    store.get_all_hebbian_connections = MagicMock(side_effect=RuntimeError("db down"))
    assert store.get_all_connections() == {}
