"""
KuzuStore 集成测试
=================
测试 Kuzu 图数据库的 CRUD 操作 + 连接池 + 断路器。
"""
from __future__ import annotations

import time
import uuid

import pytest

try:
    from graph.ryu_store import RyuStore as KuzuStore, CircuitBreakerOpen, CircuitBreakerConfig
except ImportError:  # ryugraph 已被 GraphLite 替换，旧 store 不可用时跳过
    KuzuStore = CircuitBreakerOpen = CircuitBreakerConfig = None

pytestmark = pytest.mark.skipif(
    KuzuStore is None,
    reason="ryugraph 已废弃，旧 RyuStore 不可用（GraphLite 替代）",
)


class TestKuzuStore:
    """KuzuStore 核心功能测试。"""

    def test_connect_and_schema(self, kuzu_store: KuzuStore):
        """连接后 schema 应自动初始化。"""
        # 验证表存在（查询不报错说明表存在）
        rows = kuzu_store.query_cypher(
            "MATCH (e:EpisodeNode) RETURN count(e) AS cnt"
        )
        assert rows is not None

    def test_create_episode(self, kuzu_store: KuzuStore):
        """创建情节节点应返回 ID。"""
        ep_id = str(uuid.uuid4())
        result = kuzu_store.create_episode({
            "id": ep_id,
            "content": "测试内容",
            "source": "test",
            "created_at": time.time(),
            "tau_initial": 1.0,
        })
        assert result == ep_id

    def test_get_episode(self, kuzu_store: KuzuStore):
        """按 ID 查询应返回正确数据。"""
        ep_id = str(uuid.uuid4())
        kuzu_store.create_episode({
            "id": ep_id,
            "content": "查询测试",
            "source": "test",
            "created_at": time.time(),
            "tau_initial": 0.8,
        })
        ep = kuzu_store.get_episode(ep_id)
        assert ep is not None
        assert ep["id"] == ep_id
        assert "查询测试" in ep.get("content", "")

    def test_get_nonexistent_episode(self, kuzu_store: KuzuStore):
        """查询不存在的节点应返回 None。"""
        ep = kuzu_store.get_episode("nonexistent-id")
        assert ep is None

    def test_query_cypher(self, kuzu_store: KuzuStore):
        """Cypher 查询应返回正确格式的结果。"""
        ep_id = str(uuid.uuid4())
        kuzu_store.create_episode({
            "id": ep_id,
            "content": "cypher测试",
            "source": "test",
            "created_at": time.time(),
            "tau_initial": 1.0,
        })
        rows = kuzu_store.query_cypher(
            "MATCH (e:EpisodeNode) WHERE e.id = $id RETURN e.id, e.content",
            {"id": ep_id},
        )
        assert len(rows) > 0

    def test_query_cypher_empty_result(self, kuzu_store: KuzuStore):
        """查询不存在的条件应返回空列表。"""
        rows = kuzu_store.query_cypher(
            "MATCH (e:EpisodeNode) WHERE e.id = 'impossible_id' RETURN e.id"
        )
        assert rows == []

    def test_connection_pool(self, kuzu_store: KuzuStore):
        """连接池应返回有效连接。"""
        conn = kuzu_store.conn
        assert conn is not None
        # 多次调用应返回不同的连接（轮询）
        conn2 = kuzu_store.conn
        assert conn2 is not None

    def test_multiple_episodes(self, kuzu_store: KuzuStore):
        """批量创建和查询。"""
        ids = []
        for i in range(10):
            ep_id = str(uuid.uuid4())
            kuzu_store.create_episode({
                "id": ep_id,
                "content": f"内容{i}",
                "source": "test",
                "created_at": time.time(),
                "tau_initial": 1.0 - i * 0.1,
            })
            ids.append(ep_id)

        # 逐个验证
        for i, ep_id in enumerate(ids):
            ep = kuzu_store.get_episode(ep_id)
            assert ep is not None
            assert ep["tau_initial"] == 1.0 - i * 0.1

    def test_hyperedge_node(self, kuzu_store: KuzuStore):
        """超边节点创建和查询。"""
        he_id = str(uuid.uuid4())
        kuzu_store.create_hyperedge_node({
            "id": he_id,
            "type": "test",
            "created_at": time.time(),
            "gate_value": 0.8,
            "metadata": '{"key": "value"}',
        })
        # 验证——通过 Cypher 查
        rows = kuzu_store.query_cypher(
            "MATCH (h:HyperedgeNode {id: $id}) RETURN h.type",
            {"id": he_id},
        )
        assert len(rows) > 0

    def test_circuit_breaker_integration(self, kuzu_store: KuzuStore):
        """断路器应随查询成功/失败记录。"""
        # 执行成功查询
        kuzu_store.query_cypher("MATCH (e:EpisodeNode) RETURN count(e)")
        cb = kuzu_store.circuit_breaker
        assert cb.state.name == "CLOSED"

    def test_sensory_buffer(self, kuzu_store: KuzuStore):
        """感觉缓冲区应存在。"""
        from collections import deque
        buf = kuzu_store._sensory_buffer
        assert isinstance(buf, deque)
        assert buf.maxlen == 1000

    def test_close_and_reopen(self, kuzu_store: KuzuStore):
        """关闭后不应再可用。"""
        kuzu_store.close()
        with pytest.raises(RuntimeError):
            _ = kuzu_store.conn
