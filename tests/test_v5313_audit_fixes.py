"""
v5.31.3 审计瑕疵回归测试
========================
覆盖：
  · (a) 熔断窗口不被写路径污染 —— _flush_hebbian_batch 走 execute_cypher（写路径，
        熔断中立，不 record_success/failure），失败 raise 也不打点；读路径
        query_cypher 不再被写调用
  · (b) ontology 幂等短路条件精确化 —— 部分类型缺失/实体缺失时不全跳过
        （v5.31.2 的 count>0 过宽：中断后剩余类型/IS_A 边永不补齐）
  · (c) 特殊字符 id 建边不坏 GQL —— 含 ' / \\ 的 id 经 _gql_value 转义（H1 风格）

运行: python -m pytest tests/test_v5313_audit_fixes.py -v
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from graph.graphlite_store import _gql_value, CircuitBreakerState
from api.routes.system import _flush_hebbian_batch
from core.ontology_validator import OntologyValidator, OntologyConfig


# ─── (a) 写路径熔断中立 ─────────────────────────────────────────


class TestFlushHebbianWritePathNeutral:

    def _fresh_store(self, overgraph_store):
        """隔离探针：验证 query_cypher（读路径）绝不被写调用触碰。"""
        store = overgraph_store
        orig_query = store.query_cypher

        def _read_path_guard(*args, **kwargs):
            raise AssertionError("query_cypher（读路径）不得用于 Hebbian 写操作")

        store.query_cypher = _read_path_guard
        return store, orig_query

    def test_flush_hebbian_uses_execute_cypher(self, overgraph_store):
        """写路径必须走 execute_cypher；成功不 record_success（窗口不变）。"""
        store, orig = self._fresh_store(overgraph_store)
        # 真实建两个节点 → 真实批量建边
        for nid in ("n1", "n2"):
            store.execute_cypher(
                f"INSERT (a:EpisodeNode {{id: {_gql_value(nid)}, content: 'c'}})"
            )
        ok = _flush_hebbian_batch(store, [("n1", "n2", 0.5)])
        assert ok is True
        # 写路径成功 → 熔断窗口无样本（P2-2：不 record_success，不稀释读失败率）
        assert store.circuit_breaker._window == []
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED
        # 读路径从未被调用（被 guard 拦截即测试失败）
        store.query_cypher = orig
        rows = store.query_cypher(
            f"MATCH (a:EpisodeNode {{id: {_gql_value('n1')}}})-[r:HEBBIAN_CONNECTION]->(b) "
            "RETURN count(*) AS c"
        )
        assert rows[0]["c"] == 1  # 边确实建成

    @pytest.mark.graphlite  # 【v6.0.0 legacy】GraphLite 专属语义/引擎约束（默认排除，addopts -m 'not graphlite'）
    def test_flush_hebbian_failure_raises_and_neutral_to_breaker(self, overgraph_store):

        """execute_cypher 不吞异常：真实 SDK QueryError 上抛；失败不打点窗口。"""
        from graphlite_sdk.error import QueryError

        store, orig = self._fresh_store(overgraph_store)
        store._session = MagicMock()
        store._session.query.side_effect = QueryError("real sdk query error")

        with pytest.raises(QueryError):
            _flush_hebbian_batch(store, [("n1", "n2", 0.5)])
        # 写失败不 record_failure → 窗口不变（v5.31.1 熔断风暴路径：写失败打满窗口）
        assert store.circuit_breaker._window == []
        assert store.circuit_breaker.state == CircuitBreakerState.CLOSED
        store.query_cypher = orig

    def test_flush_hebbian_empty_pairs_shortcircuits(self, overgraph_store):
        """空批直接返回 True，不触碰图库。"""
        store, orig = self._fresh_store(overgraph_store)
        assert _flush_hebbian_batch(store, []) is True
        store.query_cypher = orig


# ─── (c) 特殊字符 id 建边不坏 GQL ───────────────────────────────


class TestFlushHebbianSpecialCharIds:

    @pytest.mark.parametrize("src,dst", [
        ("it's a ' quote", 'double"quote'),
        (r"back\slash id", r"another\path"),
        ("中文字符 id", "emoji 🚀 id"),
        ("mixed'quote\\slash 中文", "spaces and chars"),
    ])
    def test_special_char_ids_build_edge(self, overgraph_store, src: str, dst: str):
        """含 ' / \\ / 中文的 id 建边成功且可回读（不坏 GQL）。"""
        store = overgraph_store
        for nid in (src, dst):
            store.execute_cypher(
                f"INSERT (a:EpisodeNode {{id: {_gql_value(nid)}, content: 'x'}})"
            )
        ok = _flush_hebbian_batch(store, [(src, dst, 0.5)])
        assert ok is True
        rows = store.query_cypher(
            f"MATCH (a:EpisodeNode {{id: {_gql_value(src)}}})-[r:HEBBIAN_CONNECTION]->(b) "
            "RETURN count(*) AS c"
        )
        assert rows[0]["c"] == 1, f"特殊字符 id 建边失败: {src!r} -> {dst!r}"


# ─── (b) ontology 幂等短路精确化 ────────────────────────────────


class TestOntologyShortCircuitPrecise:

    def _validator(self, overgraph_store) -> OntologyValidator:
        return OntologyValidator(
            graphlite_store=overgraph_store,
            config=OntologyConfig(enabled=True),
        )

    def test_partial_types_do_not_shortcircuit(self, overgraph_store):
        """库里只有部分类型（上次同步中断）→ 不得短路，必须全量补齐。"""
        store = overgraph_store
        # 只预置 1 个类型，模拟同步中途中断
        store.execute_cypher("INSERT (t:OntologyType {name: 'ml_model'})")
        v = self._validator(store)
        n = v.sync_entity_types_to_graphlite()
        assert n == len(v.ENTITY_TYPE_MAP), (
            f"部分类型缺失时短路（返回 {n}）——v5.31.2 count>0 过宽缺陷"
        )
        # 全部类型 + 全部实体现已就位（_ontology_synced 由调用方 extract_and_relate/
        # retrieve 在 sync 返回后置位，本函数不负责——见 L793/1192）
        rows = store.execute_cypher("MATCH (t:OntologyType) RETURN t.name")
        names = {r.get("t.name") or r.get("name") for r in rows}
        assert set(v.ENTITY_TYPE_MAP.values()).issubset(names)

    def test_complete_shortcircuits_to_zero(self, overgraph_store):
        """完整同步后第二次调用 → 短路返回 0（保留 v5.31.2 短路目的）。"""
        store = overgraph_store
        v = self._validator(store)
        assert v.sync_entity_types_to_graphlite() == len(v.ENTITY_TYPE_MAP)
        assert v.sync_entity_types_to_graphlite() == 0

    def test_types_full_but_entities_missing_not_shortcircuit(self, overgraph_store):
        """类型齐全但实体缺失（中断在实体建完前）→ 不得短路。"""
        store = overgraph_store
        v = self._validator(store)
        # 只建全部类型、不建实体
        for etype in set(v.ENTITY_TYPE_MAP.values()):
            store.execute_cypher(
                "INSERT (t:OntologyType {name: $type})", {"type": etype}
            )
        n = v.sync_entity_types_to_graphlite()
        assert n == len(v.ENTITY_TYPE_MAP), (
            f"类型齐全但实体缺失时短路（返回 {n}）——剩余实体/IS_A 边永不补齐"
        )
        rows = store.execute_cypher("MATCH (e:OntologyEntity) RETURN count(*) AS cnt")
        cnt = rows[0].get("cnt") if isinstance(rows[0], dict) else rows[0][0]
        assert cnt == len(v.ENTITY_TYPE_MAP)
