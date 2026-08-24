"""AtomicFact 事实级中间层测试（P0-③）"""
import pytest
from unittest.mock import MagicMock

from core.dream_pipeline import DreamPipeline
from graph.overgraph_store import OverGraphStore
from graph.overgraph_store import LABEL_FACT, LABEL_FACT_MENTIONS


class TestFactExtraction:
    def test_extract_spo_basic(self):
        p = DreamPipeline.__new__(DreamPipeline)
        facts = p._extract_facts_rules(
            "Caroline is a graduate student at Stanford University."
        )
        assert facts, "应抽取到 SPO 事实"
        f = facts[0]
        assert f["subject"] == "Caroline"
        assert f["predicate"] == "is"
        assert "Stanford" in f["object"]

    def test_extract_spo_works_at(self):
        p = DreamPipeline.__new__(DreamPipeline)
        facts = p._extract_facts_rules(
            "Melanie works at a tech company in San Francisco."
        )
        assert facts
        assert facts[0]["subject"] == "Melanie"

    def test_extract_valid_time(self):
        p = DreamPipeline.__new__(DreamPipeline)
        facts = p._extract_facts_rules(
            "Caroline graduated from MIT in 2019."
        )
        assert facts
        assert facts[0]["valid_time"] == "2019"

    def test_extract_empty_content(self):
        p = DreamPipeline.__new__(DreamPipeline)
        assert p._extract_facts_rules("") == []
        assert p._extract_facts_rules(None) == []


class TestFactStore:
    def test_create_fact_idempotent(self):
        store = OverGraphStore.__new__(OverGraphStore)
        store._locked_upsert_node = MagicMock(return_value=None)
        store._require_internal_id = MagicMock(side_effect=lambda *a: 1)
        store._ensure_edge = MagicMock(return_value=None)

        fid1 = store.create_atomic_fact(
            "Caroline", "graduated from", "MIT", valid_time="2019",
            source_episode="ep1")
        fid2 = store.create_atomic_fact(
            "Caroline", "graduated from", "MIT", valid_time="2019",
            source_episode="ep2")
        assert fid1 == fid2, "同事实同版本应幂等（确定性 key）"

        fid3 = store.create_atomic_fact(
            "Caroline", "graduated from", "MIT", valid_time="2020",
            source_episode="ep1")
        assert fid1 != fid3, "不同时间版本应不同 key"

    def test_create_fact_requires_fields(self):
        store = OverGraphStore.__new__(OverGraphStore)
        with pytest.raises(Exception):
            store.create_atomic_fact("", "is", "x")
        with pytest.raises(Exception):
            store.create_atomic_fact("Caroline", "", "x")


class TestPersistFacts:
    def test_persist_atomic_facts(self):
        p = DreamPipeline.__new__(DreamPipeline)
        store = MagicMock()
        store.create_atomic_fact = MagicMock(return_value="fact_x")
        communities = [{
            "episodes": [
                {"content": "Caroline is a graduate student at Stanford University."},
                {"content": "Melanie works at a tech company."},
                {"content": "No facts here."},
            ]
        }]
        import asyncio
        n = asyncio.run(p._persist_atomic_facts(store, communities))
        assert n >= 2, f"应落库至少 2 条事实，实际 {n}"
        # 跨调用幂等：seen 仅轮内去重；跨轮由 store 的 sha1 确定性 key 保证
        # （create_atomic_fact 同 key 复用不重复建节点）——此处只验证调用可重复。
        n2 = asyncio.run(p._persist_atomic_facts(store, communities))
        assert n2 >= n, "重复调用应可再次触发（store 侧幂等）"

    def test_persist_degraded_on_store_missing(self):
        p = DreamPipeline.__new__(DreamPipeline)
        import asyncio
        n = asyncio.run(p._persist_atomic_facts(None, []))
        assert n == 0, "store 缺失应静默返回 0"
