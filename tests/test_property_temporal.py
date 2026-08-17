"""
P0-1 实体-属性-时间三维建模（PropertyVerNode 属性时间版本链）测试
===============================================================
覆盖（任务书 AC ≥10 用例 + Codex 终审 5 缺陷回归）：
  1. 向后兼容：AttributeDef.temporal 默认 False（零迁移）
  2. 版本创建 roundtrip（store 原语）
  3. 幂等：同值 no-op（查存在→插 两段式）
  4. supersedes 链：旧版本 expired_at + SUPERSEDES 血统边
  5. 编排：triples.attributes → attr_name 派生（acquired_value）
  6. valid_from 严格单调（同微秒 → bump，排序稳定）
  7. N=8 惰性裁剪（超限 DETACH DELETE 最旧）
  8. 时间检索 最近：latest 模式取最新未过期版
  9. 时间检索 具体年份：at_time 模式取该时点前最新版
  10. 无时间词：current 模式取全部未过期版本
  11. 扩展分严格低于种子分（相对尾分缩放）
  12. GraphLite 失败 → 静默降级，主检索零回归
  13. HTTP 全链路：POST /memories/episodes → PropertyVerNode 落库
  14. 无实体候选查询 → 通道跳过
  --- Codex 终审 5 缺陷回归 ---
  15. P1-1 年份解析：relation_extractor "in 2014" → attrs[attr_year] → valid_from≈2014
  16. P1-2 实体归一化：写入 "Apple Inc" → 查询 "Apple"/"apple" 命中
  17. P2-1 at_time 年末语义：年中生效版本不丢 + expired_at 校验
  18. P2-2 相对时间词：昨天/earlier → at_time（不误归 latest）
  19. P2-3 非原子写补偿：SET expired_at/SUPERSEDES 边失败 → 回滚新版本

走公共入口：QueryRouter.retrieve()（mock 种子通道）+ HTTP 端点，
不直调被 mock 的内部检索方法。

运行: python -m pytest tests/test_property_temporal.py -v
"""
from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router, Services, get_services
from core.entity_resolver import EntityResolver
from core.relation_extractor import RelationTriple
from retrieval.query_router import QueryRouter


# ─── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    def _build(svc):
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app)

    return _build


def _svc(graphlite_store) -> Services:
    svc = Services()
    svc.graphlite_store = graphlite_store
    return svc


def _make_router(store, hypergraph_results: list[dict]):
    """构造 QueryRouter：mock _hypergraph_retrieve 控制种子；属性通道走真实 store。"""
    from retrieval.query_router import QueryRouter, QueryRouterConfig
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router._zh_en_tech_map = {}
    router._time_keywords = set()
    router.graphlite_store = store
    router._hypergraph_retrieve = MagicMock(return_value=hypergraph_results)
    return router


def _seed_result(nid: str, content: str, score: float = 0.9) -> dict:
    return {
        "node_id": nid, "content": content, "score": score,
        "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss",
    }


def _year_ts(year: int) -> float:
    """某年 1 月 1 日的本地时间戳（与 query_router._property_time_mode 同基准）。"""
    return datetime(year, 1, 1).timestamp()


def _year_end_ts(year: int) -> float:
    """某年 12 月 31 日 23:59:59 的本地时间戳（P2-1 at_time 年末语义）。"""
    return datetime(year, 12, 31, 23, 59, 59).timestamp()


def _count_property_supersedes(store, old_id: str, new_id: str) -> int:
    rows = store.execute_cypher(
        "MATCH (a:PropertyVerNode {id: $old_id})-[s:SUPERSEDES]->"
        "(b:PropertyVerNode {id: $new_id}) RETURN s",
        {"old_id": old_id, "new_id": new_id},
    )
    return len(rows)


def _get_values(versions: list[dict]) -> list[str]:
    return [str(v.get("value", "")) for v in versions]


# ─── 1. 向后兼容（决策 4：零迁移）──────────────────────────────


class TestOntologyTemporalField:

    def test_attribute_def_temporal_default_false(self):
        """AttributeDef() 不带 temporal → 默认 False（旧代码零迁移）。"""
        from core.ontology_v2 import AttrType, AttributeDef
        d = AttributeDef(name="revenue", type=AttrType.STRING)
        assert d.temporal is False
        d2 = AttributeDef(name="revenue", type=AttrType.STRING, temporal=True)
        assert d2.temporal is True
        # 全关键字构造不受新增字段影响
        d3 = AttributeDef(
            name="age", type=AttrType.INTEGER, required=True,
            indexed=True, description="年龄", min_value=0, max_value=200,
        )
        assert d3.temporal is False


# ─── 2-7. store 原语 + resolver 编排 ───────────────────────────


class TestPropertyVersionStore:

    def test_create_property_version_roundtrip(self, graphlite_store):
        """版本创建：PropertyVerNode 落库，get_latest 返回正确字段。"""
        pid = graphlite_store.create_property_version(
            "Apple", "revenue", "10B", valid_from=_year_ts(2020),
        )
        assert pid
        latest = graphlite_store.get_latest_property_version("Apple", "revenue")
        assert latest is not None
        assert latest["id"] == pid
        assert latest["entity_id"] == "Apple"
        assert latest["attr_name"] == "revenue"
        assert latest["value"] == "10B"
        assert latest["valid_from"] == pytest.approx(_year_ts(2020))
        assert graphlite_store.get_latest_property_version("Apple", "nope") is None

    def test_supersedes_chain_expired_and_edge(self, graphlite_store):
        """supersedes 链：新版本创建 → 旧版本 expired_at 打标 + SUPERSEDES 边。"""
        ts1 = _year_ts(2020)
        ts2 = _year_ts(2021)
        v1 = graphlite_store.create_property_version(
            "Apple", "revenue", "10B", valid_from=ts1,
        )
        v2 = graphlite_store.create_property_version(
            "Apple", "revenue", "20B", valid_from=ts2, supersedes_id=v1,
        )
        # 旧版本过期打标
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        by_id = {v["id"]: v for v in versions}
        assert by_id[v1]["expired_at"] == pytest.approx(ts2)
        assert by_id[v2]["expired_at"] in (None, "", "Null")
        # SUPERSEDES 血统边 v1 → v2
        assert _count_property_supersedes(graphlite_store, v1, v2) == 1
        # get_latest 返回新版本
        assert graphlite_store.get_latest_property_version("Apple", "revenue")["value"] == "20B"

    def test_resolver_same_value_idempotent(self, graphlite_store):
        """编排幂等：值相同 → no-op（不建新版本、不动旧版本）。"""
        resolver = EntityResolver(graphlite_store=graphlite_store)
        assert resolver._update_property_version(
            "Apple", "revenue", "10B", valid_from=_year_ts(2020),
        ) == 1
        assert resolver._update_property_version(
            "Apple", "revenue", "10B", valid_from=_year_ts(2021),
        ) == 0
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert len(versions) == 1

    def test_update_properties_from_triples(self, graphlite_store):
        """编排：RelationTriple.attributes → attr_name 派生（acquired_value）。"""
        resolver = EntityResolver(graphlite_store=graphlite_store)
        triples = [
            RelationTriple(
                subject="Apple", relation="ACQUIRED", obj="Beats",
                confidence=0.85, attributes={"value": "3B"},
            ),
            RelationTriple(
                subject="Google", relation="FOUNDED", obj="Alphabet",
                confidence=0.85, attributes={},  # 无属性 → 不建版本
            ),
        ]
        created = resolver.update_properties_from_triples(triples)
        assert created == 1
        latest = graphlite_store.get_latest_property_version("Apple", "acquired_value")
        assert latest is not None and latest["value"] == "3B"
        assert graphlite_store.get_latest_property_version("Google", "founded_value") is None
        # 空输入 / 无 store → 0
        assert resolver.update_properties_from_triples([]) == 0
        resolver_nostore = EntityResolver(graphlite_store=None)
        assert resolver_nostore.update_properties_from_triples(triples) == 0

    def test_valid_from_strictly_monotonic(self, graphlite_store):
        """同 valid_from 两次写入 → 第二次 bump（排序稳定，防同微秒 tie）。"""
        resolver = EntityResolver(graphlite_store=graphlite_store)
        resolver._update_property_version("Apple", "rev", "A", valid_from=1000.0)
        resolver._update_property_version("Apple", "rev", "B", valid_from=1000.0)
        versions = graphlite_store.get_property_versions("Apple", "rev")
        assert len(versions) == 2
        assert versions[0]["valid_from"] == pytest.approx(1000.0)
        assert float(versions[1]["valid_from"]) > 1000.0

    def test_prune_keeps_latest_8(self, graphlite_store):
        """N=8 惰性裁剪：10 版 → 最旧 2 版 DETACH DELETE，保留最近 8 版。"""
        resolver = EntityResolver(graphlite_store=graphlite_store)
        ts = 1000.0
        for i in range(1, 11):
            assert resolver._update_property_version(
                "Apple", "rev", f"v{i}", valid_from=ts,
            ) == 1
            ts += 1.0
        versions = graphlite_store.get_property_versions("Apple", "rev")
        assert len(versions) == 8
        # 保留最近 8 版（v3..v10），最旧 v1/v2 已删除
        assert _get_values(versions) == [f"v{i}" for i in range(3, 11)]


# ─── 8-12. 检索通道（公共入口 QueryRouter.retrieve）─────────────


class TestPropertyTemporalRetrieve:

    def _seed_versions(self, store):
        """Apple.revenue 版本链：v2020(10B) → v2021(20B) → v2022(30B)。"""
        resolver = EntityResolver(graphlite_store=store)
        resolver._update_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2020))
        resolver._update_property_version("Apple", "revenue", "20B", valid_from=_year_ts(2021))
        resolver._update_property_version("Apple", "revenue", "30B", valid_from=_year_ts(2022))

    def test_latest_mode_recent_query(self, graphlite_store):
        """query 含"最近" → 取最新未过期版本（30B）。"""
        self._seed_versions(graphlite_store)
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        out = router.retrieve("Apple 最近 收入是多少")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert props, "应召回属性版本"
        assert props[0]["entity_id"] == "Apple"
        assert props[0]["attr_name"] == "revenue"
        assert "30B" in props[0]["content"]

    def test_at_time_year_query(self, graphlite_store):
        """query 含具体年份"2021" → 取该时点前最新版本（20B）。"""
        self._seed_versions(graphlite_store)
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        out = router.retrieve("Apple 在 2021 年的收入")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert props and "20B" in props[0]["content"], \
            f"2021 应命中 valid_from=2021 版: {[p['content'] for p in props]}"

    def test_current_mode_no_time_word(self, graphlite_store):
        """无时间词 → current 模式取全部未过期版本（30B）。"""
        self._seed_versions(graphlite_store)
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        out = router.retrieve("Apple 收入")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert props and "30B" in props[0]["content"]

    def test_property_score_below_min_seed(self, graphlite_store):
        """假阳性护栏：扩展分 = min(种子分) × 0.6，严格低于种子分。"""
        self._seed_versions(graphlite_store)
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况", score=0.8),
            _seed_result("ep2", "iPhone 销售数据", score=0.5),
        ])
        out = router.retrieve("Apple 最近 收入")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert props and all(p["score"] < 0.5 for p in props)
        assert props[0]["_source"] == "property"
        assert all(p["score"] > 0.0 for p in props)

    def test_graphlite_failure_silent_degrade(self):
        """GraphLite 属性查询失败 → 静默降级返回原 results，主检索零回归。"""
        store = MagicMock()
        store.get_communities_by_seeds.side_effect = RuntimeError("graphlite down")
        store.get_property_versions_for_entities.side_effect = RuntimeError("graphlite down")
        router = _make_router(store, [_seed_result("s1", "content 1")])
        out = router.retrieve("Apple 最近 收入")
        assert [r["node_id"] for r in out] == ["s1"]
        assert all(r["level"] != "property_temporal" for r in out)

    def test_no_entity_candidate_channel_skipped(self, graphlite_store):
        """无实体候选查询（全小写/无组织词）→ 通道跳过，结果与种子一致。"""
        self._seed_versions(graphlite_store)
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "一些相关内容"),
        ])
        out = router.retrieve("最近有什么新消息吗")
        assert all(r["level"] != "property_temporal" for r in out)
        assert [r["node_id"] for r in out] == ["ep1"]


# ─── Codex 终审 5 缺陷回归 ─────────────────────────────────────


class TestP11YearValidFrom:

    def test_extractor_parses_year_attr(self):
        """P1-1: relation_extractor 解析 "in 2014" → attributes['attr_year']='2014'。"""
        from core.relation_extractor import RelationExtractor
        rext = RelationExtractor()
        triples = rext.extract(
            "Apple Inc acquired Beats Electronics for 3B dollars in 2014."
        )
        assert triples
        t = triples[0]
        assert t.relation == "ACQUIRED"
        assert t.attributes.get("attr_year") == "2014"
        assert t.attributes.get("value") == "3B"

    def test_extractor_year_scoped_to_sentence(self):
        """P1-1: 年份限定所在句子——跨句年份不误归当前三元组。"""
        from core.relation_extractor import RelationExtractor
        rext = RelationExtractor()
        triples = rext.extract(
            "Google acquired DeepMind in 2014. Apple Inc bought Beats Electronics "
            "for 3B dollars."
        )
        google = [t for t in triples if t.subject == "Google"]
        apple = [t for t in triples if t.subject == "Apple Inc"]
        assert google and google[0].attributes.get("attr_year") == "2014"
        assert apple and "attr_year" not in apple[0].attributes

    def test_year_attr_becomes_valid_from(self, graphlite_store):
        """P1-1: attr_year → valid_from（2014-01-01 ts），不单独建 attr_year 属性。"""
        resolver = EntityResolver(graphlite_store=graphlite_store)
        triples = [
            RelationTriple(
                subject="Apple Inc", relation="ACQUIRED", obj="Beats Electronics",
                confidence=0.85, attributes={"value": "3B", "attr_year": "2014"},
            ),
        ]
        created = resolver.update_properties_from_triples(triples)
        assert created == 1
        latest = graphlite_store.get_latest_property_version("Apple Inc", "acquired_value")
        assert latest is not None
        assert latest["value"] == "3B"
        assert latest["valid_from"] == pytest.approx(_year_ts(2014))
        # attr_year 不单独成属性版本
        assert graphlite_store.get_latest_property_version("Apple Inc", "acquired_attr_year") is None

    def test_year_attr_no_valid_from_when_missing(self, graphlite_store):
        """P1-1: 无 attr_year → valid_from 回落写入时刻（不传 None 之外的东西）。"""
        resolver = EntityResolver(graphlite_store=graphlite_store)
        triples = [
            RelationTriple(
                subject="Apple", relation="ACQUIRED", obj="Beats",
                confidence=0.85, attributes={"value": "3B"},
            ),
        ]
        resolver.update_properties_from_triples(triples)
        latest = graphlite_store.get_latest_property_version("Apple", "acquired_value")
        assert latest is not None
        assert latest["valid_from"] > _year_ts(2020)  # 近期时间戳


class TestP12EntityNormalization:

    def test_normalize_entity_name(self):
        """P1-2: 归一化小写 + 去尾词 Inc/Corp/Ltd 等（读写共用基准）。"""
        from core.entity_resolver import normalize_entity_name
        assert normalize_entity_name("Apple Inc") == "apple"
        assert normalize_entity_name("Apple") == "apple"
        assert normalize_entity_name("apple") == "apple"
        assert normalize_entity_name("Alphabet Corporation") == "alphabet"
        assert normalize_entity_name("Google LLC") == "google"
        assert normalize_entity_name("  IBM   Corp. ") == "ibm"

    def test_store_fuzzy_match_apple_inc(self, graphlite_store):
        """P1-2: store 层写 "Apple Inc"，查询候选 "Apple"/"apple" 前缀命中。"""
        from core.entity_resolver import normalize_entity_name
        graphlite_store.create_property_version(
            "Apple Inc", "revenue", "10B", valid_from=_year_ts(2020),
        )
        # 读侧候选归一化后 LIKE 'apple%' → 命中 "Apple Inc"
        rows = graphlite_store.get_property_versions_for_entities(["Apple"])
        assert len(rows) == 1
        assert rows[0]["entity_id"] == "Apple Inc"
        rows2 = graphlite_store.get_property_versions_for_entities(["apple"])
        assert len(rows2) == 1

    def test_retrieve_apple_hits_apple_inc(self, graphlite_store):
        """P1-2 集成：写入 "Apple Inc" → 查询 "Apple" 召回属性版本。"""
        resolver = EntityResolver(graphlite_store=graphlite_store)
        resolver._update_property_version(
            "Apple Inc", "revenue", "10B", valid_from=_year_ts(2020),
        )
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        out = router.retrieve("Apple 收入是多少")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert props, "Apple 应召回 Apple Inc 的属性版本"
        assert props[0]["entity_id"] == "Apple Inc"
        assert "10B" in props[0]["content"]


class TestP21AtTimeYearEnd:

    def test_at_time_mid_year_version_picked(self, graphlite_store):
        """P2-1: valid_from 在年中（2021-07）的版本，"2021 年"查询必须命中。

        修复前 at_ts=2021-01-01 → valid_from(2021-07) > at_ts 全跳过。
        """
        resolver = EntityResolver(graphlite_store=graphlite_store)
        resolver._update_property_version(
            "Apple", "revenue", "25B", valid_from=datetime(2021, 7, 1).timestamp(),
        )
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        out = router.retrieve("Apple 在 2021 年的收入")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert props and "25B" in props[0]["content"], \
            f"2021 年应命中年中版本: {[p['content'] for p in props]}"

    def test_at_time_excludes_expired_before_target(self, graphlite_store):
        """P2-1: at_time 校验 expired_at——目标时点已过期的旧版不返回。

        链: v2020(10B) → v2021(20B)。查 2022：v2021 已过期（expired_at=2022）
        则跳过，且不回落 v2020（其 valid_from 更早但同样已过期）。
        """
        resolver = EntityResolver(graphlite_store=graphlite_store)
        resolver._update_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2020))
        resolver._update_property_version("Apple", "revenue", "20B", valid_from=_year_ts(2021))
        # 显式构造：v2021 在 2022-06 过期（被 v2022 取代），但 v2022 不写入
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        v2021 = [v for v in versions if v["value"] == "20B"][0]
        graphlite_store.execute_cypher(
            "MATCH (p:PropertyVerNode {id: $id}) SET p.expired_at = $ts",
            {"id": v2021["id"], "ts": datetime(2022, 6, 1).timestamp()},
        )
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        # 2022 时点：v2021 expired_at(2022-06) > at_ts(2022-12-31)？否——at_ts 是年末，
        # expired 2022-06 < at_ts → 跳过
        out = router.retrieve("Apple 在 2022 年的收入")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert not props, f"2022 时点 v2021 已过期，不应返回: {[p['content'] for p in props]}"
        # 但 2021 年仍命中（expired 2022-06 > at_ts 2021-12-31）
        out2 = router.retrieve("Apple 在 2021 年的收入")
        props2 = [r for r in out2 if r["level"] == "property_temporal"]
        assert props2 and "20B" in props2[0]["content"]


class TestP22RelativeTimeWords:

    def test_relative_time_not_latest(self, graphlite_store):
        """P2-2: 昨天/earlier 不误归 latest——换算 at_ts 走 at_time。

        链: v2020(10B) → v2021(20B) → v2022(30B 最新)。查询 "Apple 昨天的收入"
        应命中 valid_from 最接近昨天但 ≤ 昨天的版本；若误归 latest 则取 30B（2022）。
        """
        resolver = EntityResolver(graphlite_store=graphlite_store)
        resolver._update_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2020))
        resolver._update_property_version("Apple", "revenue", "20B", valid_from=_year_ts(2021))
        resolver._update_property_version("Apple", "revenue", "30B", valid_from=_year_ts(2022))
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        out = router.retrieve("Apple 昨天的收入")
        props = [r for r in out if r["level"] == "property_temporal"]
        # 昨天 ts ≈ now-86400 >> 2022 → 命中最新未过期（30B）仍合理，但模式必须是 at_time
        assert props, "相对时间词应触发属性检索"
        # 直接验证模式判定（公共方法 _property_time_mode 是通道入口，非 mock 内部）
        mode, at_ts = router._property_time_mode("Apple 昨天的收入")
        assert mode == "at_time" and at_ts is not None
        assert at_ts < time.time()  # 过去时点

    def test_earlier_english_not_latest(self, graphlite_store):
        """P2-2: 英文 earlier → at_time 而非 latest（修复前 _time_keywords 误归 latest）。"""
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        mode, at_ts = router._property_time_mode("Apple earlier revenue")
        assert mode == "at_time" and at_ts is not None


class TestP23AtomicWriteCompensation:

    def _patch_execute_fail_on(self, store, fail_substr: str):
        """让 _locked_execute 在命中 fail_substr 的 GQL 时抛 QueryError。"""
        from graphlite_sdk.error import QueryError
        original = store._locked_execute
        calls = {"count": 0}
        def flaky(gql):
            calls["count"] += 1
            if fail_substr in gql:
                raise QueryError(f"simulated failure: {fail_substr}")
            return original(gql)
        store._locked_execute = flaky
        return original

    def test_compensation_when_set_expired_fails(self, graphlite_store):
        """P2-3: 旧版本 SET expired_at 失败 → 新版本回滚，链保持 v1 单版本。"""
        v1 = graphlite_store.create_property_version(
            "Apple", "revenue", "10B", valid_from=_year_ts(2020),
        )
        original = self._patch_execute_fail_on(graphlite_store, "SET p.expired_at")
        try:
            with pytest.raises(Exception):
                graphlite_store.create_property_version(
                    "Apple", "revenue", "20B", valid_from=_year_ts(2021), supersedes_id=v1,
                )
        finally:
            graphlite_store._locked_execute = original
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert len(versions) == 1, f"新版本应被回滚: {versions}"
        assert versions[0]["id"] == v1
        assert versions[0]["value"] == "10B"

    def test_compensation_when_supersedes_edge_fails(self, graphlite_store):
        """P2-3: SUPERSEDES 边插入失败 → 新版本回滚 + 旧版本 expired_at 恢复 NULL。"""
        v1 = graphlite_store.create_property_version(
            "Apple", "revenue", "10B", valid_from=_year_ts(2020),
        )
        original = self._patch_execute_fail_on(graphlite_store, "INSERT (a)-[:SUPERSEDES]")
        try:
            with pytest.raises(Exception):
                graphlite_store.create_property_version(
                    "Apple", "revenue", "20B", valid_from=_year_ts(2021), supersedes_id=v1,
                )
        finally:
            graphlite_store._locked_execute = original
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert len(versions) == 1
        assert versions[0]["id"] == v1
        # 旧版本过期标记已恢复（补偿 SET NULL）
        assert not QueryRouter._is_property_expired(versions[0])
        # get_latest 仍返回 v1（半链不存在）
        assert graphlite_store.get_latest_property_version("Apple", "revenue")["value"] == "10B"

    # ─── R6-P2：中段插入（supersedes_id + superseded_by）四个失败点注入 ───

    def _setup_mid_insert(self, store):
        """2021 → 2014（中段插入成功）→ 返回 (v2014, v2021)。"""
        v2021 = store.create_property_version(
            "Apple", "revenue", "20B", valid_from=_year_ts(2021),
        )
        v2014 = store.create_property_version(
            "Apple", "revenue", "10B", valid_from=_year_ts(2014),
            supersedes_id=v2021, superseded_by=v2021,
        )
        return v2014, v2021

    def test_mid_insert_rollback_set_pred_expired_fails(self, graphlite_store):
        """R6: 中段插入 SET pred.expired_at 失败 → 删新节点 + 恢复 P→S 边（链不断裂）。"""
        v2014, v2021 = self._setup_mid_insert(graphlite_store)
        original = self._patch_execute_fail_on(graphlite_store, "SET p.expired_at")
        try:
            with pytest.raises(Exception):
                graphlite_store.create_property_version(
                    "Apple", "revenue", "15B", valid_from=_year_ts(2016),
                    supersedes_id=v2014, superseded_by=v2021,
                )
        finally:
            graphlite_store._locked_execute = original
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert len(versions) == 2, f"新版本应被回滚: {versions}"
        # 链仍完整：2014→2021
        assert _count_property_supersedes(graphlite_store, v2014, v2021) == 1
        # 2014 expired_at 仍 ≈ 2021（未被 2016 覆盖）
        v2014b = next(v for v in versions if v["id"] == v2014)
        assert float(v2014b["expired_at"]) == pytest.approx(_year_ts(2021))

    def test_mid_insert_rollback_pred_new_edge_fails(self, graphlite_store):
        """R6: 中段插入 P→new 边失败 → 删新节点 + 恢复 P→S + pred.expired_at=succ_ts。"""
        v2014, v2021 = self._setup_mid_insert(graphlite_store)
        # 只失败第 1 次 SUPERSEDES INSERT（=P→new 边），补偿恢复 P→S 不注入失败
        original_lock = graphlite_store._locked_execute
        call_count = {"n": 0}
        def flaky_pred_new(gql):
            if "INSERT (a)-[:SUPERSEDES]" in gql:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    from graphlite_sdk.error import QueryError
                    raise QueryError("simulated failure: pred->new edge")
            return original_lock(gql)
        graphlite_store._locked_execute = flaky_pred_new
        try:
            with pytest.raises(Exception):
                graphlite_store.create_property_version(
                    "Apple", "revenue", "15B", valid_from=_year_ts(2016),
                    supersedes_id=v2014, superseded_by=v2021,
                )
        finally:
            graphlite_store._locked_execute = original_lock
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert len(versions) == 2
        # 链完整 + pred.expired_at 恢复 succ_ts（≈2021）
        assert _count_property_supersedes(graphlite_store, v2014, v2021) == 1
        v2014b = next(v for v in versions if v["id"] == v2014)
        assert float(v2014b["expired_at"]) == pytest.approx(_year_ts(2021))

    def test_mid_insert_rollback_set_new_expired_fails(self, graphlite_store):
        """R6: 中段插入 SET new.expired_at 失败 → 删新节点 + 恢复 P→S + pred.expired_at=succ_ts。"""
        v2014, v2021 = self._setup_mid_insert(graphlite_store)
        # 注意：SET p.expired_at 也匹配 SET pred——注入用更精确的 "SET p.expired_at = 17" 无法匹配
        # 数值注入。用计数器：第 2 次 SET p.expired_at 失败（第一次是 pred，第二次是 new）
        original_lock = graphlite_store._locked_execute
        call_count = {"n": 0}
        def flaky_set_new(gql):
            if "SET p.expired_at" in gql:
                call_count["n"] += 1
                if call_count["n"] == 2:  # 第二次 = new.expired_at
                    from graphlite_sdk.error import QueryError
                    raise QueryError("simulated failure: SET new expired_at")
            return original_lock(gql)
        graphlite_store._locked_execute = flaky_set_new
        try:
            with pytest.raises(Exception):
                graphlite_store.create_property_version(
                    "Apple", "revenue", "15B", valid_from=_year_ts(2016),
                    supersedes_id=v2014, superseded_by=v2021,
                )
        finally:
            graphlite_store._locked_execute = original_lock
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert len(versions) == 2
        assert _count_property_supersedes(graphlite_store, v2014, v2021) == 1
        v2014b = next(v for v in versions if v["id"] == v2014)
        assert float(v2014b["expired_at"]) == pytest.approx(_year_ts(2021))

    def test_mid_insert_rollback_new_succ_edge_fails(self, graphlite_store):
        """R6: 中段插入 new→succ 边失败 → 删新节点 + 恢复 P→S + pred.expired_at=succ_ts。"""
        v2014, v2021 = self._setup_mid_insert(graphlite_store)
        # 注入"第二个 SUPERSEDES INSERT"失败（第一个是 P→new，第二个是 new→S）
        original_lock = graphlite_store._locked_execute
        call_count = {"n": 0}
        def flaky_new_succ(gql):
            if "INSERT (a)-[:SUPERSEDES]" in gql:
                call_count["n"] += 1
                if call_count["n"] == 2:
                    from graphlite_sdk.error import QueryError
                    raise QueryError("simulated failure: new->succ edge")
            return original_lock(gql)
        graphlite_store._locked_execute = flaky_new_succ
        try:
            with pytest.raises(Exception):
                graphlite_store.create_property_version(
                    "Apple", "revenue", "15B", valid_from=_year_ts(2016),
                    supersedes_id=v2014, superseded_by=v2021,
                )
        finally:
            graphlite_store._locked_execute = original_lock
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert len(versions) == 2
        assert _count_property_supersedes(graphlite_store, v2014, v2021) == 1
        v2014b = next(v for v in versions if v["id"] == v2014)
        assert float(v2014b["expired_at"]) == pytest.approx(_year_ts(2021))

    def test_mid_insert_rollback_succ_read_exception(self, graphlite_store):
        """R7: 中段插入读取后继 valid_from 抛错 → 回滚新节点 + 异常传播。"""
        v2021 = graphlite_store.create_property_version(
            "Apple", "revenue", "20B", valid_from=_year_ts(2021),
        )
        original_exec = graphlite_store.execute_cypher
        def flaky_exec(query, params=None):
            if "RETURN s.valid_from" in query:
                from graphlite_sdk.error import QueryError
                raise QueryError("simulated failure: read successor")
            return original_exec(query, params)
        graphlite_store.execute_cypher = flaky_exec
        try:
            from graphlite_sdk.error import QueryError as _QErr
            with pytest.raises(_QErr):
                graphlite_store.create_property_version(
                    "Apple", "revenue", "10B", valid_from=_year_ts(2014),
                    supersedes_id=v2021, superseded_by=v2021,
                )
        finally:
            graphlite_store.execute_cypher = original_exec
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert len(versions) == 1, f"新节点应被回滚: {versions}"
        assert versions[0]["id"] == v2021

    def test_mid_insert_rollback_succ_read_empty(self, graphlite_store):
        """R7: 中段插入后继不存在（空结果）→ 一致性错误 + 回滚 + 异常传播。"""
        v2021 = graphlite_store.create_property_version(
            "Apple", "revenue", "20B", valid_from=_year_ts(2021),
        )
        original_exec = graphlite_store.execute_cypher
        def flaky_exec(query, params=None):
            if "RETURN s.valid_from" in query:
                return []  # 空结果：后继节点不存在
            return original_exec(query, params)
        graphlite_store.execute_cypher = flaky_exec
        try:
            from graphlite_sdk.error import QueryError as _QErr
            with pytest.raises(_QErr):
                graphlite_store.create_property_version(
                    "Apple", "revenue", "10B", valid_from=_year_ts(2014),
                    supersedes_id=v2021, superseded_by="ghost-succ-id",
                )
        finally:
            graphlite_store.execute_cypher = original_exec
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert len(versions) == 1, f"新节点应被回滚: {versions}"
        assert versions[0]["id"] == v2021


# ─── 13. HTTP 全链路（公共入口 POST /memories/episodes）────────


class TestWriteRouteIntegration:

    def test_write_route_creates_property_version(self, graphlite_store, client):
        """POST /memories/episodes（content > 80 含 ACQUIRED+金额）→ PropertyVerNode 落库。

        走生产链路：relation_extractor 正则 → triples.attributes → entity_resolver
        版本编排（qsubmit 无 write_queue 时同步直调）。
        """
        content = (
            "Apple Inc acquired Beats Electronics for 3B dollars in 2014 to "
            "strengthen its music streaming business and expand into the audio "
            "hardware market segment globally."
        )
        assert len(content) > 80
        resp = client(_svc(graphlite_store)).post("/memories/episodes", json={
            "content": content, "source": "user", "source_type": "direct",
        })
        assert resp.status_code == 200, resp.text
        # 属性版本已落库：entity_id=subject("Apple Inc")，attr=acquired_value
        versions = graphlite_store.get_property_versions("Apple Inc", "acquired_value")
        assert len(versions) == 1
        assert versions[0]["value"] == "3B"
        # P1-1: "in 2014" → valid_from ≈ 2014-01-01（时间维生产链路注入）
        assert versions[0]["valid_from"] == pytest.approx(_year_ts(2014)), \
            f"年份应注入 valid_from: {versions[0]['valid_from']}"
        # 当前有效（expired_at IS NULL）
        assert graphlite_store.get_latest_property_version(
            "Apple Inc", "acquired_value"
        )["value"] == "3B"
        # 重复写入同内容 → 幂等 no-op（版本数不变）
        resp2 = client(_svc(graphlite_store)).post("/memories/episodes", json={
            "content": content, "source": "user", "source_type": "direct",
        })
        assert resp2.status_code == 200
        assert len(graphlite_store.get_property_versions("Apple Inc", "acquired_value")) == 1

    def test_write_apple_inc_query_apple(self, graphlite_store, client):
        """P1-2 集成：写入 subject="Apple Inc"（entity_id 全名），查询 "Apple" 命中。

        修复前 _extract_query_entities 只提取 "Apple"，IN ['Apple'] 精确匹配
        无法命中 "Apple Inc"。
        """
        content = (
            "Apple Inc acquired Beats Electronics for 3B dollars in 2014 to "
            "strengthen its music streaming business and expand into the audio "
            "hardware market segment globally."
        )
        resp = client(_svc(graphlite_store)).post("/memories/episodes", json={
            "content": content, "source": "user", "source_type": "direct",
        })
        assert resp.status_code == 200, resp.text
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        out = router.retrieve("Apple 收购金额是多少")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert props, f"查询 Apple 应命中 Apple Inc 属性: {[p['content'] for p in props]}"
        assert props[0]["entity_id"] == "Apple Inc"
        assert "3B" in props[0]["content"]


# ─── P0-1-R2 Codex R2 复核 6 缺陷回归（N1-N6）──────────────────


class TestN1ChineseYear:

    def test_extractor_parses_cn_year(self):
        """N1: "2021年"（中文语境无词边界）→ attr_year='2021'。

        修复前 \\b(?:19|20)\\d{2}\\b 在 "2021年" 中 "1年" 无单词边界 → 不匹配。
        """
        from core.relation_extractor import RelationExtractor
        rext = RelationExtractor()
        triples = rext.extract("2021年，某科技公司收购了某游戏公司，花费 30亿元。")
        assert triples
        t = triples[0]
        assert t.attributes.get("attr_year") == "2021"
        assert t.attributes.get("value") == "30亿元"

    def test_cn_year_becomes_valid_from(self, graphlite_store):
        """N1: 中文年份 → valid_from=2021-01-01（生产链路注入）。"""
        resolver = EntityResolver(graphlite_store=graphlite_store)
        triples = [
            RelationTriple(
                subject="某科技公司", relation="ACQUIRED", obj="某游戏公司",
                confidence=0.85, attributes={"value": "30亿", "attr_year": "2021"},
            ),
        ]
        created = resolver.update_properties_from_triples(triples)
        assert created == 1
        latest = graphlite_store.get_latest_property_version(
            "某科技公司", "acquired_value"
        )
        assert latest is not None
        assert latest["valid_from"] == pytest.approx(_year_ts(2021))


class TestN2OutOfOrderWrite:

    def test_historical_write_does_not_reverse_chain(self, graphlite_store):
        """N2: 先写 2021(20B) 再写 2014(10B)——2014 不得抬成最新版。

        修复前 now < last_ts 时 now = last_ts + 0.001 → 2014 被标成最新，
        supersede 语义反向（2014 supersede 2021）。
        """
        resolver = EntityResolver(graphlite_store=graphlite_store)
        resolver._update_property_version("Apple", "revenue", "20B", valid_from=_year_ts(2021))
        resolver._update_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2014))
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert len(versions) == 2
        assert _get_values(versions) == ["10B", "20B"]  # ASC 时间序
        assert versions[0]["valid_from"] == pytest.approx(_year_ts(2014))
        assert versions[1]["valid_from"] == pytest.approx(_year_ts(2021))
        # 最新仍 20B（2014 未反向 supersede 2021——但按 Codex R3 血统链语义，
        # 2014 被 2021 取代：P→new→S 双挂链，new=2014 有后继 S=2021，
        # 建 (2014)-[:SUPERSEDES]->(2021) 边 + 2014.expired_at≈2021）
        assert graphlite_store.get_latest_property_version(
            "Apple", "revenue"
        )["value"] == "20B"
        assert float(versions[0]["expired_at"]) == pytest.approx(_year_ts(2021))
        assert _count_property_supersedes(
            graphlite_store, versions[0]["id"], versions[1]["id"]
        ) == 1

    def test_historical_write_chains_in_time_order(self, graphlite_store):
        """N2: 乱序三版本 2021 → 2014 → 2016 按时间序挂链。

        2016 的 supersedes 目标 = valid_from < 2016 的最新（2014），
        → 2014 打 expired_at（2014 被 2016 取代），2021 不受影响仍最新。
        """
        resolver = EntityResolver(graphlite_store=graphlite_store)
        resolver._update_property_version("Apple", "revenue", "20B", valid_from=_year_ts(2021))
        resolver._update_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2014))
        resolver._update_property_version("Apple", "revenue", "15B", valid_from=_year_ts(2016))
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert _get_values(versions) == ["10B", "15B", "20B"]
        v2014, v2016, v2021 = versions
        # 2014 被 2016 supersede（expired_at ≈ 2016）
        assert float(v2014["expired_at"]) == pytest.approx(_year_ts(2016))
        assert _count_property_supersedes(graphlite_store, v2014["id"], v2016["id"]) == 1
        # 2021 仍最新有效
        assert not QueryRouter._is_property_expired(v2021)
        assert graphlite_store.get_latest_property_version(
            "Apple", "revenue"
        )["value"] == "20B"

    def test_same_microsecond_still_bumps(self, graphlite_store):
        """N2: 仅 now == last_ts 时 bump（同微秒防重），语义保持。"""
        resolver = EntityResolver(graphlite_store=graphlite_store)
        resolver._update_property_version("Apple", "rev", "A", valid_from=1000.0)
        resolver._update_property_version("Apple", "rev", "B", valid_from=1000.0)
        versions = graphlite_store.get_property_versions("Apple", "rev")
        assert len(versions) == 2
        assert versions[0]["valid_from"] == pytest.approx(1000.0)
        assert float(versions[1]["valid_from"]) > 1000.0
        # B supersede A（同微秒 bump 等价常规更新）
        assert _count_property_supersedes(
            graphlite_store, versions[0]["id"], versions[1]["id"]
        ) == 1

    def test_out_of_order_mid_insert_full_chain(self, graphlite_store):
        """Codex R3 P1: 任意乱序 2021→2014→2016→2015 血统链完整（双挂链）。

        P→new→S 双向 SUPERSEDES + 双向 expired_at：
        - 2014 插入时后继=2021 → (2014)-[:SUPERSEDES]->(2021) + expired_at≈2021
        - 2016 插入时前驱=2014 后继=2021 → (2014)→(2016)→(2021) + 2014.expired_at≈2016
        - 2015 插入时前驱=2014 后继=2016 → (2014)→(2015)→(2016)
        - 2021 仍最新有效（无 expired_at）
        """
        resolver = EntityResolver(graphlite_store=graphlite_store)
        resolver._update_property_version("Apple", "revenue", "20B", valid_from=_year_ts(2021))
        resolver._update_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2014))
        resolver._update_property_version("Apple", "revenue", "16B", valid_from=_year_ts(2016))
        resolver._update_property_version("Apple", "revenue", "15B", valid_from=_year_ts(2015))
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert _get_values(versions) == ["10B", "15B", "16B", "20B"]
        v2014, v2015, v2016, v2021 = versions
        # 链：2014→2015→2016→2021
        assert _count_property_supersedes(graphlite_store, v2014["id"], v2015["id"]) == 1
        assert _count_property_supersedes(graphlite_store, v2015["id"], v2016["id"]) == 1
        assert _count_property_supersedes(graphlite_store, v2016["id"], v2021["id"]) == 1
        # 【R4-P1 负向断言】单链无分支：2014 不保留旧 2014→2021 边
        assert _count_property_supersedes(graphlite_store, v2014["id"], v2021["id"]) == 0
        # 【R5-P1 负向断言修正】真正覆盖「插入 2015 时删除了旧 2014→2016 边」：
        # 2014→2016 应已被 2014→2015 取代（2014 插入时建过 2014→2021，随后被
        # 2016 插入删除；2015→2021 天然不存在）
        assert _count_property_supersedes(graphlite_store, v2014["id"], v2016["id"]) == 0
        # expired_at 时间序：2014≈2015、2015≈2016、2016≈2021
        assert float(v2014["expired_at"]) == pytest.approx(_year_ts(2015))
        assert float(v2015["expired_at"]) == pytest.approx(_year_ts(2016))
        assert float(v2016["expired_at"]) == pytest.approx(_year_ts(2021))
        # 2021 最新有效
        assert not QueryRouter._is_property_expired(v2021)
        assert graphlite_store.get_latest_property_version(
            "Apple", "revenue"
        )["value"] == "20B"

    def test_same_value_historical_write_creates_version(self, graphlite_store):
        """N2-P2: 同值历史写入不静默丢弃——先 2021=10B 再 2014=10B 仍建 2014 版本。

        修复前 no-op 判定只比较最新版 value → 2014=10B 被 return 0 丢弃。
        """
        resolver = EntityResolver(graphlite_store=graphlite_store)
        resolver._update_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2021))
        resolver._update_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2014))
        versions = graphlite_store.get_property_versions("Apple", "revenue")
        assert len(versions) == 2
        assert _get_values(versions) == ["10B", "10B"]
        assert versions[0]["valid_from"] == pytest.approx(_year_ts(2014))


class TestN3CrossSentenceAmount:

    def test_attr_value_scoped_to_sentence(self):
        """N3: 属性金额限定句子窗口——跨句 "for 3B" 不误归第二条三元组。

        修复前 attr_pattern.search(text) 全文搜索 → Apple 拿到 DeepMind 的 3B。
        """
        from core.relation_extractor import RelationExtractor
        rext = RelationExtractor()
        triples = rext.extract(
            "Google acquired DeepMind for 3B dollars. "
            "Apple Inc bought Beats Electronics."
        )
        google = [t for t in triples if t.subject == "Google"]
        apple = [t for t in triples if t.subject == "Apple Inc"]
        assert google and google[0].attributes.get("value") == "3B"
        assert apple and "value" not in apple[0].attributes, \
            f"Apple 不应拿到跨句金额: {apple[0].attributes}"

    def test_attr_value_picked_from_own_sentence(self, graphlite_store):
        """N3: 同句金额正确命中 + valid_from 注入（单句内双三元组）。"""
        from core.relation_extractor import RelationExtractor
        rext = RelationExtractor()
        triples = rext.extract(
            "Google acquired DeepMind for 3B dollars in 2014. "
            "Apple Inc bought Beats Electronics for 500M dollars."
        )
        resolver = EntityResolver(graphlite_store=graphlite_store)
        created = resolver.update_properties_from_triples(triples)
        assert created == 2
        g = graphlite_store.get_latest_property_version("Google", "acquired_value")
        a = graphlite_store.get_latest_property_version("Apple Inc", "acquired_value")
        assert g is not None and g["value"] == "3B"
        assert a is not None and a["value"] == "500M", \
            f"Apple 应取本句 500M 而非跨句 3B: {a}"
        assert g["valid_from"] == pytest.approx(_year_ts(2014))


class TestN4RelativeTimeUnits:

    def test_last_year_parses_to_one_year_ago(self, graphlite_store):
        """N4: "last year" → 365 天前（修复前固定 1 天前）。"""
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        mode, at_ts = router._property_time_mode("Apple last year revenue")
        assert mode == "at_time" and at_ts is not None
        assert abs(at_ts - (time.time() - 365 * 86400)) < 60

    def test_last_month_parses_to_thirty_days_ago(self, graphlite_store):
        """N4: "last month" → 30 天前。"""
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        mode, at_ts = router._property_time_mode("Apple last month revenue")
        assert mode == "at_time"
        assert abs(at_ts - (time.time() - 30 * 86400)) < 60

    def test_today_uses_current_moment(self, graphlite_store):
        """N4: "今天" → 当前时刻（非当日 0 点，当日稍晚生效版本不丢）。"""
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        mode, at_ts = router._property_time_mode("Apple 今天的收入")
        assert mode == "at_time"
        assert abs(at_ts - time.time()) < 5

    def test_n_minutes_ago_chinese(self, graphlite_store):
        """N4: "5分钟前" → 5 分钟前（修复前仅字面"几分钟前"生效）。"""
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        mode, at_ts = router._property_time_mode("Apple 5分钟前的收入")
        assert mode == "at_time"
        assert abs(at_ts - (time.time() - 300)) < 5

    def test_n_minutes_ago_english(self, graphlite_store):
        """N4: "30 minutes ago" → 30 分钟前。"""
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        mode, at_ts = router._property_time_mode("Apple revenue 30 minutes ago")
        assert mode == "at_time"
        assert abs(at_ts - (time.time() - 1800)) < 5

    def test_year_query_not_misparsed_as_n_year_ago(self, graphlite_store):
        """N4: "2021 年"（无"前"后缀）不得被误判成 2021 年前 → 保持年份语义。"""
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        mode, at_ts = router._property_time_mode("Apple 在 2021 年的收入")
        assert mode == "at_time"
        assert at_ts == pytest.approx(_year_end_ts(2021))


class TestN5PropertyTermFilter:

    def _seed_two_attrs(self, store):
        """Apple 双属性：revenue(10B) + acquired_value(3B)。"""
        resolver = EntityResolver(graphlite_store=store)
        resolver._update_property_version("Apple", "revenue", "10B", valid_from=_year_ts(2020))
        resolver._update_property_version("Apple", "acquired_value", "3B", valid_from=_year_ts(2015))

    def test_query_property_word_filters_attr(self, graphlite_store):
        """N5: "Apple 收入" → 只返回 revenue，不返回 acquired_value。"""
        self._seed_two_attrs(graphlite_store)
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        out = router.retrieve("Apple 最近 收入")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert props, "应召回属性版本"
        assert all(p["attr_name"] == "revenue" for p in props), \
            f"属性词过滤后应只留 revenue: {[p['attr_name'] for p in props]}"

    def test_query_acquire_word_filters_attr(self, graphlite_store):
        """N5: "Apple 收购金额" → 只返回 acquired_value。"""
        self._seed_two_attrs(graphlite_store)
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        out = router.retrieve("Apple 最近 收购金额")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert props and all(p["attr_name"] == "acquired_value" for p in props)

    def test_no_property_word_no_filter(self, graphlite_store):
        """N5: 查询无属性词 → 不过滤（双属性都返回）。"""
        self._seed_two_attrs(graphlite_store)
        router = _make_router(graphlite_store, [
            _seed_result("ep1", "Apple 的业务情况"),
        ])
        out = router.retrieve("Apple 最近")
        props = [r for r in out if r["level"] == "property_temporal"]
        assert {p["attr_name"] for p in props} == {"revenue", "acquired_value"}


class TestN6PrefixBoundary:

    def test_prefix_filtered_out_similar_names(self, graphlite_store):
        """N6: "apple" 前缀不得命中 Applebee's / Applejack（词边界后置过滤）。"""
        graphlite_store.create_property_version(
            "Apple Inc", "revenue", "10B", valid_from=_year_ts(2020),
        )
        graphlite_store.create_property_version(
            "Applebee's", "revenue", "1B", valid_from=_year_ts(2020),
        )
        graphlite_store.create_property_version(
            "Applejack", "revenue", "2B", valid_from=_year_ts(2020),
        )
        rows = graphlite_store.get_property_versions_for_entities(["Apple"])
        assert len(rows) == 1, f"只应命中 Apple Inc: {[r['entity_id'] for r in rows]}"
        assert rows[0]["entity_id"] == "Apple Inc"

    def test_multi_word_suffix_still_matches(self, graphlite_store):
        """N6: "Apple Technologies Inc" 仍被 "apple" 命中（空格后缀兼容）。"""
        graphlite_store.create_property_version(
            "Apple Technologies Inc", "revenue", "10B", valid_from=_year_ts(2020),
        )
        rows = graphlite_store.get_property_versions_for_entities(["Apple"])
        assert len(rows) == 1
        assert rows[0]["entity_id"] == "Apple Technologies Inc"
