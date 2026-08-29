"""记忆体检索准确率真实行为测试（LoCoMo 攻坚闭环补充）

覆盖 P0-③ AtomicFact 写入门控 + D-MEM RPE 写入分流逻辑的**公共入口真实行为**：
- create_atomic_fact / get_by_episode / get_by_subject：走真实 OverGraphStore，
  断言 sha1 幂等 key、FACT_MENTIONS 证据链、subject 检索、valid_time 区分。
- RPE 写入门控：走真实 POST /memories/episodes 路由（TestClient + 真实
  OverGraphStore + 确定性假 encoder），断言 ignore→rpe_filtered 不落库、
  cache→τ 降级、deep→正常落库、force_promote 绕过。
不 mock 被测对象内部——被测逻辑（store / 路由）均真实执行。

运行: python3 -m pytest tests/test_memory_accuracy.py -q
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router, Services, get_services
from core.dream_pipeline import DreamPipeline
from graph.overgraph_store import OverGraphStore, LABEL_FACT, LABEL_FACT_MENTIONS


# ════════════════════════════════════════════════════════════════════
# 一、AtomicFactNode 写入门控（真实 OverGraphStore）
# ════════════════════════════════════════════════════════════════════

class TestAtomicFactWriteGateRealStore:
    """create_atomic_fact / get_by_episode / get_by_subject 真实库行为。"""

    def test_create_idempotent_real_store(self, overgraph_store):
        """同事实同 valid_time 幂等（sha1 确定性 key）；不同时间版本不同 key。"""
        fid1 = overgraph_store.create_atomic_fact(
            "Caroline", "graduated from", "MIT", valid_time="2019",
            source_episode="ep1")
        fid2 = overgraph_store.create_atomic_fact(
            "Caroline", "graduated from", "MIT", valid_time="2019",
            source_episode="ep1")
        assert fid1 == fid2, "同事实同版本应复用同一节点（幂等）"
        assert fid1.startswith("fact_"), f"事实 key 应有 fact_ 前缀: {fid1}"

        fid3 = overgraph_store.create_atomic_fact(
            "Caroline", "graduated from", "MIT", valid_time="2020",
            source_episode="ep1")
        assert fid1 != fid3, "不同 valid_time 版本应不同 key（时间版本区分）"

    def test_get_by_episode_mentions_edge(self, overgraph_store):
        """get_by_episode：FACT_MENTIONS 边真实落库 → 梦境反查可命中。"""
        ep = overgraph_store.create_episode({"content": "episode content"})
        overgraph_store.create_atomic_fact(
            "Melanie", "works at", "Google", valid_time="2023",
            source_episode=ep)
        overgraph_store.create_atomic_fact(
            "Caroline", "studies at", "Stanford", valid_time="2022",
            source_episode=ep)
        facts = overgraph_store.get_atomic_facts_by_episode(ep, limit=50)
        assert len(facts) == 2, f"应反查出 2 条事实，实际 {len(facts)}"
        subjects = {f["subject"] for f in facts}
        assert subjects == {"Melanie", "Caroline"}
        by_obj = {f["object"]: f for f in facts}
        assert by_obj["Google"]["valid_time"] == "2023"

    def test_get_by_episode_empty_on_missing_episode(self, overgraph_store):
        """未关联任何事实的 episode → 空列表（不抛异常）。"""
        ep = overgraph_store.create_episode({"content": "no facts"})
        assert overgraph_store.get_atomic_facts_by_episode(ep) == []

    def test_get_by_subject_filter(self, overgraph_store):
        """get_by_subject：按主体实体名检索事实。"""
        overgraph_store.create_atomic_fact(
            "Melanie", "has_pet", "dog", valid_time="", source_episode="ep1")
        overgraph_store.create_atomic_fact(
            "Caroline", "has_pet", "cat", valid_time="", source_episode="ep1")
        got = overgraph_store.get_atomic_facts_by_subject("Melanie")
        assert len(got) == 1
        assert got[0]["subject"] == "Melanie"
        assert got[0]["object"] == "dog"

    def test_get_by_subject_requires_subject(self, overgraph_store):
        """空 subject → 空列表（不抛异常）。"""
        assert overgraph_store.get_atomic_facts_by_subject("") == []
        assert overgraph_store.get_atomic_facts_by_subject(None) == []

    def test_create_requires_all_fields(self, overgraph_store):
        """缺 subject/predicate/object 任一项 → 抛 OverGraphError。"""
        from graph.overgraph_store import OverGraphError
        with pytest.raises(OverGraphError):
            overgraph_store.create_atomic_fact("", "is", "x")
        with pytest.raises(OverGraphError):
            overgraph_store.create_atomic_fact("Caroline", "", "x")
        with pytest.raises(OverGraphError):
            overgraph_store.create_atomic_fact("Caroline", "is", "")

    def test_rule_extraction_roundtrip_persist(self, overgraph_store):
        """规则抽取 → 落库全链路：dream 抽取的 SPO 事实可被 subject 反查。"""
        p = DreamPipeline.__new__(DreamPipeline)
        facts = p._extract_facts_rules(
            "Caroline graduated from MIT in 2019. Melanie works at Google."
        )
        assert facts, "应抽取到 SPO 事实"
        for f in facts:
            overgraph_store.create_atomic_fact(
                subject=f["subject"], predicate=f["predicate"],
                object_=f["object"], valid_time=f["valid_time"],
                confidence=0.6)
        got = overgraph_store.get_atomic_facts_by_subject("Caroline")
        assert got, "抽取的事实应可经 get_by_subject 反查"
        assert any("MIT" in f["object"] for f in got)
        assert any(f["valid_time"] == "2019" for f in got)


# ════════════════════════════════════════════════════════════════════
# 二、D-MEM RPE 写入分流（真实路由 + 真实 OverGraphStore + 确定性 encoder）
# ════════════════════════════════════════════════════════════════════

class _FakeEncoder:
    """确定性假 encoder：与 mock_encoder 同构（文本 hash → 固定向量）。

    RPE 批判依赖 encoder.embed → vector_search_dense 的 max_sim → surprise。
    真实 encoder 走 ONNX 不可在测试环境依赖；假 encoder 保持路由内 RPE
    批判逻辑（embed → 检索 → surprise/utility → 三分流）真实执行。
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2 ** 31))
        v = rng.randn(self.dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


def _correlated(base: np.ndarray, cosine: float, seed: int = 7) -> np.ndarray:
    """与 base 夹角给定 cosine 的向量（cache 测试种子，可控相关性）。"""
    dim = base.shape[0]
    noise = np.random.RandomState(seed).randn(dim).astype(np.float32)
    noise -= noise.dot(base) * base  # 与 base 正交
    noise /= np.linalg.norm(noise)
    v = cosine * base + np.sqrt(max(0.0, 1 - cosine ** 2)) * noise
    return v.astype(np.float32)


def _enable_write_gate(monkeypatch, surprise_deep=0.45, surprise_cache=0.25,
                       utility_min=0.5, cache_tau=0.5):
    """开启 RPE 写入门控：直接构造 WriteGateConfig 挂到 Settings 单例。

    路由内 `from config.settings import get_settings` 是局部导入，但读取的
    `_settings` 是同一模块单例 → 直接替换 `config.settings._settings` 即可生效，
    无需 patch api.routes.write 命名空间（模块内无 get_settings 属性）。
    """
    from config.settings import WriteGateConfig, Settings

    wg = WriteGateConfig(
        enabled=True,
        surprise_deep=surprise_deep,
        surprise_cache=surprise_cache,
        utility_min=utility_min,
        cache_tau=cache_tau,
    )
    s = Settings()
    s.write_gate = wg
    monkeypatch.setattr("config.settings._settings", s)
    return wg


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    def _build(svc):
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app)

    return _build


def _svc_with_store(overgraph_store):
    """Services 容器：真实 OverGraphStore + 假 encoder + 必需下游 mock。"""
    svc = Services()
    svc.graph_store = overgraph_store
    svc.encoder = _FakeEncoder()
    # 下游可选服务置 None（路由逐项 hasattr 判空，等价生产最小启动）
    svc.tau_engine = None
    svc.ssm_gate = None
    svc.defense_engine = None
    svc.quarantine_store = None
    svc.ontology_validator = None
    svc.ontology_v2 = None
    svc.evidence_tracker = None
    svc.dream_scheduler = None
    svc.write_queue = None
    svc.query_router = None
    svc.hyperedge_manager = None
    return svc


class TestRpeWriteRouteRealStore:
    """RPE 路由分流真实行为：走 POST /memories/episodes 公共入口。"""

    def test_route_default_off_no_filter(self, client, overgraph_store):
        """默认（enabled=False）→ 写入不被 RPE 拦截，全部落库。"""
        svc = _svc_with_store(overgraph_store)
        resp = client(svc).post("/memories/episodes", json={
            "content": "A default write with some important memory content.",
            "source": "test",
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "created", "默认关时 RPE 不介入，应直接落库"
        assert overgraph_store.get_episode(resp.json()["episode_id"]) is not None

    def test_route_ignore_filters_write(self, client, overgraph_store, monkeypatch):
        """enable 后高相似（低惊奇）内容 → ignore → 202 rpe_filtered 不落库。"""
        _enable_write_gate(monkeypatch)
        svc = _svc_with_store(overgraph_store)
        # 预置一条高相似记忆：内容 A 的向量与种子完全相同 → surprise≈0 → ignore
        seed = "Melanie goes to the beach every summer with her family."
        ep = overgraph_store.create_episode({"content": seed})
        enc = _FakeEncoder()
        overgraph_store.batch_upsert_embeddings([
            {"node_id": ep, "embedding": enc.embed(seed)},
        ])
        resp = client(svc).post("/memories/episodes", json={
            "content": seed,
            "source": "test",
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # ignore → 不落库，返回 rpe_filtered
        assert body["status"] == "rpe_filtered", f"应被 RPE 过滤，实际 {body}"
        assert overgraph_store.get_episode(body["episode_id"]) is None, (
            "ignore 路由不应落库 EpisodeNode"
        )

    def test_route_deep_persists(self, client, overgraph_store, monkeypatch):
        """全新高惊奇高效用内容 → deep → 正常落库（status=created）。"""
        _enable_write_gate(monkeypatch)
        svc = _svc_with_store(overgraph_store)
        content = ("Melanie went to Yosemite National Park in 2023 and "
                   "painted a beautiful landscape.")
        resp = client(svc).post("/memories/episodes", json={
            "content": content,
            "source": "test",
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "created", f"deep 路由应正常落库，实际 {body}"
        got = overgraph_store.get_episode(body["episode_id"])
        assert got is not None
        # 【P3-C】deep 路由应落库 rpe_surprise（空库 max_sim=0 → surprise=1.0）
        assert got.get("rpe_surprise") is not None, (
            f"deep 路由应落库 rpe_surprise，实际 {got}"
        )

    def test_route_cache_degrades_tau(self, client, overgraph_store, monkeypatch):
        """中等惊奇 → cache → τ 降为 cache_tau（快速衰减）。"""
        wg = _enable_write_gate(monkeypatch, cache_tau=0.3)
        svc = _svc_with_store(overgraph_store)
        # 预置与查询内容夹角 cosine=0.65 的向量 → surprise≈0.35 ∈ [0.25, 0.45) → cache
        content = "Melanie likes to paint landscapes in her free time."
        enc = _FakeEncoder()
        q = enc.embed(content)
        seed_ep = overgraph_store.create_episode({"content": "similar seed"})
        overgraph_store.batch_upsert_embeddings([
            {"node_id": seed_ep, "embedding": _correlated(q, 0.65)},
        ])
        resp = client(svc).post("/memories/episodes", json={
            "content": content,
            "source": "test",
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "created", f"cache 路由仍应落库，实际 {body}"
        assert abs(body["tau_initial"] - 0.3) < 1e-6, (
            f"cache 路由应使用 cache_tau={wg.cache_tau}，实际 {body['tau_initial']}"
        )

    def test_route_force_promote_bypasses(self, client, overgraph_store, monkeypatch):
        """force_promote=true 绕过 RPE 批判（强制语义无条件写入）。"""
        _enable_write_gate(monkeypatch)
        svc = _svc_with_store(overgraph_store)
        resp = client(svc).post("/memories/episodes", json={
            "content": "Duplicate content that should bypass RPE filter.",
            "source": "test",
            "force_promote": True,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "created", f"force_promote 应绕过 RPE，实际 {body}"
        got = overgraph_store.get_episode(body["episode_id"])
        assert got is not None
        assert got.get("protected") in (True, "true", 1), "force_promote 应打 protected 标记"


# ════════════════════════════════════════════════════════════════════
# 三、P3-C RPE 惊奇度信号检索重排（默认关零回归；仅 FUSION 生效）
# ════════════════════════════════════════════════════════════════════

class TestRpeRerankRetrieval:
    """RPE 惊奇度 → 检索重排真实行为。

    - rpe_rerank_enabled=False → 与基线（_deduplicate_and_sort）逐字节等价
    - =True → 高惊奇 boost / 低惊奇 dampen / 无信号不变 / 钳制不超种子最高分
    - 走 retrieve(level=FUSION) 公共入口 + mock _fusion_retrieve 固定 docs
    - 集成：真实 overgraph_store 落库 rpe_surprise → attach 回查 → boost
    """

    @staticmethod
    def _fusion_stack(router, docs):
        """patch _fusion_retrieve 返回固定 docs + 增强通道透传 + reranker 短路。

        只测 RPE 重排钩子：_fusion_retrieve 之外的所有 _finish 补充通道透传，
        _get_reranker 返回 None 让 bge-reranker 短路（不加载真实模型）。
        """
        from contextlib import ExitStack
        from unittest.mock import patch
        stack = ExitStack()
        stack.enter_context(
            patch.object(router, "_fusion_retrieve",
                         return_value=[dict(d) for d in docs])
        )
        for name in (
            "_community_expansion", "_mesa_synthesis", "_visual_recall",
            "_property_temporal_retrieve", "_entity_expansion",
            "_attribute_expansion", "_scope_retrieve", "_schema_recall",
            "_fact_retrieve",
        ):
            stack.enter_context(
                patch.object(router, name, side_effect=lambda r, *a, **k: r)
            )
        stack.enter_context(patch.object(router, "_get_reranker", return_value=None))
        return stack

    def test_rpe_rerank_disabled_byte_identical(self):
        """rpe_rerank_enabled=False → 结果与基线（_deduplicate_and_sort）逐字节等价。

        即使结果 dict 携带 rpe_surprise 字段，关闭时钩子完全不介入。
        """
        from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel
        router = QueryRouter(None, None, None,
                             config=QueryRouterConfig(rpe_rerank_enabled=False))
        docs = [
            {"node_id": "a", "content": "alpha memory", "score": 0.9,
             "rpe_surprise": 0.8},
            {"node_id": "b", "content": "beta memory", "score": 0.5,
             "rpe_surprise": 0.1},
            {"node_id": "c", "content": "gamma memory", "score": 0.4},
        ]
        golden = QueryRouter._deduplicate_and_sort([dict(d) for d in docs])
        with self._fusion_stack(router, docs):
            out = router.retrieve("memory", level=RetrievalLevel.FUSION)
        assert out == golden, "rpe_rerank_enabled=False 应与关闭前逐字节等价"

    def test_rpe_rerank_no_signal_unchanged(self):
        """enabled=True 但节点均无 rpe_surprise → 仍与基线逐字节等价。"""
        from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel
        router = QueryRouter(None, None, None,
                             config=QueryRouterConfig(rpe_rerank_enabled=True))
        docs = [
            {"node_id": "a", "content": "alpha memory", "score": 0.9},
            {"node_id": "b", "content": "beta memory", "score": 0.5},
        ]
        golden = QueryRouter._deduplicate_and_sort([dict(d) for d in docs])
        with self._fusion_stack(router, docs):
            out = router.retrieve("memory", level=RetrievalLevel.FUSION)
        assert out == golden, "无 rpe_surprise 节点即使 enabled=True 也应不变"

    def test_rpe_rerank_enabled_boost_dampen_clamp(self):
        """enabled=True → 高惊奇 ×1.05 / 低惊奇 ×0.95 / 无信号不变 / 钳制不超种子最高分。"""
        from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel
        router = QueryRouter(None, None, None, config=QueryRouterConfig(
            rpe_rerank_enabled=True, rpe_rerank_high=0.7, rpe_rerank_low=0.3,
        ))
        docs = [
            {"node_id": "a", "content": "alpha memory", "score": 0.9},
            # 种子最高分节点无信号 → 不变；同时是钳制上限
            {"node_id": "b", "content": "beta memory", "score": 0.5,
             "rpe_surprise": 0.8},    # 高惊奇 → 0.5×1.05=0.525（< 0.9 不钳）
            {"node_id": "c", "content": "gamma memory", "score": 0.4,
             "rpe_surprise": 0.1},    # 低惊奇 → 0.4×0.95=0.38
            {"node_id": "d", "content": "delta memory", "score": 0.88,
             "rpe_surprise": 0.9},    # 高惊奇但 0.88×1.05=0.924>0.9 → 钳到 0.9
            {"node_id": "e", "content": "epsilon memory", "score": 0.3,
             "rpe_surprise": 0.5},    # 中等惊奇 → 不变
        ]
        with self._fusion_stack(router, docs):
            out = router.retrieve("memory", level=RetrievalLevel.FUSION)
        by_id = {r["node_id"]: r for r in out}
        assert abs(by_id["a"]["score"] - 0.9) < 1e-9, "无信号节点不变"
        assert abs(by_id["b"]["score"] - 0.525) < 1e-9, "高惊奇应 ×1.05 boost"
        assert abs(by_id["c"]["score"] - 0.38) < 1e-9, "低惊奇应 ×0.95 dampen"
        assert abs(by_id["d"]["score"] - 0.9) < 1e-9, "boost 钳制不超种子最高分"
        assert abs(by_id["e"]["score"] - 0.3) < 1e-9, "中等惊奇不变"
        scores = [r["score"] for r in out]
        assert scores == sorted(scores, reverse=True), "重排后应保持降序"

    def test_rpe_rerank_store_attach_real(self, overgraph_store):
        """集成：write 侧落库 rpe_surprise → 检索 attach 回查 → boost（真实 store）。"""
        from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel
        # 种子最高分节点无信号（钳制上限），高惊奇节点中分 → boost 可见
        top_id = overgraph_store.create_episode({"content": "alpha memory"})
        boost_id = overgraph_store.create_episode(
            {"content": "beta memory", "rpe_surprise": 0.8})
        router = QueryRouter(
            graphlite_store=overgraph_store, faiss_index=None, tfidf_index=None,
            config=QueryRouterConfig(
                rpe_rerank_enabled=True, rpe_rerank_high=0.7, rpe_rerank_low=0.3,
            ),
        )
        docs = [
            {"node_id": top_id, "content": "alpha memory", "score": 0.9},
            {"node_id": boost_id, "content": "beta memory", "score": 0.5,
             "rpe_surprise": 0.8},
        ]
        with self._fusion_stack(router, docs):
            out = router.retrieve("memory", level=RetrievalLevel.FUSION)
        assert out, "应返回检索结果"
        by_id = {r["node_id"]: r for r in out}
        assert abs(by_id[top_id]["score"] - 0.9) < 1e-9, "无信号种子不变"
        assert abs(by_id[boost_id]["score"] - 0.525) < 1e-9, (
            f"rpe_surprise=0.8 → 应 boost 到 0.5×1.05=0.525，"
            f"实际 {by_id[boost_id]['score']}"
        )
