"""
手术刀 R2b 批量修复测试（v5.31.0）
==================================
覆盖：
  · P3  RESOLVE 记忆化：_find_and_merge_conflicts 嵌入按文本缓存 + Jaccard 预筛
       （jac < 0.15 跳过余弦）+ _resolve_step 包 to_thread
  · P4  BM25 构建 to_thread：fit_transform 经 asyncio.to_thread + _bm25_building 标志，
       构建期间 _bm25_search 返回旧索引/None 而非同步阻塞
  · P6  fusion 三路 ThreadPoolExecutor 并行：vector/bm25/entity 并发执行，
       CJK 跳过实体通道逻辑不破坏
  · P8  merge 截断：_persist_merge 先读现有 content，拼 (old + ' | merged: ' + v)[:2000] 再 SET
  · P9  types 缓存：_build_nx_graph 的 _extract_types 每节点只算一次

运行: python -m pytest tests/test_v531_perf_batch.py -v
"""
from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from core.dream_pipeline import (
    DreamPipeline,
    _JACCARD_PRESCREEN_THRESHOLD,
)
from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel


def run(coro):
    return asyncio.run(coro)


# ─── P3 RESOLVE 记忆化 ─────────────────────────────────────

class CountingEncoder:
    """计数 embed 调用次数的假编码器。"""

    def __init__(self, dim: int = 8):
        self.calls = 0
        self.dim = dim

    def embed(self, text: str):
        self.calls += 1
        # 确定性向量：同一文本 → 同一向量；文本越长向量越相似（便于测试预筛决策）
        rng = np.random.RandomState(hash(text) % (2**32))
        v = rng.rand(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


class TestP3ResolveMemoization:
    def _pipe(self, encoder=None):
        pipe = DreamPipeline(encoder=encoder)
        return pipe

    def test_combined_similarity_cache_skips_reencode(self):
        """同一文本重复比较只编码 1 次（emb_cache 命中）。"""
        enc = CountingEncoder()
        pipe = self._pipe(enc)
        cache: dict = {}
        # jac = 1.0（相同文本）≥ 0.15，走余弦路径
        sim = pipe._combined_similarity("完全相同内容", "完全相同内容", emb_cache=cache)
        assert sim >= 0.8
        assert enc.calls == 1, f"相同文本应只编码 1 次，实际 {enc.calls}"
        # 第二次比较：全部命中缓存，0 次新编码
        sim2 = pipe._combined_similarity("完全相同内容", "完全相同内容", emb_cache=cache)
        assert sim2 == sim
        assert enc.calls == 1, f"缓存命中后不应再编码，实际 {enc.calls}"

    def test_jaccard_prescreen_skips_encoder(self):
        """jac < 0.15 预筛跳过余弦：encoder.embed 零调用，返回纯 jac。"""
        enc = CountingEncoder()
        pipe = self._pipe(enc)
        cache: dict = {}
        sim = pipe._combined_similarity(
            "alpha beta gamma delta epsilon",
            "zeta eta theta iota kappa",
            emb_cache=cache,
        )
        jac = pipe._jaccard_similarity(
            "alpha beta gamma delta epsilon",
            "zeta eta theta iota kappa",
        )
        assert jac < _JACCARD_PRESCREEN_THRESHOLD
        assert sim == jac, "预筛跳过余弦后应返回纯 Jaccard"
        assert enc.calls == 0, f"预筛跳过不应调用 encoder，实际 {enc.calls}"

    def test_prescreen_decision_equivalent_with_encoder(self):
        """随机文本对：带预筛（新）与全量余弦（旧）的合并决策一致。

        旧逻辑（无条件编码）：sim_old = 0.4·jac + 0.6·cos
        新逻辑（预筛跳过）：jac < 0.15 时返回 jac；否则同旧逻辑。
        预筛返回 jac ≤ 0.15 < 0.8（合并阈值）→ 两版是否 ≥ 0.8 恒一致。
        """
        rng = np.random.RandomState(42)
        pipe = self._pipe(None)

        class _DummyEncoder:
            """固定向量编码器：cos = 0.9（相似文本会拉高 sim 至 ≥0.8 候选）。"""

            def embed(self, text: str):
                return np.array([0.9, 0.3], dtype=np.float32)

        pipe.encoder = _DummyEncoder()

        def _old_sim(a: str, b: str) -> float:
            """旧逻辑（不预筛、不缓存）：无条件编码 + 余弦。"""
            jac = pipe._jaccard_similarity(a, b)
            emb_a = pipe.encoder.embed(a)
            emb_b = pipe.encoder.embed(b)
            norm_a = np.linalg.norm(emb_a)
            norm_b = np.linalg.norm(emb_b)
            cos = float(np.dot(emb_a, emb_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
            return 0.4 * jac + 0.6 * max(0.0, cos)

        words = ["apple", "banana", "cherry", "date", "elderberry", "fig"]
        for _ in range(30):
            a = " ".join(rng.choice(words, size=3, replace=False))
            b = " ".join(rng.choice(words, size=3, replace=False))
            jac = pipe._jaccard_similarity(a, b)
            old = _old_sim(a, b)
            # 新逻辑：预筛 + 缓存
            new = pipe._combined_similarity(a, b, emb_cache={})
            # 决策等价：是否 ≥ 0.8（合并阈值）两版一致
            assert (old >= 0.8) == (new >= 0.8), (
                f"决策不一致: old={old:.3f} new={new:.3f} jac={jac:.3f}"
            )

    def test_resolve_step_runs_in_thread(self):
        """_resolve_step 整块包 asyncio.to_thread：不在事件循环内同步执行。"""
        pipe = DreamPipeline()
        ran_in_thread = []

        async def fake_to_thread(fn, *args, **kwargs):
            ran_in_thread.append(fn.__name__)
            return fn(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            run(pipe.run(
                nodes=[{"id": "n1", "content": "x", "created_at": time.time()},
                       {"id": "n2", "content": "x", "created_at": time.time()}],
                connections={},
                trigger_mode="explicit",
                graphlite_store=MagicMock(),
                candidate_store=None,
            ))

        assert "_resolve_step" in ran_in_thread, (
            f"_resolve_step 应经 to_thread 执行，实际 {ran_in_thread}"
        )


# ─── P4 BM25 构建 to_thread ─────────────────────────────────

class FakeStore:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def query_cypher(self, *args, **kwargs) -> list[dict]:
        return self.rows


def _bare_router(store) -> QueryRouter:
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router.graphlite_store = store
    router._bm25_doc_ids = []
    router._bm25_doc_contents = []
    router._bm25_doc_tau = []
    router._bm25_built = False
    router._bm25_ready = False
    router._bm25_last_attempt = 0.0
    router._bm25_building = False
    router._bm25_empty_warned = False
    router._bm25_vectorizer = None
    return router


class TestP4Bm25ToThread:
    def test_async_build_runs_fit_in_thread(self):
        """_build_bm25_index_async：fit_transform 经 asyncio.to_thread 执行。"""
        store = FakeStore([{"node_id": "n1", "content": "记忆系统测试", "tau_value": 1.0}])
        router = _bare_router(store)
        ran_in_thread = []

        async def fake_to_thread(fn, *args, **kwargs):
            ran_in_thread.append(fn.__name__)
            return fn(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            run(router._build_bm25_index_async())

        assert "_build_bm25_index_core" in ran_in_thread, (
            f"BM25 构建应经 to_thread 执行，实际 {ran_in_thread}"
        )
        assert router._bm25_ready and router._bm25_built
        assert router._bm25_search("记忆"), "异步构建后 BM25 应可检索"

    def test_building_flag_returns_empty_without_blocking(self):
        """_bm25_building 置位期间 _bm25_search 返回 [] 而非阻塞等待构建。"""
        store = FakeStore([{"node_id": "n1", "content": "记忆系统测试", "tau_value": 1.0}])
        router = _bare_router(store)
        router._bm25_building = True
        entered = []

        def spy_build():
            entered.append(True)
            return None

        router._build_bm25_index = spy_build  # type: ignore[method-assign]
        results = router._bm25_search("记忆")
        assert results == [], "构建中应返回空（不阻塞）"
        assert entered == [], f"构建中不应触发构建，实际 {entered}"

    def test_building_flag_reuses_old_index(self):
        """构建进行中但旧索引已就绪 → 直接用旧索引检索（不空等）。"""
        store = FakeStore([{"node_id": "n1", "content": "记忆系统测试", "tau_value": 1.0}])
        router = _bare_router(store)
        # 先构建出旧索引
        router._build_bm25_index()
        assert router._bm25_ready
        # 模拟新一轮构建进行中
        router._bm25_building = True
        entered = []
        router._build_bm25_index = lambda: entered.append(True)  # type: ignore[method-assign]
        results = router._bm25_search("记忆")
        assert results, "旧索引可用时应直接检索"
        assert entered == [], "旧索引可用时不触发重建"

    def test_concurrent_builds_no_double(self):
        """_bm25_building 标志防并发构建：第二路构建直接跳过。"""
        store = FakeStore([{"node_id": "n1", "content": "记忆系统测试", "tau_value": 1.0}])
        router = _bare_router(store)
        calls = {"core": 0}

        orig_core = router._build_bm25_index_core

        def counting_core():
            calls["core"] += 1
            return orig_core()

        router._build_bm25_index_core = counting_core  # type: ignore[method-assign]
        # 手动模拟第一路构建进行中
        router._bm25_building = True
        router._build_bm25_index()
        assert calls["core"] == 0, "构建中再调 _build_bm25_index 应直接跳过"

    def test_sync_build_still_works(self):
        """同步 _build_bm25_index 保持兼容（test_bm25_chinese 依赖）。"""
        store = FakeStore([{"node_id": "n1", "content": "记忆系统测试", "tau_value": 1.0}])
        router = _bare_router(store)
        router._build_bm25_index()
        assert router._bm25_ready and router._bm25_built
        assert router._bm25_search("记忆")


# ─── P6 fusion 三路并行 ─────────────────────────────────────

class _BlockingChannel:
    """协作式 mock：记录进入状态，阻塞至另一通道进入才返回（验证并行）。"""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.entered.set()
        # 等待对侧进入（若对侧永远不进入 → 死锁超时保护）
        self.release.wait(timeout=2.0)
        return []


def _fusion_router() -> QueryRouter:
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig()
    router._cjk_warned = False
    return router


class TestP6FusionParallel:
    def test_three_channels_run_concurrently(self):
        """vector/bm25/entity 三通道并发：A 阻塞时 B 仍能进入。"""
        router = _fusion_router()
        vector = _BlockingChannel()
        bm25 = _BlockingChannel()
        entity = _BlockingChannel()

        router._vector_retrieve = vector  # type: ignore[method-assign]
        router._bm25_search = bm25  # type: ignore[method-assign]
        router._entity_match = entity  # type: ignore[method-assign]
        router._fuse_results = lambda *a, **k: []

        def run_fusion():
            result = router._fusion_retrieve("memory system", raw_query="memory system")
            # 等所有通道都进入过（证明并行：A 阻塞时 B/C 已进入）
            for ch in (vector, bm25, entity):
                ch.release.set()
            return result

        t = threading.Thread(target=run_fusion)
        t.start()
        try:
            # 等 vector 通道进入（阻塞中）
            assert vector.entered.wait(timeout=2.0), "vector 通道未进入"
            # vector 阻塞中，bm25/entity 应已进入（并行）—— 若串行会卡在 vector
            assert bm25.entered.wait(timeout=2.0), "bm25 通道未并发进入（串行？）"
            assert entity.entered.wait(timeout=2.0), "entity 通道未并发进入（串行？）"
        finally:
            for ch in (vector, bm25, entity):
                ch.release.set()
        t.join(timeout=5.0)
        assert not t.is_alive(), "fusion 应正常完成"

    def test_cjk_skips_entity_task(self):
        """CJK 查询：不提交 entity 任务，且一次性 warning 保留。"""
        router = _fusion_router()
        entity_calls = {"n": 0}
        router._vector_retrieve = lambda *a, **k: []  # type: ignore[method-assign]
        router._bm25_search = lambda *a, **k: []  # type: ignore[method-assign]
        router._fuse_results = lambda *a, **k: []  # type: ignore[method-assign]

        def fake_entity(*a, **k):
            entity_calls["n"] += 1
            return []

        router._entity_match = fake_entity  # type: ignore[method-assign]
        router._fusion_retrieve("记忆系统测试", raw_query="记忆系统测试")
        assert entity_calls["n"] == 0, f"CJK 查询不应进入 entity 通道，实际 {entity_calls['n']}"
        assert router._cjk_warned is True, "CJK 一次性 warning 标志应置位"

    def test_ascii_still_runs_entity(self):
        """英文查询：entity 通道照常执行。"""
        router = _fusion_router()
        entity_calls = {"n": 0}
        router._vector_retrieve = lambda *a, **k: []  # type: ignore[method-assign]
        router._bm25_search = lambda *a, **k: []  # type: ignore[method-assign]
        router._fuse_results = lambda *a, **k: []  # type: ignore[method-assign]

        def fake_entity(*a, **k):
            entity_calls["n"] += 1
            return []

        router._entity_match = fake_entity  # type: ignore[method-assign]
        router._fusion_retrieve("memory system test", raw_query="memory system test")
        assert entity_calls["n"] == 1, f"英文查询应进入 entity 通道，实际 {entity_calls['n']}"

    def test_channel_exception_degrades(self):
        """单通道异常不影响其他通道（try/except 降级保留）。"""
        router = _fusion_router()

        def boom(*a, **k):
            raise RuntimeError("channel down")

        router._vector_retrieve = boom  # type: ignore[method-assign]
        router._bm25_search = lambda *a, **k: [{"node_id": "b1", "content": "bm25", "score": 0.5}]  # type: ignore[method-assign]
        router._entity_match = lambda *a, **k: []  # type: ignore[method-assign]
        router._fuse_results = lambda *a, **k: "FUSED"  # type: ignore[method-assign]
        assert router._fusion_retrieve("memory", raw_query="memory") == "FUSED"

    def test_entity_channel_queryerror_tags_degraded(self):
        """P1-1 loud-store 旁路用例：store 自身 query_cypher 直接抛 SDK QueryError →
        融合结果打 _degradation_level=fusion_channel_skip。

        【P3a R7】此用例降级为 loud-store 旁路（store 直接抛异常的旁路），不再是
        降级信号唯一依据：真实 GraphLiteStore.query_cypher 有永不抛异常契约（返回
        [] + thread-local 信号），由 TestP3aR7EntityDegradation 走真实 query_cypher
        契约验证（打桩 _locked_query，不替换 query_cypher）。
        """
        from graphlite_sdk.error import QueryError

        router = _fusion_router()
        router._vector_retrieve = lambda *a, **k: [{"node_id": "v1", "content": "vector hit", "score": 0.9}]  # type: ignore[method-assign]
        router._bm25_search = lambda *a, **k: []  # type: ignore[method-assign]

        class _BoomStore:
            def query_cypher(self, *a, **k):
                raise QueryError("entity channel down")

        router.graphlite_store = _BoomStore()  # type: ignore[attr-defined]
        router._fuse_results = lambda *a, **k: [{"node_id": "v1", "content": "vector hit", "score": 0.9}]  # type: ignore[method-assign]

        result = router._fusion_retrieve("memory system", raw_query="memory system")
        assert any(
            r.get("_degradation_level") == "fusion_channel_skip" for r in result
        ), "entity 通道抛 QueryError 应使融合结果带 fusion_channel_skip 降级标记"


# ─── P3a R7: entity 通道降级信号（thread-local）──────────────────

class _FakeRows:
    def __init__(self, rows):
        self.rows = rows


class TestP3aR7EntityDegradation:
    """P3a R7：thread-local 降级信号——query_cypher 永不抛异常契约下，基础设施
    降级（熔断 open / 重试耗尽）以返回 [] 表现，_entity_match 读同线程标志区分
    「正常无匹配」与「基础设施降级」。测试打桩 _locked_query 走真实 query_cypher
    契约（不替换 query_cypher，杜绝假绿）。"""

    def _store(self):
        from graph.graphlite_store import GraphLiteStore
        return GraphLiteStore()

    def test_retry_exhausted_sets_degraded_flag(self):
        """重试耗尽：_locked_query 抛 QueryError → query_cypher 返回 [] 且标志 True。"""
        from graphlite_sdk.error import QueryError

        store = self._store()
        store._locked_query = MagicMock(side_effect=QueryError("infra down"))
        with patch("time.sleep", return_value=None):
            rows = store.query_cypher("MATCH (n) RETURN n")
        assert rows == []
        assert store.last_query_infra_degraded() is True

    def test_circuit_open_sets_degraded_flag(self):
        """熔断 open：allow_request False → query_cypher 返回 [] 且标志 True。"""
        store = self._store()
        store.circuit_breaker.allow_request = lambda: False
        rows = store.query_cypher("MATCH (n) RETURN n")
        assert rows == []
        assert store.last_query_infra_degraded() is True

    def test_success_clears_degraded_flag(self):
        """成功路径：_locked_query 返回 rows → query_cypher 返回 rows 且标志 False。"""
        store = self._store()
        store._locked_query = MagicMock(return_value=_FakeRows([{"node_id": "n1"}]))
        rows = store.query_cypher("MATCH (n) RETURN n")
        assert rows == [{"node_id": "n1"}]
        assert store.last_query_infra_degraded() is False

    def test_app_error_does_not_flag_degraded(self):
        """应用错误：_locked_query 抛 RuntimeError → query_cypher 返回 [] 且标志 False（不误报）。"""
        store = self._store()
        store._locked_query = MagicMock(side_effect=RuntimeError("bad gql"))
        rows = store.query_cypher("MATCH (n) RETURN n")
        assert rows == []
        assert store.last_query_infra_degraded() is False

    def test_fusion_entity_infra_degradation_tags_channel_skip(self):
        """端到端（关键）：真实 store 注入 QueryRouter 走 _fusion_retrieve（ThreadPoolExecutor），
        断言融合结果带 fusion_channel_skip——验证 thread-local 在同池线程正确传递。

        不 monkeypatch _entity_match / query_cypher：_locked_query 抛 QueryError →
        真实 query_cypher 重试耗尽置 thread-local 信号 → _entity_match 读同线程信号
        抛 _EntityChannelDegraded → per-channel handler 打标（杜绝假绿）。
        """
        from graphlite_sdk.error import QueryError

        store = self._store()
        store._locked_query = MagicMock(side_effect=QueryError("entity channel down"))

        router = _fusion_router()
        router.graphlite_store = store
        router._vector_retrieve = lambda *a, **k: [{"node_id": "v1", "content": "vector hit", "score": 0.9}]  # type: ignore[method-assign]
        router._bm25_search = lambda *a, **k: []  # type: ignore[method-assign]
        router._fuse_results = lambda *a, **k: [{"node_id": "v1", "content": "vector hit", "score": 0.9}]  # type: ignore[method-assign]

        with patch("time.sleep", return_value=None):
            result = router._fusion_retrieve("memory system", raw_query="memory system")

        assert any(
            r.get("_degradation_level") == "fusion_channel_skip" for r in result
        ), "entity 通道基础设施降级应使融合结果带 fusion_channel_skip 降级标记"


# ─── P8 merge 截断 ─────────────────────────────────────────

class RecordingStore:
    """记录 query_cypher 调用的假 store，模拟现有 content。"""

    def __init__(self, existing_content: str = ""):
        self.existing_content = existing_content
        self.calls: list[tuple[str, dict]] = []
        self.deleted: list[str] = []
        self.archived: list[tuple[str, str]] = []

    def query_cypher(self, cypher: str, params: dict):
        self.calls.append((cypher, params))
        if "SET target.content" in cypher:
            return []
        if "DETACH DELETE" in cypher:
            self.deleted.append(params["id"])
            return []
        # 读现有 content
        return [{"content": self.existing_content}]

    def archive_node(self, node_id: str, replacement_id: str = None) -> bool:
        self.archived.append((node_id, replacement_id))
        return True

    def last_set_params(self):
        for cypher, params in reversed(self.calls):
            if "SET target.content" in cypher:
                return params
        return None


class TestP8MergeTruncation:
    def _op(self, loser="loser", winner="winner", content="被合并内容"):
        from core.audit_chain import AuditOperation
        return AuditOperation(
            op_type="update",
            node_id=loser,
            old_value=content,
            new_value=winner,
            reason="community_merge",
        )

    def test_merge_content_truncated_to_2000(self):
        """连续合并后 content 不超过 2000 字符，前段不丢。"""
        store = RecordingStore(existing_content="base-" + "x" * 1900)
        pipe = DreamPipeline()
        op = self._op(content="append-" + "y" * 500)
        pipe._persist_merge(store, [op])

        params = store.last_set_params()
        assert params is not None, "应执行 SET target.content = $content"
        merged = params["content"]
        assert len(merged) <= 2000, f"merge 后 content 应 ≤2000，实际 {len(merged)}"
        assert merged.startswith("base-"), "前段内容不应丢失"
        assert "| merged:" in merged

    def test_merge_reads_then_sets(self):
        """每 merge 多 1 次读 + 1 次 SET（不再用 GQL 内联拼接）。"""
        store = RecordingStore(existing_content="old")
        pipe = DreamPipeline()
        op = self._op(content="new")
        pipe._persist_merge(store, [op])

        reads = [c for c, _ in store.calls if "RETURN target.content AS content" in c]
        sets = [c for c, _ in store.calls if "SET target.content = $content" in c]
        assert len(reads) == 1, f"应读 1 次现有 content，实际 {len(reads)}"
        assert len(sets) == 1, f"应 SET 1 次，实际 {len(sets)}"
        assert "target.content + ' | merged: '" not in " ".join(c for c, _ in store.calls), (
            "不应再使用 GQL 内联拼接（无界追加路径已移除）"
        )
        assert store.archived == [("loser", "winner")], "被合并节点应归档并建 SUPERSEDES 血统边"

    def test_merge_no_existing_content(self):
        """目标节点不存在/无 content → 空 base 正常拼接。"""
        store = RecordingStore(existing_content="")
        pipe = DreamPipeline()
        pipe._persist_merge(store, [self._op(content="新内容")])
        params = store.last_set_params()
        assert params["content"].startswith(" | merged: 新内容")


# ─── P9 types 缓存 ─────────────────────────────────────────

class FakeOntologyValidator:
    """计数 _extract_types 调用次数的假 validator。

    与真实 _extract_types 语义一致：文本命中多个实体类型时全部返回。
    """

    def __init__(self):
        self.calls = 0

    def _extract_types(self, text: str):
        self.calls += 1
        result = []
        if "PyTorch" in text:
            result.append({"entity": "PyTorch", "type": "ml_framework"})
        if "CPU" in text or "cpu" in text:
            result.append({"entity": "CPU", "type": "hardware"})
        return result


class TestP9TypesCache:
    def test_extract_types_called_once_per_node(self):
        """_extract_types 每节点只调 1 次（不再 O(N²) 正则）。"""
        validator = FakeOntologyValidator()
        pipe = DreamPipeline(ontology_validator=validator)

        # 3 个节点：node1( PyTorch ) + node2( CPU ) + node3( PyTorch )
        # 修复前：每对内调用 2 次 _extract_types → 3 对 × 2 = 6 次
        # 修复后：每节点 1 次 → 3 次
        G = pipe._build_nx_graph(
            [
                {"id": "n1", "content": "PyTorch"},
                {"id": "n2", "content": "CPU"},
                {"id": "n3", "content": "PyTorch"},
            ],
            {},
        )
        assert validator.calls == 3, (
            f"_extract_types 应按节点只调 3 次，实际 {validator.calls}"
        )
        # 共享类型边：n1-n3 共享 ml_framework → 1 条本体边
        assert G.has_edge("n1", "n3"), "共享类型节点间应添加本体边"
        assert not G.has_edge("n1", "n2"), "无共享类型不建边"

    def test_types_cache_edge_set_equivalent(self):
        """小图 old/new 边集全等（缓存重构纯等价）。"""
        validator = FakeOntologyValidator()
        pipe = DreamPipeline(ontology_validator=validator)
        nodes = [
            {"id": "n1", "content": "PyTorch"},
            {"id": "n2", "content": "CPU"},
            {"id": "n3", "content": "PyTorch CPU"},
        ]
        G = pipe._build_nx_graph(nodes, {})

        # 手工计算期望：n1(PyTorch) n2(CPU) n3(PyTorch+CPU)
        # n1∩n2 = ∅, n1∩n3 = {ml_framework}, n2∩n3 = {hardware}
        assert G.has_edge("n1", "n3")
        assert G.has_edge("n2", "n3")
        assert not G.has_edge("n1", "n2")
        assert len([e for e in G.edges]) == 2, f"期望 2 条本体边，实际 {G.edges}"
