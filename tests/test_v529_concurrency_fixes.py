"""
v5.29.0 梦境与写库并发卡死修复测试
=================================
覆盖三项修复的关键路径：

· F1 — core/dream_scheduler.py: _run_dream 内两处 overgraph_store.query_cypher
        （EpisodeNode 拉取 + HEBBIAN_CONNECTION 拉取）改为 asyncio.to_thread，
        慢查询不再阻塞事件循环。
        · test_dream_fetch_uses_to_thread: 慢查询在工作线程（非事件循环线程）执行
        · test_dream_fetch_loop_responsive: 慢查询期间事件循环心跳正常（无 ≈5s 卡顿）

· F2 — core/dream_pipeline.py: 新增可选构造参数 write_queue + 助手 _persist_async；
        直接模式（candidate_store=None）PERSIST 步骤（_persist_prune + 社区
        切块 _persist_one_community/阶段3 _persist_communities_prune_edges +
        _persist_merge/_persist_hyperedges）经 write_queue 串行提交，纯函数
        _persist_merge_get_removed 不入队；无 write_queue 时回退 asyncio.to_thread。
        · test_persist_routed_through_write_queue: 步骤按序入队 + 纯函数不入队
        · test_persist_fallback_to_thread_without_queue: 无队列时回退 to_thread

· F5 — graph/overgraph_store.py: _session_lock = threading.RLock() + 全部
        _session.query/execute 统一经 _locked_query/_locked_execute 串行化。
        · test_session_lock_is_rlock: 锁类型为 RLock（可重入，写线程内嵌套不死锁）
        · test_concurrent_session_access_serialized: 4 线程 × 5 写无异常/无挂起/无丢写

运行: python -m pytest tests/test_v529_concurrency_fixes.py -q
"""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.dream_pipeline import DreamPipeline, DreamReport
from core.dream_scheduler import DreamScheduler, DreamSchedulerConfig, TriggerMode


def run(coro):
    """同步运行异步协程（测试用）。"""
    return asyncio.run(coro)


def _make_report(dream_id: str = "d1") -> DreamReport:
    return DreamReport(
        dream_id=dream_id,
        trigger_mode="explicit",
        timestamp=time.time(),
        duration_seconds=0.01,
        stats={"created": 1, "updated": 0, "deleted": 0},
        community_count=1,
        prune_count=0,
        conflict_count=0,
        audit_block_hash="",
    )


async def _fast_pipeline(*args, **kwargs):
    """快速成功的伪管道。"""
    return _make_report()


class _RecordingQueue:
    max_pending = 100

    def pending_count(self) -> int:
        return 0
    """记录 submit 的 fn 名并按同步方式执行（模拟单写线程）。"""

    def __init__(self):
        self.calls: list[str] = []

    async def submit(self, fn, *args, **kwargs):
        # 【v5.40】与真实 WriteQueue 一致：priority 是队列参数，不传给 fn
        kwargs.pop("priority", None)
        self.calls.append(fn.__name__)
        return fn(*args, **kwargs)  # 同步执行，模拟写线程


def _extract_ids(rows) -> set:
    """GraphLite 行格式兼容解析（RETURN e.id → {'e.id': ...} / list）。"""
    out = set()
    for r in rows:
        if isinstance(r, dict):
            v = r.get("e.id") or r.get("id") or next(iter(r.values()), "")
            out.add(str(v))
        elif isinstance(r, (list, tuple)):
            out.add(str(r[0]))
    return out


class TestF1DreamFetchToThread:
    """F1: _run_dream 的 query_cypher 拉取不阻塞事件循环。"""

    def test_dream_fetch_uses_to_thread(self):
        """慢查询记录执行线程 id → 断言全部 != 事件循环线程（to_thread 工作线程执行）。"""
        entered = threading.Event()
        release = threading.Event()
        idents: list[int] = []

        def slow_query(*args, **kwargs):
            idents.append(threading.get_ident())
            entered.set()
            release.wait(5)  # 模拟慢查询阻塞工作线程，等主协程释放
            return []

        sched = DreamScheduler(
            config=DreamSchedulerConfig(),
            pipeline_fn=_fast_pipeline,
        )
        sched._graphlite_store = SimpleNamespace(query_cypher=slow_query)

        async def main():
            loop_ident = threading.get_ident()
            task = asyncio.create_task(sched._run_dream(TriggerMode.IDLE))
            # 等慢查询进入 to_thread 工作线程（最多 2s），再释放避免测试挂起
            await asyncio.to_thread(entered.wait, 2)
            release.set()
            await asyncio.wait_for(task, 5)
            return loop_ident, list(idents)

        loop_ident, idents = run(main())

        assert idents, "slow query 应被调用（两次拉取都经 to_thread）"
        assert all(i != loop_ident for i in idents), (
            f"query_cypher 在事件循环线程执行（回归为同步直调）: idents={idents} "
            f"loop_ident={loop_ident}"
        )

    def test_dream_fetch_loop_responsive(self):
        """慢查询期间事件循环保持响应：sleep(0.05) 实际耗时 < 0.5s（同步直调会 ≈ 5s）。"""
        entered = threading.Event()
        release = threading.Event()

        def slow_query(*args, **kwargs):
            entered.set()
            release.wait(5)  # 阻塞工作线程 5s
            return []

        sched = DreamScheduler(
            config=DreamSchedulerConfig(),
            pipeline_fn=_fast_pipeline,
        )
        sched._graphlite_store = SimpleNamespace(query_cypher=slow_query)

        async def main():
            task = asyncio.create_task(sched._run_dream(TriggerMode.IDLE))
            # 确保慢查询已进入工作线程（最多 2s）
            await asyncio.to_thread(entered.wait, 2)
            t0 = time.monotonic()
            await asyncio.sleep(0.05)
            elapsed = time.monotonic() - t0
            release.set()
            await asyncio.wait_for(task, 5)
            return elapsed

        elapsed = run(main())

        assert elapsed < 0.5, (
            f"事件循环被同步 query_cypher 阻塞（回归）: sleep(0.05) 实际耗时 "
            f"{elapsed:.3f}s（同步直调会 ≈ 5s）"
        )


class TestF2PersistWriteQueue:
    """F2: 梦境 PERSIST 经 write_queue 串行 / 无队列回退 to_thread。"""

    @staticmethod
    def _make_store():
        store = MagicMock()
        store.query_cypher.return_value = []
        store.execute_cypher.return_value = []
        return store

    def test_persist_routed_through_write_queue(self):
        """直接模式 PERSIST 步骤按序经 write_queue；纯函数不入队。

        【v5.40】社区 PERSIST 切块：nodes=[] → CLUSTER 产出 1 个空社区
        （members=[]）→ 恰 1 次 _persist_one_community 块；空社区无成员 →
        member_sets 空 → 无阶段 3 _persist_communities_prune_edges。剩余
        步骤（prune/merge/hyperedges）按序入队。"""
        q = _RecordingQueue()
        pipe = DreamPipeline(write_queue=q)
        store = self._make_store()

        report = run(pipe.run(
            nodes=[],
            connections={},
            trigger_mode="explicit",
            graphlite_store=store,
            candidate_store=None,
        ))

        assert q.calls == [
            "_persist_prune",
            "_persist_one_community",
            "_persist_merge",
            "_persist_hyperedges",
            "_persist_entities",
            "_persist_schema_evolution",
            "_persist_atomic_facts",
        ], f"write_queue 提交顺序/集合不符: {q.calls}"
        assert "_persist_communities_prune_edges" not in q.calls, (
            "空社区（members=[]）不应触发阶段 3 湮灭"
        )
        assert "_persist_merge_get_removed" not in q.calls, (
            "纯函数 _persist_merge_get_removed 不应经 write_queue"
        )
        assert report is not None and report.degraded is False

    def test_persist_entities_runs_in_candidate_mode(self):
        """【v6.3.1】候选模式下实体落库 + Schema 演化也执行（幂等只增写）。

        v6.2.0 P0-① 生产缺陷：生产用候选模式（candidate_store 非 None），
        PERSIST 直接模式（PRUNE/MERGE/HYPEREDGES）不跑 → _persist_entities
        永不落库 → EntityNode=0 → Schema 自演化（P0-②）无消费对象。
        修复：候选分支同样经 write_queue 提交幂等的实体落库步骤。
        """
        q = _RecordingQueue()
        pipe = DreamPipeline(write_queue=q)
        store = self._make_store()
        candidate_store = MagicMock()

        report = run(pipe.run(
            nodes=[],
            connections={},
            trigger_mode="explicit",
            graphlite_store=store,
            candidate_store=candidate_store,
        ))

        assert "_persist_entities" in q.calls, q.calls
        assert "_persist_schema_evolution" in q.calls, q.calls
        # 破坏性操作不因候选模式提前执行（仍经 apply 人工放行）
        assert "_persist_prune" not in q.calls, q.calls
        assert "_persist_merge" not in q.calls, q.calls
        candidate_store.save_candidate.assert_called_once()
        assert report is not None

    def test_persist_fallback_to_thread_without_queue(self):
        """无 write_queue 时 _persist_async 回退 asyncio.to_thread（run 正常完成）。"""
        pipe = DreamPipeline()
        store = self._make_store()
        called: list[str] = []

        async def fake_to_thread(fn, *args, **kwargs):
            called.append(fn.__name__)
            return fn(*args, **kwargs)

        # 注意：CLUSTER 步同样走 asyncio.to_thread，patch 会一并记录
        # （含 _cluster_step），因此断言用 `in` 而非精确集合相等。
        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            report = run(pipe.run(
                nodes=[],
                connections={},
                trigger_mode="explicit",
                graphlite_store=store,
                candidate_store=None,
            ))

        assert "_persist_prune" in called, (
            f"回退路径未生效: asyncio.to_thread 调用记录={called}"
        )
        assert report is not None and report.degraded is False


class TestF5SessionLock:
    """F5: GraphLiteStore session 访问锁（RLock 可重入 + 并发写串行化）。"""

    def test_session_lock_is_rlock(self, overgraph_store):
        """_session_lock 应为 threading.RLock（写线程内嵌套调用不死锁）。

        注意：threading.RLock 在本环境是工厂函数而非类，isinstance 第二参
        用 RLock() 实例的类型。
        """
        assert isinstance(overgraph_store._session_lock, type(threading.RLock()))

    def test_concurrent_session_access_serialized(self, overgraph_store):
        """4 线程 × 5 次 create_episode：无异常、无挂起、20 条全部落库。"""
        errors: list[str] = []

        def write(i: int) -> None:
            try:
                for j in range(5):
                    overgraph_store.create_episode({
                        "id": f"ep-{i}-{j}",
                        "content": f"episode {i}-{j}",
                        "created_at": time.time(),
                        "tau_initial": 1.0,
                        "tau_value": 1.0,
                        "source": "v529",
                    })
            except Exception as e:  # noqa: BLE001 — 并发冒烟必须捕获一切异常
                errors.append(f"thread-{i}: {e!r}")

        threads = [threading.Thread(target=write, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert errors == [], f"并发写入抛出异常: {errors[:5]}"
        assert all(not t.is_alive() for t in threads), "并发写入线程挂起（>15s 未退出）"

        rows = overgraph_store.query_cypher("MATCH (e:EpisodeNode) RETURN e.id", {})
        ids = _extract_ids(rows)
        expected = {f"ep-{i}-{j}" for i in range(4) for j in range(5)}
        assert expected <= ids, f"丢写: 缺失 {sorted(expected - ids)}"
