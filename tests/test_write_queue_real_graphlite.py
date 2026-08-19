"""v5.23 写串行化队列 — 真实 GraphLite 冒烟 + 基准（Codex F1/F2/F3 条件）。

Codex 审核条件（.trio-review-v523.md）:
  · F1: 同一 GraphLite session 被事件循环读线程（query_cypher）+ 专用写线程
        （WriteQueue）**并发**访问的安全性 — 真实冒烟：写线程写 + loop 读
        1000 次，无挂起/损坏/丢写。
  · F2: _persist_hyperedge 边 INSERT 走 execute_cypher（不吞异常）——边写入
        失败必须上抛，不得落入 query_cypher 永不抛异常契约静默返回 []。
  · F3: 验收口径重定义——端到端延迟受单写者物理上限（N × 单写耗时）约束，
        队列本身零额外开销：8 并发 × 80 条真实写总耗时 ≈ 串行基线。

真实引擎 → 临时库（conftest graphlite_store fixture），不触碰真实记忆。

运行: python -m pytest tests/test_write_queue_real_graphlite.py -v
"""
import asyncio
import time
import uuid
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.graphlite  # 【v6.0.0 legacy】GraphLite 专属语义测试（默认排除，addopts -m 'not graphlite'）

from core.write_queue import WriteQueue
from graph.hyperedge import HyperedgeManager


def _make_episode(tag) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "content": f"smoke {tag} 中文",
        "created_at": time.time(),
        "tau_initial": 1.0,
        "tau_value": 0.5,
        "source": "smoke",
        "trust_score": 0.8,
    }


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


class TestRealReadWriteConcurrency:
    """F1: 真实 GraphLite 读写并发冒烟（计划最高优先级风险）。"""

    def test_write_thread_writes_loop_reads_concurrent(self, graphlite_store):
        """写线程写 + 事件循环读并发：无挂起、无损坏、无丢写、loop 不被阻塞。"""
        store = graphlite_store
        q = WriteQueue(wait_timeout=30.0)
        try:
            n_writes, n_reads = 30, 1000
            written: list[str] = []
            read_rows = [0]
            errors: list[str] = []
            heartbeats: list[float] = []

            def write_ep(tag):
                ep = _make_episode(tag)
                store.create_episode(ep)
                return ep["id"]

            async def heartbeat():
                for _ in range(50):
                    heartbeats.append(time.monotonic())
                    await asyncio.sleep(0.01)

            async def reader_loop():
                for i in range(n_reads):
                    try:
                        rows = store.query_cypher(
                            "MATCH (e:EpisodeNode) RETURN e.id LIMIT 50", {}
                        )
                        read_rows[0] += len(rows)
                    except Exception as e:  # noqa: BLE001 — 冒烟必须捕获一切异常
                        errors.append(str(e))
                    if i % 100 == 0:
                        await asyncio.sleep(0)

            async def writer_loop():
                for i in range(n_writes):
                    written.append(await q.submit(write_ep, i))

            async def main():
                hb = asyncio.create_task(heartbeat())
                t0 = time.monotonic()
                await asyncio.gather(reader_loop(), writer_loop())
                dt = time.monotonic() - t0
                await hb
                return dt

            dt = asyncio.run(main())

            # 无 SDK 异常 / 挂起（有界完成）
            assert errors == [], f"concurrent reads raised: {errors[:5]}"
            assert dt < 10.0, f"smoke too slow ({dt:.1f}s) — possible cross-thread hang"
            # 读确实看到数据（并发期间有节点可读）
            assert read_rows[0] > 0, "loop reads returned no rows during concurrent writes"
            # 无丢写：全部排队写已持久化
            persisted = _extract_ids(store.query_cypher("MATCH (e:EpisodeNode) RETURN e.id", {}))
            assert len(written) == n_writes
            assert set(written) <= persisted, f"lost {n_writes - len(set(written) & persisted)} writes"
            # 写线程占用期间事件循环保持响应（心跳无大间隔 → 读路径不被写阻塞）
            gaps = [b - a for a, b in zip(heartbeats, heartbeats[1:])]
            assert max(gaps) < 0.12, f"event loop stalled: max heartbeat gap {max(gaps):.3f}s"
        finally:
            q.shutdown()

    def test_hyperedge_edge_insert_fails_loudly(self):
        """F2: _persist_hyperedge 边 INSERT 走 execute_cypher — 失败上抛不静默。

        回归防护：若改回 query_cypher（永不抛异常契约），本测试将失败。
        """
        store = MagicMock()
        store.execute_cypher.side_effect = RuntimeError("edge insert failed")
        mgr = HyperedgeManager(store)

        with pytest.raises(RuntimeError, match="edge insert failed"):
            mgr.create_episode_hyperedge(member_ids=["m1", "m2"], topic="f2")
        # 断言走 execute_cypher（fail-loud）而非 query_cypher（静默）
        store.execute_cypher.assert_called()
        store.query_cypher.assert_not_called()


class TestRealThroughput:
    """F3: 真实端到端吞吐 vs 串行基线（验收口径重定义）。"""

    def test_80_real_writes_match_serial_baseline(self, graphlite_store):
        """8 并发 × 10 = 80 条真实 GraphLite 写经队列：总耗时 ≈ 串行基线。

        验收口径（物理现实）: 端到端延迟由单写者上限约束（N × 单写耗时），
        队列只做串行化、不放大延迟。avg<500ms 的绝对口径与单写 237ms 物理
        矛盾（80×237ms 串行 → 平均等待 ≈8s），故以「队列开销≈0」重定义。
        """
        store = graphlite_store
        q = WriteQueue(wait_timeout=30.0)
        try:
            def write_ep(tag):
                store.create_episode(_make_episode(tag))

            # 串行基线：同一写函数在主线程写 80 条
            t0 = time.monotonic()
            for i in range(80):
                write_ep(f"base_{i}")
            baseline = time.monotonic() - t0

            # 8 并发 × 10 条经写队列（专用写线程串行）
            async def writer(w):
                for j in range(10):
                    await q.submit(write_ep, f"{w}_{j}")

            async def main():
                await asyncio.gather(*[writer(w) for w in range(8)])

            t0 = time.monotonic()
            asyncio.run(main())
            queued = time.monotonic() - t0

            # 队列总耗时不应显著超过串行基线（50% 调度容差 + 0.5s 固定裕量）
            assert queued <= baseline * 1.5 + 0.5, (
                f"queue added overhead: baseline {baseline*1000:.0f}ms vs "
                f"queued {queued*1000:.0f}ms"
            )
            # 全部 160 条落库（80 基线 + 80 排队）
            persisted = _extract_ids(store.query_cypher("MATCH (e:EpisodeNode) RETURN e.id", {}))
            assert len(persisted) == 160, f"expected 160 nodes, got {len(persisted)}"
        finally:
            q.shutdown()
