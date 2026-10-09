"""OverGraph 锁公平性验收测试（v6.26 写队列 503 / 健康检查劣化修复）
====================================================================

背景（生产实测 + py-spy 采样实证）：
  OverGraphStore 单把全局锁 `_session_lock`（RLock）被检索路径长期占用——社区
  扩召回对每个相关社区各发一次 `get_community_members`（各取锁），写线程
  `shm-writer-worker` 反复 blocked → 写队列积压 → WriteQueueFullError → 503；
  /health 自身查图（COUNT 全扫）同样排队 → 10~23 秒。

修复（AC 锚点）：
  - AC1/AC2：写线程优先锁（_WriterPriorityLock，绑定写队列 worker ident）+
    社区成员批量一次取回（N 次 get_community_members → 1 次）。
  - AC3：health_probe(timeout=) 带超时探测；/health COUNT 全扫 offload + 超时降级。
  - AC5：批量取回与逐社区取回结果等价（真实 OverGraph 引擎对拍）。

本文件是「修复前红、修复后绿」的验收锚点：修复前 `bind_writer_thread` /
`health_probe` / `get_community_members_batch` 均不存在 → AttributeError（红）；
修复后逐条断言真实并发行为（无 503、写者不被饿死、/health 亚秒、结果一致）。

运行: python -m pytest tests/test_overgraph_lock_fairness.py -q
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from core.write_queue import WriteQueue, WriteQueueFullError
from graph.overgraph_store import OverGraphStore


# ─── 假引擎（确定性制造锁竞争；真实 OverGraph 引擎读为微秒级无法稳定复现）───

class _SlowFakeEngine:
    """慢读快写假引擎：execute_gql 每次 sleep read_latency（制造读侧锁竞争）。"""

    def __init__(self, read_latency: float = 0.02, write_latency: float = 0.001):
        self.read_latency = read_latency
        self.write_latency = write_latency
        self.read_calls = 0
        self.write_calls = 0
        self.writes: list[tuple[str, str]] = []
        self._cnt_lock = threading.Lock()

    def execute_gql(self, gql, params=None, mode="auto", allow_full_scan=True):
        time.sleep(self.read_latency)
        with self._cnt_lock:
            self.read_calls += 1
        return {"rows": []}

    def upsert_node(self, label, key, props=None, dense_vector=None):
        time.sleep(self.write_latency)
        with self._cnt_lock:
            self.write_calls += 1
            self.writes.append((str(label), str(key)))

    def get_node_by_key(self, label, key):
        return None


class _GatedFakeEngine:
    """闸门假引擎：主读持锁阻塞在 first_read_gate，写者/读者执行顺序可观测。"""

    def __init__(self):
        self.first_read_entered = threading.Event()
        self.first_read_gate = threading.Event()
        self.reader_gate = threading.Event()
        self.write_entered = threading.Event()
        self.write_gate = threading.Event()
        self._seq_lock = threading.Lock()
        self._read_seq = 0
        self.read_count = 0
        self.write_count = 0
        self.writes: list[tuple[str, str]] = []

    def execute_gql(self, gql, params=None, mode="auto", allow_full_scan=True):
        with self._seq_lock:
            self._read_seq += 1
            seq = self._read_seq
        if seq == 1:  # 主读：持锁阻塞，制造「锁被长读占用」
            self.first_read_entered.set()
            self.first_read_gate.wait(timeout=10)
        else:  # 后续读者：进入即阻塞（用于观察「写者先于读者入场」）
            self.reader_gate.wait(timeout=10)
        with self._seq_lock:
            self.read_count += 1
        return {"rows": []}

    def upsert_node(self, label, key, props=None, dense_vector=None):
        self.write_entered.set()
        self.write_gate.wait(timeout=10)
        with self._seq_lock:
            self.write_count += 1
            self.writes.append((str(label), str(key)))

    def get_node_by_key(self, label, key):
        return None


def _make_store(engine) -> OverGraphStore:
    """构造未连接 OverGraphStore 并注入假引擎（跳过真实建库）。"""
    store = OverGraphStore()
    store._db = engine
    return store


# ─── AC2：并发检索 + 持续写入，写队列零拒绝 ──────────────────────────

def test_writes_not_rejected_under_retrieval_flood():
    """8 路并发检索（慢读引擎）+ WriteQueue 持续写入：无 503、全部落库。"""
    store = _make_store(_SlowFakeEngine(read_latency=0.02, write_latency=0.001))
    wq = WriteQueue(max_pending=8, wait_timeout=15.0)
    store.bind_writer_thread(wq.worker_ident)

    stop = threading.Event()

    def _reader() -> None:
        while not stop.is_set():
            store.get_community_members("c0", limit=10)

    readers = [threading.Thread(target=_reader) for _ in range(8)]
    for t in readers:
        t.start()

    try:
        async def _writes() -> tuple[int, int]:
            ok = 0
            rejected = 0
            for i in range(20):
                try:
                    await wq.submit(
                        store.create_episode,
                        {"id": f"ep-{i}", "content": f"w{i}", "tau_initial": 1.0},
                        priority="high",
                    )
                    ok += 1
                except WriteQueueFullError:
                    rejected += 1
            return ok, rejected

        t0 = time.monotonic()
        ok, rejected = asyncio.run(_writes())
        elapsed = time.monotonic() - t0

        assert rejected == 0, f"写队列满/503: {rejected} 次拒绝"
        assert ok == 20, f"应完成 20 次写入，实际 {ok}"
        assert store._db.write_calls == 20, f"写未全部落库: {store._db.write_calls}/20"
        # 写者优先下 20 次写（每次至多让行 1 个读者）远快于被 8 路读者饿死
        assert elapsed < 5.0, f"写线程疑似被饿死: {elapsed:.2f}s"
    finally:
        stop.set()
        for t in readers:
            t.join(timeout=10)
        # 清理写线程：避免 worker 泄漏进后续用例（full suite 下 MED-3 断言
        # 精确匹配单一 shm-writer-worker 线程）。
        wq.shutdown()


# ─── AC2（确定性）：写者优先于已排队读者 ──────────────────────────────

def test_writer_priority_over_queued_readers():
    """锁释放瞬间，写者先于所有等待读者入场（无计时依赖的公平性证明）。

    修复前（RLock）主读释放后 FIFO 服务队首读者 → 写者被饿死 → write_entered
    永不置位 → 超时失败。修复后（写者优先）写者先入场，读者让行。
    """
    engine = _GatedFakeEngine()
    store = _make_store(engine)

    # 1. 主读线程持锁阻塞在 first_read_gate（模拟长检索）
    t_main = threading.Thread(
        target=lambda: store.get_community_members("c0", limit=10)
    )
    t_main.start()
    assert engine.first_read_entered.wait(timeout=5), "主读未进入 execute_gql（锁未被持有）"

    # 2. 8 个读者 + 1 个写者排队等锁
    def _reader() -> None:
        store.get_community_members("c0", limit=10)

    readers = [threading.Thread(target=_reader) for _ in range(8)]
    for t in readers:
        t.start()

    t_writer = threading.Thread(
        target=lambda: store.create_episode(
            {"id": "ep-w", "content": "w", "tau_initial": 1.0}
        )
    )
    t_writer.start()

    try:
        # 给写者时间登记为 waiting writer
        time.sleep(0.1)

        # 3. 释放主读 → 写者应优先入场
        engine.first_read_gate.set()
        assert engine.write_entered.wait(timeout=5), (
            "写线程未被优先调度（被读者饿死）"
        )
        # 写者已进入 upsert_node（持锁）→ 无任何读者执行过 execute_gql
        assert engine.read_count == 1, (
            f"读者先于写者入场，公平性失效: read_count={engine.read_count}"
        )
    finally:
        engine.write_gate.set()
        engine.reader_gate.set()
        engine.first_read_gate.set()
        for t in [t_main, t_writer] + readers:
            t.join(timeout=10)

    assert engine.write_count == 1
    assert engine.writes == [("EpisodeNode", "ep-w")]


# ─── AC3：健康检查亚秒 ──────────────────────────────────────────────

def test_health_probe_true_when_lock_free():
    """锁空闲时 health_probe 应成功（引擎可达 → True）。"""
    store = _make_store(_SlowFakeEngine())
    assert store.health_probe(timeout=1.0) is True


def test_health_probe_bounded_under_lock_contention():
    """持锁竞争下 health_probe(timeout=0.2) 亚秒返回 False（不持锁排队）。"""
    store = _make_store(_SlowFakeEngine())
    held = threading.Event()
    release = threading.Event()

    def _holder() -> None:
        with store._session_lock:
            held.set()
            release.wait(timeout=5)

    t = threading.Thread(target=_holder)
    t.start()
    assert held.wait(timeout=5), "holder 未取得锁"

    t0 = time.monotonic()
    ok = store.health_probe(timeout=0.2)
    elapsed = time.monotonic() - t0

    release.set()
    t.join(timeout=5)

    assert ok is False, "锁竞争下探针应超时降级（不持锁返回 False）"
    assert elapsed < 0.6, f"探针超时上界被突破: {elapsed:.2f}s"


def test_health_route_bounded_under_contention():
    """/health 处理器在持锁竞争下亚秒返回（COUNT 全扫 offload + 超时降级）。"""
    from api.routes._deps import Services
    from api.routes.system import health_check

    store = _make_store(_SlowFakeEngine(read_latency=0.05))
    held = threading.Event()
    release = threading.Event()

    def _holder() -> None:
        with store._session_lock:
            held.set()
            release.wait(timeout=5)

    t = threading.Thread(target=_holder)
    t.start()
    assert held.wait(timeout=5), "holder 未取得锁"

    svc = Services()
    svc.graph_store = store
    svc.faiss_index = None
    svc.audit_chain = None
    svc.dream_scheduler = None

    t0 = time.monotonic()
    result = asyncio.run(health_check(deps=svc))
    elapsed = time.monotonic() - t0

    release.set()
    t.join(timeout=5)

    assert result is not None
    assert elapsed < 1.5, f"/health 在锁竞争下超亚秒: {elapsed:.2f}s"


# ─── AC5：检索质量不回归（批量 == 逐社区，真实引擎对拍）──────────────

def _seed_episode(store, nid: str, content: str, tau: float) -> None:
    store.create_episode({
        "id": nid, "content": content, "tau_initial": tau,
        "fact_track": "active", "archived": False,
    })


def _seed_community(store, cid: str, members: list[str], report: str) -> None:
    from core.dream_pipeline import DreamPipeline
    DreamPipeline()._persist_one_community(
        store, {"id": cid, "members": members, "report": report}, "eval", 0,
    )


def test_community_members_batch_matches_serial(overgraph_store):
    """批量一次取回与逐社区 get_community_members 结果逐社区一致（含排序）。"""
    store = overgraph_store
    # 不同 tau_initial 保证 ORDER BY tau_initial DESC 排序确定（避免并列歧义）
    _seed_episode(store, "ep1", "K8s 集群 flannel 问题", 3.0)
    _seed_episode(store, "ep2", "flannel 被 calico 替换", 1.0)
    _seed_episode(store, "ep3", "calico BGP 配置", 2.0)
    _seed_episode(store, "ep4", "多会话记忆排障", 0.5)
    _seed_community(store, "c1", ["ep1", "ep2"], "K8s 网络 flannel calico")
    _seed_community(store, "c2", ["ep3", "ep4"], "calico BGP 排障")

    batch = store.get_community_members_batch(["c1", "c2"], limit=10)
    assert set(batch.keys()) == {"c1", "c2"}

    for cid in ["c1", "c2"]:
        serial = store.get_community_members(cid, limit=10)
        assert serial, f"社区 {cid} 应有成员（种子数据未生效）"
        assert batch[cid] == serial, (
            f"社区 {cid} 批量取回与逐社区取回不一致:\n"
            f"  batch={batch[cid]}\n  serial={serial}"
        )

    # 非空结果：确认批量路径确实取回了成员（而非两侧同时为空）
    assert {m["member_id"] for m in batch["c1"]} == {"ep1", "ep2"}
    assert {m["member_id"] for m in batch["c2"]} == {"ep3", "ep4"}
