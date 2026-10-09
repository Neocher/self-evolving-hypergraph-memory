"""OverGraph 锁公平性压测脚本（v6.26 写队列 503 / 健康检查劣化修复验收）
========================================================================

用途（AC2/AC3 实测数字）：
  在 ≥8 路并发检索 + 持续写入的锁竞争下，量化两件事：
    AC2 —— 写队列 503 拒绝次数（修复后应为 0）；
    AC3 —— /health 处理器延迟 P50/P95/max（修复后应亚秒）。

设计：注入慢读假引擎（每次 execute_gql 阻塞 read_latency），确定性复现
「社区扩召回逐社区取锁 → 读洪峰持续持锁」的生产场景（生产实测 /health
23.2s、写队列 503 11~13 次/10min，py-spy 5 次采样实证根因为全局锁饿死）。

运行（必须清洗环境变量，见项目 CLAUDE.md 已知坑）：
  env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \\
      /home/admin/shm/.venv/bin/python scripts/stress_overgraph_lock_fairness.py
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

# 脚本直跑时把仓库根加入 sys.path（scripts/ 不是包，docstring 的运行命令依赖此行）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routes._deps import Services
from api.routes.system import health_check
from core.write_queue import WriteQueue, WriteQueueFullError
from graph.overgraph_store import OverGraphStore


class _SlowFakeEngine:
    """慢读快写假引擎：execute_gql 每次 sleep read_latency（制造读侧锁竞争）。"""

    def __init__(self, read_latency: float = 0.02, write_latency: float = 0.001):
        self.read_latency = read_latency
        self.write_latency = write_latency
        self.read_calls = 0
        self.write_calls = 0
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

    def get_node_by_key(self, label, key):
        return None


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, int(round((len(sorted_vals) - 1) * p)))
    return sorted_vals[idx]


def main() -> None:
    read_latency = 0.02
    store = OverGraphStore()
    store._db = _SlowFakeEngine(read_latency=read_latency, write_latency=0.001)

    wq = WriteQueue(max_pending=8, wait_timeout=15.0)
    store.bind_writer_thread(wq.worker_ident)

    stop = threading.Event()

    def _reader() -> None:
        while not stop.is_set():
            store.get_community_members("c0", limit=10)

    n_readers = 8
    readers = [threading.Thread(target=_reader) for _ in range(n_readers)]
    for t in readers:
        t.start()

    try:
        # ── AC2：持续写入，统计 503 拒绝 ──
        async def _writes(n: int) -> tuple[int, int, float]:
            ok = 0
            rejected = 0
            t0 = time.monotonic()
            for i in range(n):
                try:
                    await wq.submit(
                        store.create_episode,
                        {"id": f"ep-{i}", "content": f"w{i}", "tau_initial": 1.0},
                        priority="high",
                    )
                    ok += 1
                except WriteQueueFullError:
                    rejected += 1
            return ok, rejected, time.monotonic() - t0

        n_writes = 50
        ok, rejected, write_elapsed = asyncio.run(_writes(n_writes))

        # ── AC3：同一锁竞争下 /health 延迟（每次强制 cache-miss 走全扫路径）──
        import api.routes.system as system_mod

        svc = Services()
        svc.graph_store = store
        svc.faiss_index = None
        svc.audit_chain = None
        svc.dream_scheduler = None

        latencies: list[float] = []
        for _ in range(20):
            system_mod._HEALTH_STATS_CACHE_TIME = 0.0  # 强制 cache miss
            t0 = time.monotonic()
            asyncio.run(health_check(deps=svc))
            latencies.append(time.monotonic() - t0)
        latencies.sort()

        print("=" * 68)
        print("OverGraph 锁公平性压测结果（修复后）")
        print("=" * 68)
        print(f"并发读者线程      : {n_readers}")
        print(f"写队列容量        : max_pending=8, wait_timeout=15.0s")
        print(f"持续写入数        : {n_writes}")
        print("-" * 68)
        print(f"AC2 写入完成      : {ok}/{n_writes}（全部落库: {store._db.write_calls}）")
        print(f"AC2 写队列 503    : {rejected} 次拒绝")
        print(f"AC2 写入总耗时    : {write_elapsed:.2f}s")
        print("-" * 68)
        print(f"AC3 /health P50   : {_pct(latencies, 0.50):.3f}s")
        print(f"AC3 /health P95   : {_pct(latencies, 0.95):.3f}s")
        print(f"AC3 /health max   : {max(latencies):.3f}s")
        print("-" * 68)
        print("基线（修复前，生产实测）: /health 23.2s；503 11~13 次/10min")
        print("=" * 68)
    finally:
        stop.set()
        for t in readers:
            t.join(timeout=10)
        wq.shutdown()


if __name__ == "__main__":
    main()
