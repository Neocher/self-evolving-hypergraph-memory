"""[H1/B-P0-1] 候选模式下被剪节点的 FAISS 向量清理链路。

基线缺陷（B-core 审计 P0-1，实跑验证）：
    生产恒走候选模式（app.py 注入 candidate_store）→ run() 的候选分支从不
    填充 all_removed_ids → DreamReport.pruned_node_ids 恒为 [] → 调度器
    `_run_dream` 的 FAISS 增量清理 `if removed and ...` 永不触发 →
    auto_apply 物理删除（DETACH DELETE）后：
      - legacy FAISS 后端：向量永驻，ghost 行占 top-k；
      - OverGraph (D7)：faiss_id_map 残留累积、ntotal/_count 漂移。

本测试证明修复后：
    H1-1 候选模式 report.pruned_node_ids 携带被剪节点 id；
    H1-2 调度器 _run_dream 实际把该 id 列表交给 incremental_update_fn；
    H1-3 清理不完整（count < 请求数）→ 计数 + warning，失败不静默；
    H1-4 auto_apply 物理删除单条失败 → per-op warning + 汇总计数。
"""
from __future__ import annotations

import asyncio
import logging
import time

from core.audit_chain import AuditOperation
from core.dream_candidate_store import DreamCandidateStore
from core.dream_pipeline import DreamPipeline
from core.dream_scheduler import DreamScheduler, DreamSchedulerConfig, TriggerMode
from core.tau_decay import TauDecayEngine, TauDecayConfig


def _nodes(now: float) -> list[dict]:
    """4 个老节点（3-4 天前，无连接 → τ≈0 可剪）+ 4 个新节点（<2h 保护）。

    4/8 = 50%，恰好不触发 _MAX_PRUNE_RATIO(>50%) 批量中止。
    """
    nodes = []
    for i in range(4):
        nodes.append({
            "id": f"old{i}",
            "content": f"old memory payload number {i} alpha beta gamma delta",
            "created_at": now - (3 + i) * 86400,
            "tau_initial": 0.5,
        })
    for i in range(4):
        nodes.append({
            "id": f"fresh{i}",
            "content": f"fresh memory payload number {i} alpha beta gamma",
            "created_at": now - 3600,
            "tau_initial": 1.0,
        })
    return nodes


def _run_async(coro):
    return asyncio.run(coro)


# ─── H1-1 ─────────────────────────────────────────────────────


def test_candidate_mode_report_carries_pruned_ids(tmp_path):
    """候选模式下 report.pruned_node_ids == 被剪节点 id（向量清理输入）。"""
    now = time.time()
    tau = TauDecayEngine(TauDecayConfig())
    pipe = DreamPipeline(tau_engine=tau)
    store = DreamCandidateStore(storage_dir=str(tmp_path / "cand"))

    report = _run_async(pipe.run(
        _nodes(now), {}, trigger_mode="idle", candidate_store=store,
    ))
    assert set(report.pruned_node_ids) == {"old0", "old1", "old2", "old3"}
    assert not any(nid.startswith("fresh") for nid in report.pruned_node_ids)


# ─── H1-2 / H1-3 ─────────────────────────────────────────────


class _FakeGraphStore:
    """_run_dream 数据源桩：节点行查询返回 dict 行，边查询返回空。"""

    def __init__(self, nodes: list[dict]):
        self._nodes = nodes

    def _flatten_row(self, row: dict, label: str = "") -> dict:
        return dict(row)

    def query_cypher(self, query: str, params: dict | None = None):
        if "HEBBIAN_CONNECTION" in query:
            return []
        return [dict(n) for n in self._nodes]


class _FaissFn:
    """incremental_faiss_update(deps, ids) 桩：记录收到的 id 列表。"""

    def __init__(self, removed_ratio: float = 1.0):
        self.calls: list[list[str]] = []
        self.removed_ratio = removed_ratio

    def __call__(self, deps, removed_node_ids: list[str]) -> int:
        self.calls.append(list(removed_node_ids))
        return int(len(removed_node_ids) * self.removed_ratio)


def _make_scheduler(fn, nodes) -> DreamScheduler:
    pipe = DreamPipeline(tau_engine=TauDecayEngine(TauDecayConfig()))
    sched = DreamScheduler(
        DreamSchedulerConfig(min_interval_seconds=0),
        pipeline_fn=pipe.run,
    )
    sched._graphlite_store = _FakeGraphStore(nodes)
    sched._candidate_store = DreamCandidateStore(
        storage_dir="/tmp/shm_h1_test_cand")
    sched._faiss_index = object()
    sched._faiss_id_map = {}
    sched._incremental_update_fn = fn
    return sched


def test_scheduler_feeds_pruned_ids_to_faiss_cleanup(tmp_path):
    """H1-2：候选模式梦境结束后，调度器把被剪 id 交给向量清理函数。"""
    import shutil
    shutil.rmtree("/tmp/shm_h1_test_cand", ignore_errors=True)
    now = time.time()
    fn = _FaissFn()
    sched = _make_scheduler(fn, _nodes(now))

    _run_async(sched._run_dream(TriggerMode.IDLE))

    assert fn.calls, "incremental_update_fn 从未被调用（ghost 向量场景复现）"
    assert set(fn.calls[0]) == {"old0", "old1", "old2", "old3"}
    assert sched._faiss_cleanup_fail_count == 0


def test_scheduler_counts_incomplete_faiss_cleanup(tmp_path, caplog):
    """H1-3：清理不完整（0/4 删除成功）→ fail_count=1 + warning 可见。"""
    import shutil
    shutil.rmtree("/tmp/shm_h1_test_cand", ignore_errors=True)
    now = time.time()
    fn = _FaissFn(removed_ratio=0.0)  # 全部删除失败
    sched = _make_scheduler(fn, _nodes(now))

    with caplog.at_level(logging.WARNING, logger="core.dream_scheduler"):
        _run_async(sched._run_dream(TriggerMode.IDLE))

    assert sched._faiss_cleanup_fail_count == 1
    assert any(
        "FAISS cleanup incomplete: 0/4" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]


# ─── H1-4 ─────────────────────────────────────────────────────


class _AutoApplyStore:
    """auto_apply 用图存储桩：e2 删除恒失败；batch_write_txn 最小实现。"""

    def __init__(self):
        self.deleted: list[str] = []

    def query_cypher(self, query: str, params: dict | None = None):
        nid = (params or {}).get("id", "")
        if "DETACH DELETE" in query and nid == "e2":
            raise RuntimeError("simulated delete failure")
        if "DETACH DELETE" in query and nid:
            self.deleted.append(nid)
        return [{"status": "ok"}]

    def batch_write_txn(self):
        class _Txn:
            def stage(self, ops):
                pass

        class _Db:
            def get_node_by_key(self, label, key):
                return None

            def get_edge_by_triple(self, *a):
                return None

        class _Ctx:
            def __enter__(self):
                return (_Txn(), _Db())

            def __exit__(self, *a):
                return False

        return _Ctx()


def _save_candidate_with_ops(store: DreamCandidateStore, dream_id: str):
    store.save_candidate(
        dream_id=dream_id,
        communities=[
            {"id": "c1", "members": ["e1"], "report": "r" * 40},
        ],
        prune_ops=[
            AuditOperation(op_type="delete", node_id="e1", reason="tau_decay"),
            AuditOperation(op_type="delete", node_id="e2", reason="tau_decay"),
        ],
        merge_ops=[],
        dream_report_kwargs={
            "trigger_mode": "idle", "stats": {}, "community_count": 1,
            "prune_count": 2, "conflict_count": 0,
        },
    )


def test_auto_apply_prune_failure_counted_and_logged(tmp_path, caplog):
    """H1-4：auto_apply 单条物理删除失败 → warning + 汇总计数，不静默。"""
    store = DreamCandidateStore(storage_dir=str(tmp_path / "cand"))
    _save_candidate_with_ops(store, "d1")
    # auto_apply 阈值：候选文件 >= 20
    for i in range(19):
        (tmp_path / "cand" / f"dummy{i:02d}.json").write_text("{}")

    graph = _AutoApplyStore()
    with caplog.at_level(logging.WARNING):
        applied, comm_created, files, summaries = store.auto_apply_candidates(graph)

    assert applied == 1
    assert graph.deleted == ["e1"]  # e2 删除失败
    assert any(
        "auto-apply prune failed for e2" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]
