"""[H2/B-P0-2] 候选 apply 路径 merge 永久丢失。

基线缺陷（B-core 审计 P0-2，实跑验证）：
    RESOLVE 里 conflict_count 恒等于 len(merge_ops)（dream_pipeline
    `_resolve_step`: conflict_count += len(ops)）→ auto_apply 质量门
    `conflict_count > 0 → discard` 等效"凡有 merge 的梦整梦丢弃"，而
    门控自身不执行 merge、auto_apply 也没有 merge 循环 → 生产路径上
    merge_ops 永久丢失；且 apply_candidate 与直接模式 _persist_merge
    是两条分叉的持久化实现（Cypher 拼接无截断 vs Python 拼接 [:2000]）。

本测试证明修复后：
    H2-1 含冲突(conflict_count=1)+merge 的候选被 auto_apply 真正应用，
         merge 执行（winner 拼接 + loser 删除）——修复前该候选必被丢弃；
    H2-2 merge 拼接按 [:2000] 截断（防连续合并无界增长，对齐 _persist_merge）；
    H2-3 质量门结构性丢弃时，未应用 merge/prune 数量有 warning（不静默），
         候选文件保持 discarded 供人工审查；
    H2-4 手动 apply_candidate 走同一入口 _apply_merge_ops 并计数。
"""
from __future__ import annotations

import json
import logging

from core.audit_chain import AuditOperation
from core.dream_candidate_store import DreamCandidateStore, _apply_merge_ops


class _MemGraph:
    """内存图存储桩：content 读 / SET / DETACH DELETE / batch_write_txn。"""

    def __init__(self, nodes: dict[str, dict]):
        self.nodes = {k: dict(v) for k, v in nodes.items()}

    def query_cypher(self, query: str, params: dict | None = None):
        p = params or {}
        if "SET t.content" in query:
            if p.get("id") in self.nodes:
                self.nodes[p["id"]]["content"] = p.get("content", "")
            return [{"status": "ok"}]
        if "DETACH DELETE" in query:
            self.nodes.pop(p.get("id"), None)
            return [{"status": "ok"}]
        if "RETURN" in query:
            n = self.nodes.get(p.get("id"))
            if n is None:
                return []
            return [{"content": n.get("content", "")}]
        return []

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


def _save_conflict_candidate(store: DreamCandidateStore, dream_id: str,
                             merge_ops: list):
    """merge_ops 传 AuditOperation——与生产 RESOLVE 输出一致（save_candidate
    从 op.new_value 序列化 target；传裸 dict 会丢 target）。"""
    store.save_candidate(
        dream_id=dream_id,
        communities=[{"id": "c1", "members": ["e2"], "report": "r" * 40}],
        prune_ops=[],
        merge_ops=merge_ops,
        dream_report_kwargs={
            "trigger_mode": "idle", "stats": {}, "community_count": 1,
            "prune_count": 0, "conflict_count": len(merge_ops),
        },
    )


def _pad_dummies(store: DreamCandidateStore, n: int = 19):
    for i in range(n):
        with open(store.storage_dir.rstrip("/") + f"/dummy{i:02d}.json", "w") as f:
            f.write("{}")


# ─── H2-1 ─────────────────────────────────────────────────────


def test_auto_apply_executes_merge_of_conflict_dream(tmp_path):
    """H2-1（关键回归）：conflict_count=1 + merge 的候选不再被质量门丢弃，
    auto_apply 真正执行 merge（winner 拼接、loser 删除）。"""
    store = DreamCandidateStore(storage_dir=str(tmp_path / "cand"))
    _save_conflict_candidate(
        store, "d1", [AuditOperation(
            op_type="update", node_id="e1", old_value="source payload",
            new_value="e2", reason="community_merge")])
    _pad_dummies(store)

    graph = _MemGraph({"e1": {"content": "source payload"},
                       "e2": {"content": "target payload"}})
    applied, comm_created, files, summaries = store.auto_apply_candidates(graph)

    assert applied == 1, "含冲突的梦在修复前被质量门 discard（merge 丢失）"
    assert "e1" not in graph.nodes, "loser 未被 DETACH DELETE"
    assert graph.nodes["e2"]["content"] == "target payload | merged: source payload"
    assert files == 1


# ─── H2-2 ─────────────────────────────────────────────────────


def test_merge_content_truncated_to_2000(tmp_path):
    """H2-2：连续合并下 content 有界（[:2000]，对齐 _persist_merge P8）。"""
    graph = _MemGraph({"e1": {"content": "S" * 1500},
                       "e2": {"content": "T" * 1500}})
    ok, failed = _apply_merge_ops(
        graph, [{"node_id": "e1", "target": "e2", "reason": "community_merge"}])

    assert (ok, failed) == (1, 0)
    content = graph.nodes["e2"]["content"]
    assert len(content) <= 2000
    assert " | merged: " in content


# ─── H2-3 ─────────────────────────────────────────────────────


def test_gate_discard_logs_unapplied_merge_count(tmp_path, caplog):
    """H2-3：结构性质量门失败（无有效社区）→ 丢弃时 warning 显式计数
    未应用的 merge/prune，候选文件保持 discarded 供人工审查。"""
    store = DreamCandidateStore(storage_dir=str(tmp_path / "cand"))
    store.save_candidate(
        dream_id="d1",
        communities=[],  # 无社区 → valid_communities=0 → 门失败
        prune_ops=[],
        merge_ops=[AuditOperation(
            op_type="update", node_id="e1", new_value="e2",
            reason="community_merge")],
        dream_report_kwargs={
            "trigger_mode": "idle", "stats": {}, "community_count": 0,
            "prune_count": 0, "conflict_count": 1,
        },
    )
    _pad_dummies(store)

    graph = _MemGraph({"e1": {"content": "s"}, "e2": {"content": "t"}})
    with caplog.at_level(logging.WARNING, logger="core.dream_candidate_store"):
        applied, _, _, _ = store.auto_apply_candidates(graph)

    assert applied == 0
    assert any(
        "1 merge_ops + 0 prune_ops not applied" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]
    # 候选文件仍在（discarded=true）——merge op 持久化可人工审查
    with open(store.storage_dir.rstrip("/") + "/d1.json") as f:
        data = json.load(f)
    assert data["discarded"] is True
    assert len(data["merge_ops"]) == 1


# ─── H2-4 ─────────────────────────────────────────────────────


def test_apply_candidate_uses_shared_merge_entry(tmp_path, caplog):
    """H2-4：手动 apply_candidate 与 auto_apply 收口同一入口，成败计数可见。"""
    store = DreamCandidateStore(storage_dir=str(tmp_path / "cand"))
    _save_conflict_candidate(
        store, "d1", [AuditOperation(
            op_type="update", node_id="e1", old_value="source",
            new_value="e2", reason="community_merge")])

    graph = _MemGraph({"e1": {"content": "source"},
                       "e2": {"content": "target"}})
    with caplog.at_level(logging.INFO, logger="core.dream_candidate_store"):
        ok = store.apply_candidate("d1", graph)

    assert ok is True
    assert "e1" not in graph.nodes
    assert graph.nodes["e2"]["content"] == "target | merged: source"
    assert any(
        "1 merges ok / 0 merge failed" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]
