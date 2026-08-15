#!/usr/bin/env python3
"""
SHM v5.41.0 社区扩召回（Community-Expansion）评测脚本 v3
========================================================
真实 GraphLite 写入 + _persist_one_community 造真实社区边，
对比扩召回开/关的 multi-session Recall@10。

场景设计（3 个多会话问题）：每个问题的答案分散在不同会话——
  - 会话 1 内容进入 FAISS（直接检索命中，作扩召回种子）
  - 会话 2 答案**不进 FAISS**（直接检索必漏，仅能经社区边召回）
  → 扩召回开：答案进入 top-10（Recall@10 提升）；关：答案缺失。

用法:
  cd /home/admin/shm && GRAPHLITE_BINDINGS=/home/admin/GraphLite/bindings/python \
    GRAPHLITE_SDK=/home/admin/GraphLite/sdk-python/src \
    .venv/bin/python scripts/eval_community_expansion.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core.dream_pipeline import DreamPipeline


# ─── 3 个多会话问题（topic / 会话1种子 / 会话2答案 / 社区摘要） ───────────
SCENARIOS = [
    {
        "topic": "K8s 集群网络 flannel 问题",
        "s1": ["K8s 集群搭建时遇到 flannel 网络插件无法启动", "查看 flannel pod 日志发现网段冲突"],
        "s2": "最后用 calico 替换 flannel 解决了集群网络问题",
        "summary": "K8s 集群网络 flannel calico 排障 多会话 网段冲突 替换方案",
    },
    {
        "topic": "Python 异步任务队列 卡死",
        "s1": ["Python asyncio 任务队列出现卡死", "怀疑是 run_in_executor 占满了线程池"],
        "s2": "改用 asyncio.shield 包装 future 后队列不再卡死",
        "summary": "Python asyncio 任务队列 卡死 排障 executor shield 多会话",
    },
    {
        "topic": "数据库 迁移 数据丢失",
        "s1": ["数据库迁移后部分记录丢失", "检查日志发现迁移脚本漏了 where 条件"],
        "s2": "补跑增量迁移脚本恢复了全部丢失记录",
        "summary": "数据库 迁移 数据丢失 恢复 增量脚本 多会话 排障",
    },
]


class _MockEncoder:
    """确定性 mock 编码器（同文本恒同向量）。"""

    def __init__(self):
        self.dim = 384

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2 ** 31))
        return rng.randn(self.dim).astype(np.float32)


class _MockFaiss:
    """最小 FAISS 桩：按 L2 距离返回 top-k（只含会话 1 种子 → 直接检索漏会话 2）。"""

    def __init__(self):
        self.vectors: dict[int, np.ndarray] = {}

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        for vec, fid in zip(vectors, ids):
            self.vectors[int(fid)] = vec.astype(np.float32)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.vectors:
            return (np.array([[float("inf")]]), np.array([[-1]]))
        q = query.reshape(1, -1) if query.ndim == 1 else query
        ids_arr = np.array(list(self.vectors.keys()), dtype=np.int64)
        vecs_arr = np.array(list(self.vectors.values()), dtype=np.float32)
        dists = np.linalg.norm(vecs_arr - q, axis=1)
        top_k = min(k, len(dists))
        idx = np.argsort(dists)[:top_k]
        return (dists[idx].reshape(1, -1), ids_arr[idx].reshape(1, -1))


def _insert_episode(store, nid: str, content: str) -> None:
    store.create_episode({
        "id": nid, "content": content, "tau_initial": 1.0,
        "fact_track": "active", "archived": False,
    })


def _build_router(store, encoder, faiss, faiss_id_map):
    from retrieval.query_router import QueryRouter, QueryRouterConfig
    return QueryRouter(
        graphlite_store=store,
        faiss_index=faiss,
        tfidf_index=None,
        encoder=encoder,
        faiss_id_map=faiss_id_map,
    )


def _recall_at_10(out: list[dict], answer_id: str) -> tuple[bool, list[str]]:
    """答案 id 是否出现在 top-10（按 score 降序）。"""
    ranked = sorted(out, key=lambda r: r.get("score", 0.0), reverse=True)[:10]
    ids = [r.get("node_id", "") for r in ranked]
    return answer_id in ids, ids


def main() -> int:
    from graph.graphlite_store import GraphLiteStore

    tmpdir = tempfile.mkdtemp(prefix="shm_eval_v541_")
    db_path = os.path.join(tmpdir, "eval_graphlite")
    config = type("cfg", (), {"database_path": db_path, "max_threads": 2})()
    store = GraphLiteStore(config=config)
    store.connect()

    encoder = _MockEncoder()
    # 每个场景独立 FAISS（只含该场景会话 1 种子）——隔离 mock 编码器跨主题
    # 向量噪声，精确模拟「直接检索命中会话 1、会话 2 答案仅经社区可达」
    scenario_faiss: dict[str, _MockFaiss] = {}
    scenario_fmap: dict[str, dict[int, str]] = {}
    fid = 0
    expected: dict[str, str] = {}  # topic → 答案 episode id

    try:
        # 1) 写 3 个多会话问题到同一真实 GraphLite 库
        for sc in SCENARIOS:
            faiss = _MockFaiss()
            fmap: dict[int, str] = {}
            seed_ids = []
            for content in sc["s1"]:
                nid = f"{uuid.uuid4().hex[:10]}"
                _insert_episode(store, nid, content)
                seed_ids.append(nid)
                faiss.add_with_ids(
                    encoder.embed(content).reshape(1, -1),
                    np.array([fid], dtype=np.int64),
                )
                fmap[fid] = nid
                fid += 1
            ans_id = f"{uuid.uuid4().hex[:10]}"
            _insert_episode(store, ans_id, sc["s2"])
            # 2) _persist_one_community 造真实社区边（会话 1 + 会话 2 同社区）
            DreamPipeline()._persist_one_community(
                store,
                {"id": f"comm_{sc['topic'][:6]}", "members": seed_ids + [ans_id],
                 "report": sc["summary"]},
                "eval",
                0,
            )
            scenario_faiss[sc["topic"]] = faiss
            scenario_fmap[sc["topic"]] = fmap
            expected[sc["topic"]] = ans_id

        # 3) 扩召回开（默认配置）：每场景独立 FAISS 检索 + Recall@10
        print(f"{'问题':<28} {'扩召回':<8} {'Recall@10':<10} 命中")
        print("─" * 78)
        on_hits = 0
        off_hits = 0
        for sc in SCENARIOS:
            router = _build_router(store, encoder, scenario_faiss[sc["topic"]],
                                   scenario_fmap[sc["topic"]])
            t0 = time.monotonic()
            out = router.retrieve(sc["topic"])
            lat = (time.monotonic() - t0) * 1000
            hit, ids = _recall_at_10(out, expected[sc["topic"]])
            on_hits += 1 if hit else 0
            lv = {r.get("level") for r in out}
            print(f"{sc['topic']:<26} ON       {'✓' if hit else '✗':<10} "
                  f"{lat:6.1f}ms levels={sorted(lv)}")

            # 4) 扩召回关（monkeypatch get_settings）：同库同检索对比
            from config.settings import Settings, CommunityExpansionConfig
            import retrieval.query_router as qr_mod
            orig = qr_mod.get_settings
            s = Settings()
            s.retrieval.community_expansion = CommunityExpansionConfig(enabled=False)
            qr_mod.get_settings = lambda: s
            try:
                out_off = router.retrieve(sc["topic"])
                hit_off, _ = _recall_at_10(out_off, expected[sc["topic"]])
                off_hits += 1 if hit_off else 0
                print(f"{sc['topic']:<26} OFF      {'✓' if hit_off else '✗'}")
            finally:
                qr_mod.get_settings = orig

        print("─" * 78)
        print(f"multi-session Recall@10: 扩召回 ON = {on_hits}/{len(SCENARIOS)}"
              f"   OFF = {off_hits}/{len(SCENARIOS)}")
        ok = on_hits > off_hits and off_hits == 0
        print(f"评测结论: {'✅ 社区扩召回提升 multi-session Recall' if ok else '❌ 未达预期'}")
        return 0 if ok else 1
    finally:
        store.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
