#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""τ 衰减形式 A/B 检索召回门 (zero-LLM, read-only)

用途: 判断把 τ 形式从 exp 换成 pow / frac 是否会改变 top-k 证据召回。
做法: 对每个问题只跑 **一次** 生产 FUSION 检索 (hyde=False, agentic_enabled=False,
     零 LLM 调用)，拿到固定候选集; 然后离线按三种 τ 形式各自的 boost 对同一候选集
     重新打分排序，比较 hit@{1,3,5,10,20,40}。这样三种形式共享同一候选集，差异只
     来自 τ 形式本身。

数据:
  conversations.jsonl / questions.jsonl (LoCoMo-Refined)
  长程子集 gap>3d, 短程子集 gap<1d (gap = convlast - max(evidence epoch))

不写库、不调 LLM、不 commit。

用法:
  python3 scripts/tau_form_recall_gate.py [N] [OUT]
  N=0 → 全量 (默认 0); OUT 默认 /tmp/tau_form_recall_gate.json
环境变量: DB_PATH DATA_PATH CONV_PATH N TAU_DECAY_SECONDS OUT
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from collections import Counter

SHM_ROOT = os.environ.get("SHM_ROOT", "/home/user/self-evolving-hypergraph-memory")
if SHM_ROOT not in sys.path:
    sys.path.insert(0, SHM_ROOT)

import numpy as np  # noqa: E402

DB_PATH = os.environ.get("DB_PATH", "/home/user/LoCoMo_refined/results/eval_db_p1")
DATA_PATH = os.environ.get("DATA_PATH", "/home/user/LoCoMo_refined/data/public/questions.jsonl")
CONV_PATH = os.environ.get("CONV_PATH", "/home/user/LoCoMo_refined/data/public/conversations.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("N", "0"))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
    "OUT", "/tmp/tau_form_recall_gate.json")
TAU_DECAY_SECONDS = float(os.environ.get("TAU_DECAY_SECONDS", "300"))
TXT_OUT = os.path.splitext(OUT)[0] + ".txt"

KS = (1, 3, 5, 10, 20, 40)
FORMS = ("exp", "pow", "frac")
SUBSETS = ("long", "short", "all")
DAY = 86400.0

from core.tau_decay import TauDecayConfig, TauDecayEngine  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# 0. 日期解析 (宽容解析: 带时间 → 仅日期)
# ──────────────────────────────────────────────────────────────────────────
def _epoch(dt_str):
    """'1:56 pm on 8 May, 2023' / '8 May, 2023' → epoch 秒; 失败返回 None。"""
    import datetime as _dt

    if not dt_str:
        return None
    for fmt in ("%I:%M %p on %d %B, %Y", "%d %B, %Y"):
        try:
            return _dt.datetime.strptime(dt_str.strip(), fmt).timestamp()
        except (ValueError, TypeError):
            continue
    return None


def load_gaps():
    """→ (sessdt, convlast): (ci, session_index) → epoch, ci → 最后会话 epoch。"""
    convs = [json.loads(l) for l in open(CONV_PATH, encoding="utf-8") if l.strip()]
    sessdt: dict = {}
    convlast: dict = {}
    for c in convs:
        ci = c.get("conversation_idx")
        if ci is None:
            continue
        best = 0.0
        for s in c.get("sessions") or []:
            e = _epoch(s.get("date_time"))
            sessdt[(ci, s.get("session_index"))] = e
            if e is not None and e > best:
                best = e
        if best > 0:
            convlast[ci] = best
    return sessdt, convlast


def probes_of(ev_texts):
    """证据 25 字符滑窗探针 (与 hitk_overgraph.py 一致)，≤8 条，len≥15。"""
    probes = []
    for et in ev_texts:
        t = (et or "").strip()
        if not t:
            continue
        probes += [t[i:i + 25] for i in range(0, max(1, len(t) - 24), 20)][:8]
    return [p for p in probes if len(p) >= 15]


# ──────────────────────────────────────────────────────────────────────────
# 1. 构建 tau 引擎 (无状态: 不 register_node, enable_adaptive=False)
# ──────────────────────────────────────────────────────────────────────────
def build_engines():
    return {
        "exp": TauDecayEngine(TauDecayConfig(
            tau_form="exp", tau_initial=1.0,
            tau_decay_seconds=TAU_DECAY_SECONDS, enable_adaptive=False)),
        "pow": TauDecayEngine(TauDecayConfig(
            tau_form="pow", alpha=0.5, tau_initial=1.0,
            tau_decay_seconds=TAU_DECAY_SECONDS, enable_adaptive=False)),
        "frac": TauDecayEngine(TauDecayConfig(
            tau_form="frac", frac_K=8, frac_scale_factor=10.0, tau_initial=1.0,
            tau_decay_seconds=TAU_DECAY_SECONDS, enable_adaptive=False)),
    }


def _boost(tau):
    return 1.0 + 1.0 / (1.0 + math.exp(-tau / 60.0))


# ──────────────────────────────────────────────────────────────────────────
# 2. 检索链路 (复用 hitk_overgraph.py fusion 装配)
# ──────────────────────────────────────────────────────────────────────────
def build_retriever():
    from graph.overgraph_store import OverGraphStore
    from embedding.encoder import TextEncoder
    from retrieval.vector_index import VectorIndexAdapter
    from retrieval.query_router import (
        QueryRouter, QueryRouterConfig, RetrievalLevel,
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    cfg = type("cfg", (), {
        "database_path": DB_PATH, "dense_vector_dimension": 512,
        "dense_vector_metric": "cosine", "ef_search": 64, "max_threads": 4})()
    gstore = OverGraphStore(config=cfg)
    gstore.connect()

    rows = gstore._locked_execute_gql(
        "MATCH (e:EpisodeNode) RETURN e.id AS id, e.content AS content, "
        "e.created_at AS ts, e.session_id AS session_id", {})["rows"]
    msg_by_id = {r["id"]: r.get("content", "") for r in rows}
    msg_created = {}
    for r in rows:
        try:
            msg_created[r["id"]] = float(r.get("ts"))
        except (TypeError, ValueError):
            msg_created[r["id"]] = None

    enc = TextEncoder(device=os.environ.get("ENC_DEVICE", "auto"))
    enc.load()

    class TfidfSearchIndex:
        def __init__(self):
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), max_features=5000)
            self._fitted = False

        def fit(self, texts):
            if texts:
                self.matrix = self.vectorizer.fit_transform(texts)
                self._fitted = True
                self.texts = texts

        def search(self, query, k=20):
            if not self._fitted:
                return []
            q_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self.matrix)[0]
            top_k = min(k, len(scores))
            if top_k == 0:
                return []
            top_indices = np.argsort(scores)[-top_k:][::-1]
            return [(self.texts[i], float(scores[i])) for i in top_indices]

    tfidf_index = TfidfSearchIndex()
    tfidf_index.fit(list(msg_by_id.values()))
    faiss_id_map = {i: f"ep_{i}" for i in range(len(msg_by_id))}
    faiss_index = VectorIndexAdapter(store=gstore, dimension=512, faiss_id_map=faiss_id_map)

    cfg_router = QueryRouterConfig()
    cfg_router.agentic_enabled = False
    cfg_router.hyde_timeout = 10.0
    cfg_router.mesa_enabled = True
    cfg_router.mesa_boost = 0.4
    cfg_router.mesa_threshold = 0.5
    cfg_router.mesa_max_nodes = 5
    qr = QueryRouter(graphlite_store=gstore, faiss_index=faiss_index,
                     tfidf_index=tfidf_index, encoder=enc, config=cfg_router,
                     faiss_id_map=faiss_id_map, episode_cache={})

    # 【关键】绕过生产近常数 boost: 基础 pass 分数保持纯融合分。
    import retrieval.query_router as _qrmod
    _qrmod.QueryRouter._apply_time_decay = staticmethod(
        lambda results, now_ts=None: results)

    return qr, gstore, msg_by_id, msg_created, RetrievalLevel


# ──────────────────────────────────────────────────────────────────────────
# 3. 主流程
# ──────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    sessdt, convlast = load_gaps()
    questions = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8") if l.strip()]
    if N > 0:
        questions = questions[:N]
    print(f"[1] 数据: conv={len(sessdt)} (session,date), questions={len(questions)} "
          f"(N={N}) ({time.time()-t0:.0f}s)", flush=True)

    engines = build_engines()
    qr, _gstore, msg_by_id, msg_created, RetrievalLevel = build_retriever()
    print(f"[2] 检索器就绪 + 引擎 {list(engines)} (tau_decay={TAU_DECAY_SECONDS}s) "
          f"({time.time()-t0:.0f}s)", flush=True)

    # stats[subset][form] = {"tot": int, "hits": {k: int}}
    stats = {s: {f: {"tot": 0, "hits": {k: 0 for k in KS}} for f in FORMS}
             for s in SUBSETS}
    sens = {f: {"taus": [], "boosts": []} for f in FORMS}
    boost_delta = {f: [] for f in FORMS}

    n_used = 0
    n_no_date = 0
    n_no_probes = 0
    t_retr = 0.0

    for i, q in enumerate(questions):
        ci = q.get("conversation_idx")
        evs = q.get("evidence_messages") or []
        ev_epochs = [sessdt.get((ci, m.get("session_index"))) for m in evs]
        ev_epochs = [e for e in ev_epochs if e is not None]
        if not ev_epochs or convlast.get(ci) is None:
            n_no_date += 1
            continue
        probes = probes_of([e.get("text", "") for e in evs])
        if not probes:
            n_no_probes += 1
            continue

        gap_days = (convlast[ci] - max(ev_epochs)) / DAY
        sub = "long" if gap_days > 3 else ("short" if gap_days < 1 else None)

        anchor = convlast[ci]
        _t = time.time()
        raw = qr.retrieve(q["question"], level=RetrievalLevel.FUSION,
                          session_ts=anchor, hyde=False)
        t_retr += time.time() - _t

        cands = []
        seen = set()
        for r in raw:
            nid = r.get("node_id")
            if not nid or nid in seen:
                continue
            seen.add(nid)
            content = r.get("content") or msg_by_id.get(nid, "")
            base = float(r.get("score") or 0.0)
            cands.append((nid, content, msg_created.get(nid), base))

        # τ / boost per candidate per form
        tau_by_cand = []
        boost_by_cand = []
        for (nid, _content, ca, _base) in cands:
            taus = {}
            boosts = {}
            for form in FORMS:
                if ca is not None:
                    tau = engines[form].compute_tau(
                        nid, created_at=float(ca), force_now=anchor)
                else:
                    tau = 0.0
                taus[form] = tau
                boosts[form] = _boost(tau)
            tau_by_cand.append(taus)
            boost_by_cand.append(boosts)

        for form in FORMS:
            for idx in range(len(cands)):
                sens[form]["taus"].append(tau_by_cand[idx][form])
                sens[form]["boosts"].append(boost_by_cand[idx][form])
                boost_delta[form].append(
                    abs(boost_by_cand[idx][form] - boost_by_cand[idx]["exp"]))

        # per-form 离线重排 + hit@k
        per_form_topk = {}
        for form in FORMS:
            scored = [
                (cands[idx][3] * boost_by_cand[idx][form], cands[idx][1])
                for idx in range(len(cands))
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            per_form_topk[form] = [c for _s, c in scored]

        for s in ([sub] if sub else []) + ["all"]:
            for form in FORMS:
                topk = per_form_topk[form]
                for k in KS:
                    if any(any(p in d for d in topk[:k]) for p in probes):
                        stats[s][form]["hits"][k] += 1
                stats[s][form]["tot"] += 1

        n_used += 1
        if (i + 1) % 100 == 0 or i == len(questions) - 1:
            print(f"  [{i+1}/{len(questions)}] used={n_used} "
                  f"retr={t_retr:.0f}s elapsed={time.time()-t0:.0f}s", flush=True)

    # ── 汇总 ──
    def pct(sub, form, k):
        t = stats[sub][form]["tot"]
        return 100.0 * stats[sub][form]["hits"][k] / t if t else 0.0

    long_did = {f: pct("long", f, 10) - pct("long", "exp", 10) for f in FORMS}
    short_drop = {f: pct("short", "exp", 10) - pct("short", f, 10) for f in FORMS}
    headroom = any(long_did[f] > 4.4 and short_drop[f] <= 2.0 for f in FORMS)

    sensout = {}
    for form in FORMS:
        taus = sens[form]["taus"]
        boosts = sens[form]["boosts"]
        sensout[form] = {
            "tau_min": min(taus) if taus else None,
            "tau_median": statistics.median(taus) if taus else None,
            "tau_max": max(taus) if taus else None,
            "boost_min": min(boosts) if boosts else None,
            "boost_max": max(boosts) if boosts else None,
            "boost_spread": (max(boosts) - min(boosts)) if boosts else None,
            "max_abs_score_delta_vs_exp": max(boost_delta[form]) if boost_delta[form] else None,
        }

    banner = (
        "MISSION rule: τ-form swap is admitted ONLY if long_did(hit@10) > 4.4 pts "
        "AND short_drop(hit@10) <= 2.0 pts. "
        "A NULL result with tiny boost spread is attributable to τ-boost "
        "insensitivity: the production boost 1+1/(1+exp(-τ/60)) is near-constant "
        "because τ∈[0,1] maps to a narrow band (~1.5), so re-ranking cannot move "
        "the candidate order."
    )

    payload = {
        "config": {
            "db_path": DB_PATH, "data_path": DATA_PATH, "conv_path": CONV_PATH,
            "N": N, "tau_decay_seconds": TAU_DECAY_SECONDS,
            "forms": list(FORMS), "ks": list(KS),
            "n_evaluated": n_used, "elapsed_s": round(time.time() - t0, 2),
            "retrieval_s": round(t_retr, 2),
        },
        "subset_sizes": {
            "long": stats["long"]["exp"]["tot"],
            "short": stats["short"]["exp"]["tot"],
            "all": stats["all"]["exp"]["tot"],
            "skipped_no_date": n_no_date,
            "skipped_no_probes": n_no_probes,
        },
        "hit_at_k": {
            f: {s: {str(k): round(pct(s, f, k), 4) for k in KS} for s in SUBSETS}
            for f in FORMS
        },
        "long_did_hit10": {f: round(long_did[f], 4) for f in FORMS},
        "short_drop_hit10": {f: round(short_drop[f], 4) for f in FORMS},
        "headroom": headroom,
        "sensitivity": sensout,
        "mission_banner": banner,
    }

    # ── 文本表格 ──
    L = []
    L.append("=== τ-form 检索召回门 (zero-LLM, read-only) ===")
    L.append(f"库: {DB_PATH}")
    L.append(f"问题: {len(questions)} | 有效评测: {n_used} | "
             f"跳过期(无日期/无证据探针): {n_no_date}/{n_no_probes}")
    L.append(f"子集: long(gap>3d)={payload['subset_sizes']['long']} "
             f"short(gap<1d)={payload['subset_sizes']['short']} "
             f"all={payload['subset_sizes']['all']} "
             f"(mid 1–3d={payload['subset_sizes']['all'] - payload['subset_sizes']['long'] - payload['subset_sizes']['short']})")
    L.append(f"tau_decay_seconds={TAU_DECAY_SECONDS} | 耗时={payload['config']['elapsed_s']}s "
             f"(检索 {payload['config']['retrieval_s']}s)")

    for s in SUBSETS:
        L.append("")
        L.append(f"[{s}] n={stats[s]['exp']['tot']}")
        L.append(f"  {'form':<6}" + "".join(f"{'@'+str(k):>9}" for k in KS))
        for f in FORMS:
            row = f"  {f:<6}" + "".join(f"{pct(s, f, k):>9.2f}" for k in KS)
            if s == "long":
                row += f"   Δ@10={long_did[f]:+.2f}"
            if s == "short":
                row += f"   drop@10={short_drop[f]:+.2f}"
            L.append(row)

    L.append("")
    L.append("τ / boost 敏感性诊断 (全候选):")
    L.append(f"  {'form':<6}{'tau_min':>12}{'tau_med':>12}{'tau_max':>12}"
             f"{'boost_min':>12}{'boost_max':>12}{'spread':>12}{'Δboost_exp':>12}")
    for f in FORMS:
        d = sensout[f]
        L.append(f"  {f:<6}{d['tau_min']:>12.5f}{d['tau_median']:>12.5f}"
                 f"{d['tau_max']:>12.5f}{d['boost_min']:>12.5f}{d['boost_max']:>12.5f}"
                 f"{d['boost_spread']:>12.6f}{d['max_abs_score_delta_vs_exp']:>12.6f}")

    L.append("")
    L.append(f"headroom (long_did>4.4 AND short_drop<=2.0): {headroom}")
    L.append("")
    L.append(banner)

    report = "\n".join(L)
    print("\n" + report)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nJSON → {OUT}\nTXT  → {TXT_OUT}")


if __name__ == "__main__":
    main()
