# -*- coding: utf-8 -*-
"""HITK 检索责任分层 (OverGraph 直连版, 无本地 faiss)

用途: 达摩院 §4.4 责任分层 — 判断 LoCoMo-Refined 错题是检索责任(证据没召回)
      还是生成责任(召回了没用)。纯检索不调 LLM。

链路: questions.jsonl → bge-m3 embed → OverGraph dense HNSW (vector_search_dense)
      → top-k 消息文本 → 证据 25 字符滑动探针命中检测 → 按 cat 统计 hit@k

用法:
  SHM_ROOT=/home/user/self-evolving-hypergraph-memory \
  DB_PATH=/home/user/LoCoMo_refined/results/eval_db_v610 \
  DATA_PATH=/home/user/LoCoMo_refined/data/public/questions.jsonl \
  HITK_MODE_CHANNEL=vector|fusion \
  python3 scripts/hitk_overgraph.py [N] [OUT]
  N 默认 1382 (全量), OUT 默认 /tmp/hitk_report.txt
  HITK_MODE_CHANNEL: vector=纯 OverGraph dense | fusion=QueryRouter FUSION 生产链路 (hyde=False)
"""
import json, os, sys, time

SHM_ROOT = os.environ.get("SHM_ROOT", "/home/user/self-evolving-hypergraph-memory")
sys.path.insert(0, SHM_ROOT)
import numpy as np

DB_PATH = os.environ.get("DB_PATH", "/home/user/LoCoMo_refined/results/eval_db_v610")
DATA_PATH = os.environ.get("DATA_PATH", "/home/user/LoCoMo_refined/data/public/questions.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1382
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hitk_report.txt"
K_MAX = 40
MODE = os.environ.get("HITK_MODE_CHANNEL", "vector")  # vector=纯向量 | fusion=QueryRouter 生产链路
# P1 (2026-09-02): HITK_SESSION_TS=1 → fusion 检索传 conv 级真实时间锚 (由库内
# episode.session_id + created_at 聚合; P1 新库才有 session_id)。HITK_ANCHOR=first|last
# (默认 last=会话结束=回顾性问询时刻)。旧合成库无 session_id → 锚点空 → 回落 None。
HITK_SESSION_TS = os.environ.get("HITK_SESSION_TS") == "1"
HITK_ANCHOR = os.environ.get("HITK_ANCHOR", "last")

from graph.overgraph_store import OverGraphStore
from embedding.encoder import TextEncoder

# ── 1. OverGraph 直连 (生产检索链路) ──
cfg = type("cfg", (), {"database_path": DB_PATH, "dense_vector_dimension": 512,
                       "dense_vector_metric": "cosine", "ef_search": 64, "max_threads": 4})()
gstore = OverGraphStore(config=cfg)
gstore.connect()
t0 = time.time()

# 全量加载 EpisodeNode 消息文本 (id → content) + 会话时间元数据
rows = gstore._locked_execute_gql(
    "MATCH (e:EpisodeNode) RETURN e.id AS id, e.content AS content, "
    "e.created_at AS ts, e.session_id AS session_id", {})
rows = rows["rows"]
msg_by_id = {r["id"]: r.get("content", "") for r in rows}
conv_ts = {}   # conversation_idx → 真实时间锚 epoch (库内 session_id 聚合)
if HITK_SESSION_TS:
    _grp: dict = {}
    for r in rows:
        _sid = r.get("session_id")
        if _sid is None:
            continue
        try:
            _t = float(r.get("ts") or 0)
        except (TypeError, ValueError):
            continue
        if _t > 0:
            _grp.setdefault(int(_sid), []).append(_t)
    for _sid, _ts in _grp.items():
        conv_ts[_sid] = (max if HITK_ANCHOR == "last" else min)(_ts)
    if not conv_ts:
        print("[P1] HITK_SESSION_TS=1 但库无 session_id (旧合成库) → session_ts 回落 None", flush=True)
print(f"[1] OverGraph 加载: {len(msg_by_id)} 条 EpisodeNode, conv 时间锚 {len(conv_ts)} 会话"
      f" ({time.time()-t0:.0f}s)", flush=True)

# ── 2. 加载评测问题 ──
qa_all = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8") if l.strip()][:N]
print(f"[2] 问题集: {len(qa_all)} 条", flush=True)

# ── 3. encoder (与灌库同模型 bge-m3; 设备 auto: cuda 优先) ──
enc = TextEncoder(device=os.environ.get("ENC_DEVICE", "auto"))
enc.load()
print(f"[3] encoder 就绪 ({time.time()-t0:.0f}s)", flush=True)

# ── 3.5 检索器: vector=OverGraph dense 直查 | fusion=QueryRouter 生产链路 ──
if MODE == "fusion":
    from retrieval.vector_index import VectorIndexAdapter
    from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    class TfidfSearchIndex:
        def __init__(self):
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=5000)
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
    qr = QueryRouter(graphlite_store=gstore, faiss_index=faiss_index, tfidf_index=tfidf_index,
                     encoder=enc, config=cfg_router, faiss_id_map=faiss_id_map, episode_cache={})
    print(f"[3.5] fusion 检索器就绪 (QueryRouter FUSION, hyde=False) ({time.time()-t0:.0f}s)", flush=True)


def _retrieve_topk(question, conversation_idx=None):
    """返回 top-k 消息文本列表 (含 score 排序)。"""
    if MODE == "fusion":
        from retrieval.query_router import RetrievalLevel
        session_ts = conv_ts.get(conversation_idx) if HITK_SESSION_TS else None
        raw = qr.retrieve(question, level=RetrievalLevel.FUSION, session_ts=session_ts, hyde=False)
        docs = []
        for r in raw:
            if isinstance(r, dict):
                c = r.get("content") or msg_by_id.get(r.get("node_id", ""), "")
            else:
                c = str(r)
            if c and c not in docs:
                docs.append(c)
        return docs
    qv = np.asarray(enc.embed(question), dtype=np.float32).reshape(1, -1)
    qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    hits = gstore.vector_search_dense(k=K_MAX, query_vec=qv[0])
    return [msg_by_id.get(ep_id, "") for ep_id, _s in hits]

# ── 4. 逐题: 检索 top-k → 证据探针 ──
hitk_stats = {"total": 0, "by_cat": {}}
miss_retrieval = []   # 证据完全没进 top-40 (检索责任候选)
hit_generation = []   # 证据进了 top-40 (检索 OK, 责任在生成/重排侧)


def probes_of(ev_texts):
    probes = []
    for et in ev_texts:
        t = (et or "").strip()
        if not t:
            continue
        probes += [t[i:i + 25] for i in range(0, max(1, len(t) - 24), 20)][:8]
    return [p for p in probes if len(p) >= 15]


for i, q in enumerate(qa_all):
    cat_s = str(q.get("category", 0))
    ev_texts = [e.get("text", "") for e in (q.get("evidence_messages") or [])]
    c = hitk_stats["by_cat"].setdefault(cat_s, {"total": 0, "k1": 0, "k3": 0, "k5": 0,
                                                "k10": 0, "k20": 0, "k40": 0})
    c["total"] += 1
    hitk_stats["total"] += 1

    topk = _retrieve_topk(q["question"], q.get("conversation_idx"))
    probes = probes_of(ev_texts)
    for kk in (1, 3, 5, 10, 20, 40):
        if any(any(p in d for d in topk[:kk]) for p in probes):
            c[f"k{kk}"] += 1

    if probes and not any(any(p in d for d in topk) for p in probes):
        miss_retrieval.append(q.get("qa_id", i))
    elif probes:
        hit_generation.append(q.get("qa_id", i))

    if (i + 1) % 100 == 0 or i == len(qa_all) - 1:
        print(f"  [{i+1}/{len(qa_all)}] elapsed={time.time()-t0:.0f}s", flush=True)

# ── 5. 汇总报告 ──
lines = []
lines.append("=== HITK 检索责任分层报告 (OverGraph 直连, bge-m3) ===")
lines.append(f"模式: {MODE} | 库: {DB_PATH} | 消息: {len(msg_by_id)} | 问题: {len(qa_all)}")
lines.append("")
lines.append("类别命中率 (证据 25-char 探针是否进入 top-k 消息):")
for cat_s in sorted(hitk_stats["by_cat"], key=lambda x: int(x) if x.isdigit() else 99):
    d = hitk_stats["by_cat"][cat_s]
    t = max(1, d["total"])
    lines.append(
        f"  cat{cat_s}: n={d['total']} hit@1={d['k1']/t*100:.1f}% "
        f"@3={d['k3']/t*100:.1f}% @5={d['k5']/t*100:.1f}% "
        f"@10={d['k10']/t*100:.1f}% @20={d['k20']/t*100:.1f}% @40={d['k40']/t*100:.1f}%")
t = max(1, hitk_stats["total"])
lines.append(
    f"  全部: n={hitk_stats['total']} hit@1={hitk_stats['by_cat'] and sum(d['k1'] for d in hitk_stats['by_cat'].values())/t*100:.1f}% "
    f"@3={sum(d['k3'] for d in hitk_stats['by_cat'].values())/t*100:.1f}% "
    f"@5={sum(d['k5'] for d in hitk_stats['by_cat'].values())/t*100:.1f}% "
    f"@10={sum(d['k10'] for d in hitk_stats['by_cat'].values())/t*100:.1f}% "
    f"@20={sum(d['k20'] for d in hitk_stats['by_cat'].values())/t*100:.1f}% "
    f"@40={sum(d['k40'] for d in hitk_stats['by_cat'].values())/t*100:.1f}%")
lines.append("")
n_ev = len(miss_retrieval) + len(hit_generation)
lines.append(f"责任分层 (有证据问题 n={n_ev}):")
lines.append(f"  检索责任 (证据未进 top-40): {len(miss_retrieval)} ({len(miss_retrieval)/max(1,n_ev)*100:.1f}%)")
lines.append(f"  生成/重排责任 (证据已进 top-40): {len(hit_generation)} ({len(hit_generation)/max(1,n_ev)*100:.1f}%)")
lines.append(f"  检索责任 qa_id 示例: {miss_retrieval[:15]}")
lines.append("")
lines.append(f"耗时: {time.time()-t0:.0f}s")
report = "\n".join(lines)
print(report)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(f"\n报告已存: {OUT}")
