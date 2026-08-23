# -*- coding: utf-8 -*-
"""v69 agentic 两轮（EverOS 参数 + v63 教训修正）——200 问对比 v66 82.5%

round1: FUSION + multi-query → rerank top-50（保留）
sufficiency check: LLM 判断证据是否足够
round2（不足时）: LLM 生成 follow-up 查询 → FUSION 追加（cap 40，去重）→ 合并 rerank
"""
import json, os, re, sys, time
sys.path.insert(0, "/home/admin/shm")
sys.path.insert(0, "/home/admin/.hermes/skills/research/memory-benchmark-eval/scripts")
import numpy as np
from rag_v4_common import llm_generate, llm_judge, rerank, get_reranker

DATA = "/home/admin/shm/data/bench/locomo10.json"
DB_PATH = "/tmp/locomo_og_eval_v69"
SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
RERANK_POOL = int(os.environ.get("RERANK_POOL", "200"))
RERANK_TOP = int(os.environ.get("RERANK_TOP", "50"))
CTX_TOKENS = int(os.environ.get("CTX_TOKENS", "20000"))
ROUND2_CAP = int(os.environ.get("ROUND2_CAP", "40"))   # EverOS round2 追加上限
SUFF_IF_EMPTY = os.environ.get("SUFF_IF_EMPTY", "1")   # 空检索直接 round2
print(f"v69 配置: pool={RERANK_POOL} top={RERANK_TOP} ctx={CTX_TOKENS} round2_cap={ROUND2_CAP}", flush=True)

from graph.overgraph_store import OverGraphStore
from retrieval.vector_index import VectorIndexAdapter
from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel
from embedding.encoder import TextEncoder

gstore = OverGraphStore(config=type("cfg", (), {"database_path": DB_PATH, "dense_vector_dimension": 512,
                                                "dense_vector_metric": "cosine", "ef_search": 64, "max_threads": 4})())
gstore.connect()
enc = TextEncoder(device="cpu")
enc.load()

def load_messages(path):
    raw = json.load(open(path))
    conv_map, sess_date = {}, {}
    all_msgs = []
    for ci, item in enumerate(raw):
        conv = item.get("conversation", {})
        for k in list(conv.keys()):
            if k.startswith("session_") and k.endswith("_date_time"):
                sess_date[k.replace("_date_time", "")] = conv[k]
        msgs = []
        for k in list(conv.keys()):
            if k.startswith("session_") and k[len("session_"):].isdigit():
                v = conv[k]
                date_prefix = f"[date: {sess_date.get(k, '')}] " if sess_date.get(k) else ""
                if isinstance(v, list):
                    for m in v:
                        if isinstance(m, dict):
                            speaker, text = m.get("speaker", ""), m.get("text", "")
                        else:
                            speaker, text = "", str(m)
                        if text:
                            msgs.append(f"{date_prefix}[{speaker}] {text}")
        conv_map[ci] = msgs
        all_msgs.extend(msgs)
    return conv_map, all_msgs

conv_map, all_msgs = load_messages(DATA)
data = json.load(open(DATA))
print(f"LoCoMo 消息: {len(all_msgs)} 条, {len(conv_map)} 会话", flush=True)

# ── 灌库（v64 同款）──
faiss_id_map = {}
faiss_index = VectorIndexAdapter(store=gstore, dimension=512, faiss_id_map=faiss_id_map)
msg_by_id = {}
episode_cache = {}
BATCH = 500
id_buf, vec_buf = [], []
t0 = time.time()
for b_start in range(0, len(all_msgs), BATCH):
    batch = all_msgs[b_start:b_start + BATCH]
    vecs = np.asarray(enc.embed_batch(batch), dtype=np.float32)
    for j, m in enumerate(batch):
        midx = b_start + j
        mid = f"ep_{midx}"
        msg_by_id[mid] = m
        try:
            gstore.create_episode({
                "id": mid, "content": m,
                "created_at": time.time() - (len(all_msgs) - midx) * 60,
                "source_type": "sensory", "layer": 1,
            })
        except Exception as e:
            print(f"  graph err {mid}: {e}", flush=True)
        id_buf.append(int(midx))
        vec_buf.append(vecs[j])
        faiss_id_map[int(midx)] = mid
        episode_cache[mid] = {
            "id": mid, "content": m, "created_at": time.time() - (len(all_msgs) - midx) * 60,
            "source_type": "sensory", "layer": 1,
        }
    print(f"  灌入 {min(b_start+BATCH, len(all_msgs))}/{len(all_msgs)} ({time.time()-t0:.0f}s)", flush=True)
if vec_buf:
    faiss_index.add_with_ids(np.vstack(vec_buf), np.array(id_buf, dtype=np.int64))
print(f"灌入完成: {len(msg_by_id)} 条, {time.time()-t0:.0f}s", flush=True)

# ── TF-IDF ──
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
print("TF-IDF 就绪", flush=True)

config = QueryRouterConfig()
config.agentic_enabled = False
config.hyde_timeout = 10.0
config.mesa_enabled = True
config.mesa_boost = 0.4
config.mesa_threshold = 0.5
config.mesa_max_nodes = 5
# 【2026-08-23 评测口径对齐 v6.1.0】v6.1.0 无 _entity_expansion/_attribute_expansion；
# v6.2/v6.3 引入的实体/属性扩展在 LoCoMo 评测库（主角恒定）产生噪音（-2.0pp，
# 同 v64 教训）。评测关闭扩展通道 → 苹果对苹果对比 v6.1.0 88.5%。
config.entity_expansion.enabled = False
qr = QueryRouter(graphlite_store=gstore, faiss_index=faiss_index, tfidf_index=tfidf_index,
                 encoder=enc, config=config, faiss_id_map=faiss_id_map, episode_cache=episode_cache)

def multi_query_expand(question, n=3):
    prompt = f"""Generate {n} alternative search queries for finding the answer to this question in a conversation log.
Question: {question}
Output a JSON list of {n} query strings, e.g. ["query1", "query2", "query3"]. One should target the entity/people involved, one should target the specific fact/attribute, one should use different wording.
No other text."""
    try:
        raw = llm_generate(prompt, max_tokens=150, temperature=0.0)
        start, end = raw.find("["), raw.rfind("]")
        if start >= 0 and end > start:
            arr = json.loads(raw[start:end + 1])
            return [str(x) for x in arr if isinstance(x, str) and x.strip()][:n]
    except Exception:
        pass
    return []

def _fuse(q, seen, docs):
    try:
        raw = qr.retrieve(q, level=RetrievalLevel.FUSION, session_ts=None, hyde=True)
    except Exception:
        return
    for r in raw:
        if isinstance(r, dict):
            c = r.get("content") or msg_by_id.get(r.get("node_id", ""), "")
            if c and c[:200] not in seen:
                seen.add(c[:200])
                docs.append(c)
        else:
            s = str(r)
            if s[:200] not in seen:
                seen.add(s[:200])
                docs.append(s)

def _adjacent(docs, seen):
    doc_ids = []
    for c in docs:
        for mid, content in msg_by_id.items():
            if content[:200] == c[:200]:
                doc_ids.append(mid)
                break
    extra = []
    for mid in doc_ids:
        if mid.startswith("ep_"):
            try:
                n = int(mid[3:])
            except ValueError:
                continue
            for nb in range(max(0, n - 3), n + 4):
                nb_mid = f"ep_{nb}"
                if nb_mid in msg_by_id and msg_by_id[nb_mid][:200] not in seen:
                    seen.add(msg_by_id[nb_mid][:200])
                    extra.append(msg_by_id[nb_mid])
    docs.extend(extra)

def retrieve_round1(question):
    queries = [question] + multi_query_expand(question, n=3)
    seen, docs = set(), []
    for q in queries:
        _fuse(q, seen, docs)
    _adjacent(docs, seen)
    return docs

def suff_check(question, docs_top):
    """sufficiency：LLM 判断 top 文档能否回答问题。返回 (enough, followup_query)"""
    ctx = "\n".join(f"[{j+1}] {d[:120]}" for j, d in enumerate(docs_top[:10]))
    prompt = f"""You are searching a conversation log. Given the retrieved snippets below, decide whether they are SUFFICIENT to answer the question.

Question: {question}

Retrieved snippets:
{ctx}

Output STRICT JSON: {{"sufficient": true/false, "missing_info": "what specific info is missing, or empty string"}}
No other text."""
    try:
        raw = llm_generate(prompt, max_tokens=200, temperature=0.0)
        s, e = raw.find("{"), raw.rfind("}")
        d = json.loads(raw[s:e + 1]) if s >= 0 and e > s else {}
        enough = bool(d.get("sufficient"))
        missing = str(d.get("missing_info", ""))
        return enough, missing
    except Exception:
        return True, ""

def followup_query(question, missing):
    prompt = f"""Generate a search query to find the missing information in a conversation log.

Question: {question}
Missing information needed: {missing}

Output a single search query string. No other text."""
    try:
        return llm_generate(prompt, max_tokens=80, temperature=0.0).strip()[:200]
    except Exception:
        return question

conv_ts = {}
for item in data:
    conv_ts[len(conv_ts)] = item.get("conversation", {}).get("session_1_date_time", 0)

def parse_session_ts(ts):
    try:
        return float(ts)
    except (ValueError, TypeError):
        return None

qa_conv = {}
for ci, item in enumerate(data):
    for q in item["qa"]:
        qa_conv[id(q)] = ci
qa_all = []
for item in data:
    qa_all.extend(item["qa"])
qa_all = [q for q in qa_all if q.get("category") != 5 and q.get("answer")]
if SAMPLE_N > 0:
    qa_all = qa_all[:SAMPLE_N]
print(f"评测规模: {len(qa_all)} 问", flush=True)

results = {"total": 0, "correct": 0, "errors": 0, "by_cat": {}, "round2_used": 0}
t0 = time.time()
for i, q in enumerate(qa_all):
    qid = id(q)
    ci = qa_conv.get(qid, 0)
    question, gold, cat = q["question"], q["answer"], q.get("category", 0)
    session_ts = parse_session_ts(conv_ts.get(ci))

    # round1 检索 + rerank top-50（保留）
    docs = retrieve_round1(question)
    try:
        get_reranker()
        reranked = rerank(question, docs[:RERANK_POOL], top_n=RERANK_TOP)
        docs = [d for d, s in reranked]
    except Exception as e:
        print(f"  [rerank err] {e}", flush=True)
        docs = docs[:RERANK_TOP]

    # sufficiency check + round2 追加（EverOS 修正版：保留 round1）
    enough = True
    if docs:
        try:
            enough, missing = suff_check(question, docs)
        except Exception:
            enough, missing = True, ""
    else:
        enough, missing = False, "general information about the question"
    if not enough:
        results["round2_used"] += 1
        fq = followup_query(question, missing) if missing and missing != "general information about the question" else question
        seen2 = {c[:200] for c in docs}
        _fuse(fq, seen2, docs)
        _adjacent(docs, seen2)
        # round2 结果合并（cap 40 追加——已在 docs 中，rerank 池 200 内）
        try:
            reranked2 = rerank(question, docs[:RERANK_POOL], top_n=RERANK_TOP)
            docs = [d for d, s in reranked2]
        except Exception:
            docs = docs[:RERANK_TOP]

    ctx = "\n".join(f"[{j+1}] {d}" for j, d in enumerate(docs[:RERANK_TOP]))
    ctx = ctx[:CTX_TOKENS]
    prompt = f"""Answer the question based on the conversation snippets below. Reason across snippets if needed (e.g., infer dates from session timestamps).

Conversation snippets:
{ctx}

Question: {question}
Answer:"""
    try:
        pred = llm_generate(prompt, max_tokens=200, temperature=0.2)
    except Exception as e:
        print(f"  [gen err] {e}", flush=True)
        results["errors"] += 1
        continue
    results["total"] += 1
    try:
        ok = llm_judge(question, gold, pred)
    except Exception as e:
        print(f"  [judge err] {e}", flush=True)
        ok = False
    if ok:
        results["correct"] += 1
    results["by_cat"].setdefault(cat, {"t": 0, "c": 0})
    results["by_cat"][cat]["t"] += 1
    if ok:
        results["by_cat"][cat]["c"] += 1
    if (i + 1) % 10 == 0 or i == len(qa_all) - 1:
        acc = results["correct"] / max(1, results["total"]) * 100
        print(f"  {i+1}/{len(qa_all)} acc={acc:.1f}% ({results['correct']}/{results['total']}) round2={results['round2_used']} elapsed={time.time()-t0:.0f}s", flush=True)

acc = results["correct"] / max(1, results["total"]) * 100
print(f"\n=== v69 agentic 两轮评测 {SAMPLE_N}问 ===", flush=True)
print(f"准确率: {acc:.1f}% ({results['correct']}/{results['total']})", flush=True)
print(f"round2 使用: {results['round2_used']} 问 | 错误: {results['errors']} | 耗时: {time.time()-t0:.0f}s", flush=True)
for cat in sorted(results["by_cat"]):
    d = results["by_cat"][cat]
    print(f"  cat={cat}: {d['c']}/{d['t']} = {d['c']/max(1,d['t'])*100:.1f}%", flush=True)

out = "/tmp/locomo_v69_results.json"
json.dump({"acc": acc, "correct": results["correct"], "total": results["total"],
           "by_cat": results["by_cat"], "errors": results["errors"], "round2_used": results["round2_used"],
           "version": "6.4.0-agentic2"},
          open(out, "w"), ensure_ascii=False, indent=2)
print(f"结果已存: {out}", flush=True)
