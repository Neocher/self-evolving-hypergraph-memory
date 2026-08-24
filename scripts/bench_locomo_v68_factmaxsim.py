# -*- coding: utf-8 -*-
"""v68 融合方案（MindMemOS 大池 × EverOS 事实级 MaxSim）——200 问对比 v66

管道：灌库 → 全库 LLM 实体-属性抽取（triples）→ 事实向量索引（FAISS）
     → query: 消息级 FUSION（v66 大池 200）⊕ 事实级 MaxSim（query→事实 top-20→聚合 episode top-10）
     → rerank 池 = FUSION + 事实聚合（去重）→ top-50 → LLM
"""
import json, os, re, sys, time
sys.path.insert(0, "/home/admin/shm")
sys.path.insert(0, "/home/admin/.hermes/skills/research/memory-benchmark-eval/scripts")
import numpy as np
from rag_v4_common import llm_generate, llm_judge, rerank, get_reranker

DATA = "/home/admin/shm/data/bench/locomo10.json"
DB_PATH = "/tmp/locomo_og_eval_v68"
SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
RERANK_POOL = int(os.environ.get("RERANK_POOL", "200"))
RERANK_TOP = int(os.environ.get("RERANK_TOP", "50"))
CTX_TOKENS = int(os.environ.get("CTX_TOKENS", "20000"))
FACT_TOP = int(os.environ.get("FACT_TOP", "20"))        # 事实检索 top-k
FACT_EP_CAP = int(os.environ.get("FACT_EP_CAP", "10"))  # MaxSim 聚合 episode 上限
EXTRACT_LIMIT = int(os.environ.get("EXTRACT_LIMIT", "1300"))  # 覆盖前 200 问
print(f"v68 配置: pool={RERANK_POOL} top={RERANK_TOP} ctx={CTX_TOKENS} fact_top={FACT_TOP} ep_cap={FACT_EP_CAP} extract={EXTRACT_LIMIT}", flush=True)

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
qr = QueryRouter(graphlite_store=gstore, faiss_index=faiss_index, tfidf_index=tfidf_index,
                 encoder=enc, config=config, faiss_id_map=faiss_id_map, episode_cache=episode_cache)

# ── 实体-属性抽取（v64 4.5 段同款，产出 triples）──
import re as _re
def _parse_triple_objects(raw):
    objs = []
    try:
        s, e = raw.find("["), raw.rfind("]")
        if s >= 0 and e > s:
            arr = json.loads(raw[s:e + 1])
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
    except Exception:
        pass
    for m in _re.finditer(r'\{([^{}]*)\}', raw):
        seg = m.group(1)
        d = {}
        for key in ("entity", "attribute", "value", "dia_id"):
            km = _re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', seg)
            if km:
                d[key] = km.group(1).replace('\\"', '"')
        if d.get("entity") and d.get("attribute") and d.get("dia_id"):
            objs.append(d)
    return objs

_EXTRACT_PROMPT = """Extract ALL factual entity-attribute-value triples from these conversation messages. Be aggressive and exhaustive.
Rules:
- entity: person/place/thing proper noun (Caroline, Melanie, Paris, Yosemite)
- attribute: verb phrase or property (went_to, camped_at, bought, painted, plays, has_pet, birthday, works_at, favorite)
- value: the specific fact, include dates/places/names when present
- Extract ALL facts per message: activities, opinions, preferences, family, work, school, travel, pets, hobbies, purchases. 2-5 per message.
- dia_id MUST be the [D1:N] tag of the message the fact came from.
Output STRICT JSON list only: [{"entity": "...", "attribute": "...", "value": "...", "dia_id": "D1:N"}]
No other text. If nothing, output [].

Messages:
{messages}"""

def _extract_batch(batch_msgs):
    msgs_text = "\n".join(f"[{m['dia_id']}] {m['speaker']}: {m['text'][:200]}" for m in batch_msgs)
    try:
        raw = llm_generate(_EXTRACT_PROMPT.replace("{messages}", msgs_text), max_tokens=1800, temperature=0.0)
        return _parse_triple_objects(raw)
    except Exception:
        return []

# dia_id → ep 映射
_dia_to_ep = {}
for _ci in range(len(conv_map)):
    for _j in range(len(conv_map[_ci])):
        _dia_to_ep[f"D{_ci+1}:{_j+1}"] = f"ep_{sum(len(conv_map[c]) for c in range(_ci)) + _j}"

print("全库实体-属性抽取（LLM 分批）...", flush=True)
_triples_all = []
_mid_msgs = []
for _ci in range(len(conv_map)):
    for _j, _m in enumerate(conv_map[_ci]):
        _mid_msgs.append({"dia_id": f"D{_ci+1}:{_j+1}", "speaker": _m.split("]")[0].replace("[", "").strip() if "]" in _m else "", "text": _m})
if EXTRACT_LIMIT > 0:
    _mid_msgs = _mid_msgs[:EXTRACT_LIMIT]
_BATCH = 25
for _i in range(0, len(_mid_msgs), _BATCH):
    _got = _extract_batch(_mid_msgs[_i:_i + _BATCH])
    _triples_all.extend(_got)
    if (_i // _BATCH) % 5 == 0:
        print(f"  抽取 {min(_i+_BATCH, len(_mid_msgs))}/{len(_mid_msgs)} (cum {len(_triples_all)} triples)", flush=True)
print(f"抽取完成: {len(_triples_all)} triples", flush=True)
json.dump(_triples_all, open("/tmp/p01_triples_v68.json", "w"), ensure_ascii=False)

# ── 事实索引（EverOS AtomicFact 式）：triple → 事实文本 → 向量 FAISS ──
facts = []  # (fact_text, ep_id)
for _t in _triples_all:
    _e = (_t.get("entity") or "").strip()
    _a = (_t.get("attribute") or "").strip()
    _v = (_t.get("value") or "").strip()
    _ep = _dia_to_ep.get((_t.get("dia_id") or "").strip(), "")
    if not (_e and _a and _v and _ep):
        continue
    facts.append((f"{_e} {_a}: {_v}", _ep))
print(f"事实表: {len(facts)} 条", flush=True)

import faiss
if facts:
    fact_texts = [f[0] for f in facts]
    fact_vecs = np.asarray(enc.embed_batch(fact_texts), dtype=np.float32)
    fact_vecs = fact_vecs / np.linalg.norm(fact_vecs, axis=1, keepdims=True)
    idx_fact = faiss.IndexFlatIP(512)
    idx_fact.add(fact_vecs)
    print(f"事实索引: {idx_fact.ntotal} 向量", flush=True)
else:
    idx_fact = None

def fact_maxsim(question):
    """EverOS MaxSim：query → 事实 top-k → 按 episode 聚合（取 max score）→ 证据 episode"""
    if idx_fact is None or not facts:
        return []
    qv = np.asarray(enc.embed(question), dtype=np.float32).reshape(1, -1)
    qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    scores, fidx = idx_fact.search(qv, FACT_TOP)
    # 聚合：ep → max score
    ep_score = {}
    for s, fi in zip(scores[0], fidx[0]):
        ep = facts[fi][1]
        ep_score[ep] = max(ep_score.get(ep, -1.0), float(s))
    ranked = sorted(ep_score.items(), key=lambda x: -x[1])[:FACT_EP_CAP]
    out = []
    for ep, sc in ranked:
        c = msg_by_id.get(ep, "")
        if c:
            out.append((c, sc))
    return out

_MQ_STOP = {"According", "Apr", "April", "Aug", "August", "Dec", "December", "Did", "Does",
            "Feb", "February", "Friday", "How", "Jan", "January", "Jul", "July", "Jun", "June",
            "Mar", "March", "May", "Monday", "Nov", "November", "Oct", "October", "Saturday",
            "Sep", "Sept", "September", "Sunday", "The", "Thursday", "Tuesday", "Wednesday",
            "What", "When", "Where", "Which", "Who", "Whose"}

def multi_query_expand(question):
    prompt = f"""Generate 2 alternative search queries for finding the answer to this question in a conversation log.
Question: {question}
Output a JSON list of 2 query strings, e.g. ["query1", "query2"]. One should target the entity/people involved, one should target the specific fact/attribute.
No other text."""
    try:
        raw = llm_generate(prompt, max_tokens=120, temperature=0.0)
        start, end = raw.find("["), raw.rfind("]")
        if start >= 0 and end > start:
            arr = json.loads(raw[start:end + 1])
            return [str(x) for x in arr if isinstance(x, str) and x.strip()][:2]
    except Exception:
        pass
    return []

def retrieve_docs(question):
    queries = [question] + multi_query_expand(question)
    seen, docs = set(), []
    for q in queries:
        try:
            raw = qr.retrieve(q, level=RetrievalLevel.FUSION, session_ts=None, hyde=True)
        except Exception:
            continue
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
    # 相邻拉取
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
    # 事实级 MaxSim 通道（EverOS）——结果进 rerank 池（去重）
    try:
        for c, sc in fact_maxsim(question):
            if c[:200] not in seen:
                seen.add(c[:200])
                docs.append(c)
    except Exception:
        pass
    return docs

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

results = {"total": 0, "correct": 0, "errors": 0, "by_cat": {}}
t0 = time.time()
for i, q in enumerate(qa_all):
    qid = id(q)
    ci = qa_conv.get(qid, 0)
    question, gold, cat = q["question"], q["answer"], q.get("category", 0)
    session_ts = parse_session_ts(conv_ts.get(ci))
    docs = retrieve_docs(question)
    try:
        get_reranker()
        reranked = rerank(question, docs[:RERANK_POOL], top_n=RERANK_TOP)
        docs = [d for d, s in reranked]
    except Exception as e:
        print(f"  [rerank err] {e}", flush=True)
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
        print(f"  {i+1}/{len(qa_all)} acc={acc:.1f}% ({results['correct']}/{results['total']}) elapsed={time.time()-t0:.0f}s", flush=True)

acc = results["correct"] / max(1, results["total"]) * 100
print(f"\n=== v68 事实级融合评测 {SAMPLE_N}问 ===", flush=True)
print(f"准确率: {acc:.1f}% ({results['correct']}/{results['total']})", flush=True)
print(f"错误: {results['errors']} | 耗时: {time.time()-t0:.0f}s", flush=True)
for cat in sorted(results["by_cat"]):
    d = results["by_cat"][cat]
    print(f"  cat={cat}: {d['c']}/{d['t']} = {d['c']/max(1,d['t'])*100:.1f}%", flush=True)

out = "/tmp/locomo_v68_results.json"
json.dump({"acc": acc, "correct": results["correct"], "total": results["total"],
           "by_cat": results["by_cat"], "errors": results["errors"],
           "version": "6.4.0-factmaxsim", "facts": len(facts)},
          open(out, "w"), ensure_ascii=False, indent=2)
print(f"结果已存: {out}", flush=True)
