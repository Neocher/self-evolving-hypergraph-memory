# -*- coding: utf-8 -*-
"""v64: Schema 自演化评测（v61 管道 + 图库 EntityNode 直查通道）
灌库 → 梦境（含实体持久化）→ multi-query + FUSION + schema直查候选 + 相邻拉取 + rerank top-20
"""
import json, os, re, shutil, sys, time
import numpy as np

DATA = "/home/admin/shm/data/bench/locomo10.json"
DB_PATH = "/tmp/locomo_og_eval_v64"
SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
# P0-② 池扩大参数（环境变量可配；默认 = v66 扩大配置，v61 原值 80/20/12000）
RERANK_POOL = int(os.environ.get("RERANK_POOL", "200"))
RERANK_TOP = int(os.environ.get("RERANK_TOP", "50"))
CTX_TOKENS = int(os.environ.get("CTX_TOKENS", "20000"))
print(f"P0-② 配置: pool={RERANK_POOL} top={RERANK_TOP} ctx={CTX_TOKENS}", flush=True)
sys.path.insert(0, "/home/admin/shm")
sys.path.insert(0, "/home/admin/.hermes/skills/research/memory-benchmark-eval/scripts")
from rag_v4_common import llm_generate, llm_judge, rerank, get_reranker

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH, ignore_errors=True)

from graph.overgraph_store import OverGraphStore
try:
    from graph.graphlite_store import EpisodeCache
except ImportError:
    from graph.common import EpisodeCache
from retrieval.vector_index import VectorIndexAdapter
store_cfg = type("cfg", (), {"database_path": DB_PATH, "dense_vector_dimension": 512,
                             "dense_vector_metric": "cosine", "ef_search": 64, "max_threads": 4})()
gstore = OverGraphStore(config=store_cfg)
gstore.connect()
episode_cache = EpisodeCache()
faiss_id_map: dict[int, str] = {}
faiss_index = VectorIndexAdapter(store=gstore, dimension=512, faiss_id_map=faiss_id_map)

# ─── 1. 编码器 ───
from embedding.encoder import TextEncoder
enc = TextEncoder(device="cpu")
enc.load()
print("encoder ready", flush=True)

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
print(f"LoCoMo 消息: {len(all_msgs)} 条, {len(conv_map)} 会话", flush=True)

msg_by_id = {}
id_buf, vec_buf = [], []
t0 = time.time()
BATCH = 1000
for b_start in range(0, len(all_msgs), BATCH):
    batch = all_msgs[b_start:b_start + BATCH]
    try:
        vecs = enc.embed_batch(batch)
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
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
    except Exception as e:
        print(f"  batch err {b_start}: {e}", flush=True)
if vec_buf:
    faiss_index.add_with_ids(np.vstack(vec_buf), np.array(id_buf, dtype=np.int64))
    print(f"FAISS 写入: {len(id_buf)} 条", flush=True)
print(f"灌入完成: {len(msg_by_id)} 条, {time.time()-t0:.0f}s", flush=True)

# ─── 2. TF-IDF ───
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

# ─── 3. QueryRouter ───
from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel
config = QueryRouterConfig()
config.agentic_enabled = False
config.hyde_timeout = 10.0
config.mesa_enabled = True
config.mesa_boost = 0.4
config.mesa_threshold = 0.5
config.mesa_max_nodes = 5
qr = QueryRouter(
    graphlite_store=gstore, faiss_index=faiss_index, tfidf_index=tfidf_index,
    encoder=enc, config=config, faiss_id_map=faiss_id_map, episode_cache=episode_cache,
)
print("QueryRouter ready", flush=True)

# ─── 4. 梦境（生产签名，含实体持久化）—— SKIP_DREAM=1 时跳过（梦境慢且评测价值低，实体由 4.5 段全库抽取）───
import os as _os
if _os.environ.get("SKIP_DREAM", "0") != "1":
    from core.dream_pipeline import DreamPipeline
    from core.tau_decay import TauDecayEngine, TauDecayConfig
    from core.hebbian import SparseHebbianUpdater
    from core.audit_chain import AuditChain
    from core.confidence_calibrator import ConfidenceCalibrator
    from core.llm_client import LLMClient
    llm_client = LLMClient()
    llm_client.hot_reload()
    dp = DreamPipeline(
        tau_engine=TauDecayEngine(TauDecayConfig()),
        hebbian_updater=SparseHebbianUpdater(),
        audit_chain=AuditChain(),
        llm_client=llm_client,
        confidence_calibrator=ConfidenceCalibrator(),
    )
    print("梦境开始...", flush=True)
    try:
        import asyncio
        nodes = []
        rows = gstore.query_cypher("MATCH (e:EpisodeNode) WHERE (e.archived IS NULL OR e.archived = false) RETURN e ORDER BY e.created_at DESC LIMIT 10000")
        for row in rows or []:
            if isinstance(row, dict):
                flat = gstore._flatten_row(row, "e")
                if flat:
                    try:
                        flat["created_at"] = float(flat.get("created_at", 0))
                    except (ValueError, TypeError):
                        flat["created_at"] = 0.0
                    try:
                        flat["tau_initial"] = float(flat.get("tau_initial", 1.0))
                    except (ValueError, TypeError):
                        flat["tau_initial"] = 1.0
                    nodes.append(flat)
        connections = {}
        edge_rows = gstore.query_cypher("MATCH (a)-[r:HEBBIAN_CONNECTION]->(b) RETURN a.id AS src, b.id AS dst, r.weight AS w LIMIT 5000")
        for row in edge_rows or []:
            if isinstance(row, dict):
                s, d = row.get("src", ""), row.get("dst", "")
                try:
                    w = float(row.get("w", 0))
                except (ValueError, TypeError):
                    w = 0.0
                if s and d:
                    connections.setdefault(s, {})[d] = w
        print(f"梦境数据源: {len(nodes)} nodes, {len(connections)} connections", flush=True)
        asyncio.run(dp.run(nodes, connections, "explicit", graphlite_store=gstore, candidate_store=None))
        print("梦境完成", flush=True)
    except Exception as e:
        print(f"梦境异常（继续）: {e}", flush=True)

ents = gstore.get_entities()
print(f"EntityNode 落库: {len(ents)}", flush=True)

# ─── 4.5 评测层全库实体抽取+持久化（SKIP_EXTRACT=1 时跳过——纯池扩大对比用）───
if _os.environ.get("SKIP_EXTRACT", "0") != "1":
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

if _os.environ.get("SKIP_EXTRACT", "0") == "1":
    print("跳过实体抽取（SKIP_EXTRACT=1）", flush=True)
    _triples_all = []
else:
    print("全库实体抽取（LLM 分批）...", flush=True)
    _BATCH = 25
    _triples_all = []
    _mid_msgs = []
    _EXTRACT_LIMIT = int(os.environ.get("EXTRACT_LIMIT", "0"))  # 0 = 全量；N = 只抽前 N 条消息（冒烟）
    for _ci in range(len(conv_map)):
        for _j, _m in enumerate(conv_map[_ci]):
            _mid_msgs.append({"dia_id": f"D{_ci+1}:{_j+1}", "speaker": _m.split("]")[0].replace("[", "").strip() if "]" in _m else "", "text": _m})
    if _EXTRACT_LIMIT > 0:
        _mid_msgs = _mid_msgs[:_EXTRACT_LIMIT]
    # dia_id 对齐：conv_map 展开序与灌入 ep_ 一致（msg_by_id ep_i ↔ dia_id D{ci+1}:{j+1}）
    _dia_to_ep = {}
    for _ci in range(len(conv_map)):
        for _j in range(len(conv_map[_ci])):
            _dia_to_ep[f"D{_ci+1}:{_j+1}"] = f"ep_{sum(len(conv_map[c]) for c in range(_ci)) + _j}"
    for _i in range(0, len(_mid_msgs), _BATCH):
        _got = _extract_batch(_mid_msgs[_i:_i + _BATCH])
        _triples_all.extend(_got)
        if (_i // _BATCH) % 5 == 0:
            print(f"  抽取 {min(_i+_BATCH, len(_mid_msgs))}/{len(_mid_msgs)} (cum {len(_triples_all)} triples)", flush=True)
    print(f"抽取完成: {len(_triples_all)} triples", flush=True)

    _persisted = 0
    for _t in _triples_all:
        _e = (_t.get("entity") or "").strip()
        _a = (_t.get("attribute") or "").strip()
        _d = (_t.get("dia_id") or "").strip()
        _ep = _dia_to_ep.get(_d, "")
        if not _e or not _a or not _ep:
            continue
        try:
            _eid = gstore.create_entity(_e, entity_type="Person")
            # 属性版本链
            try:
                gstore.create_property_version(_eid, _a, (_t.get("value") or "")[:200])
            except Exception:
                pass
            gstore.link_entity_to_episode(_e, _ep)
            _persisted += 1
        except Exception:
            continue
    ents = gstore.get_entities()
    print(f"实体持久化完成: {_persisted} triples → EntityNode {len(ents)}", flush=True)
    json.dump(_triples_all, open("/tmp/p01_triples_v64.json", "w"), ensure_ascii=False)

# ─── 5. 评测（multi-query + FUSION + schema直查 + 相邻拉取 + rerank top-20）───
_MQ_STOP = {"According", "Apr", "April", "Aug", "August", "Dec", "December", "Did", "Does",
            "Feb", "February", "Friday", "How", "Jan", "January", "Jul", "July", "Jun", "June",
            "Mar", "March", "May", "Monday", "Nov", "November", "Oct", "October", "Saturday",
            "Sep", "Sept", "September", "Sunday", "The", "Thursday", "Tuesday", "Wednesday",
            "What", "When", "Where", "Which", "Who", "Whose"}

# ── schema 属性级直查全局（v65 A'）──
_ATTR_PREFIXES = ["went_to_", "went_", "has_", "likes_", "loves_", "wants_to_", "is_",
                  "plays_", "playing_", "painted_", "visited_", "visit_", "go_to_",
                  "going_to_", "took_", "made_", "worked_at_", "works_at_", "born_in_",
                  "studied_", "studies_", "interested_in_", "planning_", "plans_to_"]
def _attr_core(a):
    a = (a or "").lower().strip()
    for _p in _ATTR_PREFIXES:
        if a.startswith(_p):
            a = a[len(_p):]
            break
    for _suf in ("ing", "ed", "es", "s"):
        if len(a) > 4 and a.endswith(_suf):
            a = a[: -len(_suf)]
            break
    return a.strip(" _-")

_attr_cache = {}
def _build_attr_cache(gstore):
    """实体 → 属性名列表（PropertyVerNode 查询，全局一次）"""
    global _attr_cache
    _attr_cache = {}
    try:
        for _ae in gstore.get_entities(limit=500):
            _eid = _ae.get("id", "")
            if not _eid:
                continue
            try:
                _rows = gstore.query_cypher(
                    "MATCH (p:PropertyVerNode) WHERE p.entity_id = $eid RETURN DISTINCT p.attr_name AS a",
                    {"eid": _eid})
                _attr_cache[_eid] = [r.get("a") for r in (_rows or []) if r.get("a")]
            except Exception:
                _attr_cache[_eid] = []
    except Exception:
        pass

def _attr_lookup(question, gstore, msg_by_id, seen, ents_max=3, attr_cap=6):
    """LLM 解析 (entity, attribute) → 属性版本 → 属性值定位消息（限流防噪音）"""
    try:
        _raw = llm_generate(
            f"""Parse this question into a structured query.
Question: {question}
Output STRICT JSON: {{"entities": ["entity1", ...], "attribute": "attribute"}}
No other text.""", max_tokens=120, temperature=0.0)
        _s, _e = _raw.find("{"), _raw.rfind("}")
        _parsed = json.loads(_raw[_s:_e + 1]) if _s >= 0 and _e > _s else {}
    except Exception:
        return []
    _ents = [str(x).strip() for x in _parsed.get("entities", []) if str(x).strip()]
    _qa = _attr_core(_parsed.get("attribute", ""))
    _out = []
    for _en in _ents[:ents_max]:
        try:
            _ent = gstore.get_entity(_en)
            if not _ent:
                continue
            _eid = _ent["id"]
            _found_attr = None
            _versions = []
            for _ta in _attr_cache.get(_eid, []):
                if not _qa or _attr_core(_ta) == _qa or _attr_core(_ta) in _qa or _qa in _attr_core(_ta):
                    _found_attr = _ta
                    break
            if _found_attr:
                _versions = gstore.get_property_versions(_eid, _found_attr)
            for _v in _versions:
                _val = str(_v.get("value", ""))[:60]
                if not _val:
                    continue
                for _mid, _ct in msg_by_id.items():
                    if _val.lower()[:40] in _ct.lower():
                        if _ct[:200] not in seen:
                            seen.add(_ct[:200])
                            _out.append(_ct)
                        if len(_out) >= attr_cap:
                            return _out
        except Exception:
            continue
    return []

_build_attr_cache(gstore)
print(f"属性直查缓存: {sum(len(v) for v in _attr_cache.values())} attrs / {len(_attr_cache)} entities", flush=True)

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
    # ── schema 属性级直查（v65 A'）—— SCHEMA_OFF=1 时跳过（P0-② 纯池扩大对比）──
    if _os.environ.get("SCHEMA_OFF", "0") != "1":
        _schema_docs = _attr_lookup(question, gstore, msg_by_id, seen)
        docs.extend(_schema_docs)
    # 相邻拉取（简化：命中 ep_ 前缀取 ±3）
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
                if nb_mid in msg_by_id and nb_mid[:200] not in seen:
                    seen.add(msg_by_id[nb_mid][:200])
                    extra.append(msg_by_id[nb_mid])
    docs.extend(extra)
    return docs

data = json.load(open(DATA))
qa_all = []
for item in data:
    qa_all.extend(item["qa"])
qa_all = [q for q in qa_all if q.get("category") != 5 and q.get("answer")]
qa_all = qa_all[:SAMPLE_N]
print(f"评测规模: {len(qa_all)} 问", flush=True)

conv_ts = {}
for ci, item in enumerate(data):
    conv = item.get("conversation", {})
    for k in list(conv.keys()):
        if k.startswith("session_") and k.endswith("_date_time"):
            conv_ts[ci] = conv[k]
            break

def parse_session_ts(s):
    if not s:
        return None
    from datetime import datetime
    for fmt in ("%I:%M %p on %d %B, %Y", "%d %B %Y", "%I:%M %p on %d %B %Y"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except Exception:
            continue
    return None

qa_conv = {}
for ci, item in enumerate(data):
    for q in item["qa"]:
        qa_conv[id(q)] = ci

results = {"total": 0, "correct": 0, "errors": 0, "by_cat": {}}
t0 = time.time()

# 全量模式：checkpoint + retry + resume（FULL=1）
CKPT_PATH = "/tmp/locomo_v66_full_ckpt.json"
done_qids = set()
if _os.environ.get("FULL", "0") == "1" and os.path.exists(CKPT_PATH):
    try:
        ck = json.load(open(CKPT_PATH))
        results = ck.get("results", results)
        done_qids = set(ck.get("done_qids", []))
        print(f"checkpoint 恢复: {len(done_qids)} 问已完成", flush=True)
    except Exception as e:
        print(f"checkpoint 读取失败（从头跑）: {e}", flush=True)

def _gen_retry(prompt, max_tokens=200, temperature=0.2, tries=3):
    last: Exception | None = None
    for _t in range(tries):
        try:
            return llm_generate(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            last = e
            print(f"  [gen retry {_t+1}] {str(e)[:60]}", flush=True)
    if last is None:
        raise RuntimeError("llm_generate failed without exception")
    raise last

def _judge_retry(question, gold, pred, tries=3):
    last: Exception | None = None
    for _t in range(tries):
        try:
            return llm_judge(question, gold, pred)
        except Exception as e:
            last = e
            print(f"  [judge retry {_t+1}] {str(e)[:60]}", flush=True)
    if last is None:
        raise RuntimeError("llm_judge failed without exception")
    raise last

for i, q in enumerate(qa_all):
    qid = id(q)
    if qid in done_qids:
        continue
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
        pred = _gen_retry(prompt, max_tokens=200, temperature=0.2)
    except Exception as e:
        print(f"  [gen err] {e}", flush=True)
        results["errors"] += 1
        done_qids.add(qid)
        continue
    results["total"] += 1
    try:
        ok = _judge_retry(question, gold, pred)
    except Exception as e:
        print(f"  [judge err] {e}", flush=True)
        ok = False
    if ok:
        results["correct"] += 1
    results["by_cat"].setdefault(cat, {"t": 0, "c": 0})
    results["by_cat"][cat]["t"] += 1
    if ok:
        results["by_cat"][cat]["c"] += 1
    done_qids.add(qid)
    if _os.environ.get("FULL", "0") == "1" and (i + 1) % 50 == 0:
        try:
            json.dump({"results": results, "done_qids": list(done_qids)},
                      open(CKPT_PATH, "w"), ensure_ascii=False)
            print(f"  [ckpt] {len(done_qids)} 问已保存", flush=True)
        except Exception as e:
            print(f"  [ckpt err] {e}", flush=True)
    if (i + 1) % 10 == 0 or i == len(qa_all) - 1:
        acc = results["correct"] / max(1, results["total"]) * 100
        print(f"  {i+1}/{len(qa_all)} acc={acc:.1f}% ({results['correct']}/{results['total']}) elapsed={time.time()-t0:.0f}s", flush=True)

acc = results["correct"] / max(1, results["total"]) * 100
print(f"\n=== v64 schema 评测 {SAMPLE_N}问 ===", flush=True)
print(f"准确率: {acc:.1f}% ({results['correct']}/{results['total']})", flush=True)
print(f"错误: {results['errors']} | 耗时: {time.time()-t0:.0f}s", flush=True)
for cat in sorted(results["by_cat"]):
    d = results["by_cat"][cat]
    print(f"  cat={cat}: {d['c']}/{d['t']} = {d['c']/max(1,d['t'])*100:.1f}%", flush=True)

out = "/tmp/locomo_v66_full_results.json" if _os.environ.get("FULL", "0") == "1" else "/tmp/locomo_v64_schema_results.json"
json.dump({"acc": acc, "correct": results["correct"], "total": results["total"],
           "by_cat": results["by_cat"], "errors": results["errors"],
           "version": "6.4.0-schema", "entities": len(ents)},
          open(out, "w"), ensure_ascii=False, indent=2)
print(f"结果已存: {out}", flush=True)
