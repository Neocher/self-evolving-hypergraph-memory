"""方案 A-OverGraph：生产数据评测 —— 真实 OverGraph 引擎 + LoCoMo 灌入 + 完整 QueryRouter 评测

SHM v6.0.0 阶段2：/tmp/bench_locomo_p3b.py 的 overgraph 后端版。
变更点（其余逻辑逐行复用）:
  - 图后端: GraphLiteStore → OverGraphStore（config backend 切换语义）
  - 向量: FaissStore → VectorIndexAdapter（OverGraph HNSW，faiss.Index 鸭子类型）
  - 灌入: create_episode（OverGraph 写路径）+ adapter.add_with_ids
           → store.batch_upsert_embeddings（dense_vector 一等字段写路径验证）

用法: python /tmp/bench_locomo_p3b_overgraph.py <N> <mode>
  mode: baseline (FUSION 无梦境) / mesa (社区+MESA) / attr (属性+Schema) / fusion (全通道)
"""
import os, sys, json, time, uuid, shutil, tempfile
import numpy as np

# 独立进程需显式设 GraphLite 环境（systemd 服务自带）；overgraph 无环境依赖
os.environ.setdefault("GRAPHLITE_BINDINGS", "/home/admin/GraphLite/bindings/python")
os.environ.setdefault("GRAPHLITE_SDK", "/home/admin/GraphLite/sdk-python/src")
# 【P3b】加载 .env 的 DEEPSEEK_API_KEY 到环境（生产 hyde.py 从 os.environ 读）
for _envp in ("/home/admin/.hermes/.env", "/home/admin/shm/.env"):
    if os.path.exists(_envp):
        for _line in open(_envp):
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL"):
                    os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))

sys.path.insert(0, "/home/admin/shm")
os.chdir("/home/admin/shm")

SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MODE = sys.argv[2] if len(sys.argv) > 2 else "baseline"
DATA = "/home/admin/shm/data/bench/locomo10.json"
DB_PATH = f"/tmp/locomo_og_eval_{MODE}"  # 独立临时 OverGraph 库（零污染）

print(f"=== 方案A-OverGraph 生产数据评测 [{MODE}] N={SAMPLE_N} ===", flush=True)
print(f"DB: {DB_PATH}", flush=True)

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH, ignore_errors=True)
print("旧库已清理", flush=True)

# ─── 1. 构造真实组件（生产代码）───
from config.settings import get_settings
from graph.graphlite_store import EpisodeCache
from graph.overgraph_store import OverGraphStore
from embedding.encoder import TextEncoder
from retrieval.vector_index import VectorIndexAdapter

cfg = get_settings()
print("settings loaded", flush=True)

store_cfg = type("cfg", (), {
    "database_path": DB_PATH,
    "dense_vector_dimension": 512,
    "dense_vector_metric": "cosine",
    "ef_search": 64,
})()
gstore = OverGraphStore(config=store_cfg)
gstore.connect()
print("OverGraphStore connected (real engine)", flush=True)

enc = TextEncoder(device="cpu")
enc.load()
print("encoder loaded", flush=True)

faiss_id_map: dict[int, str] = {}
faiss_index = VectorIndexAdapter(store=gstore, dimension=512, faiss_id_map=faiss_id_map)
episode_cache = EpisodeCache()

# ─── 2. 灌入 LoCoMo 消息（走生产写路径）───
def load_messages(path):
    raw = json.load(open(path))
    conv_map = {}
    sess_date = {}
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
texts_all = all_msgs[:]
for b_start in range(0, len(texts_all), BATCH):
    batch = texts_all[b_start:b_start + BATCH]
    try:
        vecs = enc.embed_batch(batch)
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        for j, m in enumerate(batch):
            midx = b_start + j
            mid = f"ep_{midx}"
            msg_by_id[mid] = m
            # OverGraph 写入（同步，embedding 已批量）
            try:
                gstore.create_episode({
                    "id": mid,
                    "content": m,
                    "created_at": time.time() - (len(all_msgs) - midx) * 60,
                    "source_type": "sensory",
                    "layer": 1,
                })
            except Exception as e:
                print(f"  graph err {mid}: {e}", flush=True)
            id_buf.append(int(midx))
            vec_buf.append(vecs[j])
            faiss_id_map[int(midx)] = mid          # faiss int id → node_id 映射
            episode_cache[mid] = {
                "id": mid, "content": m, "created_at": time.time() - (len(all_msgs) - midx) * 60,
                "source_type": "sensory", "layer": 1,
            }
        print(f"  灌入 {min(b_start+BATCH, len(texts_all))}/{len(texts_all)} (embed_batch {len(batch)}条, {time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  batch err {b_start}: {e}", flush=True)
if vec_buf:
    # VectorIndexAdapter.add_with_ids → OverGraphStore.batch_upsert_embeddings
    # （dense_vector 一等字段写路径）
    added = faiss_index.add_with_ids(np.vstack(vec_buf), np.array(id_buf, dtype=np.int64))
    print(f"dense_vector 写入: {added} 条 (batch_upsert_embeddings)", flush=True)
print(f"灌入完成: {len(msg_by_id)} 条, HNSW {len(id_buf)} 向量, {time.time()-t0:.0f}s", flush=True)

# ─── 3. TF-IDF（真实 BM25 通道）───
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
        if not self._fitted or not hasattr(self, "matrix"):
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
print("TF-IDF 索引就绪", flush=True)

# ─── 4. QueryRouter（完整生产代码 + OverGraph 引擎）───
from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel
config = QueryRouterConfig()
config.agentic_enabled = False
config.hyde_timeout = 10.0
config.mesa_enabled = (MODE in ("mesa", "fusion"))
if config.mesa_enabled:
    config.mesa_boost = 0.4
    config.mesa_threshold = 0.5
    config.mesa_max_nodes = 5
qr = QueryRouter(
    graphlite_store=gstore,
    faiss_index=faiss_index,
    tfidf_index=tfidf_index,
    encoder=enc,
    config=config,
    faiss_id_map=faiss_id_map,
    episode_cache=episode_cache,
)
print(f"QueryRouter 构造完成 (mesa={config.mesa_enabled})", flush=True)

# ─── 5. 真实梦境（跑生产 DreamPipeline：聚类+LLM 摘要+属性抽取）───
if MODE in ("mesa", "attr", "fusion"):
    from core.dream_pipeline import DreamPipeline
    from core.tau_decay import TauDecayEngine, TauDecayConfig
    from core.hebbian import SparseHebbianUpdater
    from core.audit_chain import AuditChain
    from core.confidence_calibrator import ConfidenceCalibrator
    from core.llm_client import LLMClient
    # 【FIX 2026-08-20】LLMClient(cfg) 会把 settings 对象当 api_key 位置参数 → 401。
    # 生产签名：LLMClient() 无参 + hot_reload() 从 ~/.hermes/.env 加载权威 key。
    llm_client = LLMClient()
    llm_client.hot_reload()
    dp = DreamPipeline(
        tau_engine=TauDecayEngine(TauDecayConfig()),
        hebbian_updater=SparseHebbianUpdater(),
        audit_chain=AuditChain(),
        llm_client=llm_client,
        confidence_calibrator=ConfidenceCalibrator(),
    )
    print("梦境开始（真实聚类+LLM 摘要）...", flush=True)
    try:
        import asyncio
        # 【FIX 2026-08-20】生产签名：run(nodes, connections, trigger_mode, graphlite_store, candidate_store)
        # 照抄 dream_scheduler._run_dream 取数逻辑（OverGraph 兼容）：
        #   EpisodeNode 全量（未归档）+ HEBBIAN_CONNECTION 边
        nodes: list[dict] = []
        connections: dict[str, dict[str, float]] = {}
        rows = gstore.query_cypher(
            "MATCH (e:EpisodeNode) "
            "WHERE (e.archived IS NULL OR e.archived = false) "
            "RETURN e "
            "ORDER BY e.created_at DESC LIMIT 10000"
        )
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
        edge_rows = gstore.query_cypher(
            "MATCH (a)-[r:HEBBIAN_CONNECTION]->(b) "
            "RETURN a.id AS src, b.id AS dst, r.weight AS w LIMIT 5000"
        )
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
        asyncio.run(dp.run(
            nodes, connections,
            "explicit",
            graphlite_store=gstore,
            candidate_store=None,
        ))
        print("梦境完成", flush=True)
    except Exception as e:
        print(f"梦境异常（继续评测）: {e}", flush=True)

# ─── 6. 评测（v5.48 完整管道：HyDE + retrieve(FUSION) + rerank + LLM judge）───
sys.path.insert(0, "/tmp")
from rag_v4_common import llm_generate, llm_judge, rerank, get_reranker

def embed_one(text):
    v = enc.embed(text)
    v = np.asarray(v, dtype=np.float32)
    return v if v.ndim == 1 else v[0]

data = json.load(open(DATA))
qa_all = []
for item in data:
    qa_all.extend(item["qa"])
qa_all = [q for q in qa_all if q.get("category") != 5 and q.get("answer")]
qa_all = qa_all[:SAMPLE_N]

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

for i, q in enumerate(qa_all):
    qid = id(q)
    ci = qa_conv.get(qid, 0)
    question = q["question"]
    gold = q["answer"]
    cat = q.get("category", 0)
    session_ts = parse_session_ts(conv_ts.get(ci))

    try:
        raw = qr.retrieve(
            question, level=RetrievalLevel.FUSION, session_ts=session_ts,
            hyde=True,
        )
    except Exception as e:
        print(f"  [retrieve err] {e}", flush=True)
        results["errors"] += 1
        continue

    docs = []
    for r in raw:
        if isinstance(r, dict):
            c = r.get("content") or msg_by_id.get(r.get("node_id", ""), "")
            if c:
                docs.append(c)
        else:
            docs.append(str(r))

    try:
        get_reranker()
        reranked = rerank(question, docs[:40], top_n=12)
        docs = [d for d, s in reranked]
    except Exception as e:
        print(f"  [rerank err] {e}", flush=True)
        docs = docs[:12]

    ctx = "\n".join(f"[{j+1}] {d}" for j, d in enumerate(docs[:12]))
    ctx = ctx[:6000]

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
print(f"\n=== v6.0.0 OverGraph 生产数据 LoCoMo {SAMPLE_N}问 [{MODE}] ===", flush=True)
print(f"准确率: {acc:.1f}% ({results['correct']}/{results['total']})", flush=True)
print(f"错误: {results['errors']} | 耗时: {time.time()-t0:.0f}s", flush=True)
for cat in sorted(results["by_cat"]):
    d = results["by_cat"][cat]
    print(f"  cat={cat}: {d['c']}/{d['t']} = {d['c']/max(1,d['t'])*100:.1f}%", flush=True)

out = f"/tmp/locomo_v60_overgraph_{MODE}_results.json"
json.dump({"acc": acc, "correct": results["correct"], "total": results["total"],
           "by_cat": results["by_cat"], "errors": results["errors"],
           "version": "6.0.0", "mode": MODE, "backend": "overgraph"},
          open(out, "w"), ensure_ascii=False, indent=2)
print(f"结果已存: {out}", flush=True)
print("完成", flush=True)
