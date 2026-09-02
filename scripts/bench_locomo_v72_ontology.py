# -*- coding: utf-8 -*-
"""v71 三源融合（MindMemOS 记忆块 × EverOS agentic × semantica 图遍历）

= v70 LLM 压缩记忆块 + v69 agentic 两轮 + semantica 图遍历通道
管道：灌库 → LLM 压缩块 → 实体抽取（triples → EntityNode）→ 图遍历索引
     → query: 消息级 FUSION ⊕ 块级检索 ⊕ 图遍历（实体→属性关系/共现 episode）
     → rerank top-50 → sufficiency → round2 追加 → LLM
"""
import json, os, re, sys, time
sys.path.insert(0, os.environ.get("SHM_ROOT", "/home/admin/shm"))

import numpy as np
from rag_v4_common import llm_generate, llm_judge, rerank, get_reranker

DATA = os.environ.get("DATA_PATH", "/home/admin/shm/data/bench/locomo10.json")
DB_PATH = os.environ.get("DB_PATH", "/tmp/locomo_og_eval_v71")
SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
RERANK_POOL = int(os.environ.get("RERANK_POOL", "200"))
RERANK_TOP = int(os.environ.get("RERANK_TOP", "50"))
RERANK_LIGHT = os.environ.get("RERANK_LIGHT") == "1"  # 轻量重排 A/B: 词重叠信号(零依赖)
RERANK_MODE = os.environ.get("RERANK_MODE", "")       # auto=分类混合(cat1/4 cross-enc, cat2 轻量, cat3 原序)
CTX_TOKENS = int(os.environ.get("CTX_TOKENS", "20000"))
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "15"))
BLOCK_TOP = int(os.environ.get("BLOCK_TOP", "8"))
GRAPH_TOP = int(os.environ.get("GRAPH_TOP", "10"))      # 图遍历 episode 上限
LIMIT_BLOCKS = int(os.environ.get("LIMIT_BLOCKS", "0"))
EXTRACT_LIMIT = int(os.environ.get("EXTRACT_LIMIT", "1300"))
# v6.5.1 并行评测: judge 切换 (openrouter=GPT-4o-mini 严格 / deepseek=默认) + 类别过滤
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "deepseek")
CAT_FILTER = os.environ.get("CAT_FILTER", "")          # 逗号分隔, 如 "1,2,3,4"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct")
# 预测生成模式 (对接 LoCoMo-Refined 官方判卷): 输出 predictions.jsonl 不判卷
PREDICT_MODE = os.environ.get("PREDICT_MODE") == "1"
PREDICT_QUESTIONS = os.environ.get("PREDICT_QUESTIONS", "")
PREDICT_OUT = os.environ.get("PREDICT_OUT", "/tmp/locomo_refined_predictions.jsonl")
PREDICT_RANGE = os.environ.get("PREDICT_RANGE", "")
PREDICT_MAX_TOKENS = int(os.environ.get("PREDICT_MAX_TOKENS", "512"))  # 生成答案上限: 200→512 (枚举型答案截断修复)
HITK_MODE = os.environ.get("HITK_MODE") == "1"  # 纯检索 hit@k 基线 (P2: 证据是否进 top-k docs, 不调 LLM)
print(f"v72 配置: pool={RERANK_POOL} top={RERANK_TOP} ctx={CTX_TOKENS} block_size={BLOCK_SIZE} graph_top={GRAPH_TOP}", flush=True)
print(f"judge: {JUDGE_PROVIDER} ({JUDGE_MODEL}) | CAT_FILTER={CAT_FILTER or '全部'}", flush=True)

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
    """Load conversation messages. Supports both legacy locomo10.json and official LoCoMo-Refined jsonl format.

    Legacy format:  JSON list, each item has ``conversation.session_N`` dicts with ``session_N_date_time``.
    Official format: JSONL (one conversation per line), each line has ``sessions[].messages[]`` with dia_ids.

    Returns: (conv_map: {conv_idx: [formatted_msg_str, ...]}, all_msgs: [str, ...], dia_ids: [str, ...])
        — ``dia_ids[i]`` is the official dia_id (e.g. 'D1:3') for ``all_msgs[i]``;
          None when the legacy format is used (caller falls back to synthesising D{ci+1}:{j+1}).
    """
    # ── detect format ──
    is_jsonl = path.endswith(".jsonl")
    conv_map, sess_date = {}, {}
    all_msgs = []
    dia_ids = []          # parallel to all_msgs; official dia_id or None (legacy)
    conv_first_dt = {}    # conv_idx → first session date_time string (for conv_ts)

    if is_jsonl:
        raw = []
        with open(path) as _f:
            for _l in _f:
                _l = _l.strip()
                if _l:
                    raw.append(json.loads(_l))
        for ci, item in enumerate(raw):
            sessions = item.get("sessions", [])
            msgs = []
            for sess in sessions:
                si = sess.get("session_index", 0)
                dt = sess.get("date_time", "")
                if ci not in conv_first_dt and dt:
                    conv_first_dt[ci] = dt
                date_prefix = f"[date: {dt}] " if dt else ""
                for m in sess.get("messages", []):
                    speaker = m.get("speaker", "")
                    text = m.get("text", "")
                    did = m.get("dia_id") or f"D{si}:{m.get('message_index', '')}"
                    if text:
                        fm = f"{date_prefix}[{speaker}] {text}"
                        msgs.append(fm)
                        dia_ids.append(did)
            conv_map[ci] = msgs
            all_msgs.extend(msgs)
        # legacy callers expect 2-tuple; return 3-tuple with dia_ids
        return conv_map, all_msgs, dia_ids, raw, conv_first_dt

    # ── legacy locomo10.json ──
    raw = json.load(open(path))
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
    return conv_map, all_msgs, None, raw, sess_date

conv_map, all_msgs, _dia_ids_official, data, conv_first_dt = load_messages(DATA)
print(f"LoCoMo 消息: {len(all_msgs)} 条, {len(conv_map)} 会话", flush=True)

# ── 灌库 ──
# ── 包公两阶段灌库 (v6.5.1): INGEST_LOADED=1 跳过灌库 load 索引; INGEST_ONLY=1 灌库后退出
INGEST_SKIP = os.environ.get("INGEST_LOADED") == "1"
INGEST_ONLY = os.environ.get("INGEST_ONLY") == "1"
INGEST_PKL = DB_PATH + "_index.pkl"

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
faiss_index = None  # 各分支构造 (SKIP load / 灌库)

if INGEST_SKIP:
    import pickle as _pkl
    _restored = False
    if os.path.exists(INGEST_PKL) and os.path.getsize(INGEST_PKL) > 0:
        try:
            with open(INGEST_PKL, "rb") as _f:
                _st = _pkl.load(_f)
            faiss_id_map = _st["faiss_id_map"]
            msg_by_id = _st["msg_by_id"]; episode_cache = _st["episode_cache"]
            blocks = _st["blocks"]; block_vecs = _st["block_vecs"]
            entity_eps = _st["entity_eps"]; entity_co = _st["entity_co"]
            _triples_all = _st["triples"]; _dia_to_ep = _st["dia_to_ep"]
            _ontology_facts = _st["ontology_facts"]
            _schema_classes = _st["schema_classes"]; _top_classes = _st["top_classes"]
            tfidf_index = _st["tfidf_index"]
            faiss_index = VectorIndexAdapter(store=gstore, dimension=512,
                                             faiss_id_map=faiss_id_map)
            faiss_index.add_with_ids(_st["faiss_vecs"], _st["faiss_ids"])
            import faiss
            idx_blk = faiss.IndexFlatIP(512)
            idx_blk.add(block_vecs)
            print(f"[INGEST_LOADED] pkl 索引恢复: {len(msg_by_id)} 条, blocks={len(blocks)}", flush=True)
            _restored = True
        except Exception as _e:
            print(f"[INGEST_LOADED] pkl 损坏 ({_e}), 走 OverGraph 直连恢复", flush=True)
    if not _restored:
        # OverGraph 直连恢复 (2026-09-02): 本地 pkl 快照不可依赖 (被误覆盖/丢失),
        # 数据本体在 OverGraph DB — EpisodeNode 全量 + 引擎 dense HNSW 即生产检索链路。
        _rows = gstore._locked_execute_gql(
            "MATCH (e:EpisodeNode) RETURN e.id AS id, e.content AS content, e.created_at AS ts", {})["rows"]
        msg_by_id = {r["id"]: r.get("content", "") for r in _rows}
        faiss_id_map = {}
        episode_cache = {}
        for _r in _rows:
            _n = int(_r["id"].split("_")[1])
            faiss_id_map[_n] = _r["id"]
            episode_cache[_r["id"]] = {"id": _r["id"], "content": _r.get("content", ""),
                                       "created_at": _r.get("ts") or time.time(),
                                       "source_type": "sensory", "layer": 1}
        blocks = []
        block_vecs = np.zeros((0, 512), dtype=np.float32)
        entity_eps = {}
        entity_co = {}
        _triples_all = []
        _dia_to_ep = {}
        _ontology_facts = {}
        _schema_classes = {}
        _top_classes = []  # [(cls, vals), ...] 列表 (灌库路径 sorted items 产物)
        tfidf_index.fit(list(msg_by_id.values()))
        faiss_index = VectorIndexAdapter(store=gstore, dimension=512, faiss_id_map=faiss_id_map)
        import faiss
        idx_blk = faiss.IndexFlatIP(512)  # 块级向量缺失 → 空索引, B 通道检索自然为空
        print(f"[INGEST_LOADED] OverGraph 直连恢复: {len(msg_by_id)} 条 EpisodeNode "
              f"(pkl 不可用, blocks/entity 空 — 评测口径同 hitk OverGraph 版)", flush=True)
else:
    faiss_id_map = {}
    faiss_index = VectorIndexAdapter(store=gstore, dimension=512, faiss_id_map=faiss_id_map)
    msg_by_id = {}
    episode_cache = {}
    BATCH = 500
    id_buf, vec_buf = [], []
    t0 = time.time()
    for b_start in range(0, len(all_msgs), BATCH):
        batch = all_msgs[b_start:b_start + BATCH]
        # v6.5.1 提速: 超长消息截断再 embedding (16706 chars 单条会拖垮整批 padding)
        vecs = np.asarray(enc.embed_batch([m[:1500] for m in batch]), dtype=np.float32)
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
    tfidf_index.fit(list(msg_by_id.values()))

    # ═══ A. LLM 压缩记忆块（MindMemOS）═══
    _BLOCK_PROMPT = """Compress the following conversation messages into a memory block that preserves ALL key facts for later retrieval.
Keep: entities (names, places), events, activities, preferences, dates/times, relationships, objects, amounts.
Structure the block as:
FACTS:
- [entity] [verb phrase] [detail with dates/places/names]
- ... (exhaustive, 1 per fact, include speaker and rough date when present)
ENTITIES: comma-separated list

Conversation messages:
{messages}

Output only the FACTS/ENTITIES block. No preamble."""

    def _compress_block(msgs):
        msgs_text = "\n".join(f"[{i+1}] {m[:250]}" for i, m in enumerate(msgs))
        try:
            raw = llm_generate(_BLOCK_PROMPT.replace("{messages}", msgs_text), max_tokens=1200, temperature=0.0)
            return raw.strip()
        except Exception:
            return "\n".join(msgs)

    print(f"LLM 压缩记忆块（{BLOCK_SIZE} 条/块）...", flush=True)
    blocks = []
    t0 = time.time()
    _n_total = (len(all_msgs) + BLOCK_SIZE - 1) // BLOCK_SIZE

    # config 由公共段 (if/else 后) 定义
    for b_start in range(0, len(all_msgs), BLOCK_SIZE):
        if LIMIT_BLOCKS > 0 and len(blocks) >= LIMIT_BLOCKS:
            break
        b_end = min(b_start + BLOCK_SIZE, len(all_msgs))
        summary = _compress_block(all_msgs[b_start:b_end])
        blocks.append((f"blk_{len(blocks)}", b_start, b_end, summary))
        if len(blocks) % 25 == 0:
            print(f"  块 {len(blocks)}/{_n_total} ({time.time()-t0:.0f}s)", flush=True)
    print(f"记忆块完成: {len(blocks)} 块 ({time.time()-t0:.0f}s)", flush=True)
    json.dump([{"id": b[0], "start": b[1], "end": b[2], "summary": b[3]} for b in blocks],
              open("/tmp/v71_blocks.json", "w"), ensure_ascii=False)
    import faiss
    block_vecs = np.asarray(enc.embed_batch([b[3][:1500] for b in blocks]), dtype=np.float32)
    block_vecs = block_vecs / np.linalg.norm(block_vecs, axis=1, keepdims=True)
    idx_blk = faiss.IndexFlatIP(512)
    idx_blk.add(block_vecs)
    print(f"块索引: {idx_blk.ntotal} 向量", flush=True)

    # ═══ B. semantica 图遍历索引（实体 → 属性关系/共现 episode）═══
    # 实体-属性抽取（v68 同款，产出 triples）
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
        for m in re.finditer(r'\{([^{}]*)\}', raw):
            seg = m.group(1)
            d = {}
            for key in ("entity", "attribute", "value", "dia_id"):
                km = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', seg)
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

    _dia_to_ep = {}
    if _dia_ids_official:
        # Official jsonl format: dia_ids are parallel to all_msgs
        for _i, _did in enumerate(_dia_ids_official):
            _dia_to_ep[_did] = f"ep_{_i}"
    else:
        # Legacy format: synthesise D{ci+1}:{j+1} (global message index)
        for _ci in range(len(conv_map)):
            for _j in range(len(conv_map[_ci])):
                _dia_to_ep[f"D{_ci+1}:{_j+1}"] = f"ep_{sum(len(conv_map[c]) for c in range(_ci)) + _j}"

    print("实体-属性抽取（LLM 分批）...", flush=True)
    _triples_all = []
    _mid_msgs = []
    if _dia_ids_official:
        # Official format: use real dia_ids from the flat list
        _global_idx = 0
        for _ci in range(len(conv_map)):
            for _j, _m in enumerate(conv_map[_ci]):
                _did = _dia_ids_official[_global_idx]
                _mid_msgs.append({"dia_id": _did, "speaker": _m.split("]")[0].replace("[", "").strip() if "]" in _m else "", "text": _m})
                _global_idx += 1
    else:
        for _ci in range(len(conv_map)):
            for _j, _m in enumerate(conv_map[_ci]):
                _mid_msgs.append({"dia_id": f"D{_ci+1}:{_j+1}", "speaker": _m.split("]")[0].replace("[", "").strip() if "]" in _m else "", "text": _m})
    if EXTRACT_LIMIT > 0:
        _mid_msgs = _mid_msgs[:EXTRACT_LIMIT]
    for _i in range(0, len(_mid_msgs), 25):
        _got = _extract_batch(_mid_msgs[_i:_i + 25])
        _triples_all.extend(_got)
        if (_i // 25) % 5 == 0:
            print(f"  抽取 {min(_i+25, len(_mid_msgs))}/{len(_mid_msgs)} (cum {len(_triples_all)} triples)", flush=True)
    print(f"抽取完成: {len(_triples_all)} triples", flush=True)

    # 图遍历索引：实体 → 属性关系 episode（属性值中含实体的消息）；实体 → 共现实体
    entity_eps = {}   # entity → {ep: score}
    entity_co = {}    # entity → {co_entity: count}
    for _t in _triples_all:
        _e = (_t.get("entity") or "").strip()
        _ep = _dia_to_ep.get((_t.get("dia_id") or "").strip(), "")
        if not _e or not _ep:
            continue
        entity_eps.setdefault(_e, {})
        entity_eps[_e][_ep] = entity_eps[_e].get(_ep, 0) + 1.0
        # 共现：同 ep 的其他实体
        entity_co.setdefault(_e, {})
        for _t2 in _triples_all:
            _e2 = (_t2.get("entity") or "").strip()
            _ep2 = _dia_to_ep.get((_t2.get("dia_id") or "").strip(), "")
            if _e2 and _e2 != _e and _ep2 == _ep:
                entity_co[_e][_e2] = entity_co[_e].get(_e2, 0) + 1
    print(f"图遍历索引: {len(entity_eps)} 实体, 关联 episodes {sum(len(v) for v in entity_eps.values())}", flush=True)

    # ── 本体论组织（v72：实体事实簇 + 关系证据 + 全局线索 + 动态 schema）──
    def triples_by_entity_map():
        """triples → {entity: [fact_text]}（本体属性事实）"""
        m = {}
        for _t in _triples_all:
            _e = (_t.get("entity") or "").strip()
            _a = (_t.get("attribute") or "").strip()
            _v = (_t.get("value") or "").strip()
            if _e and _a and _v:
                m.setdefault(_e, []).append(f"{_e} {_a}: {_v}")
        return m

    _ontology_facts = triples_by_entity_map()

    # ═══ schema 自进化（Ontology v2 思想）：从 triples 动态发现高频属性模式 → 动态类 ═══
    def discover_schema(triples, min_support=3):
        """动态 schema：统计 (attribute → 计数)，高频属性成为 schema 类。
        非硬编码——schema 从数据自适应涌现（自进化）。"""
        attr_cnt, attr_vals = {}, {}
        for _t in triples:
            _a = (_t.get("attribute") or "").strip()
            _v = (_t.get("value") or "").strip()
            if _a and _v:
                attr_cnt[_a] = attr_cnt.get(_a, 0) + 1
                attr_vals.setdefault(_a, []).append(_v)
        # 高频属性 → schema 类（支持度 ≥ min_support）
        classes = {a: vals for a, vals in attr_vals.items() if attr_cnt[a] >= min_support}
        return classes

    _schema_classes = discover_schema(_triples_all)
    print(f"schema 自进化: {len(_schema_classes)} 个动态类（高频属性模式）", flush=True)
    _top_classes = sorted(_schema_classes.items(), key=lambda x: -len(x[1]))[:15]
    # 注: faiss add 已在灌库循环内完成, 这里只 dump
    import pickle as _pkl
    with open(INGEST_PKL, "wb") as _f:
        _pkl.dump({"faiss_id_map": faiss_id_map, "faiss_vecs": np.vstack(vec_buf),
                   "faiss_ids": np.array(id_buf, dtype=np.int64),
                   "msg_by_id": msg_by_id, "episode_cache": episode_cache,
                   "blocks": blocks, "block_vecs": block_vecs,
                   "entity_eps": entity_eps, "entity_co": entity_co,
                   "triples": _triples_all, "dia_to_ep": _dia_to_ep,
                   "ontology_facts": _ontology_facts,
                   "schema_classes": _schema_classes, "top_classes": _top_classes,
                   "tfidf_index": tfidf_index}, _f)
    print(f"[INGEST_DONE] 索引已存: {INGEST_PKL}", flush=True)
    if INGEST_ONLY:
        print("[INGEST_ONLY] 灌库完成, 退出", flush=True)
        sys.exit(0)

config = QueryRouterConfig()
config.agentic_enabled = False
config.hyde_timeout = 10.0
config.mesa_enabled = True
config.mesa_boost = 0.4
config.mesa_threshold = 0.5
config.mesa_max_nodes = 5
qr = QueryRouter(graphlite_store=gstore, faiss_index=faiss_index, tfidf_index=tfidf_index,
                 encoder=enc, config=config, faiss_id_map=faiss_id_map, episode_cache=episode_cache)

def extract_query_entities(question):
    """从问题提取实体（与 entity_eps 键匹配：直接词匹配 + LLM 兜底）"""
    found = [e for e in entity_eps if e.lower() in question.lower() and len(e) > 2][:5]
    if found:
        return found
    try:
        prompt = f"""Extract the main proper-noun entities (people/places/things) from this question. Output JSON list of strings.
Question: {question}
No other text."""
        raw = llm_generate(prompt, max_tokens=80, temperature=0.0)
        s, e = raw.find("["), raw.rfind("]")
        if s >= 0 and e > s:
            arr = json.loads(raw[s:e + 1])
            return [str(x) for x in arr if x and x.strip() in entity_eps][:5]
    except Exception:
        pass
    return []

def graph_walk(question):
    """semantica 图遍历：query 实体 → 属性关系 episode（加权）→ 共现实体 episode → top-k"""
    ents = extract_query_entities(question)
    if not ents:
        return []
    cand = {}
    for e in ents:
        # 属性关系（直接 MENTIONS 加权 1.0）
        for ep, sc in entity_eps.get(e, {}).items():
            cand[ep] = cand.get(ep, 0) + sc
        # 共现实体（2 跳：实体 A → 同 episode 的实体 B → B 的 episode，加权 0.5）
        for co, cnt in sorted(entity_co.get(e, {}).items(), key=lambda x: -x[1])[:3]:
            for ep2, sc2 in entity_eps.get(co, {}).items():
                cand[ep2] = cand.get(ep2, 0) + 0.5 * sc2
    ranked = sorted(cand.items(), key=lambda x: -x[1])[:GRAPH_TOP]
    out = []
    for ep, sc in ranked:
        c = msg_by_id.get(ep, "")
        if c:
            out.append(c)
    return out

# v6.5.1 并行评测: OpenRouter GPT-4o-mini judge (严格判卷)
def _or_key():
    try:
        txt = open(os.path.expanduser("~/.hermes/.env")).read()
    except OSError:
        txt = ""
    m = re.search(r'OPENROUTER_API_KEY\s*=\s*"?([^"\s]+)"?', txt)
    return m.group(1) if m else os.environ.get("OPENROUTER_API_KEY", "")


def _judge_openrouter(question, ground_truth, prediction, model=JUDGE_MODEL):
    """OpenRouter LLM-as-judge (GPT-4o-mini): 对齐 LoCoMo 判定协议。"""
    import urllib.request
    key = _or_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY 缺失")
    prompt = (f"判断以下回答是否正确。\n\n问题: {question}\n标准答案: {ground_truth}\n"
              f"待评回答: {prediction}\n\n"
              "只输出 JSON: {\"correct\": true/false, \"reason\": \"一句话\"}")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 100, "temperature": 0.0}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 data=body, headers={
                                     "Content-Type": "application/json",
                                     "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    content = resp["choices"][0]["message"]["content"]
    s, e = content.find("{"), content.rfind("}")
    if s >= 0 and e > s:
        d = json.loads(content[s:e + 1])
        return bool(d.get("correct"))
    return False


def _judge(q, gold, pred):
    if JUDGE_PROVIDER == "openrouter":
        return _judge_openrouter(q, gold, pred)
    return llm_judge(q, gold, pred)


def _judge_call(q, gold, pred):
    try:
        return _judge(q, gold, pred)
    except Exception as e:
        print(f"  [judge err] {e}", flush=True)
        return False


# ═══ C. 检索融合 ═══
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

def _block_add(question, seen, docs):
    try:
        qv = np.asarray(enc.embed(question), dtype=np.float32).reshape(1, -1)
        qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
        scores, bidx = idx_blk.search(qv, BLOCK_TOP)
        for bi in bidx[0]:
            blk = blocks[bi]
            if blk[3][:300] not in seen:
                seen.add(blk[3][:300])
                docs.append(blk[3])
            for j in range(blk[1], blk[2]):
                c = msg_by_id.get(f"ep_{j}", "")
                if c and c[:200] not in seen:
                    seen.add(c[:200])
                    docs.append(c)
    except Exception:
        pass

def _graph_add(question, seen, docs):
    try:
        for c in graph_walk(question):
            if c[:200] not in seen:
                seen.add(c[:200])
                docs.append(c)
    except Exception:
        pass

def retrieve_channels(question, hitk=False):
    """三通道并行检索，各自独立打分（有机融合——不塞池竞争）"""
    # P2 基线: HITK 模式单查询(基础检索能力), 不含 multi_query 增强
    queries = [question] + ([] if hitk else multi_query_expand(question))
    seen_a, docs_a = set(), []
    for q in queries:
        _fuse(q, seen_a, docs_a)
    _adjacent(docs_a, seen_a)
    # A 通道：消息级 top-40
    ch_a = docs_a[:40]

    # B 通道：LLM 记忆块（块摘要 + 展开消息）
    ch_b_sum, ch_b_msg, seen_b = [], [], set()
    try:
        qv = np.asarray(enc.embed(question), dtype=np.float32).reshape(1, -1)
        qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
        scores, bidx = idx_blk.search(qv, BLOCK_TOP)
        for bi in bidx[0]:
            blk = blocks[bi]
            if blk[3][:300] not in seen_b:
                seen_b.add(blk[3][:300])
                ch_b_sum.append(blk[3])
            for j in range(blk[1], blk[2]):
                c = msg_by_id.get(f"ep_{j}", "")
                if c and c[:200] not in seen_b:
                    seen_b.add(c[:200])
                    ch_b_msg.append(c)
    except Exception:
        pass
    ch_b = (ch_b_sum[:5] + ch_b_msg)[:15]  # 块摘要优先 + 消息展开

    # C 通道：图遍历（实体关系）
    ch_c = []
    try:
        for c in graph_walk(question):
            if c[:200] not in seen_a and c[:200] not in seen_b:
                ch_c.append(c)
    except Exception:
        pass
    ch_c = ch_c[:8]

    return {"A": ch_a, "B_sum": ch_b_sum[:4], "B": ch_b, "C": ch_c}

def rrf_fuse(channels, k=60):
    """RRF 融合：三通道各自 top-N → 倒数排名和 → 排序（保底机制）"""
    from collections import defaultdict
    scores = defaultdict(float)
    for docs in (channels["A"], channels["B"], channels["C"]):
        for rank, d in enumerate(docs):
            scores[d] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: -scores[x])


_STOP = set("the a an is are was were be been being of to in on at for with from by about as and or not no it its this that these those what when where who which how why do does did have has had will would can could should may might".split())


def light_rerank(query, docs, top_k=30):
    """轻量重排：query 词重叠率加权（零依赖；alpha=0.5 与 RRF 原序混合，tiebreak 保 RRF 顺序）"""
    q_tokens = set(re.findall(r"[a-z']+", query.lower())) - _STOP
    if not q_tokens:
        return docs[:top_k]
    # 稳定排序：overlap 降序，同分保持原 RRF 序
    scored = sorted(((sum(1 for t in q_tokens if t in d.lower()) / len(q_tokens), i, d)
                     for i, d in enumerate(docs)), key=lambda x: (-x[0], x[1]))
    return [d for _, _, d in scored][:top_k]

# ── 本体论组织（v72：实体事实簇 + 关系证据 + 全局线索 + 动态 schema）──

_cur_cat = "0"  # 当前问题类别（RERANK_MODE=auto 分类路由用）


def ontology_organize(question, channels):
    """本体论核心融合：实体/属性/关系/事实/动态schema 五维度组织（非排名平铺）"""
    ents = extract_query_entities(question)
    parts = []

    # 1. 实体事实簇（本体属性——按动态 schema 类分组展示）
    for e in ents[:3]:
        facts = _ontology_facts.get(e, [])
        if not facts:
            continue
        # 属性按 schema 类分组（自进化结果）
        cls_lines = []
        for cls, vals in _top_classes:
            evs = [v for v in vals if v.lower() in " ".join(facts).lower()]
            if evs and len(evs) <= 3:
                cls_lines.append(f"  · {cls}: {evs[0][:150]}")
        lines = [f"- {f}" for f in facts[:8]]
        if cls_lines:
            lines = cls_lines + ["  · (full facts below)"] + lines[:6]
        # 实体相关块摘要（全局记忆）
        for blk in blocks:
            if e.lower() in blk[3].lower():
                lines.append(f"- [block] {blk[3][:200]}")
                break
        parts.append(f"[ENTITY: {e}]\n" + "\n".join(lines))

    # 2. 关系证据（实体间共现/关联）
    if len(ents) >= 2:
        rel_parts = []
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                cnt = entity_co.get(ents[i], {}).get(ents[j], 0)
                if cnt > 0:
                    rel_parts.append(f"- {ents[i]} — {ents[j]}: co-occur in {cnt} fact(s)")
        if rel_parts:
            parts.append("[RELATIONS]\n" + "\n".join(rel_parts))

    # 3. 事实全景（全局 triples 精华——按动态 schema 类组织）
    fact_lines = []
    for cls, vals in _top_classes[:10]:
        fact_lines.append(f"- [{cls}] {vals[0][:150]}")
    if fact_lines:
        parts.append("[FACT TYPES (auto-discovered schema)]\n" + "\n".join(fact_lines))

    # 4. 全局线索（记忆块摘要——精简 2 块，标注来源）
    summ = "\n".join(f"[MEMORY BLOCK {j+1}] {s[:300]}" for j, s in enumerate(channels["B_sum"][:2]))
    if summ:
        parts.append("[GLOBAL CONTEXT (LLM-compressed memory blocks)]\n" + summ)

    # 5. 直接证据（消息级 rerank——不加回 B/C，避免污染）
    fused = rrf_fuse(channels)
    try:
        if RERANK_MODE == "auto":
            # 分类混合: cat1/4 cross-encoder, cat2 轻量(词重叠), cat3 原序(推理类重排退化)
            if str(_cur_cat) == "3":
                direct = fused[:30]
            elif str(_cur_cat) == "2":
                direct = light_rerank(question, fused[:RERANK_POOL], top_k=30)
            else:
                get_reranker()
                direct = rerank(question, fused[:RERANK_POOL], top_n=30)
        elif RERANK_LIGHT:
            direct = light_rerank(question, fused[:RERANK_POOL], top_k=30)
        else:
            get_reranker()
            direct = rerank(question, fused[:RERANK_POOL], top_n=30)
    except Exception:
        direct = fused[:30]
    ev_sec = "\n".join(f"[{j+1}] {d}" for j, d in enumerate(direct[:30]))
    parts.append("[DIRECT EVIDENCE]\n" + ev_sec)

    ctx = "\n\n".join(parts)
    return ctx[:CTX_TOKENS], direct

def build_ctx(question, channels, rerank_top=40):
    """v72 本体论核心：ontology_organize（保留接口兼容）"""
    return ontology_organize(question, channels)

# ═══ D. agentic 两轮（EverOS）═══
def suff_check(question, docs_top):
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
        return bool(d.get("sufficient")), str(d.get("missing_info", ""))
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
if _dia_ids_official:
    # Official jsonl format: use conv_first_dt from first session date_time
    conv_ts = conv_first_dt
else:
    for item in data:
        conv_ts[len(conv_ts)] = item.get("conversation", {}).get("session_1_date_time", 0)

def parse_session_ts(ts):
    try:
        return float(ts)
    except (ValueError, TypeError):
        return None

qa_conv = {}
for ci, item in enumerate(data):
    for q in item.get("qa", []):
        qa_conv[id(q)] = ci
qa_all = []
if PREDICT_QUESTIONS:
    with open(PREDICT_QUESTIONS) as _fq:
        for _l in _fq:
            _l = _l.strip()
            if not _l:
                continue
            _q = json.loads(_l)
            qa_all.append({"qa_id": _q.get("qa_id"), "question": _q.get("question"), "answer": _q.get("answer"),
                           "category": _q.get("category"),
                           "evidence": _q.get("evidence"), "evidence_messages": _q.get("evidence_messages")})
    _sample_map = {}
    for _ci, _item in enumerate(data):
        _sample_map[_item.get("sample_id")] = _ci
    for _q in qa_all:
        _sid = str(_q.get("qa_id", "")).split("#")[0]
        qa_conv[id(_q)] = _sample_map.get(_sid, 0)
    if PREDICT_RANGE:
        _s, _e = PREDICT_RANGE.split(":")
        qa_all = qa_all[int(_s):int(_e)]
    print(f"[PREDICT] 加载 LoCoMo-Refined 问题: {len(qa_all)} 问 (range={PREDICT_RANGE or 'all'})", flush=True)
else:
    for item in data:
        qa_all.extend(item.get("qa", []))
qa_all = [q for q in qa_all if q.get("category") != 5 and q.get("answer")]
if CAT_FILTER:
    cats = {int(c) for c in CAT_FILTER.split(",") if c.strip().isdigit()}
    qa_all = [q for q in qa_all if q.get("category", 0) in cats]
    print(f"类别过滤: {sorted(cats)} → {len(qa_all)} 问", flush=True)
if SAMPLE_N > 0:
    qa_all = qa_all[:SAMPLE_N]
print(f"评测规模: {len(qa_all)} 问", flush=True)

results = {"total": 0, "correct": 0, "errors": 0, "by_cat": {}, "round2_used": 0}
hitk_stats = {"total": 0, "by_cat": {}}  # P2 hit@k 基线
t0 = time.time()

# ── 断点续跑 (RESUME=1): 跳过 PREDICT_OUT 已存在的 qa_id，不重复调 LLM ──
_done_qids = set()
if PREDICT_MODE and os.environ.get("RESUME") == "1" and os.path.exists(PREDICT_OUT):
    try:
        with open(PREDICT_OUT, encoding="utf-8", errors="ignore") as _f:
            for _l in _f:
                _l = _l.strip().strip("\x00")
                if not _l:
                    continue
                try:
                    _d = json.loads(_l)
                except Exception:
                    continue  # 跳过被中断的坏行/NUL 残留, 不中断续跑
                if _d.get("qa_id"):
                    _done_qids.add(_d["qa_id"])
        print(f"[RESUME] 已有 {len(_done_qids)} 条预测, 跳过继续", flush=True)
    except Exception as _e:
        print(f"[RESUME] 读取已有预测失败, 从头跑: {_e}", flush=True)

for i, q in enumerate(qa_all):
    qid = id(q)
    if PREDICT_MODE and q.get("qa_id") in _done_qids:
        continue
    ci = qa_conv.get(qid, 0)
    question, gold, cat = q["question"], q["answer"], q.get("category", 0)
    session_ts = parse_session_ts(conv_ts.get(ci))

    # 三通道检索（有机融合）+ 证据分区
    _cur_cat = str(cat)
    channels = retrieve_channels(question, hitk=HITK_MODE)
    # falsification 实验 (2026-09-02): oracle 注入 = 金标证据人工置顶 (人工正确排序)
    if os.environ.get("ORACLE_INJECT") == "1":
        _ev = [e.get("text", "").strip() for e in (q.get("evidence_messages") or []) if e.get("text")]
        _seen = {c[:200] for c in channels["A"]}
        channels["A"] = [e for e in _ev if e[:200] not in _seen] + channels["A"]
    ctx, docs = build_ctx(question, channels, rerank_top=40)

    # P2 hit@k 基线: evidence 消息文本是否进入 top-k docs (纯检索, 不调 LLM)
    if HITK_MODE:
        ev_msgs = q.get("evidence_messages") or []
        ev_texts = [e.get("text", "") for e in ev_msgs if e.get("text")]
        cat_s = str(cat)
        c = hitk_stats["by_cat"].setdefault(cat_s, {"total": 0, "k1": 0, "k3": 0, "k5": 0, "k10": 0, "k20": 0, "k40": 0})
        c["total"] += 1
        hitk_stats["total"] += 1
        # 滑动窗口探针: 25 字符片段 (压缩块可能改写原文, 全句匹配过严)
        probes = []
        for et in ev_texts:
            t = et.strip()
            if not t:
                continue
            probes += [t[i:i + 25] for i in range(0, max(1, len(t) - 24), 20)][:8]
        probes = [p for p in probes if len(p) >= 15]
        for kk in (1, 3, 5, 10, 20, 40):
            topk = docs[:kk]
            ok = any(any(p in d for d in topk) for p in probes)
            if ok:
                c[f"k{kk}"] += 1
        if (i + 1) % 50 == 0 or i == len(qa_all) - 1:
            print(f"  [HITK] {i+1}/{len(qa_all)} elapsed={time.time()-t0:.0f}s", flush=True)
            json.dump(hitk_stats, open("/tmp/locomo_v72_hitk.json", "w"), ensure_ascii=False, indent=2)  # checkpoint 防中断丢数据
        continue

    # agentic 协同：sufficiency 基于分区后完整证据（保守触发——v71 教训 76% 太高）
    enough = True
    if len(docs) >= 10:
        try:
            enough, missing = suff_check(question, docs[:10])
        except Exception:
            enough, missing = True, ""
        # 保守：missing 太泛（<10 字符）不触发 round2
        if not enough and len(missing) < 10:
            enough = True
    else:
        enough = False
        missing = "general information about the question"
    if not enough:
        results["round2_used"] += 1
        fq = followup_query(question, missing) if missing and missing != "general information about the question" else question
        # round2 定向补缺：重跑三通道，只追加缺失部分（去重）
        ch2 = retrieve_channels(fq)
        seen2 = {c[:200] for c in docs}
        new_docs = []
        for d in ch2["A"][:20] + ch2["B"][:8] + ch2["C"][:6]:
            if d[:200] not in seen2:
                seen2.add(d[:200])
                new_docs.append(d)
        if new_docs:
            docs = docs + new_docs[:20]
        summ2 = "\n".join(f"[MEMORY BLOCK {j+1}] {s[:400]}" for j, s in enumerate(ch2["B_sum"][:2]))
        ev_sec = "\n".join(f"[{j+1}] {d}" for j, d in enumerate(docs[:40]))
        ctx = (summ2 + "\n\n" + ev_sec) if summ2 else ev_sec
        ctx = ctx[:CTX_TOKENS]

    prompt = f"""Answer the question based on the conversation snippets below. Reason across snippets if needed (e.g., infer dates from session timestamps).

Conversation snippets:
{ctx}

Question: {question}
Answer:"""
    try:
        pred = llm_generate(prompt, max_tokens=PREDICT_MAX_TOKENS, temperature=0.2)
    except Exception as e:
        print(f"  [gen err] {e}", flush=True)
        results["errors"] += 1
        continue
    if PREDICT_MODE:
        with open(PREDICT_OUT, "a") as _pf:
            _pf.write(json.dumps({"qa_id": q.get("qa_id", f"q{i}"), "predicted_answer": pred[:4000]}) + "\n")
        if (i + 1) % 20 == 0 or i == len(qa_all) - 1:
            print(f"  [PREDICT] {i+1}/{len(qa_all)} elapsed={time.time()-t0:.0f}s", flush=True)
        continue
    results["total"] += 1
    ok = _judge_call(question, gold, pred)
    if ok:
        results["correct"] += 1
    # ── 诊断模式 (DIAG_DUMP=1): 每题落盘，供错误归因（默认不生效，不改评测口径）──
    if os.environ.get("DIAG_DUMP") == "1":
        _gt = set(t for t in re.findall(r"[A-Za-z]{4,}|\d+", str(gold).lower()) if t not in
                  {"what","when","where","who","how","many","the","and","was","did","does","they","she","he","her","his","their","have","has","been","with","from","that","this","for","are","you","your","would","could","should","will","going","went","go","about","there","its","them","him","not","one","two","three","time","day","week","month","year","said","told"})
        _dtxt = " ".join(docs).lower()
        _hit = [t for t in _gt if t in _dtxt]
        _diag = {"i": i, "q": question, "gold": gold, "pred": pred[:300], "ok": bool(ok),
                 "n_docs": len(docs), "n_gold_key": len(_gt), "key_hit": _hit[:10],
                 "round2_total": results["round2_used"]}
        with open(os.environ.get("DIAG_OUT", "/tmp/locomo_diag.jsonl"), "a") as _f:
            _f.write(json.dumps(_diag, ensure_ascii=False) + "\n")
    results["by_cat"].setdefault(cat, {"t": 0, "c": 0})
    results["by_cat"][cat]["t"] += 1
    if ok:
        results["by_cat"][cat]["c"] += 1
    if (i + 1) % 10 == 0 or i == len(qa_all) - 1:
        acc = results["correct"] / max(1, results["total"]) * 100
        print(f"  {i+1}/{len(qa_all)} acc={acc:.1f}% ({results['correct']}/{results['total']}) round2={results['round2_used']} elapsed={time.time()-t0:.0f}s", flush=True)

acc = results["correct"] / max(1, results["total"]) * 100
print(f"\n=== v72 本体论核心融合评测 {SAMPLE_N}问 ===", flush=True)
print(f"准确率: {acc:.1f}% ({results['correct']}/{results['total']})", flush=True)
print(f"round2: {results['round2_used']} | 错误: {results['errors']} | 耗时: {time.time()-t0:.0f}s", flush=True)
for cat in sorted(results["by_cat"]):
    d = results["by_cat"][cat]
    print(f"  cat={cat}: {d['c']}/{d['t']} = {d['c']/max(1,d['t'])*100:.1f}%", flush=True)

# P2 hit@k 基线输出 (HITK_MODE)
if HITK_MODE:
    print(f"\n=== P2 hit@k 基线 (evidence 进 top-k) ===", flush=True)
    for cat_s in sorted(hitk_stats["by_cat"], key=lambda x: int(x) if x.isdigit() else 99):
        d = hitk_stats["by_cat"][cat_s]
        t = max(1, d["total"])
        print(f"  cat={cat_s}: n={d['total']} hit@1={d['k1']/t*100:.1f}% @3={d['k3']/t*100:.1f}% @5={d['k5']/t*100:.1f}% @10={d['k10']/t*100:.1f}% @20={d['k20']/t*100:.1f}% @40={d['k40']/t*100:.1f}%", flush=True)
    hitk_out = f"/tmp/locomo_v72_hitk.json"
    json.dump(hitk_stats, open(hitk_out, "w"), ensure_ascii=False, indent=2)
    print(f"hit@k 结果已存: {hitk_out}", flush=True)

out = f"/tmp/locomo_v72_results_cat{CAT_FILTER or 'all'}_{JUDGE_PROVIDER}.json"
json.dump({"acc": acc, "correct": results["correct"], "total": results["total"],
           "by_cat": results["by_cat"], "errors": results["errors"], "round2_used": results["round2_used"],
           "blocks": len(blocks), "entities": len(entity_eps),
           "judge": JUDGE_PROVIDER, "judge_model": JUDGE_MODEL,
           "version": "6.4.0-ontology"},
          open(out, "w"), ensure_ascii=False, indent=2)
print(f"结果已存: {out}", flush=True)