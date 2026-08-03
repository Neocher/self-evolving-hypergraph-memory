#!/usr/bin/env python3
"""迁移脚本: Kuzu + RyuGraph 旧库 → GraphLite 新库

- 读取旧库全部 EpisodeNode / HyperedgeNode / CommunityNode / OntologyType / OntologyEntity
- 按 id 去重(查询-插入两段式,GraphLite schemaless 无主键)
- 写入 data/shm_graphlite_db
"""
import sys, os, json, time

sys.path.insert(0, os.path.expanduser("~/GraphLite/bindings/python"))
sys.path.insert(0, os.path.expanduser("~/GraphLite/sdk-python/src"))
from graphlite_sdk import GraphLite

SHM_DIR = os.path.expanduser("~/self-evolving-hypergraph-memory")
os.chdir(SHM_DIR)

def gql_val(v):
    """GQL 字面量序列化 — 与 graph/graphlite_store._gql_value 一致。

    GraphLite Rust lexer 有 UTF-8 bug: 非 ASCII 字符串必须 b64 编码存储
    (前缀 {b64}), 读时由 _flatten_row 解码。
    """
    from base64 import b64encode
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        try:
            v.encode('ascii')
            v = v.replace("\\", "\\\\").replace("'", "\\'")
            return f"'{v}'"
        except UnicodeEncodeError:
            b64 = b64encode(v.encode('utf-8')).decode('ascii')
            return f"'{{b64}}{b64}'"
    if isinstance(v, (list, dict)):
        b64 = b64encode(json.dumps(v, ensure_ascii=False).encode('utf-8')).decode('ascii')
        return f"'{{b64}}{b64}'"
    return gql_val(str(v))

def clean_row(d):
    """去掉 Kuzu/RyuGraph 内部字段"""
    out = {}
    for k, v in d.items():
        if k in ("_id", "_label"):
            continue
        if k == "embedding" and v is None:
            continue
        out[k] = v
    return out

# ── 1. 读旧库 ──────────────────────────────
episodes, communities, hyperedges = {}, {}, {}
ontology_entities, ontology_types = {}, {}
hebbian, is_a = [], []

# Kuzu 库 (单独进程, 与 ryugraph ctypes 冲突)
def read_kuzu():
    import kuzu
    db = kuzu.Database("data/shm_kuzu_db")
    conn = kuzu.Connection(db)
    eps, comms, heb = {}, {}, []
    for _, row in conn.execute("MATCH (e:EpisodeNode) RETURN e").get_as_df().iterrows():
        e = clean_row(dict(row["e"]))
        eps[e["id"]] = e
    for _, row in conn.execute("MATCH (c:CommunityNode) RETURN c").get_as_df().iterrows():
        e = clean_row(dict(row["c"]))
        comms[e["id"]] = e
    for _, row in conn.execute("MATCH ()-[r:HEBBIAN_CONNECTION]->() RETURN r").get_as_df().iterrows():
        heb.append(dict(row["r"]))
    return eps, comms, heb

import subprocess, tempfile, pickle
with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
    tmp_kuzu = tf.name
kuzu_script = '''
import sys, pickle, json
sys.path.insert(0, ".")
import kuzu
db = kuzu.Database("data/shm_kuzu_db")
conn = kuzu.Connection(db)
eps, comms, heb = {}, {}, []
for _, row in conn.execute("MATCH (e:EpisodeNode) RETURN e").get_as_df().iterrows():
    e = dict(row["e"]); e.pop("_id", None); e.pop("_label", None)
    eps[e["id"]] = e
for _, row in conn.execute("MATCH (c:CommunityNode) RETURN c").get_as_df().iterrows():
    e = dict(row["c"]); e.pop("_id", None); e.pop("_label", None)
    comms[e["id"]] = e
# 节点 offset -> id 映射
offsets = {}
for _, row in conn.execute("MATCH (e:EpisodeNode) RETURN e").get_as_df().iterrows():
    off = row["e"]["_id"]["offset"]
    offsets[(row["e"]["_id"]["table"], off)] = row["e"]["id"]
for _, row in conn.execute("MATCH (c:CommunityNode) RETURN c").get_as_df().iterrows():
    off = row["c"]["_id"]["offset"]
    offsets[(row["c"]["_id"]["table"], off)] = row["c"]["id"]
heb = []
for _, row in conn.execute("MATCH ()-[r:HEBBIAN_CONNECTION]->() RETURN r").get_as_df().iterrows():
    r = dict(row["r"])
    src = offsets.get((r["_src"]["table"], r["_src"]["offset"]), None)
    dst = offsets.get((r["_dst"]["table"], r["_dst"]["offset"]), None)
    if src and dst:
        heb.append({"src": src, "dst": dst, "weight": r.get("weight", 1.0)})
pickle.dump({"eps": eps, "comms": comms, "heb": heb}, open(sys.argv[1], "wb"))
'''
subprocess.run([sys.executable, "-c", kuzu_script, tmp_kuzu], check=True)
with open(tmp_kuzu, "rb") as f:
    kdata = pickle.load(f)
episodes.update(kdata["eps"])
communities.update(kdata["comms"])
hebbian.extend(kdata["heb"])
os.unlink(tmp_kuzu)
print(f"Kuzu: {len(kdata['eps'])} episodes, {len(kdata['comms'])} communities, {len(kdata['heb'])} hebbian")

# RyuGraph 库 (主进程独占, 字段为 _ID/_LABEL/_SRC/_DST 大写)
try:
    import ryugraph
    db = ryugraph.Database("data/shm_ryugraph_db")
    conn = ryugraph.Connection(db)
    for _, row in conn.execute("MATCH (e:EpisodeNode) RETURN e").get_as_df().iterrows():
        e = clean_row(dict(row["e"]))
        episodes.setdefault(e["id"], e)
    for _, row in conn.execute("MATCH (c:CommunityNode) RETURN c").get_as_df().iterrows():
        e = clean_row(dict(row["c"]))
        communities.setdefault(e["id"], e)
    for _, row in conn.execute("MATCH (o:OntologyType) RETURN o").get_as_df().iterrows():
        e = clean_row(dict(row["o"]))
        ontology_types[e.get("name", e.get("id"))] = e
    for _, row in conn.execute("MATCH (o:OntologyEntity) RETURN o").get_as_df().iterrows():
        e = clean_row(dict(row["o"]))
        ontology_entities[e.get("name", e.get("id"))] = e
    # RyuGraph 内部字段大写: _SRC/_DST, 需经 offset 反查 id
    ryu_offsets = {}
    for label in ("EpisodeNode", "CommunityNode", "OntologyType", "OntologyEntity"):
        try:
            for _, row in conn.execute(f"MATCH (n:{label}) RETURN n").get_as_df().iterrows():
                node = dict(row["n"])
                off = node["_ID"]["offset"]
                ryu_offsets[(node["_ID"]["table"], off)] = node.get("id", node.get("name"))
        except Exception:
            pass
    for _, row in conn.execute("MATCH ()-[r:HEBBIAN_CONNECTION]->() RETURN r").get_as_df().iterrows():
        r = dict(row["r"])
        src = ryu_offsets.get((r["_SRC"]["table"], r["_SRC"]["offset"]), None)
        dst = ryu_offsets.get((r["_DST"]["table"], r["_DST"]["offset"]), None)
        if src and dst:
            hebbian.append({"src": src, "dst": dst, "weight": r.get("weight", 1.0)})
    for _, row in conn.execute("MATCH (a)-[r:IS_A]->(b) RETURN a.name AS src, b.name AS dst").get_as_df().iterrows():
        is_a.append({"src": row["src"], "dst": row["dst"]})
    print(f"RyuGraph: {len(episodes)} episodes, {len(communities)} comm, "
          f"{len(ontology_types)} otypes, {len(ontology_entities)} oentities, "
          f"{len(hebbian)} hebbian, {len(is_a)} is_a")
except Exception as ex:
    import traceback
    print(f"RyuGraph 读取失败: {ex}")
    traceback.print_exc()

# ── 2. 写入 GraphLite ───────────────────────
gl = GraphLite.open("data/shm_graphlite_db")
s = gl.session("shm")
# 全新库初始化: 建 schema → set schema → 建 graph → set graph
# (CREATE GRAPH 前必须先 SESSION SET SCHEMA, 顺序颠倒会失败)
try:
    s.execute("CREATE SCHEMA /shm")
except Exception:
    pass  # schema 可能已存在
try:
    s.execute("SESSION SET SCHEMA /shm")
except Exception as e:
    print(f"SET SCHEMA 失败: {str(e)[:100]}")
try:
    s.execute("CREATE GRAPH /shm")
except Exception:
    pass  # graph 可能已存在
try:
    s.execute("SESSION SET GRAPH /shm")
except Exception as e:
    print(f"SET GRAPH 失败: {str(e)[:100]}")

def insert_if_missing(label, props, id_key="id"):
    nid = str(props.get(id_key, ""))
    if not nid:
        return
    try:
        r = s.query(f"MATCH (n:{label} {{{id_key}: '{nid}'}}) RETURN n.{id_key}")
        if r.rows:
            return  # 已存在
    except Exception:
        pass
    kvs = ", ".join(f"{k}: {gql_val(v)}" for k, v in props.items() if v is not None)
    s.execute(f"INSERT (n:{label} {{{kvs}}})")

cnt = {"episode": 0, "community": 0, "otype": 0, "oentity": 0}
for e in episodes.values():
    insert_if_missing("EpisodeNode", e)
    cnt["episode"] += 1
for c in communities.values():
    insert_if_missing("CommunityNode", c)
    cnt["community"] += 1
for t in ontology_types.values():
    insert_if_missing("OntologyType", t, id_key="name")
    cnt["otype"] += 1
for o in ontology_entities.values():
    insert_if_missing("OntologyEntity", o, id_key="name")
    cnt["oentity"] += 1

# Hebbian 关系(两端节点需存在,跳过缺失)
hebbian_ok = 0
for h in hebbian:
    src = h.get("src") or h.get("source") or h.get("from")
    dst = h.get("dst") or h.get("target") or h.get("to")
    w = h.get("weight", 1.0)
    if not src or not dst:
        continue
    try:
        s.execute(f"MATCH (a {{id: '{src}'}}), (b {{id: '{dst}'}}) "
                  f"INSERT (a)-[:HEBBIAN_CONNECTION {{weight: {float(w)}}}]->(b)")
        hebbian_ok += 1
    except Exception:
        pass

# IS_A 关系
is_a_ok = 0
for r in is_a:
    try:
        s.execute(f"MATCH (a {{id: '{r['src']}'}}), (b {{id: '{r['dst']}'}}) "
                  f"INSERT (a)-[:IS_A]->(b)")
        is_a_ok += 1
    except Exception:
        pass

gl.close()
print(f"\n✅ 迁移完成: episodes={cnt['episode']} communities={cnt['community']} "
      f"otypes={cnt['otype']} oentities={cnt['oentity']} hebbian={hebbian_ok} is_a={is_a_ok}")
