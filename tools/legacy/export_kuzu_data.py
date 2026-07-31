#!/usr/bin/env python3
"""Step 1: 从旧 Kuzu DB 导出数据到 JSON（修正版—正确处理嵌套字段和关系键）"""
import json
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KUZU_DB = str(DATA_DIR / "shm_kuzu_db")
EXPORT_FILE = str(DATA_DIR / "kuzu_migration_full.json")

import kuzu
import numpy as np

if not Path(KUZU_DB).exists():
    print(f"❌ Kuzu DB not found: {KUZU_DB}")
    exit(1)

db = kuzu.Database(KUZU_DB)
conn = kuzu.Connection(db)

def q(query):
    """Execute query and return list of dicts"""
    return conn.execute(query).get_as_pl().to_pandas().to_dict("records")

def flatten_rows(alias, rows):
    """Flatten Kuzu nested aliased rows into flat dicts"""
    result = []
    for row in rows:
        obj = row[alias]
        if isinstance(obj, dict):
            # Remove internal fields
            clean = {k: v for k, v in obj.items() if k not in ('_ID', '_LABEL')}
            result.append(clean)
    return result

# ─── 导出节点 ───
print("📤 导出节点...")
nodes = {}
labels_aliases = {
    "EpisodeNode": "e", "HyperedgeNode": "h", "CommunityNode": "c",
    "SessionNode": "s", "VisualNode": "v",
    "OntologyType": "o", "OntologyEntity": "oe",
}

for label, alias in labels_aliases.items():
    rows = q(f"MATCH ({alias}:{label}) RETURN {alias}")
    flat = flatten_rows(alias, rows)
    # Convert embedding bytes→list for EpisodeNode
    if label == "EpisodeNode":
        for row in flat:
            emb = row.get("embedding")
            if emb is not None and not isinstance(emb, (list, tuple)):
                try:
                    row["embedding"] = np.frombuffer(emb, dtype=np.float32).tolist()
                except Exception:
                    row["embedding"] = [0.0] * 384
    nodes[label] = flat
    print(f"  {label}: {len(flat)}")

# ─── 导出边 ───
print("\n📤 导出边...")
rels = {}
rel_queries = {
    "HEBBIAN_CONNECTION": (
        "MATCH (a:EpisodeNode)-[r:HEBBIAN_CONNECTION]->(b:EpisodeNode) "
        "RETURN a.id AS a_id, b.id AS b_id, r.weight AS weight"
    ),
    "HYPEREDGE_MEMBER": (
        "MATCH (h:HyperedgeNode)-[r:HYPEREDGE_MEMBER]->(e:EpisodeNode) "
        "RETURN h.id AS h_id, e.id AS e_id"
    ),
    "COMMUNITY_MEMBER": (
        "MATCH (c:CommunityNode)-[r:COMMUNITY_MEMBER]->(e:EpisodeNode) "
        "RETURN c.id AS c_id, e.id AS e_id"
    ),
    "TEMPORAL_LINK": (
        "MATCH (a:EpisodeNode)-[r:TEMPORAL_LINK]->(b:EpisodeNode) "
        "RETURN a.id AS a_id, b.id AS b_id, r.time_diff AS time_diff"
    ),
    "SESSION_MEMBER": (
        "MATCH (s:SessionNode)-[r:SESSION_MEMBER]->(e:EpisodeNode) "
        "RETURN s.id AS s_id, e.id AS e_id"
    ),
    "IS_A": (
        "MATCH (o:OntologyEntity)-[r:IS_A]->(t:OntologyType) "
        "RETURN o.name AS o_name, t.name AS t_name"
    ),
    "RELATES_TO": (
        "MATCH (a:OntologyEntity)-[r:RELATES_TO]->(b:OntologyEntity) "
        "RETURN a.name AS a_name, b.name AS b_name, r.relation AS relation"
    ),
}

rel_key_map = {
    "HEBBIAN_CONNECTION": {"src": "a_id", "dst": "b_id", "extra": ["weight"]},
    "HYPEREDGE_MEMBER": {"src": "h_id", "dst": "e_id", "extra": []},
    "COMMUNITY_MEMBER": {"src": "c_id", "dst": "e_id", "extra": []},
    "TEMPORAL_LINK": {"src": "a_id", "dst": "b_id", "extra": ["time_diff"]},
    "SESSION_MEMBER": {"src": "s_id", "dst": "e_id", "extra": []},
    "IS_A": {"src": "o_name", "dst": "t_name", "extra": []},
    "RELATES_TO": {"src": "a_name", "dst": "b_name", "extra": ["relation"]},
}

for rel, query in rel_queries.items():
    rows = q(query)
    km = rel_key_map[rel]
    clean = []
    for row in rows:
        item = {"src": row[km["src"]], "dst": row[km["dst"]]}
        for k in km["extra"]:
            item[k] = row.get(k)
        clean.append(item)
    rels[rel] = clean
    print(f"  {rel}: {len(clean)}")

conn.close()
db.close()

data = {
    "nodes": nodes,
    "rels": rels,
    "meta": {"exported_at": time.time(), "source": "kuzu"},
}

with open(EXPORT_FILE, "w") as f:
    json.dump(data, f, default=str)

print(f"\n✅ 导出完成: {EXPORT_FILE} ({(Path(EXPORT_FILE).stat().st_size / 1024):.0f} KB)")
