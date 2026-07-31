#!/usr/bin/env python3
"""Step 2: 导入数据到 RyuGraph"""
import json, os, shutil

with open("data/shm_export_flat2.json") as f:
    data = json.load(f)

RYU_DB = "data/shm_ryugraph_db"
if os.path.exists(RYU_DB):
    shutil.rmtree(RYU_DB)

import ryugraph as ryu
print(f"RyuGraph {ryu.__version__}")

db = ryu.Database(RYU_DB)
conn = ryu.Connection(db)

for q in [
    "CREATE NODE TABLE IF NOT EXISTS EpisodeNode (id STRING, content STRING, embedding DOUBLE[384], created_at DOUBLE, tau_initial DOUBLE, tau_value DOUBLE, trust_score DOUBLE, ontology_type STRING, source STRING, visibility STRING, PRIMARY KEY (id))",
    "CREATE NODE TABLE IF NOT EXISTS HyperedgeNode (id STRING, type STRING, created_at DOUBLE, gate_value DOUBLE, metadata STRING, PRIMARY KEY (id))",
    "CREATE NODE TABLE IF NOT EXISTS CommunityNode (id STRING, name STRING, summary STRING, leiden_score DOUBLE, created_at DOUBLE, PRIMARY KEY (id))",
    "CREATE REL TABLE IF NOT EXISTS HEBBIAN_CONNECTION (FROM EpisodeNode TO EpisodeNode, weight DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS HYPEREDGE_MEMBER (FROM HyperedgeNode TO EpisodeNode)",
]:
    conn.execute(q)

def safe(v, default):
    return v if v is not None else default

cnt = 0
for row in data["EpisodeNode"]:
    emb = safe(row.get("embedding"), [0.0]*384)
    try:
        conn.execute(
            "CREATE (e:EpisodeNode {id: $id, content: $content, embedding: $embedding, created_at: $created_at, tau_initial: $tau_initial, tau_value: $tau_value, trust_score: $trust_score, ontology_type: $ontology_type, source: $source, visibility: $visibility}) RETURN e.id",
            {"id": row["id"], "content": safe(row["content"], ""), "embedding": emb,
             "created_at": safe(row["created_at"], 0.0), "tau_initial": safe(row["tau_initial"], 1.0),
             "tau_value": safe(row["tau_value"], 1.0), "trust_score": safe(row["trust_score"], 0.5),
             "ontology_type": safe(row["ontology_type"], ""), "source": safe(row["source"], ""),
             "visibility": safe(row["visibility"], "private")}
        )
        cnt += 1
    except Exception as e:
        pid = str(row.get("id",""))[:20]
        print(f"  Skip EpisodeNode {pid}: {e}")
print(f"  EpisodeNode: {cnt}")

cnt = 0
for row in data["HyperedgeNode"]:
    try:
        conn.execute(
            "CREATE (h:HyperedgeNode {id: $id, type: $type, created_at: $created_at, gate_value: $gate_value, metadata: $metadata}) RETURN h.id",
            {"id": row["id"], "type": safe(row["type"], ""), "created_at": safe(row["created_at"], 0.0),
             "gate_value": safe(row["gate_value"], 0.0), "metadata": safe(row["metadata"], "")}
        )
        cnt += 1
    except Exception as e:
        pid = str(row.get("id",""))[:20]
        print(f"  Skip HyperedgeNode {pid}: {e}")
print(f"  HyperedgeNode: {cnt}")

cnt = 0
for row in data["CommunityNode"]:
    try:
        conn.execute(
            "CREATE (c:CommunityNode {id: $id, name: $name, summary: $summary, leiden_score: $leiden_score, created_at: $created_at}) RETURN c.id",
            {"id": row["id"], "name": safe(row["name"], ""), "summary": safe(row["summary"], ""),
             "leiden_score": safe(row["leiden_score"], 0.0), "created_at": safe(row["created_at"], 0.0)}
        )
        cnt += 1
    except Exception as e:
        pid = str(row.get("id",""))[:20]
        print(f"  Skip CommunityNode {pid}: {e}")
print(f"  CommunityNode: {cnt}")

cnt = 0
for row in data["rels"]["hebbian"]:
    try:
        conn.execute("MATCH (a:EpisodeNode {id: $src}), (b:EpisodeNode {id: $dst}) CREATE (a)-[:HEBBIAN_CONNECTION {weight: $w}]->(b)",
            {"src": row["src"], "dst": row["dst"], "w": row["weight"]})
        cnt += 1
    except:
        pass
print(f"  HEBBIAN_CONNECTION: {cnt}")

cnt = 0
for row in data["rels"]["hyperedges"]:
    try:
        conn.execute("MATCH (h:HyperedgeNode {id: $src}), (e:EpisodeNode {id: $dst}) CREATE (h)-[:HYPEREDGE_MEMBER]->(e)",
            {"src": row["src"], "dst": row["dst"]})
        cnt += 1
    except:
        pass
print(f"  HYPEREDGE_MEMBER: {cnt}")

print()
ep = conn.execute("MATCH (e:EpisodeNode) RETURN count(e)").get_next()[0]
hp = conn.execute("MATCH (h:HyperedgeNode) RETURN count(h)").get_next()[0]
cp = conn.execute("MATCH (c:CommunityNode) RETURN count(c)").get_next()[0]
hb = conn.execute("MATCH ()-[r:HEBBIAN_CONNECTION]->() RETURN count(r)").get_next()[0]
hm = conn.execute("MATCH ()-[r:HYPEREDGE_MEMBER]->() RETURN count(r)").get_next()[0]
print(f"═══ Verify ═══")
print(f"  EpisodeNode:       {ep}")
print(f"  HyperedgeNode:     {hp}")
print(f"  CommunityNode:     {cp}")
print(f"  HEBBIAN_CONNECTION: {hb}")
print(f"  HYPEREDGE_MEMBER:  {hm}")

conn.close()
db.close()
print("✅ Migration done")
