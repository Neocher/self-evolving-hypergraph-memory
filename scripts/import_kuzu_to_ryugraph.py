#!/usr/bin/env python3
"""Step 2: 从 JSON 导入数据到 RyuGraph (MERGE 模式 — 修正版)"""
import json
import shutil
import os
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RYU_DB = str(DATA_DIR / "shm_ryugraph_db")
EXPORT_FILE = str(DATA_DIR / "kuzu_migration_full.json")

if not os.path.exists(EXPORT_FILE):
    print(f"❌ 导出文件不存在: {EXPORT_FILE}")
    sys.exit(1)

# 备份当前 RyuGraph DB
backup_path = str(DATA_DIR / f"shm_ryugraph_db.backup.{int(time.time())}")
if os.path.exists(RYU_DB):
    if os.path.isdir(RYU_DB):
        shutil.copytree(RYU_DB, backup_path, dirs_exist_ok=True)
    else:
        shutil.copy2(RYU_DB, backup_path)
    wal_path = RYU_DB + ".wal"
    if os.path.exists(wal_path):
        shutil.copy2(wal_path, backup_path + ".wal")
    print(f"📦 RyuGraph DB 已备份: {backup_path}")

with open(EXPORT_FILE) as f:
    data = json.load(f)

import ryugraph as ryu

print(f"RyuGraph {ryu.__version__ if hasattr(ryu, '__version__') else ''}")

db = ryu.Database(RYU_DB)
conn = ryu.Connection(db)

# ─── 节点 PK 映射 ───
PK_MAP = {
    "EpisodeNode": "id", "HyperedgeNode": "id", "CommunityNode": "id",
    "SessionNode": "id", "VisualNode": "id",
    "OntologyType": "name", "OntologyEntity": "name",
}

# ─── 导入节点 (MERGE) ───
print("\n📥 导入节点...")
node_counts = {}
for label, rows in data["nodes"].items():
    if not rows:
        node_counts[label] = 0
        continue
    pk = PK_MAP[label]
    # Get all column names from first row
    cols = list(rows[0].keys())
    # Remove _ID, _LABEL if any (safety check)
    cols = [c for c in cols if c not in ('_ID', '_LABEL')]
    
    # Build MERGE query: label name might be 'n' which is also reserved... use label as alias
    set_clause = ", ".join([f"n.{c} = ${c}" for c in cols if c != pk])
    merge_q = f"MERGE (n:{label} {{{pk}: ${pk}}}) ON CREATE SET {set_clause}"

    cnt = 0
    errors = 0
    for row in rows:
        params = {c: row.get(c) for c in cols}
        try:
            conn.execute(merge_q, params)
            cnt += 1
        except Exception as e:
            err = str(e)
            if "duplicate" not in err.lower() and "already exists" not in err.lower() and "violate" not in err.lower():
                if errors < 3:
                    print(f"  ⚠️  {label} {row.get(pk,'')[:20]}: {err[:80]}")
                errors += 1
    node_counts[label] = cnt
    extra = f" ({errors} errors)" if errors else ""
    print(f"  {label}: {cnt}/{len(rows)}{extra}")

# ─── 导入边 (MERGE) ───
REL_MAP = {
    "HEBBIAN_CONNECTION": {
        "q": "MATCH (a:EpisodeNode), (b:EpisodeNode) WHERE a.id = $src AND b.id = $dst MERGE (a)-[e:HEBBIAN_CONNECTION]->(b) ON CREATE SET e.weight = $weight",
        "label": "HEBBIAN_CONNECTION",
    },
    "HYPEREDGE_MEMBER": {
        "q": "MATCH (h:HyperedgeNode), (e:EpisodeNode) WHERE h.id = $src AND e.id = $dst MERGE (h)-[:HYPEREDGE_MEMBER]->(e)",
        "label": "HYPEREDGE_MEMBER",
    },
    "COMMUNITY_MEMBER": {
        "q": "MATCH (c:CommunityNode), (e:EpisodeNode) WHERE c.id = $src AND e.id = $dst MERGE (c)-[:COMMUNITY_MEMBER]->(e)",
        "label": "COMMUNITY_MEMBER",
    },
    "TEMPORAL_LINK": {
        "q": "MATCH (a:EpisodeNode), (b:EpisodeNode) WHERE a.id = $src AND b.id = $dst MERGE (a)-[e:TEMPORAL_LINK]->(b) ON CREATE SET e.time_diff = $time_diff",
        "label": "TEMPORAL_LINK",
    },
    "SESSION_MEMBER": {
        "q": "MATCH (s:SessionNode), (e:EpisodeNode) WHERE s.id = $src AND e.id = $dst MERGE (s)-[:SESSION_MEMBER]->(e)",
        "label": "SESSION_MEMBER",
    },
    "IS_A": {
        "q": "MATCH (o:OntologyEntity), (t:OntologyType) WHERE o.name = $src AND t.name = $dst MERGE (o)-[:IS_A]->(t)",
        "label": "IS_A",
    },
    "RELATES_TO": {
        "q": "MATCH (a:OntologyEntity), (b:OntologyEntity) WHERE a.name = $src AND b.name = $dst MERGE (a)-[e:RELATES_TO]->(b) ON CREATE SET e.relation = $relation",
        "label": "RELATES_TO",
    },
}

print("\n📥 导入边...")
for rel in ["HEBBIAN_CONNECTION", "HYPEREDGE_MEMBER", "COMMUNITY_MEMBER",
             "TEMPORAL_LINK", "SESSION_MEMBER", "IS_A", "RELATES_TO"]:
    rows = data["rels"].get(rel, [])
    if not rows:
        continue
    rm = REL_MAP[rel]
    cnt = 0
    for row in rows:
        try:
            conn.execute(rm["q"], row)
            cnt += 1
        except Exception:
            pass
    print(f"  {rel}: {cnt}/{len(rows)}")

# ─── 验证 ───
print("\n═══ 验证 ═══")
for t in ["EpisodeNode", "HyperedgeNode", "CommunityNode", "SessionNode",
           "OntologyType", "OntologyEntity", "ProceduralNode", "ConceptualNode"]:
    try:
        c = conn.execute(f"MATCH (n:{t}) RETURN count(n)").get_next()[0]
        print(f"  {t}: {c}")
    except Exception as e:
        print(f"  {t}: {str(e)[:50]}")

# Rels
for r in ["HEBBIAN_CONNECTION", "HYPEREDGE_MEMBER", "IS_A", "RELATES_TO",
           "SESSION_MEMBER"]:
    try:
        c = conn.execute(f"MATCH ()-[e:{r}]->() RETURN count(e)").get_next()[0]
        print(f"  {r}: {c}")
    except Exception as e:
        print(f"  {r}: {str(e)[:50]}")

conn.close()
db.close()
print("\n✅ 迁移完成!")
