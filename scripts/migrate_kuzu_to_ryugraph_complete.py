#!/usr/bin/env python3
"""
SHM Kuzu → RyuGraph 完整数据迁移
=================================
将旧 Kuzu DB 的全部数据补迁移到已有的 RyuGraph DB。
使用 MERGE 避免覆盖运行时积累的新数据。

用法:
  python scripts/migrate_kuzu_to_ryugraph_complete.py
"""
import json
import os
import sys
import shutil
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KUZU_DB = str(DATA_DIR / "shm_kuzu_db")
RYU_DB = str(DATA_DIR / "shm_ryugraph_db")


def export_kuzu():
    """导出旧 Kuzu DB 全部数据"""
    import kuzu
    if not os.path.exists(KUZU_DB):
        print(f"❌ Kuzu DB not found: {KUZU_DB}")
        return None

    db = kuzu.Database(KUZU_DB)
    conn = kuzu.Connection(db)

    def q(query):
        return conn.execute(query).get_as_pl().to_pandas().to_dict("records")

    data = {
        "nodes": {
            "EpisodeNode": q("MATCH (e:EpisodeNode) RETURN *"),
            "HyperedgeNode": q("MATCH (h:HyperedgeNode) RETURN *"),
            "CommunityNode": q("MATCH (c:CommunityNode) RETURN *"),
            "SessionNode": q("MATCH (s:SessionNode) RETURN *"),
            "VisualNode": q("MATCH (v:VisualNode) RETURN *"),
            "OntologyType": q("MATCH (o:OntologyType) RETURN *"),
            "OntologyEntity": q("MATCH (o:OntologyEntity) RETURN *"),
        },
        "rels": {
            "HEBBIAN_CONNECTION": q("MATCH (a)-[r:HEBBIAN_CONNECTION]->(b) RETURN id(a) AS src, id(b) AS dst, r.weight"),
            "HYPEREDGE_MEMBER": q("MATCH (h:HyperedgeNode)-[r:HYPEREDGE_MEMBER]->(e) RETURN id(h) AS src, id(e) AS dst"),
            "COMMUNITY_MEMBER": q("MATCH (c:CommunityNode)-[r:COMMUNITY_MEMBER]->(e) RETURN id(c) AS src, id(e) AS dst"),
            "TEMPORAL_LINK": q("MATCH (a)-[r:TEMPORAL_LINK]->(b) RETURN id(a) AS src, id(b) AS dst, r.time_diff"),
            "SESSION_MEMBER": q("MATCH (s)-[r:SESSION_MEMBER]->(e) RETURN id(s) AS src, id(e) AS dst"),
            "VISUAL_HYPEREDGE_MEMBER": q("MATCH (h)-[r:VISUAL_HYPEREDGE_MEMBER]->(v) RETURN id(h) AS src, id(v) AS dst"),
            "IS_A": q("MATCH (o)-[r:IS_A]->(t) RETURN id(o) AS src, id(t) AS dst"),
            "RELATES_TO": q("MATCH (a)-[r:RELATES_TO]->(b) RETURN id(a) AS src, id(b) AS dst, r.relation"),
        },
        "meta": {
            "exported_at": time.time(),
            "source": "kuzu",
        },
    }

    conn.close()
    db.close()
    return data


def safe(v, default=None):
    return v if v is not None else default


def main():
    print("📤 从 Kuzu 导出...")
    data = export_kuzu()
    if data is None:
        sys.exit(1)

    for label, rows in data["nodes"].items():
        print(f"  {label}: {len(rows)}")
    for rel, rows in data["rels"].items():
        if rows:
            print(f"  {rel}: {len(rows)}")

    # 备份当前 RyuGraph DB
    backup_path = str(DATA_DIR / f"shm_ryugraph_db.backup.{int(time.time())}")
    if os.path.exists(RYU_DB):
        if os.path.isdir(RYU_DB):
            shutil.copytree(RYU_DB, backup_path, dirs_exist_ok=True)
        else:
            shutil.copy2(RYU_DB, backup_path)
        # 也备份 WAL
        wal_path = RYU_DB + ".wal"
        if os.path.exists(wal_path):
            shutil.copy2(wal_path, backup_path + ".wal")
        print(f"\n📦 RyuGraph DB 已备份: {backup_path}")

    print("\n📥 导入到 RyuGraph (MERGE 模式)...")
    import ryugraph as rykuzu

    db = rykuzu.Database(RYU_DB)
    conn = rykuzu.Connection(db)

    # ─── 导入节点 (MERGE) ───
    node_counts = {}
    for label, rows in data["nodes"].items():
        if not rows:
            node_counts[label] = 0
            continue

        cols = [k for k in rows[0].keys() if k != "_id"]
        pk_cols = {
            "EpisodeNode": "id", "HyperedgeNode": "id", "CommunityNode": "id",
            "SessionNode": "id", "VisualNode": "id",
            "OntologyType": "name", "OntologyEntity": "name",
        }
        pk = pk_cols.get(label, "id")

        set_clause = ", ".join([f"n.{c} = ${c}" for c in cols if c != pk])
        merge_q = f"MERGE (n:{label} {{{pk}: ${pk}}}) ON CREATE SET {set_clause}"

        cnt = 0
        for row in rows:
            params = {}
            for c in cols:
                params[c] = row.get(c)
            # Handle embedding bytes→list
            if label == "EpisodeNode" and "embedding" in params:
                emb = params["embedding"]
                if emb is not None and not isinstance(emb, (list, tuple)):
                    import numpy as np
                    try:
                        params["embedding"] = np.frombuffer(emb, dtype=np.float32).tolist()
                    except Exception:
                        params["embedding"] = [0.0] * 384
            try:
                conn.execute(merge_q, params)
                cnt += 1
            except Exception as e:
                if "duplicate" not in str(e).lower() and "already exists" not in str(e).lower():
                    print(f"  ⚠️  Skip {label} {row.get(pk,'')[:30]}: {str(e)[:60]}")
        node_counts[label] = cnt
        print(f"  {label}: {cnt} (from {len(rows)} in Kuzu)")

    # ─── 导入边 (MERGE) ───
    rel_counts = {}
    for rel, rows in data["rels"].items():
        if not rows:
            rel_counts[rel] = 0
            continue

        # Build MATCH → MERGE query per rel type
        if rel == "HEBBIAN_CONNECTION":
            q = "MATCH (a:EpisodeNode), (b:EpisodeNode) WHERE a.id = $src AND b.id = $dst MERGE (a)-[e:HEBBIAN_CONNECTION]->(b) ON CREATE SET e.weight = $weight"
        elif rel == "HYPEREDGE_MEMBER":
            q = "MATCH (h:HyperedgeNode), (e:EpisodeNode) WHERE h.id = $src AND e.id = $dst MERGE (h)-[:HYPEREDGE_MEMBER]->(e)"
        elif rel == "COMMUNITY_MEMBER":
            q = "MATCH (c:CommunityNode), (e:EpisodeNode) WHERE c.id = $src AND e.id = $dst MERGE (c)-[:COMMUNITY_MEMBER]->(e)"
        elif rel == "TEMPORAL_LINK":
            q = "MATCH (a:EpisodeNode), (b:EpisodeNode) WHERE a.id = $src AND b.id = $dst MERGE (a)-[e:TEMPORAL_LINK]->(b) ON CREATE SET e.time_diff = $time_diff"
        elif rel == "SESSION_MEMBER":
            q = "MATCH (s:SessionNode), (e:EpisodeNode) WHERE s.id = $src AND e.id = $dst MERGE (s)-[:SESSION_MEMBER]->(e)"
        elif rel == "VISUAL_HYPEREDGE_MEMBER":
            q = "MATCH (h:HyperedgeNode), (v:VisualNode) WHERE h.id = $src AND v.id = $dst MERGE (h)-[:VISUAL_HYPEREDGE_MEMBER]->(v)"
        elif rel == "IS_A":
            q = "MATCH (o:OntologyEntity), (t:OntologyType) WHERE o.name = $src AND t.name = $dst MERGE (o)-[:IS_A]->(t)"
        elif rel == "RELATES_TO":
            q = "MATCH (a:OntologyEntity), (b:OntologyEntity) WHERE a.name = $src AND b.name = $dst MERGE (a)-[e:RELATES_TO]->(b) ON CREATE SET e.relation = $relation"
        else:
            continue

        cnt = 0
        for row in rows:
            try:
                conn.execute(q, row)
                cnt += 1
            except Exception as e:
                if "duplicate" not in str(e).lower() and "already exists" not in str(e).lower():
                    if cnt < 3:  # only show first few errors
                        print(f"  ⚠️  Skip {rel}: {str(e)[:80]}")
        rel_counts[rel] = cnt
        print(f"  {rel}: {cnt} (from {len(rows)} in Kuzu)")

    # ─── 验证 ───
    print("\n═══ 验证 ═══")
    for t in ["EpisodeNode", "HyperedgeNode", "CommunityNode", "SessionNode",
               "OntologyType", "OntologyEntity"]:
        try:
            c = conn.execute(f"MATCH (n:{t}) RETURN count(n)").get_next()[0]
            print(f"  {t}: {c}")
        except Exception as e:
            print(f"  {t}: {str(e)[:40]}")

    conn.close()
    db.close()
    print("\n✅ 数据迁移完成!")


if __name__ == "__main__":
    main()
