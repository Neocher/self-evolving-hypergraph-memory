#!/usr/bin/env python3
"""Phase 2: Import exported data into new Kuzu DB."""
import json, os, sys
import kuzu

NEW_DB = "data/shm_kuzu_db"
EXPORT_DIR = "data/migration_export"

def main():
    db = kuzu.Database(NEW_DB)
    conn = kuzu.Connection(db)

    # ---- Node definitions: table -> (pk_col, [data_columns]) ----
    node_defs = {
        "OntologyType":    ("name", ["category"]),
        "OntologyEntity":  ("name", ["type", "category"]),
        "EpisodeNode":     ("id", ["content", "embedding", "created_at",
                                    "tau_initial", "tau_value", "trust_score",
                                    "ontology_type", "source"]),
        "HyperedgeNode":   ("id", ["type", "created_at", "gate_value", "metadata"]),
        "CommunityNode":   ("id", ["name", "summary", "leiden_score", "created_at"]),
        "SessionNode":     ("id", ["session_id", "created_at", "metadata"]),
        "VisualNode":      ("id", ["image_path", "caption", "embedding",
                                   "source", "created_at"]),
    }
    node_order = ["OntologyType", "OntologyEntity", "EpisodeNode",
                  "HyperedgeNode", "CommunityNode", "SessionNode", "VisualNode"]

    total_nodes = 0
    for table in node_order:
        path = f"{EXPORT_DIR}/{table}.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        if not data:
            print(f"  {table}: empty (skipped)")
            continue

        pk, cols = node_defs[table]
        # Build MERGE: MERGE (n:Table {pk: $pk}) ON CREATE SET n.col1=$col1, n.col2=$col2
        set_clause = ", ".join([f"n.{c} = ${c}" for c in cols])
        query = f"MERGE (n:{table} {{{pk}: ${pk}}}) ON CREATE SET {set_clause}"

        for row in data:
            params = {}
            for c in [pk] + cols:
                val = row.get(c)
                # Handle FLOAT[384] arrays - Kuzu expects lists
                if isinstance(val, list) and len(val) == 384:
                    val = val  # pass through as list
                params[c] = val
            try:
                conn.execute(query, params)
            except RuntimeError as e:
                err = str(e).lower()
                if "already exists" not in err and "violate" not in err:
                    print(f"  WARN {table}: {e}")

        total_nodes += len(data)
        print(f"  {table}: {len(data)} nodes ✓")

    # ---- Rel definitions ----
    rel_defs = [
        ("HEBBIAN_CONNECTION", "EpisodeNode", "id", "EpisodeNode", "id", ["weight"]),
        ("HYPEREDGE_MEMBER", "HyperedgeNode", "id", "EpisodeNode", "id", []),
        ("IS_A", "OntologyEntity", "name", "OntologyType", "name", []),
        ("RELATES_TO", "OntologyEntity", "name", "OntologyEntity", "name", ["relation"]),
        # Zero-rows: COMMUNITY_MEMBER, TEMPORAL_LINK, SESSION_MEMBER, VISUAL_HYPEREDGE_MEMBER
    ]

    total_rels = 0
    for rel, src_t, src_pk, dst_t, dst_pk, props in rel_defs:
        path = f"{EXPORT_DIR}/{rel}.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        if not data:
            continue

        # Build: MATCH (a) MATCH (b) MERGE (a)-[e:REL]->(b) ON CREATE SET ...
        set_clause = ""
        if props:
            set_clause = " ON CREATE SET " + ", ".join([f"e.{p} = ${p}" for p in props])

        query = (f"MATCH (a:{src_t}) WHERE a.{src_pk} = $src_key "
                 f"MATCH (b:{dst_t}) WHERE b.{dst_pk} = $dst_key "
                 f"MERGE (a)-[e:{rel}]->(b){set_clause}")

        for row in data:
            params = {"src_key": row["src_key"], "dst_key": row["dst_key"]}
            for p in props:
                params[p] = row.get(p)
            try:
                conn.execute(query, params)
            except RuntimeError as e:
                err = str(e).lower()
                if "already exists" not in err and "violate" not in err:
                    print(f"  WARN {rel}: {e}")

        total_rels += len(data)
        print(f"  {rel}: {len(data)} rels ✓")

    print(f"\nTotal imported: {total_nodes} nodes + {total_rels} edges ✓")
    conn.close()
    db.close()

if __name__ == "__main__":
    main()
