#!/usr/bin/env python3
"""完整数据迁移: kuzu_migration_full.json → GraphLite API"""
import json, time, sys, os, requests

BASE = "http://127.0.0.1:8000"
sys.path.insert(0, "/home/admin/GraphLite/bindings/python")
sys.path.insert(0, "/home/admin/GraphLite/sdk-python/src")
from graphlite_sdk import GraphLite

def migrate():
    with open("data/kuzu_migration_full.json") as f:
        data = json.load(f)
    
    nodes = data.get("nodes", {}).get("EpisodeNode", [])
    real = [n for n in nodes if n.get("source") != "bench"]
    hyperedges = data.get("nodes", {}).get("HyperedgeNode", [])
    
    # 使用 GraphLite 直连迁移（不走API，更可靠）
    db = GraphLite.open("/home/admin/shm/data/shm_graphlite_db")
    session = db.session("admin")
    session.execute("CREATE SCHEMA IF NOT EXISTS /shm")
    session.execute("SESSION SET SCHEMA /shm")
    session.execute("CREATE GRAPH IF NOT EXISTS default")
    session.execute("SESSION SET GRAPH default")
    
    id_map = {}  # old_id → new_id
    
    print(f"迁移 {len(real)} 个 EpisodeNode...")
    for i, n in enumerate(real):
        nid = n.get("id", "")
        content = n.get("content", "")
        if not content:
            continue
        vals = []
        for k, v in n.items():
            if v is None: continue
            if isinstance(v, (int, float)):
                vals.append(f"{k}: {v}")
            elif isinstance(v, str):
                safe = v.replace("\\","\\\\").replace("'","\\'")
                try:
                    safe.encode("ascii")
                    vals.append(f"{k}: '{safe}'")
                except UnicodeEncodeError:
                    from base64 import b64encode
                    b64 = b64encode(v.encode("utf-8")).decode("ascii")
                    vals.append(f"{k}: '{{b64}}{b64}'")
            else:
                vals.append(f"{k}: '{v}'")
        
        vstr = ", ".join(vals)
        gql = f"INSERT (e:EpisodeNode {{id: '{nid}', {vstr}}})"
        try:
            session.execute(gql)
            id_map[nid] = nid
        except Exception as e:
            print(f"  [{i}] {nid[:12]} fail: {str(e)[:60]}")
            continue
        
        if (i+1) % 10 == 0:
            print(f"  ... {i+1}/{len(real)}")
    
    print(f"迁移 {len(hyperedges)} 个 HyperedgeNode...")
    for i, h in enumerate(hyperedges):
        hid = h.get("id", "")
        htype = h.get("type", "episode")
        gate = h.get("gate_value", 1.0)
        created = h.get("created_at", time.time())
        meta = h.get("metadata", "{}")
        gql = f"INSERT (h:HyperedgeNode {{id: '{hid}', type: '{htype}', gate_value: {gate}, created_at: {created}, metadata: '{meta}'}})"
        try:
            session.execute(gql)
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"  h[{i}] fail: {str(e)[:60]}")
    
    # 迁移关系
    edges_types = {
        "HYPEREDGE_MEMBER": data.get("edges", {}).get("HYPEREDGE_MEMBER", 
            data.get("rels", {}).get("HYPEREDGE_MEMBER", [])),
        "HEBBIAN_CONNECTION": data.get("edges", {}).get("HEBBIAN_CONNECTION",
            data.get("rels", {}).get("HEBBIAN_CONNECTION", [])),
    }
    
    for rel_type, edges in edges_types.items():
        if not edges: continue
        print(f"迁移 {len(edges)} 条 {rel_type}...")
        for e in edges:
            src = e.get("_src", e.get("src", e.get("from_id", "")))
            dst = e.get("_dst", e.get("dst", e.get("to_id", "")))
            if not src or not dst:
                continue
            if rel_type == "HYPEREDGE_MEMBER":
                gql = f"MATCH (h:HyperedgeNode {{id: '{src}'}}), (e:EpisodeNode {{id: '{dst}'}}) INSERT (h)-[:HYPEREDGE_MEMBER]->(e)"
            elif rel_type == "HEBBIAN_CONNECTION":
                weight = e.get("weight", e.get("_data", {}).get("weight", 0.5))
                gql = f"MATCH (a:EpisodeNode {{id: '{src}'}}), (b:EpisodeNode {{id: '{dst}'}}) INSERT (a)-[:HEBBIAN {{weight: {weight}}}]->(b)"
            try:
                session.execute(gql)
            except:
                pass
    
    db.close()
    
    # 验证
    db2 = GraphLite.open("/home/admin/shm/data/shm_graphlite_db")
    s2 = db2.session("admin")
    s2.execute("SESSION SET SCHEMA /shm")
    s2.execute("SESSION SET GRAPH default")
    
    r1 = s2.query("MATCH (e:EpisodeNode) RETURN count(e) AS cnt")
    r2 = s2.query("MATCH (h:HyperedgeNode) RETURN count(h) AS cnt")
    r3 = s2.query("MATCH ()-[r:HYPEREDGE_MEMBER]->() RETURN count(r) AS cnt")
    
    print(f"\n迁移完成!")
    print(f"  EpisodeNode: {r1.rows[0]['cnt']}")
    print(f"  HyperedgeNode: {r2.rows[0]['cnt']}")
    print(f"  HYPEREDGE_MEMBER: {r3.rows[0]['cnt']}")
    db2.close()

if __name__ == "__main__":
    migrate()
