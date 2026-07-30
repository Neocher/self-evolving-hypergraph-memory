#!/usr/bin/env python3
"""Clean migration: kuzu_migration_full.json → fresh GraphLite DB"""
import sys, json, shutil, os
from base64 import b64encode

sys.path.insert(0, "/home/admin/GraphLite/bindings/python")
sys.path.insert(0, "/home/admin/GraphLite/sdk-python/src")
from graphlite_sdk import GraphLite

def esc(s):
    s = str(s)
    try:
        s.encode("ascii")
        return f"'{s.replace(chr(92), chr(92)+chr(92)).replace(chr(39), chr(92)+chr(39))}'"
    except UnicodeEncodeError:
        return f"'{{b64}}{b64encode(s.encode('utf-8')).decode('ascii')}'"

DB = "/home/admin/shm/data/shm_graphlite_db"
shutil.rmtree(DB, ignore_errors=True)

db = GraphLite.open(DB)
session = db.session("admin")
session.execute("CREATE SCHEMA IF NOT EXISTS /shm")
session.execute("SESSION SET SCHEMA /shm")
session.execute("CREATE GRAPH IF NOT EXISTS default")
session.execute("SESSION SET GRAPH default")

with open("data/kuzu_migration_full.json") as f:
    data = json.load(f)

nodes = [n for n in data["nodes"]["EpisodeNode"] if n.get("source") != "bench"]
for n in nodes:
    nid = n.pop("id", "")
    props = {k: v for k, v in n.items() if v is not None}
    vals = ", ".join(f"{k}: {esc(v)}" for k, v in props.items())
    session.execute(f"INSERT (e:EpisodeNode {{id: {esc(nid)}, {vals}}})")

hyps = data["nodes"].get("HyperedgeNode", [])
for h in hyps:
    hid = h.pop("id", "")
    props = {k: v for k, v in h.items() if v is not None}
    vals = ", ".join(f"{k}: {esc(v)}" for k, v in props.items())
    session.execute(f"INSERT (h:HyperedgeNode {{id: {esc(hid)}, {vals}}})")

# Fix: the rels use _src/_dst but also src/dst
for src_name, dst_name in [("src", "dst"), ("_src", "_dst")]:
    mem = data.get("rels", {}).get("HYPEREDGE_MEMBER", [])
    for e in mem:
        src, dst = e.get(src_name, ""), e.get(dst_name, "")
        if src and dst:
            try:
                session.execute(f"MATCH (h:HyperedgeNode {{id: {esc(src)}}}), (e:EpisodeNode {{id: {esc(dst)}}}) INSERT (h)-[:HYPEREDGE_MEMBER]->(e)")
            except: pass

    heb = data.get("rels", {}).get("HEBBIAN_CONNECTION", [])
    for e in heb:
        src, dst = e.get(src_name, ""), e.get(dst_name, "")
        w = e.get("weight", e.get("_data", {}).get("weight", 0.5))
        if src and dst:
            try:
                session.execute(f"MATCH (a:EpisodeNode {{id: {esc(src)}}}), (b:EpisodeNode {{id: {esc(dst)}}}) INSERT (a)-[:HEBBIAN {{weight: {w}}}]->(b)")
            except: pass

r = session.query("MATCH (e:EpisodeNode) RETURN count(e) AS c")
h = session.query("MATCH (h:HyperedgeNode) RETURN count(h) AS c")
m = session.query("MATCH ()-[:HYPEREDGE_MEMBER]->() RETURN count(*) AS c")
he = session.query("MATCH ()-[:HEBBIAN]->() RETURN count(*) AS c")
print(f"EP={r.rows[0]['c']} HE={h.rows[0]['c']} HM={m.rows[0]['c']} HEB={he.rows[0]['c']}")
db.close()
