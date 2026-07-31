#!/usr/bin/env python3
"""
SHM Kuzu → RyuGraph 数据迁移脚本
================================
从现有 Kuzu 数据库导出并导入到 RyuGraph。

用法:
  python scripts/migrate_to_ryugraph.py          # 自动迁移（默认路径）
  python scripts/migrate_to_ryugraph.py --check  # 仅检查，不迁移
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# ─── 配置 ──────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KUZU_DB = DATA_DIR / "shm_kuzu_db"
RYU_DB = DATA_DIR / "shm_ryugraph_db"
EXPORT_FILE = DATA_DIR / "shm_export_before_migration.json"


def export_from_kuzu():
    """从现有 Kuzu 数据库导出全部数据"""
    import kuzu

    if not KUZU_DB.exists():
        print(f"❌ Kuzu 数据库不存在: {KUZU_DB}")
        return None

    db = kuzu.Database(str(KUZU_DB))
    conn = kuzu.Connection(db)

    def q(query):
        return conn.execute(query).get_as_pl().to_pandas().to_dict("records")

    data = {
        "nodes": {
            "EpisodeNode": q("MATCH (e:EpisodeNode) RETURN *"),
            "HyperedgeNode": q("MATCH (h:HyperedgeNode) RETURN *"),
            "CommunityNode": q("MATCH (c:CommunityNode) RETURN *"),
        },
        "rels": {
            "HEBBIAN_CONNECTION": q("MATCH (a)-[r:HEBBIAN_CONNECTION]->(b) RETURN id(a) AS src, id(b) AS dst, r.weight"),
            "HYPEREDGE_MEMBER": q("MATCH (h:HyperedgeNode)-[r:HYPEREDGE_MEMBER]->(e) RETURN id(h) AS src, id(e) AS dst"),
            "COMMUNITY_MEMBER": q("MATCH (c:CommunityNode)-[r:COMMUNITY_MEMBER]->(e) RETURN id(c) AS src, id(e) AS dst"),
            "TEMPORAL_LINK": q("MATCH (a)-[r:TEMPORAL_LINK]->(b) RETURN id(a) AS src, id(b) AS dst, r.time_diff"),
        },
        "meta": {
            "exported_at": time.time(),
            "source": "kuzu",
            "version": kuzu.__version__ if hasattr(kuzu, "__version__") else "unknown",
        },
    }

    conn.close()
    db.close()
    return data


def import_to_ryugraph(data):
    """导入数据到 RyuGraph"""
    import ryugraph as ryu

    # 清理旧的 RyuGraph 数据库
    if RYU_DB.exists():
        shutil.rmtree(RYU_DB)

    db = ryu.Database(str(RYU_DB))
    conn = ryu.Connection(db)

    def execute(q):
        conn.execute(q)

    # ─── 创建 Schema ────────────────────────────────────
    execute(
        "CREATE NODE TABLE IF NOT EXISTS EpisodeNode ("
        "id STRING, content STRING, embedding FLOAT[384], "
        "created_at DOUBLE, tau_initial DOUBLE, tau_value DOUBLE, "
        "trust_score DOUBLE, ontology_type STRING, source STRING, "
        "visibility STRING, "
        "PRIMARY KEY (id))"
    )
    execute(
        "CREATE NODE TABLE IF NOT EXISTS HyperedgeNode ("
        "id STRING, type STRING, created_at DOUBLE, "
        "gate_value DOUBLE, metadata STRING, "
        "PRIMARY KEY (id))"
    )
    execute(
        "CREATE NODE TABLE IF NOT EXISTS CommunityNode ("
        "id STRING, name STRING, summary STRING, "
        "leiden_score DOUBLE, created_at DOUBLE, "
        "PRIMARY KEY (id))"
    )
    execute(
        "CREATE REL TABLE IF NOT EXISTS HEBBIAN_CONNECTION "
        "(FROM EpisodeNode TO EpisodeNode, weight DOUBLE)"
    )
    execute(
        "CREATE REL TABLE IF NOT EXISTS HYPEREDGE_MEMBER "
        "(FROM HyperedgeNode TO EpisodeNode)"
    )
    execute(
        "CREATE REL TABLE IF NOT EXISTS COMMUNITY_MEMBER "
        "(FROM CommunityNode TO EpisodeNode)"
    )
    execute(
        "CREATE REL TABLE IF NOT EXISTS TEMPORAL_LINK "
        "(FROM EpisodeNode TO EpisodeNode, time_diff DOUBLE)"
    )

    # ─── 导入节点 ────────────────────────────────────────
    counts = {}
    for label, rows in data["nodes"].items():
        if not rows:
            counts[label] = 0
            continue
        # 列名
        cols = [k for k in rows[0].keys() if k != "_id"]
        placeholders = ", ".join([f"${c}" for c in cols])
        col_names = ", ".join(cols)
        insert = f"CREATE (n:{label} {{{col_names}}}) RETURN n.id"
        for row in rows:
            try:
                params = {c: row[c] for c in cols}
                # 处理 embedding（可能是 bytes, list, 或 None）
                emb = params.get("embedding")
                if emb is not None and not isinstance(emb, (list, tuple)):
                    import struct
                    try:
                        # numpy bytes → list
                        import numpy as np
                        params["embedding"] = np.frombuffer(emb, dtype=np.float32).tolist()
                    except Exception:
                        params["embedding"] = [0.0] * 384
                conn.execute(insert, params)
            except Exception as e:
                print(f"  ⚠️  跳过 {label} {row.get('id','')}: {e}")
        counts[label] = len(rows)

    # ─── 导入边 ──────────────────────────────────────────
    # HEBBIAN_CONNECTION: 按 id 匹配 EpisodeNode
    for rel, rows in data["rels"].items():
        if not rows:
            counts[rel] = 0
            continue
        match = 0
        for row in rows:
            try:
                if rel == "HEBBIAN_CONNECTION":
                    # 需要用 Cypher 匹配源和目标
                    q = (
                        f"MATCH (a:EpisodeNode), (b:EpisodeNode) "
                        f"WHERE id(a) = $src AND id(b) = $dst "
                        f"CREATE (a)-[:HEBBIAN_CONNECTION {{weight: $weight}}]->(b)"
                    )
                    conn.execute(q, {"src": row["src"], "dst": row["dst"], "weight": row["weight"]})
                elif rel == "HYPEREDGE_MEMBER":
                    q = (
                        f"MATCH (h:HyperedgeNode), (e:EpisodeNode) "
                        f"WHERE id(h) = $src AND id(e) = $dst "
                        f"CREATE (h)-[:HYPEREDGE_MEMBER]->(e)"
                    )
                    conn.execute(q, {"src": row["src"], "dst": row["dst"]})
                elif rel == "COMMUNITY_MEMBER":
                    q = (
                        f"MATCH (c:CommunityNode), (e:EpisodeNode) "
                        f"WHERE id(c) = $src AND id(e) = $dst "
                        f"CREATE (c)-[:COMMUNITY_MEMBER]->(e)"
                    )
                    conn.execute(q, {"src": row["src"], "dst": row["dst"]})
                elif rel == "TEMPORAL_LINK":
                    q = (
                        f"MATCH (a:EpisodeNode), (b:EpisodeNode) "
                        f"WHERE id(a) = $src AND id(b) = $dst "
                        f"CREATE (a)-[:TEMPORAL_LINK {{time_diff: $time_diff}}]->(b)"
                    )
                    conn.execute(q, {"src": row["src"], "dst": row["dst"], "time_diff": row["time_diff"]})
                match += 1
            except Exception as e:
                print(f"  ⚠️  跳过 {rel} {row.get('src','')}→{row.get('dst','')}: {e}")
        counts[rel] = match

    conn.close()
    db.close()
    return counts


def verify():
    """验证 RyuGraph 数据库完整性"""
    import ryugraph as ryu

    db = ryu.Database(str(RYU_DB))
    conn = ryu.Connection(db)

    def c(q):
        return conn.execute(q).get_next()[0]

    print()
    print("═══ 验证结果 ═══")
    ep = c("MATCH (e:EpisodeNode) RETURN count(e)")
    hp = c("MATCH (h:HyperedgeNode) RETURN count(h)")
    cp = c("MATCH (c:CommunityNode) RETURN count(c)")
    hb = c("MATCH ()-[r:HEBBIAN_CONNECTION]->() RETURN count(r)")
    hm = c("MATCH ()-[r:HYPEREDGE_MEMBER]->() RETURN count(r)")
    print(f"  EpisodeNode:       {ep}")
    print(f"  HyperedgeNode:     {hp}")
    print(f"  CommunityNode:     {cp}")
    print(f"  HEBBIAN_CONNECTION: {hb}")
    print(f"  HYPEREDGE_MEMBER:  {hm}")
    conn.close()
    db.close()
    return {"episodes": ep, "hyperedges": hp, "communities": cp}


def main():
    parser = argparse.ArgumentParser(description="Kuzu → RyuGraph 迁移")
    parser.add_argument("--check", action="store_true", help="仅检查不迁移")
    args = parser.parse_args()

    # 备份 Kuzu DB
    backup = DATA_DIR / f"shm_kuzu_db.backup.{int(time.time())}"
    if KUZU_DB.exists():
        shutil.copy2(KUZU_DB, backup)
        print(f"📦 Kuzu DB 已备份到: {backup}")

    # 导出
    print("📤 从 Kuzu 导出...")
    data = export_from_kuzu()
    if data is None:
        sys.exit(1)

    for label, rows in data["nodes"].items():
        print(f"  {label}: {len(rows)}")
    for rel, rows in data["rels"].items():
        print(f"  {rel}: {len(rows)}")

    # 保存导出文件
    with open(DATA_DIR / "shm_export_before_migration.json", "w") as f:
        json.dump({k: {k2: v2 for k2, v2 in v.items()} if isinstance(v, dict) else v for k, v in data.items()}, f, default=str)

    if args.check:
        print("✅ 检查完成（未迁移）")
        return

    # 导入
    print()
    print("📥 导入到 RyuGraph...")
    counts = import_to_ryugraph(data)

    print()
    for label, cnt in counts.items():
        print(f"  {label}: {cnt}")

    # 验证
    verify()

    # 切换
    print()
    print(f"🔄 数据库路径: {RYU_DB}")
    print("✅ 迁移完成！现在可以切换 config 中的 database_path")


if __name__ == "__main__":
    main()
