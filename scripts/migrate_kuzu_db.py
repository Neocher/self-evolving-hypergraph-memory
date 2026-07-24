#!/usr/bin/env python3
"""
SHM Kuzu DB Migration Script
=============================
自动检测 Kuzu 版本兼容性，在需要时迁移数据。

用法:
  python scripts/migrate_kuzu_db.py          # 检测+迁移
  python scripts/migrate_kuzu_db.py --check  # 仅检测

集成方式:
  - 在 run_server.py 启动前调用
  - 或作为 systemd/Timer 的 pre-start hook

工作原理:
  1. 尝试打开现有多 Kuzzy DB
  2. 检测 ALTER TABLE 兼容性
  3. 如果不兼容且 schema 有变更 → 导出 → 重建 → 导入
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
DB_PATH = DATA_DIR / "shm_kuzu_db"
BACKUP_DIR = DATA_DIR / "migration_backups"
EXPORT_DIR = DATA_DIR / "migration_export"

# 预期 schema — 当前版本需要的完整列定义
# 如果旧 DB 不包含这些列，需要迁移
EXPECTED_EPISODE_COLUMNS = [
    "id", "content", "embedding", "created_at",
    "tau_initial", "tau_value", "trust_score", "ontology_type", "source", "visibility",
]


def check_compatibility() -> tuple[bool, str]:
    """检查 Kuzu DB 兼容性。

    Returns:
        (兼容否, 消息)
    """
    if not DB_PATH.exists():
        return True, "No existing database (fresh install)"

    try:
        import kuzu
    except ImportError:
        return False, "kuzu package not installed"

    try:
        db = kuzu.Database(str(DB_PATH))
        conn = kuzu.Connection(db)

        # Test ALTER TABLE compatibility
        try:
            # Create a temp table to test
            conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS __shm_migrate_test__ "
                "(id STRING, PRIMARY KEY(id))"
            )
            conn.execute(
                "ALTER TABLE __shm_migrate_test__ ADD COLUMN test_col STRING"
            )
            conn.execute("DROP TABLE __shm_migrate_test__")
            alter_supported = True
        except RuntimeError as e:
            err = str(e).lower()
            alter_supported = not (
                "add column" in err and "not supported" in err
            ) and not (
                "invalid input" in err and "add column" in err
            ) and not (
                "parser exception" in err and "add column" in err
            )

        # Check if EpisodeNode has all expected columns
        try:
            r = conn.execute(
                "MATCH (e:EpisodeNode) RETURN e.* LIMIT 1"
            )
            existing_cols = r.get_column_names()
            # Strip 'e.' prefix from column names
            existing_cols = [c.split(".")[-1] if "." in c else c for c in existing_cols]

            has_all = all(col in existing_cols for col in EXPECTED_EPISODE_COLUMNS)
        except RuntimeError:
            has_all = True  # Table might not exist yet

        conn.close()
        db.close()

        if not alter_supported and not has_all:
            return False, (
                "Kuzu ALTER TABLE not supported AND schema mismatch. "
                "Migration required."
            )
        return True, "Compatible"
    except Exception as e:
        return False, f"Compatibility check failed: {e}"


def export_data() -> str:
    """导出旧 DB 数据到 JSON 文件。

    Returns:
        导出目录路径
    """
    import kuzu

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.mkdir(parents=True)

    db = kuzu.Database(str(DB_PATH), read_only=True)
    conn = kuzu.Connection(db)

    # Node PK definitions
    NODE_PK = {
        "EpisodeNode": "id", "HyperedgeNode": "id", "CommunityNode": "id",
        "SessionNode": "id", "VisualNode": "id", "OntologyType": "name",
        "OntologyEntity": "name",
    }

    # Export nodes
    for table in sorted(NODE_PK):
        r = conn.execute(f"MATCH (n:{table}) RETURN n.*")
        df = r.get_as_pl()
        path = EXPORT_DIR / f"{table}.json"
        with open(path, "w") as f:
            f.write(df.write_json())
        print(f"  Exported {table}: {df.shape[0]} rows")

    # Export rels
    rel_defs = [
        ("HEBBIAN_CONNECTION", "EpisodeNode", "id", "EpisodeNode", "id", ["weight"]),
        ("HYPEREDGE_MEMBER", "HyperedgeNode", "id", "EpisodeNode", "id", []),
        ("COMMUNITY_MEMBER", "CommunityNode", "id", "EpisodeNode", "id", []),
        ("SESSION_MEMBER", "SessionNode", "id", "EpisodeNode", "id", []),
        ("TEMPORAL_LINK", "EpisodeNode", "id", "EpisodeNode", "id", ["time_diff"]),
        ("VISUAL_HYPEREDGE_MEMBER", "HyperedgeNode", "id", "VisualNode", "id", []),
        ("IS_A", "OntologyEntity", "name", "OntologyType", "name", []),
        ("RELATES_TO", "OntologyEntity", "name", "OntologyEntity", "name", ["relation"]),
    ]

    for rel, src_t, src_pk, dst_t, dst_pk, props in rel_defs:
        extra = ", " + ", ".join([f"e.{p}" for p in props]) if props else ""
        query = (f"MATCH (a:{src_t})-[e:{rel}]->(b:{dst_t}) "
                 f"RETURN a.{src_pk} AS src_key, b.{dst_pk} AS dst_key{extra}")
        r = conn.execute(query)
        df = r.get_as_pl()
        if df.shape[0] > 0:
            path = EXPORT_DIR / f"{rel}.json"
            with open(path, "w") as f:
                f.write(df.write_json())
            print(f"  Exported {rel}: {df.shape[0]} rels")

    conn.close()
    db.close()
    return str(EXPORT_DIR)


def import_data() -> tuple[int, int]:
    """从 JSON 导入数据到新的 Kuzu DB。

    Returns:
        (node_count, edge_count)
    """
    import kuzu

    # Delete old DB so server can recreate with correct schema
    if DB_PATH.exists():
        DB_PATH.unlink()
    wal = DB_PATH.with_suffix(".db.wal")
    if wal.exists():
        wal.unlink()

    # Create fresh DB with schema (same as server's _init_schema)
    db = kuzu.Database(str(DB_PATH))
    conn = kuzu.Connection(db)

    node_defs = {
        "OntologyType":    ("name", ["category"]),
        "OntologyEntity":  ("name", ["type", "category"]),
        "EpisodeNode":     ("id", ["content", "embedding", "created_at",
                                   "tau_initial", "tau_value", "trust_score",
                                   "ontology_type", "source", "visibility"]),
        "HyperedgeNode":   ("id", ["type", "created_at", "gate_value", "metadata"]),
        "CommunityNode":   ("id", ["name", "summary", "leiden_score", "created_at"]),
        "SessionNode":     ("id", ["session_id", "created_at", "metadata"]),
        "VisualNode":      ("id", ["image_path", "caption", "embedding",
                                   "source", "created_at"]),
    }
    node_order = ["OntologyType", "OntologyEntity", "EpisodeNode",
                  "HyperedgeNode", "CommunityNode", "SessionNode", "VisualNode"]

    # Create tables first (same as _init_schema)
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS EpisodeNode ("
        "id STRING, content STRING, embedding FLOAT[384], "
        "created_at DOUBLE, tau_initial DOUBLE, tau_value DOUBLE, "
        "trust_score DOUBLE, ontology_type STRING, source STRING, "
        "visibility STRING, "
        "PRIMARY KEY (id))"
    )
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS HyperedgeNode ("
        "id STRING, type STRING, created_at DOUBLE, "
        "gate_value DOUBLE, metadata STRING, "
        "PRIMARY KEY (id))"
    )
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS CommunityNode ("
        "id STRING, name STRING, summary STRING, "
        "leiden_score DOUBLE, created_at DOUBLE, "
        "PRIMARY KEY (id))"
    )
    conn.execute(
        "CREATE REL TABLE IF NOT EXISTS HEBBIAN_CONNECTION "
        "(FROM EpisodeNode TO EpisodeNode, weight DOUBLE)"
    )
    conn.execute(
        "CREATE REL TABLE IF NOT EXISTS HYPEREDGE_MEMBER "
        "(FROM HyperedgeNode TO EpisodeNode)"
    )
    conn.execute(
        "CREATE REL TABLE IF NOT EXISTS COMMUNITY_MEMBER "
        "(FROM CommunityNode TO EpisodeNode)"
    )
    conn.execute(
        "CREATE REL TABLE IF NOT EXISTS TEMPORAL_LINK "
        "(FROM EpisodeNode TO EpisodeNode, time_diff DOUBLE)"
    )
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS SessionNode ("
        "id STRING, session_id STRING, created_at DOUBLE, "
        "metadata STRING, PRIMARY KEY (id))"
    )
    conn.execute(
        "CREATE REL TABLE IF NOT EXISTS SESSION_MEMBER "
        "(FROM SessionNode TO EpisodeNode)"
    )
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS VisualNode ("
        "id STRING, image_path STRING, caption STRING, "
        "embedding FLOAT[384], source STRING, created_at DOUBLE, "
        "PRIMARY KEY (id))"
    )
    conn.execute(
        "CREATE REL TABLE IF NOT EXISTS VISUAL_HYPEREDGE_MEMBER "
        "(FROM HyperedgeNode TO VisualNode)"
    )
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS OntologyType ("
        "name STRING, category STRING, PRIMARY KEY (name))"
    )
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS OntologyEntity ("
        "name STRING, type STRING, category STRING, "
        "PRIMARY KEY (name))"
    )
    conn.execute(
        "CREATE REL TABLE IF NOT EXISTS IS_A "
        "(FROM OntologyEntity TO OntologyType)"
    )
    conn.execute(
        "CREATE REL TABLE IF NOT EXISTS RELATES_TO "
        "(FROM OntologyEntity TO OntologyEntity, relation STRING)"
    )

    # Import nodes
    total_nodes = 0
    for table in node_order:
        path = EXPORT_DIR / f"{table}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        if not data:
            continue

        pk, cols = node_defs[table]
        set_clause = ", ".join([f"n.{c} = ${c}" for c in cols])
        query = f"MERGE (n:{table} {{{pk}: ${pk}}}) ON CREATE SET {set_clause}"

        for row in data:
            params = {k: row.get(k) for k in [f"n.{pk}"] + [f"n.{c}" for c in cols]}
            # Strip 'n.' prefix
            params = {k.split(".")[-1] if "." in k else k: v for k, v in params.items()}
            try:
                conn.execute(query, params)
            except RuntimeError as e:
                err = str(e).lower()
                if "already exists" not in err and "violate" not in err:
                    print(f"  WARN {table}: {e}")

        total_nodes += len(data)
        print(f"  Imported {table}: {len(data)} nodes")

    # Import rels
    rel_defs = [
        ("HEBBIAN_CONNECTION", "EpisodeNode", "id", "EpisodeNode", "id", ["weight"]),
        ("HYPEREDGE_MEMBER", "HyperedgeNode", "id", "EpisodeNode", "id", []),
        ("IS_A", "OntologyEntity", "name", "OntologyType", "name", []),
        ("RELATES_TO", "OntologyEntity", "name", "OntologyEntity", "name", ["relation"]),
    ]

    total_edges = 0
    for rel, src_t, src_pk, dst_t, dst_pk, props in rel_defs:
        path = EXPORT_DIR / f"{rel}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        if not data:
            continue

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

        total_edges += len(data)
        print(f"  Imported {rel}: {len(data)} rels")

    conn.close()
    db.close()
    return total_nodes, total_edges


def migrate(backup: bool = True) -> bool:
    """执行完整迁移流程。

    Args:
        backup: 是否在迁移前备份旧 DB

    Returns:
        True 表示迁移成功或无需迁移
    """
    compatible, msg = check_compatibility()
    if compatible:
        print(f"✓ {msg}")
        return True

    print(f"⚠  {msg}")
    print("Starting migration...")

    # Step 1: Backup
    if backup:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"shm_kuzu_db_{timestamp}"
        shutil.copy2(DB_PATH, backup_path)
        print(f"✓ Backed up to {backup_path}")

    # Step 2: Export
    print("Exporting data from old DB...")
    export_data()

    # Step 3: Import
    print("Importing data into new DB...")
    nodes, edges = import_data()
    print(f"✓ Migration complete: {nodes} nodes + {edges} edges")

    # Step 4: Cleanup export files
    shutil.rmtree(EXPORT_DIR)
    print("✓ Cleaned up export files")

    return True


def main():
    parser = argparse.ArgumentParser(description="SHM Kuzu DB Migration Tool")
    parser.add_argument("--check", action="store_true", help="Only check compatibility")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup step")
    args = parser.parse_args()

    if args.check:
        compatible, msg = check_compatibility()
        print(f"Compatible: {compatible}")
        print(f"Message: {msg}")
        return 0 if compatible else 1

    success = migrate(backup=not args.no_backup)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
