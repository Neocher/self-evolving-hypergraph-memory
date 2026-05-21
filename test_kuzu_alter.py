"""Test Kuzu 0.11 ALTER TABLE ADD COLUMN behavior"""
import kuzu
import os
import tempfile
import shutil

tmpdir = tempfile.mkdtemp()
db_path = os.path.join(tmpdir, "test_db")
db = kuzu.Database(db_path)
conn = kuzu.Connection(db)

# Create table
conn.execute("CREATE NODE TABLE TestTable (id STRING, name STRING, PRIMARY KEY (id))")

# Try ALTER TABLE ADD COLUMN
try:
    conn.execute("ALTER TABLE TestTable ADD COLUMN ontology_type STRING")
    print("ALTER TABLE ADD COLUMN: WORKS")
except Exception as e:
    print(f"ALTER TABLE ADD COLUMN: FAILED - {e}")

# Check if we can query a non-existent property
conn.execute("CREATE (t:TestTable {id: '1', name: 'test'})")
try:
    r = conn.execute("MATCH (t:TestTable) WHERE t.ontology_type = 'foo' RETURN t.*")
    print(f"Query non-existent property: returned {r.get_as_pl().to_dicts()}")
except Exception as e:
    print(f"Query non-existent property: FAILED - {e}")

shutil.rmtree(tmpdir)
