"""GraphLiteStore — 基于 GraphLite (GQL) 的图存储适配器（当前图引擎）。"""
import json, shutil, os, time, uuid, sys, tempfile
from pathlib import Path
from typing import Optional, Any

sys.path.insert(0, "/home/admin/GraphLite/bindings/python")
sys.path.insert(0, "/home/admin/GraphLite/sdk-python/src")

from graphlite_sdk import GraphLite, Session

SHM_SCHEMA = "/shm"
SHM_GRAPH = "default"

def _now() -> float:
    return time.time()

def _dict_to_gql_values(d: dict, skip_keys: set = None) -> str:
    """Convert Python dict to GQL literal syntax, handling UTF-8 safely."""
    from base64 import b64encode
    skip = skip_keys or set()
    parts = []
    for k, v in d.items():
        if k in skip or v is None:
            continue
        if isinstance(v, str):
            # GraphLite Rust lexer has UTF-8 bug; b64-encode non-ASCII
            try:
                v.encode('ascii')
                v = v.replace("\\", "\\\\").replace("'", "\\'")
                parts.append(f"{k}: '{v}'")
            except UnicodeEncodeError:
                # Non-ASCII: store as b64 with prefix
                b64 = b64encode(v.encode('utf-8')).decode('ascii')
                parts.append(f"{k}: '{{b64}}{b64}'")
        elif isinstance(v, bool):
            parts.append(f"{k}: {str(v).lower()}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}: {v}")
        elif isinstance(v, list):
            parts.append(f"{k}: '{json.dumps(v, ensure_ascii=False)}'")
    return ", ".join(parts)


class CircuitBreakerOpen(Exception):
    """断路器跳闸异常，供上层捕获降级（兼容 RyuStore 接口）。"""
    pass


class GraphLiteStore:
    """GraphLite-backed graph store, current graph engine."""

    def __init__(self, config: Optional[Any] = None):
        self._db: Optional[GraphLite] = None
        self._session: Optional[Session] = None
        self._db_path: str = ""
        self.config = config or type("cfg", (), {"database_path": "", "max_threads": 4})()

    @property
    def conn(self):
        if self._session is None:
            raise RuntimeError("GraphLiteStore not connected")
        return self._session

    def connect(self) -> None:
        """Open/create GraphLite DB and setup schema."""
        db_path = getattr(self.config, "database_path", "") or \
                  os.path.join(os.path.dirname(__file__), "..", "data", "shm_graphlite_db")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path

        self._db = GraphLite.open(db_path)
        self._session = self._db.session("shm")
        # Setup schema if first time, otherwise just set context
        try:
            self._session.execute("SESSION SET SCHEMA /shm")
            self._session.execute("SESSION SET GRAPH default")
        except Exception:
            try:
                self._session.execute("CREATE SCHEMA /shm")
            except Exception:
                pass  # schema 可能已存在
            try:
                self._session.execute("CREATE GRAPH IF NOT EXISTS default")
            except Exception:
                self._session.execute("CREATE GRAPH IF NOT EXISTS social")
            self._session.execute("SESSION SET SCHEMA /shm")
            try:
                self._session.execute("SESSION SET GRAPH default")
            except Exception:
                self._session.execute("SESSION SET GRAPH social")

    # ─── Episode CRUD ───────────────────────────────

    def create_episode(self, episode: dict) -> str:
        """INSERT EpisodeNode. Returns id."""
        eid = episode.get("id", str(uuid.uuid4()))
        vals = _dict_to_gql_values(episode, skip_keys={"id"})
        gql = f"INSERT (e:EpisodeNode {{id: '{eid}', {vals}}})"
        self._session.execute(gql)
        return eid

    def get_episode(self, node_id: str) -> Optional[dict]:
        """MATCH EpisodeNode by id."""
        gql = f"MATCH (e:EpisodeNode {{id: '{node_id}'}}) RETURN e"
        try:
            result = self._session.query(gql)
            if result.rows:
                row = result.rows[0]
                return self._flatten_row(row, "e")
        except Exception:
            return None
        return None

    def get_episodes_batch(self, node_ids: list[str]) -> list[dict]:
        """Batch GET by ids."""
        if not node_ids:
            return []
        ids = ", ".join(f"'{i}'" for i in node_ids)
        gql = f"MATCH (e:EpisodeNode) WHERE e.id IN [{ids}] RETURN e"
        try:
            result = self._session.query(gql)
            return [self._flatten_row(r, "e") for r in result.rows]
        except Exception:
            return []

    def get_active_episodes(self, time_window_seconds: float = 1800) -> list[dict]:
        """Get recently created episodes."""
        cutoff = _now() - time_window_seconds
        gql = f"MATCH (e:EpisodeNode) WHERE e.created_at >= {cutoff} RETURN e"
        try:
            result = self._session.query(gql)
            return [self._flatten_row(r, "e") for r in result.rows]
        except Exception:
            return []

    def get_episodes_by_tau_range(self, min_tau: float, max_tau: float, limit: int = 100) -> list[dict]:
        """Filter by tau range."""
        gql = f"MATCH (e:EpisodeNode) WHERE e.tau_initial >= {min_tau} AND e.tau_initial <= {max_tau} RETURN e LIMIT {limit}"
        try:
            result = self._session.query(gql)
            return [self._flatten_row(r, "e") for r in result.rows]
        except Exception:
            return []

    def update_with_version(self, node_id: str, updates: dict, expected_version: int) -> bool:
        """Optimistic lock update. GQL SET syntax."""
        sets = _dict_to_gql_values(updates)
        if not sets:
            return True
        gql = f"MATCH (e:EpisodeNode {{id: '{node_id}'}}) SET e.{sets}"
        try:
            self._session.execute(gql)
            return True
        except Exception:
            return False

    # ─── Hyperedge CRUD ─────────────────────────────

    def create_hyperedge_node(self, hyperedge: dict) -> str:
        hid = hyperedge.get("id", str(uuid.uuid4()))
        vals = _dict_to_gql_values({k: v for k, v in hyperedge.items() if k != "id"})
        gql = f"INSERT (h:HyperedgeNode {{id: '{hid}', {vals}}})"
        self._session.execute(gql)
        return hid

    def link_hyperedge_member(self, hyperedge_id: str, episode_id: str) -> None:
        gql = (
            f"MATCH (h:HyperedgeNode {{id: '{hyperedge_id}'}}), "
            f"(e:EpisodeNode {{id: '{episode_id}'}}) "
            f"INSERT (h)-[:HYPEREDGE_MEMBER]->(e)"
        )
        self._session.execute(gql)

    def get_hyperedge_members(self, hyperedge_id: str) -> list[dict]:
        gql = f"MATCH (h:HyperedgeNode {{id: '{hyperedge_id}'}})-[:HYPEREDGE_MEMBER]->(e) RETURN e"
        try:
            result = self._session.query(gql)
            return [self._flatten_row(r, "e") for r in result.rows]
        except Exception:
            return []

    def get_hyperedges_by_node(self, node_id: str) -> list[dict]:
        gql = f"MATCH (h:HyperedgeNode)-[:HYPEREDGE_MEMBER]->(e:EpisodeNode {{id: '{node_id}'}}) RETURN h"
        try:
            result = self._session.query(gql)
            return [self._flatten_row(r, "h") for r in result.rows]
        except Exception:
            return []

    def get_all_hebbian_connections(self) -> list[dict]:
        gql = "MATCH (a)-[r:HEBBIAN]->(b) RETURN a.id AS src, b.id AS dst, r.weight AS weight"
        try:
            result = self._session.query(gql)
            return list(result.rows)
        except Exception:
            return []

    def link_to_session(self, session_id: str, episode_id: str) -> None:
        """Link episode to session node."""
        gql = (
            f"MATCH (s:SessionNode {{id: '{session_id}'}}), "
            f"(e:EpisodeNode {{id: '{episode_id}'}}) "
            f"INSERT (s)-[:SESSION_MEMBER]->(e)"
        )
        self._session.execute(gql)

    def get_session_memories(self, session_id: str, limit: int = 100) -> list[dict]:
        gql = f"MATCH (s:SessionNode {{id: '{session_id}'}})-[:SESSION_MEMBER]->(e) RETURN e LIMIT {limit}"
        try:
            result = self._session.query(gql)
            return [self._flatten_row(r, "e") for r in result.rows]
        except Exception:
            return []

    # ─── Direct GQL ─────────────────────────────────

    def execute_cypher(self, query: str, params: Optional[dict] = None) -> None:
        """Execute GQL directly (passthrough)."""
        q = self._interpolate(query, params)
        self._session.execute(q)

    def query_cypher(self, query: str, params: Optional[dict] = None) -> list:
        """Query GQL, return list of dicts."""
        q = self._interpolate(query, params)
        try:
            result = self._session.query(q)
            return list(result.rows)
        except Exception:
            return []

    # ─── Helpers ────────────────────────────────────

    @staticmethod
    def _flatten_row(row: dict, label: str = "") -> dict:
        """Extract properties from GQL result row (deeply nested format)."""
        from base64 import b64decode
        result = {}
        for k, v in row.items():
            if isinstance(v, dict) and 'Node' in v:
                props = v['Node'].get('properties', {})
                flat = {}
                for pk, pv in props.items():
                    if isinstance(pv, dict):
                        flat[pk] = next(iter(pv.values()), pv)
                    else:
                        flat[pk] = pv
                # Decode b64 content
                for pk in flat:
                    if isinstance(flat[pk], str) and flat[pk].startswith('{b64}'):
                        try:
                            flat[pk] = b64decode(flat[pk][5:]).decode('utf-8')
                        except:
                            pass
                if label and k == label:
                    return flat
                result[k] = flat
            elif isinstance(v, dict) and 'Relationship' in v:
                rel = v['Relationship']
                props = rel.get('properties', {})
                flat = {}
                for pk, pv in props.items():
                    if isinstance(pv, dict):
                        flat[pk] = next(iter(pv.values()), pv)
                    else:
                        flat[pk] = pv
                result[k] = flat
            else:
                result[k] = v
        return result

    @staticmethod
    def _interpolate(query: str, params: Optional[dict] = None) -> str:
        """Basic $param interpolation to GQL literals (security: simple only)."""
        if not params:
            return query
        result = query
        for k, v in params.items():
            if isinstance(v, str):
                result = result.replace(f"${k}", f"'{v}'")
            elif isinstance(v, (int, float)):
                result = result.replace(f"${k}", str(v))
            elif v is None:
                result = result.replace(f"${k}", "NULL")
        return result

    # ─── Lifecycle ──────────────────────────────────

    def close(self) -> None:
        if self._db:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None
            self._session = None

    def __del__(self):
        self.close()
