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

def _gql_value(v: Any) -> Optional[str]:
    """Encode a single Python value to GQL literal (UTF-8-safe). None if unsupported."""
    from base64 import b64encode
    if isinstance(v, str):
        # GraphLite Rust lexer has UTF-8 bug; b64-encode non-ASCII
        try:
            v.encode('ascii')
            v = v.replace("\\", "\\\\").replace("'", "\\'")
            return f"'{v}'"
        except UnicodeEncodeError:
            # Non-ASCII: store as b64 with prefix
            b64 = b64encode(v.encode('utf-8')).decode('ascii')
            return f"'{{b64}}{b64}'"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        # JSON 序列化后统一 b64 (GraphLite lexer UTF-8 bug: 中文 list 直插 PANIC)
        b64 = b64encode(json.dumps(v, ensure_ascii=False).encode('utf-8')).decode('ascii')
        return f"'{chr(123)}b64{chr(125)}{b64}'"
    return None


def _dict_to_gql_values(d: dict, skip_keys: set = None) -> str:
    """Convert Python dict to GQL literal syntax, handling UTF-8 safely."""
    skip = skip_keys or set()
    parts = []
    for k, v in d.items():
        if k in skip or v is None:
            continue
        lit = _gql_value(v)
        if lit is not None:
            parts.append(f"{k}: {lit}")
    return ", ".join(parts)


def _dict_to_gql_set_values(d: dict, skip_keys: set = None) -> str:
    """Convert Python dict to GQL SET clause (e.key = value, ...).

    逐字段直接构建 (不复用 split), 值含 ', ' (如 content="a, b") 不会拆坏 SQL。
    """
    skip = skip_keys or set()
    parts = []
    for k, v in d.items():
        if k in skip or v is None:
            continue
        lit = _gql_value(v)
        if lit is not None:
            parts.append(f"e.{k} = {lit}")
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
        # GraphLite 本版要求 graph 名称带 / 前缀（如 /shm），但旧库用 default（无斜杠）
        # 双名兼容：先试 default（兼容现有生产库），再试 /shm（新格式）
        self._graph_name: str = ""
        # 双名探测：default（旧生产库）→ /shm（新格式）。
        # 注意：探测在同一 try 内先 SET SCHEMA 再 SET GRAPH——若 schema 不存在
        # 则 default 候选也被跳过（落入创建路径）。当前生产库 schema=/shm，
        # 该前提成立；非 /shm schema 的 legacy 库会新建空 /shm graph。
        for candidate in (SHM_GRAPH, SHM_SCHEMA):
            try:
                self._session.execute(f"SESSION SET SCHEMA {SHM_SCHEMA}")
                self._session.execute(f"SESSION SET GRAPH {candidate}")
                self._graph_name = candidate
                break
            except Exception:
                continue
        if not self._graph_name:
            # 全新库：按序创建 schema → set schema → create graph → set graph
            # （CREATE GRAPH 前必须先 SESSION SET SCHEMA，顺序颠倒会失败）
            try:
                self._session.execute(f"CREATE SCHEMA {SHM_SCHEMA}")
            except Exception:
                pass  # schema 可能已存在
            self._session.execute(f"SESSION SET SCHEMA {SHM_SCHEMA}")
            try:
                self._session.execute(f"CREATE GRAPH {SHM_SCHEMA}")
            except Exception:
                pass  # graph 可能已存在
            self._session.execute(f"SESSION SET GRAPH {SHM_SCHEMA}")
            self._graph_name = SHM_SCHEMA

    # ─── Episode CRUD ───────────────────────────────

    def create_episode(self, episode: dict) -> str:
        """INSERT EpisodeNode. Returns id.

        注意: GraphLite INSERT 不能直接带 version 字段 (会 QUERY_ERROR)，
        必须先 INSERT 再 SET version。
        """
        eid = episode.get("id", str(uuid.uuid4()))
        vals = _dict_to_gql_values(episode, skip_keys={"id", "version"})
        gql = f"INSERT (e:EpisodeNode {{id: '{eid}', {vals}}})"
        self._session.execute(gql)
        # 乐观锁基线: 无 version 时置 1 (INSERT 带 version 会 QUERY_ERROR, 故后置 SET)
        ver = episode.get("version", 1)
        self._session.execute(
            f"MATCH (e:EpisodeNode {{id: '{eid}'}}) SET e.version = {int(ver)}"
        )
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
        """Optimistic lock update (两步法: 查 version → 匹配 SET + version 递增).

        expected_version=None 时跳过版本检查 (force 写入, 不递增 version —
        语义选择: force 后旧 expected_version 仍可通过校验, 调用方需自行保证时序)。
        节点不存在 / version 不匹配 / 旧数据无 version → False。
        """
        set_clause = _dict_to_gql_set_values(updates, skip_keys={"id", "version"})
        if not set_clause:
            return True
        # Step 1: 读当前 version
        try:
            result = self._session.query(
                f"MATCH (e:EpisodeNode {{id: '{node_id}'}}) RETURN e.version AS v"
            )
        except Exception:
            return False
        if not result.rows:
            return False  # 节点不存在
        v = result.rows[0].get("v") if isinstance(result.rows[0], dict) else None
        next_version = None
        if expected_version is not None:
            if v is None:
                return False  # 旧数据无 version 字段
            try:
                if int(v) != int(expected_version):
                    return False
                next_version = int(expected_version) + 1
            except (TypeError, ValueError):
                return False
        # Step 2: 版本匹配 (或跳过检查) → SET 更新 + version 递增
        if next_version is not None:
            set_clause = f"{set_clause}, e.version = {next_version}"
        try:
            self._session.execute(
                f"MATCH (e:EpisodeNode {{id: '{node_id}'}}) SET {set_clause}"
            )
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

    def get_all_connections(self) -> dict[str, dict[str, float]]:
        """全部 Hebbian 连接，格式 {src_id: {dst_id: weight}}（供 Hebbian 更新器使用）。"""
        conns: dict[str, dict[str, float]] = {}
        try:
            for row in self.get_all_hebbian_connections():
                src = row.get("src") or row.get("a.id")
                dst = row.get("dst") or row.get("b.id")
                if not src or not dst:
                    continue
                conns.setdefault(str(src), {})[str(dst)] = float(
                    row.get("weight") or row.get("r.weight") or 0.0
                )
        except Exception:
            pass  # 连接查询失败时返回空字典（与 get_all_hebbian_connections 一致）
        return conns

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

    def execute_cypher(self, query: str, params: Optional[dict] = None) -> list:
        """Execute GQL directly, return list of row dicts (MATCH/DML results)."""
        q = self._interpolate(query, params)
        result = self._session.query(q)
        return list(result.rows)

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
        from base64 import b64encode
        if not params:
            return query
        result = query
        for k, v in params.items():
            if isinstance(v, str):
                if not v:
                    # 空串：GraphLite 中 CONTAINS '' 恒真 → NOT CONTAINS '' 恒假，
                    # read_validate 的 $new_value 为空会导致矛盾漏检。
                    # 用哨兵值使 NOT CONTAINS 恒真（语义 = 不排除已有事实）。
                    result = result.replace(f"${k}", "'__SHM_NO_VALUE__'")
                else:
                    try:
                        v.encode('ascii')
                        result = result.replace(f"${k}", f"'{v}'")
                    except UnicodeEncodeError:
                        # GraphLite Rust lexer has UTF-8 bug; b64-encode non-ASCII
                        b64 = b64encode(v.encode('utf-8')).decode('ascii')
                        result = result.replace(f"${k}", f"'{{b64}}{b64}'")
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
