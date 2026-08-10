"""GraphLiteStore — 基于 GraphLite (GQL) 的图存储适配器（当前图引擎）。"""
import json, shutil, os, time, threading, uuid, sys, tempfile
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Optional, Any

sys.path.insert(0, os.environ.get("GRAPHLITE_BINDINGS", os.path.expanduser("~/GraphLite/bindings/python")))
sys.path.insert(0, os.environ.get("GRAPHLITE_SDK", os.path.expanduser("~/GraphLite/sdk-python/src")))

from graphlite_sdk import GraphLite, Session
from graphlite_sdk.error import (
    ConnectionError as GraphLiteConnectionError,
    QueryError,
)

from core.retry import with_retry

SHM_SCHEMA = "/shm"
SHM_GRAPH = "default"

# 熔断器计数的「基础设施异常」集合（P0: 异常类型不匹配 → 熔断器死代码）:
# - SDK 层: GraphLiteConnectionError / QueryError 是 graphlite_sdk.error 自有的异常
#   （GraphLiteError 子类），与内置 ConnectionError 无继承关系。connection.py 的
#   query()/execute() 把所有底层异常统一包装成 QueryError —— 生产环境下连接失败/
#   超时只会以 QueryError 形式出现，因此必须显式纳入。
# - P2-1: 显式枚举，不纳入 GraphLiteError 基类——基类过宽，SerializationError/
#   NotFoundError 等数据/业务错误会被误计为基础设施故障。
# - 内置 ConnectionError/TimeoutError 保留以兼容测试 mock。
# 折中（SDK 未区分「连接失败」与「坏 GQL 语法」——均为 QueryError）: 两者都计数，
# 比 P0 前永不跳闸（死代码）好；代价是坏查询可能污染窗口（写路径由 P2-2 缓解）。
_INFRA_EXCEPTIONS = (
    GraphLiteConnectionError,
    QueryError,
    ConnectionError,
    TimeoutError,
)

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


class CircuitBreakerState(str, Enum):
    """断路器状态: closed → open → half_open。"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """滑动窗口失败率断路器状态机（closed → open → half_open）。

    - 默认参数与 config/settings.CircuitBreakerConfig / config/defaults.yaml 一致；
      不强制 import settings（避免循环依赖），config 缺失字段回落默认值。
    - record_success / record_failure 维护滑动窗口 _window（list[bool]，长度 ≤ window_size）。
    - 窗口满（window_size 条样本）后计算失败率，≥ failure_threshold → open。
      窗口不满时不跳闸，避免单次瞬时故障切断整个图存储。
    - open 后 recovery_timeout 秒自动迁移 half_open。
    - half_open 放行 half_open_max_requests 个探测请求，成功 → closed，失败 → open。
    - 跳闸（进入 open）时 raise CircuitBreakerOpen 供上层降级。
    """

    def __init__(self, config: Optional[Any] = None):
        cfg = config or type("cfg", (), {})()
        self.failure_threshold: float = float(getattr(cfg, "failure_threshold", 0.5))
        self.recovery_timeout: float = float(getattr(cfg, "recovery_timeout", 30.0))
        self.half_open_max_requests: int = int(getattr(cfg, "half_open_max_requests", 1))
        self.window_size: int = int(getattr(cfg, "window_size", 10))
        # 并发访问保护: GraphLiteStore 单例被事件循环 + to_thread + ThreadPool 共享
        self._lock = threading.RLock()
        # 滑动窗口: True=成功, False=失败（长度 ≤ window_size）
        self._window: list[bool] = []
        self._state = CircuitBreakerState.CLOSED
        self._opened_at: float = 0.0
        self._half_open_probes: int = 0
        self._half_open_last_probe_at: float = 0.0  # P1-2 探针时间戳（配额重新武装用）

    # ─── 状态机 ───────────────────────────────

    @property
    def state(self) -> CircuitBreakerState:
        """当前状态；open 后经过 recovery_timeout 自动迁移 half_open。"""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN and (
                time.time() - self._opened_at >= self.recovery_timeout
            ):
                self._state = CircuitBreakerState.HALF_OPEN
                self._half_open_probes = 0
                self._half_open_last_probe_at = 0.0
            return self._state

    def is_open(self) -> bool:
        """是否处于 open（拒绝请求）；half_open 放行探测请求。"""
        with self._lock:
            return self.state == CircuitBreakerState.OPEN

    def allow_request(self) -> bool:
        """请求门控: open → 拒绝；half_open → 放行 half_open_max_requests 个探测。

        P1-2: half_open 下探针配额按时间重新武装——距上次探测超过
        recovery_timeout / half_open_max_requests 即重置配额，防止探针被消耗
        但未触发 record_* 时永久卡在 half_open。
        """
        with self._lock:
            st = self.state
            if st == CircuitBreakerState.CLOSED:
                return True
            if st == CircuitBreakerState.HALF_OPEN:
                now = time.time()
                if self._half_open_probes >= self.half_open_max_requests:
                    interval = self.recovery_timeout / max(1, self.half_open_max_requests)
                    if now - self._half_open_last_probe_at >= interval:
                        self._half_open_probes = 0
                    else:
                        return False
                self._half_open_probes += 1
                self._half_open_last_probe_at = now
                return True
            return False

    # ─── 事件记录 ─────────────────────────────

    def record_success(self) -> None:
        """请求成功：写入窗口；half_open 探测成功 → 复位 closed。"""
        with self._lock:
            self._window.append(True)
            if len(self._window) > self.window_size:
                self._window.pop(0)
            if self.state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
                self._half_open_probes = 0
                self._half_open_last_probe_at = 0.0
                self._window = []

    def record_failure(self, exc: Optional[BaseException] = None) -> None:
        """请求失败：写入窗口；触发跳闸时 raise CircuitBreakerOpen（from exc 保留原始异常链）。

        只对基础设施错误计数（_INFRA_EXCEPTIONS: SDK QueryError/ConnectionError +
        内置 ConnectionError/TimeoutError）；应用错误（RuntimeError 等）不计数，
        避免坏查询反复调用 10 次后污染整个窗口导致全图熔断。
        exc=None 视为显式失败信号，计数。
        折中: SDK 把连接失败与坏 GQL 语法统一包装成 QueryError（无子类区分），
        两者都计数——比 P0 前（SDK QueryError 永远匹配不到内置类 → 熔断器永不
        跳闸的死代码）更好。
        """
        if exc is not None and not isinstance(exc, _INFRA_EXCEPTIONS):
            return
        with self._lock:
            self._window.append(False)
            if len(self._window) > self.window_size:
                self._window.pop(0)
            st = self.state  # 触发 open → half_open 自动迁移
            if st == CircuitBreakerState.HALF_OPEN:
                self._trip()
                raise CircuitBreakerOpen("half-open probe failed, circuit re-opened") from exc
            if st == CircuitBreakerState.CLOSED and len(self._window) == self.window_size \
                    and self._failure_rate() >= self.failure_threshold:
                self._trip()
                raise CircuitBreakerOpen(
                    f"failure rate {self._failure_rate():.0%} >= "
                    f"threshold {self.failure_threshold:.0%}, circuit opened"
                ) from exc

    # ─── Helpers ──────────────────────────────

    def _failure_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(1 for r in self._window if not r) / len(self._window)

    def _trip(self) -> None:
        self._state = CircuitBreakerState.OPEN
        self._opened_at = time.time()
        self._half_open_probes = 0
        self._half_open_last_probe_at = 0.0


class GraphLiteStore:
    """GraphLite-backed graph store, current graph engine."""

    def __init__(self, config: Optional[Any] = None, cb_config: Optional[Any] = None):
        self._db: Optional[GraphLite] = None
        self._session: Optional[Session] = None
        self._db_path: str = ""
        self.config = config or type("cfg", (), {"database_path": "", "max_threads": 4})()
        self.circuit_breaker = CircuitBreaker(cb_config)

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

    @with_retry(
        max_attempts=2, base_delay=0.2, backoff=2.0,
        retryable_exceptions=_INFRA_EXCEPTIONS,
    )
    def _get_episodes_batch_retryable(self, node_ids: list[str]) -> list[dict]:
        """底层批量查询（熔断门控 + 重试）：成功返回 episodes。

        - open 状态 raise CircuitBreakerOpen（query_router L1 超图检索的
          传播链入口——L613 级联 L2；L2 向量检索静默降级）
        - 基础设施错误（_INFRA_EXCEPTIONS）直接抛出交给 with_retry 重试；
          失败计数由 get_episodes_batch 在重试耗尽后统一记录一次（P2-D）
        - 应用错误不计数、不重试，返回 []
        """
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerOpen("circuit breaker open, batch lookup rejected")
        ids = ", ".join(f"'{i}'" for i in node_ids)
        gql = f"MATCH (e:EpisodeNode) WHERE e.id IN [{ids}] RETURN e"
        try:
            result = self._session.query(gql)
        except _INFRA_EXCEPTIONS:
            raise  # 交给 with_retry 重试；失败计数由 get_episodes_batch 重试耗尽后统一记录
        except Exception:
            return []  # 应用错误不计数、不重试
        self.circuit_breaker.record_success()
        return [self._flatten_row(r, "e") for r in result.rows]

    def get_episodes_batch(self, node_ids: list[str]) -> list[dict]:
        """Batch GET by ids.

        P0-1: 熔断门控——open 状态 raise CircuitBreakerOpen（query_router
        L1 超图检索的传播链入口）；基础设施错误由 _get_episodes_batch_retryable
        重试（最多 2 次），重试耗尽后统一计 1 次失败（窗口按查询结果计数，
        与 query_cypher 一致，P2-D）；跳闸时抛 CircuitBreakerOpen 供上层级联，
        否则返回 []。应用错误不计数并返回 []。
        """
        if not node_ids:
            return []
        try:
            return self._get_episodes_batch_retryable(node_ids)
        except _INFRA_EXCEPTIONS as e:
            try:
                self.circuit_breaker.record_failure(e)  # 重试耗尽 → 统一计失败
            except CircuitBreakerOpen:
                raise CircuitBreakerOpen(
                    "circuit breaker open, batch lookup rejected"
                ) from e
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

    def ensure_session(self, session_id: str) -> None:
        """确保 SessionNode 存在（不存在则创建），供 link_to_session 前置调用。

        注意: GraphLite 是 schemaless 图库，无 MERGE（Kuzu 语法）也无主键冲突，
        重复 INSERT 会创建重复节点 —— 必须用 查询-插入 两段式保证幂等。
        """
        try:
            result = self._session.query(
                f"MATCH (s:SessionNode {{id: '{session_id}'}}) RETURN s.id"
            )
            if result.rows:
                return  # 已存在
        except Exception:
            pass  # 查询失败（如表不存在）时走创建路径
        self._session.execute(
            f"INSERT (s:SessionNode {{id: '{session_id}', "
            f"created_at: {int(time.time())}, last_seen: {int(time.time())}}})"
        )

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

    # ─── Session/Visual CRUD（P0-2 幽灵方法实现）───────────────

    def get_or_create_session(self, session_id: str, metadata: Optional[str] = None) -> str:
        """获取或创建 SessionNode，返回 session_id（两段式幂等，参照 ensure_session）。"""
        self.ensure_session(session_id)
        if metadata:
            self._session.execute(
                f"MATCH (s:SessionNode {{id: '{session_id}'}}) "
                f"SET s.metadata = '{metadata}'"
            )
        return session_id

    def link_session_member(self, session_node_id: str, episode_id: str) -> None:
        """Link episode to session node（参照 link_to_session 的 MATCH + INSERT）。"""
        gql = (
            f"MATCH (s:SessionNode {{id: '{session_node_id}'}}), "
            f"(e:EpisodeNode {{id: '{episode_id}'}}) "
            f"INSERT (s)-[:SESSION_MEMBER]->(e)"
        )
        self._session.execute(gql)

    def create_visual_node(self, node: dict) -> str:
        """INSERT VisualNode。id 为必填；embedding 是 list，_gql_value 自动 b64 序列化。"""
        vid = node.get("id", str(uuid.uuid4()))
        vals = _dict_to_gql_values(node, skip_keys={"id"})
        self._session.execute(f"INSERT (v:VisualNode {{id: '{vid}', {vals}}})")
        return vid

    def get_visual_node(self, visual_id: str) -> Optional[dict]:
        """MATCH VisualNode by id（参照 get_episode）。"""
        gql = f"MATCH (v:VisualNode {{id: '{visual_id}'}}) RETURN v"
        try:
            result = self._session.query(gql)
            if result.rows:
                return self._flatten_row(result.rows[0], "v")
        except Exception:
            return None
        return None

    def get_visual_nodes(self, limit: int = 50) -> list[dict]:
        """列出 VisualNode（flatten 后含 b64 解码的 caption）。"""
        gql = f"MATCH (v:VisualNode) RETURN v LIMIT {limit}"
        try:
            result = self._session.query(gql)
            return [self._flatten_row(r, "v") for r in result.rows]
        except Exception:
            return []

    def delete_namespace(self, namespace: str) -> int:
        """按命名空间删除：删除 SessionNode 及其 SESSION_MEMBER 关联的 EpisodeNode。

        返回删除的 EpisodeNode 数。节点有关联关系必须 DETACH DELETE(skill 记过的坑)。
        """
        # 1. 找到该 namespace(SessionNode)下的所有 EpisodeNode
        try:
            result = self._session.query(
                f"MATCH (s:SessionNode {{id: '{namespace}'}})-[:SESSION_MEMBER]->(e:EpisodeNode) "
                f"RETURN e"
            )
            ep_ids = [self._flatten_row(r, "e").get("id", "") for r in result.rows]
            ep_ids = [i for i in ep_ids if i]
        except Exception:
            return 0

        # 2. 逐个 DETACH DELETE EpisodeNode(处理 Hebbian/超边等关联)
        deleted = 0
        for eid in ep_ids:
            try:
                self._session.execute(
                    f"MATCH (e:EpisodeNode {{id: '{eid}'}}) DETACH DELETE e"
                )
                deleted += 1
            except Exception:
                pass

        # 3. 删除 SessionNode 本身(及其残留关系)
        try:
            self._session.execute(
                f"MATCH (s:SessionNode {{id: '{namespace}'}}) DETACH DELETE s"
            )
        except Exception:
            pass
        return deleted

    # ─── Direct GQL ─────────────────────────────────

    def execute_cypher(self, query: str, params: Optional[dict] = None) -> list:
        """Execute GQL directly, return list of row dicts (MATCH/DML results).

        熔断门控 + 不吞异常：
        - open 状态 raise CircuitBreakerOpen（写路径需显式失败）
        - 不加 @with_retry —— 写操作（INSERT/CREATE/SET）不自动重试，
          避免非幂等双重执行；读路径 query_cypher 保留重试。
        - P2-2: 写路径对熔断窗口完全中立（既不 record_success 也不
          record_failure）——只参与 allow_request 门控；坏 GQL/连接失败不污染
          窗口，写流量也不稀释读失败率（否则写流量 ≥ 2× 读失败时熔断永不
          跳闸）；熔断完全由读路径驱动，写失败原样上抛，由调用方处理。
        """
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerOpen("circuit breaker open, query rejected")
        q = self._interpolate(query, params)
        try:
            result = self._session.query(q)
        except Exception:
            raise
        return list(result.rows)

    @with_retry(
        max_attempts=2, base_delay=0.2, backoff=2.0,
        retryable_exceptions=_INFRA_EXCEPTIONS,
    )
    def _query_retryable(self, query: str, params: Optional[dict] = None) -> list:
        """底层查询（熔断门控 + 重试）：成功返回 rows。

        - open 状态返回 []（不抛，由 query_cypher 保持永不抛异常契约）
        - 基础设施错误（_INFRA_EXCEPTIONS: SDK QueryError 等 + 内置类）直接抛出，
          交给 with_retry 重试；失败计数由 query_cypher 在重试耗尽后统一记录一次
          ——窗口按「查询结果」计数而非「attempt」（P2-B: 重试成功 F→T 的查询
          不污染窗口，否则 5 个需重试的查询后窗口 50% 可能误跳闸）
        - 应用错误（GQL 语法等）不计数、不重试，返回 []
        - P3-1: 成功路径（record_success + rows 构造）在 try 内——畸形数据
          （如 result 无 rows）不逃出永不抛异常契约，按应用错误防御性返回 []
        """
        if not self.circuit_breaker.allow_request():
            return []
        q = self._interpolate(query, params)
        try:
            result = self._session.query(q)
            self.circuit_breaker.record_success()
            return list(result.rows)
        except _INFRA_EXCEPTIONS:
            raise  # 交给 with_retry 重试；失败计数由 query_cypher 重试耗尽后统一记录
        except Exception:
            return []  # 应用错误不计数、不重试

    def query_cypher(self, query: str, params: Optional[dict] = None) -> list:
        """Query GQL, return list of dicts. 永不抛异常契约（P0-2 关键设计决策）。

        - open 状态返回 []（静默降级；query_router 通过显式 is_open() 检查级联，
          而非异常传播）——~25 个调用方无需处理 CircuitBreakerOpen
        - 基础设施错误（_INFRA_EXCEPTIONS: SDK QueryError 等 + 内置类）由
          _query_retryable 重试（最多 2 次，base_delay=0.2s，避免同步重试冻结
          async 事件循环）；重试耗尽后统一计 1 次失败（窗口按查询结果计数，
          P2-B）→ 返回 []；跳闸（record_failure raise CircuitBreakerOpen）时
          静默返回 []，保持永不抛异常契约
        - 应用错误（GQL 语法等）不计数、不重试，返回 []
        """
        try:
            return self._query_retryable(query, params)
        except _INFRA_EXCEPTIONS as e:
            try:
                self.circuit_breaker.record_failure(e)  # 重试耗尽 → 统一计失败
            except CircuitBreakerOpen:
                pass  # 跳闸 → 静默返回 []（永不抛异常契约）
            return []  # 重试耗尽 → 静默降级

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
            elif isinstance(v, (np.integer, np.floating)):
                # numpy 标量（如 FAISS 搜索返回的 np.float32）不是 int/float 实例，
                # 直接 str() 会带类型前缀；统一转 Python 标量
                result = result.replace(f"${k}", str(v.item()))
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
