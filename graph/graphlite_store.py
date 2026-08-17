"""GraphLiteStore — 基于 GraphLite (GQL) 的图存储适配器（当前图引擎）。"""
import json, shutil, os, time, threading, uuid, sys, tempfile, logging
from collections import OrderedDict
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Optional, Any

# 【L2】硬编码 home 路径降级为回退：优先直接 import（SDK 已在标准路径/已装包）；
# 仅 ImportError 时才把 GRAPHLITE_BINDINGS / GRAPHLITE_SDK（含默认 ~/GraphLite 回退）
# 插入 sys.path——本机 SDK 只经这两个路径可见，默认路径必须保留为回退，不能删；
# 且 os.path.isdir 存在才插（避免污染 sys.path）。
try:
    from graphlite_sdk import GraphLite, Session
    from graphlite_sdk.error import (
        ConnectionError as GraphLiteConnectionError,
        QueryError,
    )
except ImportError:
    for env_key, default in (
        ("GRAPHLITE_BINDINGS", "~/GraphLite/bindings/python"),
        ("GRAPHLITE_SDK", "~/GraphLite/sdk-python/src"),
    ):
        p = os.path.expanduser(os.environ.get(env_key, default))
        if os.path.isdir(p):
            sys.path.insert(0, p)
    from graphlite_sdk import GraphLite, Session
    from graphlite_sdk.error import (
        ConnectionError as GraphLiteConnectionError,
        QueryError,
    )

logger = logging.getLogger("shm.graphlite_store")

from core.retry import with_retry

SHM_SCHEMA = "/shm"
SHM_GRAPH = "default"

# 【P0-1 实体-属性-时间】每 (entity_id, attr_name) 属性版本上限（决策 5：
# 写时惰性裁剪，超限 DETACH DELETE 最旧；不复用 tau 衰减）
PROPERTY_MAX_VERSIONS = 8

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


def _backup_corrupt_db(db_path: str) -> None:
    """open 失败时自动备份损坏库，保留崩溃现场供恢复；备份失败仅日志，不吞原始异常。"""
    try:
        if not os.path.isdir(db_path):
            logger.error("GraphLite open failed; DB path absent, nothing to back up: %s", db_path)
            return
        backup_path = f"{db_path}.corrupt.{time.strftime('%Y%m%d_%H%M%S_%f')}"
        shutil.copytree(db_path, backup_path)
        logger.error("GraphLite open failed; corrupted DB backed up: %s", backup_path)
    except Exception:
        logger.exception("Corrupt DB backup failed (original error preserved)")


def _gql_value(v: Any) -> Optional[str]:
    """Encode a single Python value to GQL literal (UTF-8-safe). None if unsupported."""
    if isinstance(v, str):
        # GraphLite lexer UTF-8 bug 已修复（fork Neocher/GraphLite 4452a96）——原生中文直写
        v = v.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{v}'"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        # JSON 序列化直写（引擎支持 UTF-8）；先转义反斜杠再转义单引号
        json_str = json.dumps(v, ensure_ascii=False)
        json_str = json_str.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{json_str}'"
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


class EpisodeCache:
    """检索侧 episode 内容缓存（OrderedDict LRU + TTL）。

    【M5】原 Services._episode_cache 是裸 dict 且无任何写入方（死缓存）。
    加界壳：maxsize 封顶（默认 4096，超限逐出最旧未访问项）、ttl 过期
    （默认 600s，过期项读取时惰性剔除）。读写均 O(1)，无锁——调用侧
    已通过 FAISS 批量 flush 单线程写入、检索线程只读，与 faiss_id_map
    相同的共享引用语义。
    """

    def __init__(self, maxsize: int = 4096, ttl: float = 600.0) -> None:
        self.maxsize = int(maxsize)
        self.ttl = float(ttl)
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    def _expired(self, entry: tuple[float, dict], now: float) -> bool:
        ts, _val = entry
        return now - ts > self.ttl

    def __getitem__(self, key: str) -> dict:
        entry = self._data[key]
        if self._expired(entry, time.time()):
            del self._data[key]
            raise KeyError(key)
        self._data.move_to_end(key)
        return entry[1]

    def __contains__(self, key: str) -> bool:
        return key in self._data and not self._expired(self._data[key], time.time())

    def get(self, key: str, default: Optional[dict] = None) -> Optional[dict]:
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: str, value: dict) -> None:
        self._data[key] = (time.time(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)


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
        # 【F5】session 访问锁：GraphLite session 无并发防护（跨线程并发访问会引擎级
        # 挂起），所有 _session.query/execute 统一经 _locked_query/_locked_execute 串行化。
        # RLock 可重入——写线程内嵌套调用不死锁。
        self._session_lock = threading.RLock()

    @property
    def conn(self):
        if self._session is None:
            raise RuntimeError("GraphLiteStore not connected")
        return self._session

    def _locked_query(self, gql: str):
        with self._session_lock:
            return self._session.query(gql)

    def _locked_execute(self, gql: str):
        with self._session_lock:
            return self._session.execute(gql)

    def connect(self) -> None:
        """Open/create GraphLite DB and setup schema."""
        db_path = getattr(self.config, "database_path", "") or \
                  os.path.join(os.path.dirname(__file__), "..", "data", "shm_graphlite_db")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path

        try:
            self._db = GraphLite.open(db_path)
        except Exception:
            # open 失败（DATABASE_OPEN_ERROR——kill -9/OOM 使 Sled 库损坏）时先备份损坏库再 re-raise
            _backup_corrupt_db(db_path)
            raise
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
                self._locked_execute(f"SESSION SET SCHEMA {SHM_SCHEMA}")
                self._locked_execute(f"SESSION SET GRAPH {candidate}")
                self._graph_name = candidate
                break
            except Exception:
                continue
        if not self._graph_name:
            # 全新库：按序创建 schema → set schema → create graph → set graph
            # （CREATE GRAPH 前必须先 SESSION SET SCHEMA，顺序颠倒会失败）
            try:
                self._locked_execute(f"CREATE SCHEMA {SHM_SCHEMA}")
            except Exception:
                pass  # schema 可能已存在
            self._locked_execute(f"SESSION SET SCHEMA {SHM_SCHEMA}")
            try:
                self._locked_execute(f"CREATE GRAPH {SHM_SCHEMA}")
            except Exception:
                pass  # graph 可能已存在
            self._locked_execute(f"SESSION SET GRAPH {SHM_SCHEMA}")
            self._graph_name = SHM_SCHEMA

        # 【P1-2】EpisodeNode (source, created_at) 复合索引 —— 超边窗口查询
        # (MATCH e WHERE e.source = $src AND e.created_at >= $cutoff) 从全表
        # 扫描降为索引扫描, 写入延迟不随节点数增长。GraphLite 支持
        # CREATE INDEX ... ON <node> (col, ...) 语法 (parser + executor 均有),
        # 已实测不抛异常且查询结果一致。尽力而为: 失败仅日志, 不影响启动
        # (性能兜底: P1-1 已把每写 2 次 MATCH 合并为每 source 1 次)。
        self._ensure_episode_index()

    def _ensure_episode_index(self) -> None:
        """P1-2: 尽力创建 (source, created_at) 复合索引, 失败仅日志。"""
        try:
            self._locked_execute(
                "CREATE INDEX IF NOT EXISTS idx_episode_src_ts "
                "ON EpisodeNode (source, created_at)"
            )
            logger.debug("EpisodeNode (source, created_at) index ensured")
        except Exception as e:
            logger.warning("EpisodeNode index creation skipped (non-fatal): %s", e)

    # ─── Episode CRUD ───────────────────────────────

    def create_episode(self, episode: dict) -> str:
        """INSERT EpisodeNode. Returns id.

        注意: GraphLite INSERT 不能直接带 version 字段 (会 QUERY_ERROR)，
        必须先 INSERT 再 SET version。
        """
        # 【Archive-Supersedes】写时基线：新节点默认 archived=false，保证检索过滤一致性
        # （一处覆盖 4 个写入点：write.py create_episode / write_sensory 兜底 /
        # write_multimodal / promote_to_episode）。
        episode.setdefault("archived", False)
        # 【Source-Trust】写时基线：默认 direct 向后兼容；防洗白降级（agent 直述
        # direct→inferred）由调用方 resolve_source_type 完成，此处仅兜底默认。
        episode.setdefault("source_type", "direct")
        # 【Dual-Track】写时基线：默认 active（保守）；core 由调用方分类后显式写入
        episode.setdefault("fact_track", "active")
        eid = episode.get("id", str(uuid.uuid4()))
        vals = _dict_to_gql_values(episode, skip_keys={"id", "version"})
        # 【H1】id 经 _gql_value 转义（含 ' / \ 的 id 不再裸插注入 GQL）
        id_lit = _gql_value(str(eid))
        gql = f"INSERT (e:EpisodeNode {{id: {id_lit}, {vals}}})"
        self._locked_execute(gql)
        # 乐观锁基线: 无 version 时置 1 (INSERT 带 version 会 QUERY_ERROR, 故后置 SET)
        ver = episode.get("version", 1)
        self._locked_execute(
            f"MATCH (e:EpisodeNode {{id: {id_lit}}}) SET e.version = {int(ver)}"
        )
        return eid

    def get_episode(self, node_id: str) -> Optional[dict]:
        """MATCH EpisodeNode by id."""
        # 【H1】id 经 _gql_value 转义（外部可达：GET /memories/episodes/{id}）
        gql = f"MATCH (e:EpisodeNode {{id: {_gql_value(str(node_id))}}}) RETURN e"
        try:
            result = self._locked_query(gql)
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
        # 【H1】批量 id 逐个 _gql_value 转义（含 ' / \ 的 id 不再裸插注入 GQL）
        ids = ", ".join(_gql_value(str(i)) for i in node_ids)
        gql = f"MATCH (e:EpisodeNode) WHERE e.id IN [{ids}] RETURN e"
        try:
            result = self._locked_query(gql)
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
            result = self._locked_query(gql)
            return [self._flatten_row(r, "e") for r in result.rows]
        except Exception:
            return []

    def get_episodes_by_tau_range(self, min_tau: float, max_tau: float, limit: int = 100) -> list[dict]:
        """Filter by tau range."""
        gql = f"MATCH (e:EpisodeNode) WHERE e.tau_initial >= {min_tau} AND e.tau_initial <= {max_tau} RETURN e LIMIT {limit}"
        try:
            result = self._locked_query(gql)
            return [self._flatten_row(r, "e") for r in result.rows]
        except Exception:
            return []

    def update_with_version(self, node_id: str, updates: dict, expected_version: int) -> bool:
        """Optimistic lock update (两步法: 查 version → 匹配 SET + version 递增).

        expected_version=None 时跳过版本检查 (force 写入, 不递增 version —
        语义选择: force 后旧 expected_version 仍可通过校验, 调用方需自行保证时序)。
        节点不存在 / version 不匹配 / 旧数据无 version → False。

        【M2】读 version + SET 包进同一个 `with self._session_lock:`（直接调
        self._session.query/execute，不再分两次 _locked_query/_locked_execute）。
        RLock 可重入，写线程内安全；锁本来就是全局唯一串行点，无新并发结构——
        消除"读后、SET 前"另一写线程抢先更新导致的乐观锁漏检窗口。
        """
        # 【H1】node_id 经 _gql_value 转义
        id_lit = _gql_value(str(node_id))
        set_clause = _dict_to_gql_set_values(updates, skip_keys={"id", "version"})
        if not set_clause:
            return True
        with self._session_lock:
            assert self._session is not None  # connect() 后必有
            if expected_version is None:
                # force 写入（跳过版本检查、不递增 version——保持 v5.31.0 语义）
                try:
                    self._session.execute(
                        f"MATCH (e:EpisodeNode {{id: {id_lit}}}) SET {set_clause}"
                    )
                    return True
                except Exception:
                    return False
            # CAS 单条 GQL（v5.31.4+ 引擎 rows_affected 检测）：
            # MATCH WHERE version 条件 + SET 单查询原子执行（SHM 单写线程 +
            # Sled 单写锁保证串行）——无需 BEGIN/COMMIT 包裹（多语句返回
            # 最后一条 COMMIT 的 rows_affected=0 会误判失败）。
            # rows_affected > 0 = 更新成功; 0 = 节点不存在/版本不匹配/旧数据无 version
            nxt = int(expected_version) + 1
            gql = (
                f"MATCH (e:EpisodeNode {{id: {id_lit}}}) "
                f"WHERE e.version = {int(expected_version)} "
                f"SET e.version = {nxt}, {set_clause}"
            )
            try:
                result = self._session.execute(gql)
                return result is not None and result > 0
            except Exception:
                return False

    def archive_node(self, node_id: str, replacement_id: Optional[str] = None) -> bool:
        """标记 EpisodeNode 为归档（archived=true）；replacement_id 非空时建 SUPERSEDES 血统边。

        GraphLite 无 MERGE：supersedes 边用双 MATCH + INSERT（参考
        link_hyperedge_member 的逗号分隔 MATCH 模式）。节点存在性用
        session.execute 的 rows_affected 判定（MATCH 无匹配时 execute 返回 0，
        与 update_with_version 的 CAS 检测一致；query_cypher 对 SET 恒返回
        status 行，无法区分存在性）。
        """
        id_lit = _gql_value(str(node_id))
        try:
            affected = self._locked_execute(
                f"MATCH (e:EpisodeNode {{id: {id_lit}}}) SET e.archived = true"
            )
        except Exception:
            logger.warning("archive_node failed for %s", str(node_id)[:12], exc_info=True)
            return False
        if affected is None or affected <= 0:
            return False
        if replacement_id:
            new_lit = _gql_value(str(replacement_id))
            try:
                self._locked_execute(
                    f"MATCH (a:EpisodeNode {{id: {id_lit}}}), "
                    f"(b:EpisodeNode {{id: {new_lit}}}) "
                    f"INSERT (a)-[:SUPERSEDES]->(b)"
                )
            except Exception:
                logger.warning("archive_node: SUPERSEDES edge insert failed for %s -> %s",
                               str(node_id)[:12], str(replacement_id)[:12], exc_info=True)
        return True

    def unarchive(self, node_id: str) -> bool:
        """撤销归档（P1 restore 端点可翻转软删）：SET archived=false。

        幂等：未归档节点 SET false 也是 no-op 成功；仅节点不存在返回 False
        （execute rows_affected=0，与 archive_node 存在性判定一致）。
        """
        id_lit = _gql_value(str(node_id))
        try:
            affected = self._locked_execute(
                f"MATCH (e:EpisodeNode {{id: {id_lit}}}) SET e.archived = false"
            )
        except Exception:
            logger.warning("unarchive failed for %s", str(node_id)[:12], exc_info=True)
            return False
        return affected is not None and affected > 0

    # ─── Property Version CRUD（P0-1 实体-属性-时间三维建模）────────

    def create_property_version(
        self,
        entity_id: str,
        attr_name: str,
        value: str,
        valid_from: Optional[float] = None,
        supersedes_id: Optional[str] = None,
        superseded_by: Optional[str] = None,
    ) -> str:
        """INSERT PropertyVerNode {id, entity_id, attr_name, value, valid_from, expired_at}。

        - 新版本不写 expired_at 字段（缺省 → expired_at IS NULL 即当前有效，同
          archived 的「缺失字段匹配 IS NULL」语义）
        - supersedes_id 非空 → 旧版本打 expired_at + 建 (old)-[:SUPERSEDES]->(new)
          血统边（复用 archive_node 双 MATCH + INSERT 范式）。GraphLite 一条
          execute 多条语句只执行第一条（多语句静默截断坑）→「SET expired_at」与
          「INSERT 边」必须拆为两条独立 execute。
        - superseded_by 非空（乱序中段插入，Codex R3 P1）→ 新版本打
          expired_at = 后继 valid_from + 建 (new)-[:SUPERSEDES]->(succ) 边，
          保证任意乱序下血统链完整：P.expired_at=now、new.expired_at=S.valid_from、
          P→new、new→S 四条语义全部落盘。
        """
        pid = str(uuid.uuid4())
        now = valid_from if valid_from is not None else time.time()
        # 【R4】superseded_by 相关变量预初始化（supersedes_id 块内可能引用）
        succ_lit: str = ""
        succ_ts: float = now
        # 【H1】id/entity_id/attr_name/value 均外部可达，经 _gql_value 转义
        id_lit = _gql_value(pid)
        eid_lit = _gql_value(str(entity_id))
        name_lit = _gql_value(attr_name)
        val_lit = _gql_value(str(value))
        self._locked_execute(
            f"INSERT (p:PropertyVerNode {{id: {id_lit}, entity_id: {eid_lit}, "
            f"attr_name: {name_lit}, value: {val_lit}, valid_from: {now}}})"
        )
        # 【R6-P1 使用时机修复】若有后继（中段插入），先读取后继 valid_from →
        # succ_ts（后续所有补偿都用正确的后继时间，而非初始化的 now）。
        # 【R6-P1b】读取失败 → 抛异常回滚（不得以 now 静默继续，否则产生
        # expired_at == valid_from 的失效版本）。
        if superseded_by:
            assert superseded_by is not None
            succ_lit = _gql_value(str(superseded_by))
            try:
                rows = self.execute_cypher(
                    "MATCH (s:PropertyVerNode {id: $sid}) RETURN s.valid_from AS vf",
                    {"sid": str(superseded_by)},
                )
                if rows and rows[0].get("vf") is not None:
                    succ_ts = float(rows[0]["vf"])
                else:
                    # 【R7-P1】读不到后继行/vf=None → 一致性错误，回滚后重抛
                    # （不得静默保留 succ_ts=now，否则产生 expired_at==valid_from 失效版本）
                    raise QueryError(
                        f"successor property version not found: {str(superseded_by)[:12]}"
                    )
            except Exception:
                # 回滚：删新节点后重抛（读取失败属一致性错误，不静默）
                try:
                    self._locked_execute(
                        f"MATCH (p:PropertyVerNode {{id: {id_lit}}}) DETACH DELETE p"
                    )
                except Exception:
                    logger.warning(
                        "create_property_version: rollback after succ read failed "
                        "new=%s", pid[:12], exc_info=True)
                logger.warning(
                    "create_property_version: read successor valid_from failed, "
                    "rolled back new=%s succ=%s",
                    pid[:12], str(superseded_by)[:12], exc_info=True)
                raise
        if supersedes_id:
            old_lit = _gql_value(str(supersedes_id))
            # 【R4-P1 中段插入】supersedes_id 与 superseded_by 同时非空时，
            # 需删除旧的 (P)-[:SUPERSEDES]->(S) 边——否则血统链是分支图而非单链
            # （2021→2014→2016 后 2014 同时保留 2014→2021 与 2014→2016）。
            if superseded_by:
                try:
                    # 【R4-P1 实测】GraphLite 不支持双 MATCH 链式 DELETE
                    # （MATCH (a),(b) MATCH (a)-[s]->(b) DELETE s 静默无效，
                    # count 仍 1）——必须用单 MATCH 边模式直接删
                    self._locked_execute(
                        f"MATCH (a:PropertyVerNode {{id: {old_lit}}})"
                        f"-[s:SUPERSEDES]->"
                        f"(b:PropertyVerNode {{id: {succ_lit}}}) DELETE s"
                    )
                except Exception:
                    logger.warning(
                        "create_property_version: DELETE old P->S edge failed "
                        "(non-fatal) old=%s succ=%s",
                        str(supersedes_id)[:12], str(superseded_by)[:12], exc_info=True)
            try:
                self._locked_execute(
                    f"MATCH (p:PropertyVerNode {{id: {old_lit}}}) SET p.expired_at = {now}"
                )
            except Exception:
                # P2-3 补偿：新版本已 INSERT 但旧版本过期标记失败 → 清理新版本，
                # 保持链一致（新版本已存在但旧版本未过期 = 半链）
                # 【R5-P1】中段插入时 P→S 边已被删除，补偿需恢复它（链不断裂）
                try:
                    self._locked_execute(
                        f"MATCH (p:PropertyVerNode {{id: {id_lit}}}) DETACH DELETE p"
                    )
                    if superseded_by:
                        assert superseded_by is not None
                        self._locked_execute(
                            f"MATCH (a:PropertyVerNode {{id: {old_lit}}}), "
                            f"(b:PropertyVerNode {{id: {_gql_value(str(superseded_by))}}}) "
                            f"INSERT (a)-[:SUPERSEDES]->(b)"
                        )
                except Exception:
                    logger.warning(
                        "create_property_version: compensation cleanup failed "
                        "for new=%s old=%s", pid[:12], str(supersedes_id)[:12], exc_info=True)
                logger.warning(
                    "create_property_version: SET expired_at failed, rolled back "
                    "new=%s old=%s", pid[:12], str(supersedes_id)[:12], exc_info=True)
                raise
            try:
                self._locked_execute(
                    f"MATCH (a:PropertyVerNode {{id: {old_lit}}}), "
                    f"(b:PropertyVerNode {{id: {id_lit}}}) "
                    f"INSERT (a)-[:SUPERSEDES]->(b)"
                )
            except Exception:
                # P2-3 补偿：血统边插入失败 → 清理新版本 + 恢复旧版本过期标记。
                # 【R4-P2】中段插入时 pred 原已过期于 succ.valid_from（非 NULL），
                # 补偿应恢复原值 succ_ts 而非 NULL。
                # 【R5-P1】中段插入时 P→S 边已被删除，补偿需恢复它（链不断裂）
                try:
                    self._locked_execute(
                        f"MATCH (p:PropertyVerNode {{id: {id_lit}}}) DETACH DELETE p"
                    )
                    if superseded_by:
                        assert superseded_by is not None
                        self._locked_execute(
                            f"MATCH (p:PropertyVerNode {{id: {old_lit}}}) "
                            f"SET p.expired_at = {succ_ts}"
                        )
                        self._locked_execute(
                            f"MATCH (a:PropertyVerNode {{id: {old_lit}}}), "
                            f"(b:PropertyVerNode {{id: {_gql_value(str(superseded_by))}}}) "
                            f"INSERT (a)-[:SUPERSEDES]->(b)"
                        )
                    else:
                        self._locked_execute(
                            f"MATCH (p:PropertyVerNode {{id: {old_lit}}}) "
                            f"SET p.expired_at = NULL"
                        )
                except Exception:
                    logger.warning(
                        "create_property_version: compensation cleanup failed "
                        "for new=%s old=%s", pid[:12], str(supersedes_id)[:12], exc_info=True)
                logger.warning(
                    "create_property_version: SUPERSEDES edge failed, rolled back "
                    "new=%s old=%s", pid[:12], str(supersedes_id)[:12], exc_info=True)
                raise
        if superseded_by:
            # 【R6】succ_lit/succ_ts 已在前置块读取（R6-P1 时机修复）
            try:
                self._locked_execute(
                    f"MATCH (p:PropertyVerNode {{id: {id_lit}}}) SET p.expired_at = {succ_ts}"
                )
            except Exception:
                # 【R4-P1 superseded_by 失败补偿】SET new.expired_at 失败 → 回滚：
                # 删新节点 + 恢复 pred 到原 expired_at（succ_ts）+ 恢复被删的 P→S 边。
                # 【R5-P1】pred.expired_at 恢复 succ_ts 而非 NULL（非最新节点必有 expired_at 不变式）
                try:
                    self._locked_execute(
                        f"MATCH (p:PropertyVerNode {{id: {id_lit}}}) DETACH DELETE p"
                    )
                    if supersedes_id:
                        old_lit2 = _gql_value(str(supersedes_id))
                        self._locked_execute(
                            f"MATCH (p:PropertyVerNode {{id: {old_lit2}}}) "
                            f"SET p.expired_at = {succ_ts}"
                        )
                        self._locked_execute(
                            f"MATCH (a:PropertyVerNode {{id: {old_lit2}}}), "
                            f"(b:PropertyVerNode {{id: {succ_lit}}}) "
                            f"INSERT (a)-[:SUPERSEDES]->(b)"
                        )
                except Exception:
                    logger.warning(
                        "create_property_version: superseded_by compensation cleanup "
                        "failed new=%s succ=%s",
                        pid[:12], str(superseded_by)[:12], exc_info=True)
                logger.warning(
                    "create_property_version: SET new expired_at failed, rolled back "
                    "new=%s succ=%s", pid[:12], str(superseded_by)[:12], exc_info=True)
                raise
            try:
                self._locked_execute(
                    f"MATCH (a:PropertyVerNode {{id: {id_lit}}}), "
                    f"(b:PropertyVerNode {{id: {succ_lit}}}) "
                    f"INSERT (a)-[:SUPERSEDES]->(b)"
                )
            except Exception:
                # 【R4-P1 superseded_by 边失败补偿】new→succ 边失败 → 回滚：
                # 删新节点 + 恢复 pred expired_at=succ_ts（非 NULL）+ 恢复 P→S 边。
                # 【R5-P1】pred.expired_at 恢复 succ_ts 而非 NULL（非最新节点必有 expired_at 不变式）
                try:
                    self._locked_execute(
                        f"MATCH (p:PropertyVerNode {{id: {id_lit}}}) DETACH DELETE p"
                    )
                    if supersedes_id:
                        old_lit2 = _gql_value(str(supersedes_id))
                        self._locked_execute(
                            f"MATCH (p:PropertyVerNode {{id: {old_lit2}}}) "
                            f"SET p.expired_at = {succ_ts}"
                        )
                        self._locked_execute(
                            f"MATCH (a:PropertyVerNode {{id: {old_lit2}}}), "
                            f"(b:PropertyVerNode {{id: {succ_lit}}}) "
                            f"INSERT (a)-[:SUPERSEDES]->(b)"
                        )
                except Exception:
                    logger.warning(
                        "create_property_version: superseded_by edge compensation "
                        "cleanup failed new=%s succ=%s",
                        pid[:12], str(superseded_by)[:12], exc_info=True)
                logger.warning(
                    "create_property_version: SUPERSEDES edge to successor failed, "
                    "rolled back new=%s succ=%s",
                    pid[:12], str(superseded_by)[:12], exc_info=True)
                raise
        return pid

    def get_latest_property_version(self, entity_id: str, attr_name: str) -> Optional[dict]:
        """当前最新（未过期）版本：expired_at IS NULL + valid_from DESC 取第一。

        写路径（entity_resolver 版本编排）调用 → execute_cypher（P2-2 写路径熔断
        中立：不 record_success/failure）；失败原样上抛由调用方 try/except 兜底。
        """
        rows = self.execute_cypher(
            "MATCH (p:PropertyVerNode) "
            "WHERE p.entity_id = $eid AND p.attr_name = $name "
            "AND (p.expired_at IS NULL) "
            "RETURN p.id AS id, p.entity_id AS entity_id, p.attr_name AS attr_name, "
            "p.value AS value, p.valid_from AS valid_from, p.expired_at AS expired_at "
            "ORDER BY p.valid_from DESC LIMIT 1",
            {"eid": str(entity_id), "name": attr_name},
        )
        return rows[0] if rows else None

    def get_property_versions(self, entity_id: str, attr_name: str) -> list[dict]:
        """(entity_id, attr_name) 全部版本（valid_from ASC 旧→新），供惰性裁剪。

        写路径调用 → execute_cypher（熔断中立）；失败原样上抛由调用方兜底。
        """
        return self.execute_cypher(
            "MATCH (p:PropertyVerNode) "
            "WHERE p.entity_id = $eid AND p.attr_name = $name "
            "RETURN p.id AS id, p.entity_id AS entity_id, p.attr_name AS attr_name, "
            "p.value AS value, p.valid_from AS valid_from, p.expired_at AS expired_at "
            "ORDER BY p.valid_from ASC",
            {"eid": str(entity_id), "name": attr_name},
        )

    def prune_property_versions(
        self, entity_id: str, attr_name: str,
        max_versions: int = PROPERTY_MAX_VERSIONS,
    ) -> int:
        """写时惰性裁剪（决策 5）：每 (entity_id, attr_name) 保留最近 N=8 版。

        超限时 DETACH DELETE 最旧版本（含其 SUPERSEDES 血统边）。单条独立
        execute（多语句截断坑）。返回删除数；未超限返回 0。
        """
        versions = self.get_property_versions(entity_id, attr_name)
        if len(versions) <= max_versions:
            return 0
        removed = 0
        for v in versions[: len(versions) - max_versions]:
            vid = v.get("id", "")
            if not vid:
                continue
            try:
                self._locked_execute(
                    f"MATCH (p:PropertyVerNode {{id: {_gql_value(str(vid))}}}) "
                    f"DETACH DELETE p"
                )
                removed += 1
            except Exception:
                logger.warning("prune_property_versions: delete failed for %s (non-fatal)",
                               str(vid)[:12])
        return removed

    def get_property_versions_for_entities(self, entity_ids: list[str]) -> list[dict]:
        """批量取多个实体的全部属性版本（检索通道 _property_temporal_retrieve 用）。

        【P1-2 实体归一化匹配】候选经 normalize_entity_name（小写 + 去尾词
        Inc/Corp/Ltd/Company 等）后与写入 entity_id 做前缀匹配——写侧原始
        subject（"Apple Inc"）与读侧短名/小写（"Apple"/"apple"）对齐，
        LOWER(entity_id) 前缀命中（'apple' 命中 'Apple Inc'）。

        复用 query_cypher 永不抛异常契约：空输入 / 查询失败 → []（静默降级，
        主检索零回归）。valid_from DESC（最新在前）。
        """
        if not entity_ids:
            return []
        from core.entity_resolver import normalize_entity_name
        # 候选归一化 → 前缀模式（'apple%'）；去重保序
        patterns: list[str] = []
        seen: set[str] = set()
        for eid in entity_ids:
            norm = normalize_entity_name(eid)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            # N6: 前缀匹配加词边界后置过滤——精确 OR 空格后缀（"apple" 命中
            # "Apple Inc" 但不命中 "Applebee's"/"Applejack"）
            norm_lit = _gql_value(norm)[1:-1]
            patterns.append(
                f"(LOWER(p.entity_id) = '{norm_lit}' "
                f"OR LOWER(p.entity_id) LIKE '{norm_lit} %')"
            )
        if not patterns:
            return []
        where = " OR ".join(patterns)
        return self.query_cypher(
            "MATCH (p:PropertyVerNode) "
            f"WHERE {where} "
            "RETURN p.id AS id, p.entity_id AS entity_id, p.attr_name AS attr_name, "
            "p.value AS value, p.valid_from AS valid_from, p.expired_at AS expired_at "
            "ORDER BY p.valid_from DESC"
        )

    def get_distinct_attr_names(self) -> list[str]:
        """【v5.50.0 P2】全部属性名清单（PropertyVerNode.attr_name distinct）。

        只读，复用 query_cypher 永不抛异常契约（GraphLite 失败 → []）；
        供 ontology_evolution 注入 prompt + 校验 attr_ops canonical 有消费方。
        Python 侧去重保序（DISTINCT 引擎侧去重 + 跨行防御）。
        """
        rows = self.query_cypher(
            "MATCH (p:PropertyVerNode) RETURN DISTINCT p.attr_name AS attr_name"
        )
        names: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("attr_name", "") or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names

    # ─── Hyperedge CRUD ─────────────────────────────

    def create_hyperedge_node(self, hyperedge: dict) -> str:
        hid = hyperedge.get("id", str(uuid.uuid4()))
        vals = _dict_to_gql_values({k: v for k, v in hyperedge.items() if k != "id"})
        # 【H1】id 经 _gql_value 转义
        id_lit = _gql_value(str(hid))
        gql = f"INSERT (h:HyperedgeNode {{id: {id_lit}, {vals}}})"
        self._locked_execute(gql)
        return hid

    def link_hyperedge_member(self, hyperedge_id: str, episode_id: str) -> None:
        # 【H1】id 经 _gql_value 转义
        hid_lit = _gql_value(str(hyperedge_id))
        eid_lit = _gql_value(str(episode_id))
        gql = (
            f"MATCH (h:HyperedgeNode {{id: {hid_lit}}}), "
            f"(e:EpisodeNode {{id: {eid_lit}}}) "
            f"INSERT (h)-[:HYPEREDGE_MEMBER]->(e)"
        )
        self._locked_execute(gql)

    def get_hyperedge_members(self, hyperedge_id: str) -> list[dict]:
        gql = f"MATCH (h:HyperedgeNode {{id: {_gql_value(str(hyperedge_id))}}})-[:HYPEREDGE_MEMBER]->(e) RETURN e"
        try:
            result = self._locked_query(gql)
            return [self._flatten_row(r, "e") for r in result.rows]
        except Exception:
            return []

    def get_hyperedges_by_node(self, node_id: str) -> list[dict]:
        gql = f"MATCH (h:HyperedgeNode)-[:HYPEREDGE_MEMBER]->(e:EpisodeNode {{id: {_gql_value(str(node_id))}}}) RETURN h"
        try:
            result = self._locked_query(gql)
            return [self._flatten_row(r, "h") for r in result.rows]
        except Exception:
            return []

    def get_hypergraph_neighbors(self, seed_ids: list[str], limit: int = 20) -> dict[str, list[dict]]:
        """向量种子 → 共享超边成员扩散（1 跳）。返回 {seed_id: [{id, content, co_occurrence}]}。

        每个种子执行单条 GQL 获取其超边邻居，避免逐邻居查询的 N+1 问题。
        GraphLite b64 编码的 content 在此解码。
        空输入 / 全部异常 → 返回 {}。
        """
        from base64 import b64decode

        if not seed_ids:
            return {}
        result: dict[str, list[dict]] = {}
        for sid in seed_ids:
            gql = (
                "MATCH (e:EpisodeNode {id: $sid})-[:HYPEREDGE_MEMBER]-(h:HyperedgeNode)-[:HYPEREDGE_MEMBER]-(e2:EpisodeNode) "
                "WHERE e2.id <> $sid "
                "RETURN DISTINCT e2.id AS id, e2.content AS content, count(h) AS co_occurrence "
                "ORDER BY co_occurrence DESC LIMIT $limit"
            )
            # query_cypher 永不抛异常（GraphLiteStore P0-2 契约），无需 try/except
            rows = self.query_cypher(gql, {"sid": sid, "limit": limit})
            if not rows:
                continue
            neighbors: list[dict] = []
            for row in rows:
                nid = row.get("id", "")
                content = row.get("content", "")
                cooc = row.get("co_occurrence", 0)
                # b64 decode if needed
                if isinstance(content, str) and content.startswith("{b64}"):
                    try:
                        content = b64decode(content[5:]).decode("utf-8")
                    except Exception:
                        pass  # decode failure → keep raw
                try:
                    cooc = int(cooc)
                except (TypeError, ValueError):
                    cooc = 0
                neighbors.append({"id": str(nid), "content": content, "co_occurrence": cooc})
            if neighbors:
                result[sid] = neighbors
        return result

    def get_communities_by_seeds(self, seed_ids: list[str]) -> list[dict]:
        """【v5.41 社区扩召回】种子节点 → 所属社区反查。

        边方向：(c:CommunityNode)-[:COMMUNITY_MEMBER]->(e:EpisodeNode)（社区→成员，
        CC 修正：任务书 v1 写反）。一条批量查询返回
        [{community_id, summary, member_ids}]；member_ids 为该社区命中种子的成员 id。

        复用 query_cypher 永不抛契约：空输入 / 查询失败 / 无命中 → []（静默降级，
        主检索零回归）。【H1】批量 id 逐个 _gql_value 转义（含 ' / \\ 的 id 不裸插）。
        """
        if not seed_ids:
            return []
        ids = ", ".join(_gql_value(str(i)) for i in seed_ids)
        gql = (
            "MATCH (c:CommunityNode)-[:COMMUNITY_MEMBER]->(e:EpisodeNode) "
            f"WHERE e.id IN [{ids}] "
            "RETURN c.id AS community_id, c.summary AS summary, e.id AS member_id"
        )
        rows = self.query_cypher(gql)  # 永不抛（P0-2 契约），失败返回 []
        if not rows:
            return []
        communities: dict[str, dict] = {}
        for row in rows:
            cid = row.get("community_id", "") or ""
            if not cid:
                continue
            entry = communities.setdefault(cid, {
                "community_id": cid,
                "summary": row.get("summary", "") or "",
                "member_ids": [],
            })
            mid = row.get("member_id", "") or ""
            if mid and mid not in entry["member_ids"]:
                entry["member_ids"].append(mid)
        return list(communities.values())

    def get_community_members(self, community_id: str, limit: int = 10) -> list[dict]:
        """【v5.41 社区扩召回】按社区批量取成员。

        返回 [{member_id, content, archived, fact_track, tau_value}]（content/归档/
        fact_track 一次取回，供 _finish 去重/归档过滤/core·画像 boost 复用，免 N+1 回查）。
        复用 query_cypher 永不抛契约：空输入 / 查询失败 → []（静默降级）。
        """
        if not community_id:
            return []
        gql = (
            "MATCH (c:CommunityNode {id: $cid})-[:COMMUNITY_MEMBER]->(e:EpisodeNode) "
            "RETURN e.id AS member_id, e.content AS content, "
            "e.archived AS archived, e.fact_track AS fact_track, "
            "e.tau_initial AS tau_value "
            "ORDER BY e.tau_initial DESC "
            "LIMIT $limit"
        )
        rows = self.query_cypher(gql, {"cid": community_id, "limit": int(limit)})
        members: list[dict] = []
        for row in rows:
            mid = row.get("member_id", "") or ""
            if not mid:
                continue
            members.append({
                "member_id": mid,
                "content": row.get("content", "") or "",
                "archived": row.get("archived", False),
                "fact_track": row.get("fact_track", "active") or "active",
                "tau_value": row.get("tau_value", 0.0) or 0.0,
            })
        return members

    def get_all_hebbian_connections(self) -> list[dict]:
        gql = "MATCH (a)-[r:HEBBIAN_CONNECTION]->(b) RETURN a.id AS src, b.id AS dst, r.weight AS weight"
        try:
            result = self._locked_query(gql)
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
        # 【H1】session_id 外部可达（X-Session-Id 头），经 _gql_value 转义
        id_lit = _gql_value(str(session_id))
        try:
            result = self._locked_query(
                f"MATCH (s:SessionNode {{id: {id_lit}}}) RETURN s.id"
            )
            if result.rows:
                return  # 已存在
        except Exception:
            pass  # 查询失败（如表不存在）时走创建路径
        self._locked_execute(
            f"INSERT (s:SessionNode {{id: {id_lit}, "
            f"created_at: {int(time.time())}, last_seen: {int(time.time())}}})"
        )

    def link_to_session(self, session_id: str, episode_id: str) -> None:
        """Link episode to session node."""
        # 【H1】session_id/episode_id 经 _gql_value 转义（外部可达：X-Session-Id 头）
        sid_lit = _gql_value(str(session_id))
        eid_lit = _gql_value(str(episode_id))
        gql = (
            f"MATCH (s:SessionNode {{id: {sid_lit}}}), "
            f"(e:EpisodeNode {{id: {eid_lit}}}) "
            f"INSERT (s)-[:SESSION_MEMBER]->(e)"
        )
        self._locked_execute(gql)

    def get_session_memories(self, session_id: str, limit: int = 100) -> list[dict]:
        gql = f"MATCH (s:SessionNode {{id: {_gql_value(str(session_id))}}})-[:SESSION_MEMBER]->(e) RETURN e LIMIT {limit}"
        try:
            result = self._locked_query(gql)
            return [self._flatten_row(r, "e") for r in result.rows]
        except Exception:
            return []

    # ─── Session/Visual CRUD（P0-2 幽灵方法实现）───────────────

    def get_or_create_session(self, session_id: str, metadata: Optional[str] = None) -> str:
        """获取或创建 SessionNode，返回 session_id（两段式幂等，参照 ensure_session）。"""
        self.ensure_session(session_id)
        if metadata:
            # 【H1】session_id/metadata 经 _gql_value 转义（metadata 为外部可达 JSON 串）
            self._locked_execute(
                f"MATCH (s:SessionNode {{id: {_gql_value(str(session_id))}}}) "
                f"SET s.metadata = {_gql_value(str(metadata))}"
            )
        return session_id

    def link_session_member(self, session_node_id: str, episode_id: str) -> None:
        """Link episode to session node（参照 link_to_session 的 MATCH + INSERT）。"""
        # 【H1】session_node_id/episode_id 经 _gql_value 转义
        sid_lit = _gql_value(str(session_node_id))
        eid_lit = _gql_value(str(episode_id))
        gql = (
            f"MATCH (s:SessionNode {{id: {sid_lit}}}), "
            f"(e:EpisodeNode {{id: {eid_lit}}}) "
            f"INSERT (s)-[:SESSION_MEMBER]->(e)"
        )
        self._locked_execute(gql)

    def create_visual_node(self, node: dict) -> str:
        """INSERT VisualNode。id 为必填；embedding 是 list，_gql_value 自动 b64 序列化。"""
        vid = node.get("id", str(uuid.uuid4()))
        vals = _dict_to_gql_values(node, skip_keys={"id"})
        # 【H1】id 经 _gql_value 转义
        id_lit = _gql_value(str(vid))
        self._locked_execute(f"INSERT (v:VisualNode {{id: {id_lit}, {vals}}})")
        return vid

    def get_visual_node(self, visual_id: str) -> Optional[dict]:
        """MATCH VisualNode by id（参照 get_episode）。"""
        gql = f"MATCH (v:VisualNode {{id: {_gql_value(str(visual_id))}}}) RETURN v"
        try:
            result = self._locked_query(gql)
            if result.rows:
                return self._flatten_row(result.rows[0], "v")
        except Exception:
            return None
        return None

    def get_visual_nodes(self, limit: int = 50) -> list[dict]:
        """列出 VisualNode（flatten 后含 b64 解码的 caption）。"""
        gql = f"MATCH (v:VisualNode) RETURN v LIMIT {limit}"
        try:
            result = self._locked_query(gql)
            return [self._flatten_row(r, "v") for r in result.rows]
        except Exception:
            return []

    def delete_namespace(self, namespace: str) -> int:
        """按命名空间删除：删除 SessionNode 及其 SESSION_MEMBER 关联的 EpisodeNode。

        返回删除的 EpisodeNode 数。节点有关联关系必须 DETACH DELETE(skill 记过的坑)。
        """
        # 【H1】namespace/eid 经 _gql_value 转义（外部可达：DELETE /namespaces/{ns}）
        ns_lit = _gql_value(str(namespace))
        # 1. 找到该 namespace(SessionNode)下的所有 EpisodeNode
        try:
            result = self._locked_query(
                f"MATCH (s:SessionNode {{id: {ns_lit}}})-[:SESSION_MEMBER]->(e:EpisodeNode) "
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
                self._locked_execute(
                    f"MATCH (e:EpisodeNode {{id: {_gql_value(str(eid))}}}) DETACH DELETE e"
                )
                deleted += 1
            except Exception:
                pass

        # 3. 删除 SessionNode 本身(及其残留关系)
        try:
            self._locked_execute(
                f"MATCH (s:SessionNode {{id: {ns_lit}}}) DETACH DELETE s"
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
            result = self._locked_query(q)
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
            result = self._locked_query(q)
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
                        # 【L1】裸 except → ValueError：b64decode 抛 binascii.Error⊂ValueError，
                        # decode 抛 UnicodeDecodeError⊂ValueError；语义不变，不再吞 KeyboardInterrupt。
                        try:
                            flat[pk] = b64decode(flat[pk][5:]).decode('utf-8')
                        except ValueError:
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
        """Basic $param interpolation to GQL literals (security: simple only).

        v5.30.0 改单次 re.sub 带回调替换 $name —— 按捕获的完整键名查 params：
        - 键序无关（旧实现按 dict 序逐键 str.replace）
        - 无前缀碰撞（旧实现 $t1 会误替换 $t10 → P0 静默数据丢失）
        - 无需 re.escape（占位符是 $word，匹配不到普通文本）
        未知键返回原 match 文本（保持"未命中不替换"语义）。
        """
        import re
        if not params:
            return query

        def _repl(match: re.Match) -> str:
            key = match.group(1)
            if key not in params:
                return match.group(0)
            v = params[key]
            if isinstance(v, str):
                if not v:
                    # 空串：GraphLite 中 CONTAINS '' 恒真 → NOT CONTAINS '' 恒假，
                    # read_validate 的 $new_value 为空会导致矛盾漏检。
                    # 用哨兵值使 NOT CONTAINS 恒真（语义 = 不排除已有事实）。
                    return "'__SHM_NO_VALUE__'"
                # GraphLite lexer UTF-8 bug 已修复（fork 4452a96）——原生中文直插
                escaped = v.replace("\\", "\\\\").replace("'", "\\'")
                return f"'{escaped}'"
            elif isinstance(v, (int, float)):
                return str(v)
            elif isinstance(v, (np.integer, np.floating)):
                # numpy 标量（如 FAISS 搜索返回的 np.float32）不是 int/float 实例，
                # 直接 str() 会带类型前缀；统一转 Python 标量
                return str(v.item())
            elif v is None:
                return "NULL"
            return match.group(0)

        return re.sub(r"\$([A-Za-z_]\w*)", _repl, query)

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
