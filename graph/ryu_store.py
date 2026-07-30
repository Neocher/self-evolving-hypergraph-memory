"""
RyuGraph 图存储封装
==============
RyuGraph 嵌入式列式图数据库（Kuzu 社区活跃 fork），Cypher 查询引擎。
drop-in 兼容 RyuGraph API，使用 import ryugraph as kuzu。

超边实现策略：将超边编码为辅助节点 (HyperedgeNode)，
用 Cypher 边连接所有成员节点。额外一跳查询，性能可接受。

[Harness Fix] 集成断路器模式：错误率 > 50% 跳闸，30s 半开探测。
"""

from __future__ import annotations

# RyuGraph（Kuzu 社区活跃 fork，drop-in 兼容）
import ryugraph as kuzu
import logging
import time
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# ─── [Harness Fix] 断路器 ───────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"       # 正常
    OPEN = "open"           # 断开
    HALF_OPEN = "half_open" # 半开探测


@dataclass
class CircuitBreakerConfig:
    """断路器配置"""
    failure_threshold: float = 0.5   # 错误率 > 50% 跳闸
    recovery_timeout: float = 30.0   # 半开等待时间
    half_open_max_requests: int = 1  # 半开时允许的探测请求数
    window_size: int = 10            # [Fix] 滑动窗口大小（最近 N 次请求）


class CircuitBreakerOpen(Exception):
    """断路器跳闸异常，供上层捕获降级"""
    pass


class CircuitBreaker:
    """
    [Harness Fix] 断路器模式实现。

    保护外部依赖（Kuzu）不被连续故障压垮，
    在故障率达到阈值时自动断开请求，30s 后尝试半开探测。
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.last_failure_time: float = 0.0
        # [Fix] 滑动窗口：记录最近 N 次请求的成功/失败
        self._window: collections.deque[bool] = __import__('collections').deque(
            maxlen=self.config.window_size
        )

    def record_success(self) -> None:
        """记录成功调用"""
        self._window.append(True)
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self._window.clear()

    def record_failure(self) -> None:
        """记录失败调用"""
        self._window.append(False)
        if len(self._window) < 2:
            return  # 样本不足，暂不跳闸
        failures = sum(1 for r in self._window if not r)
        error_rate = failures / len(self._window)
        if error_rate > self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()

    def is_available(self) -> bool:
        """
        检查断路器是否允许请求。

        Returns:
            True 如果允许请求通过

        Raises:
            CircuitBreakerOpen: 如果断路器处于 OPEN 状态且未到恢复时间
        """
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            raise CircuitBreakerOpen("Circuit breaker is OPEN, request blocked")
        # HALF_OPEN: 只允许一个探测请求
        return True


def _clean_kuzu_row(row: dict) -> dict:
    """RyuGraph RETURN e.* returns keys like 'e.id', clean to 'id'.
    Accepts a dict (from polars to_dicts()) or a tuple (from rows()).
    """
    if isinstance(row, dict):
        cleaned = {}
        for k, v in row.items():
            clean_key = k.split(".")[-1] if "." in str(k) else k
            cleaned[clean_key] = v
        return cleaned
    # tuple: don't have keys, return as-is or convert
    # This shouldn't be called with tuples - use to_dicts() instead
    return {"value": row[0]} if row else {}

@dataclass
class RyuConfig:
    """RyuGraph 连接配置"""
    database_path: str = "./data/shm_kuzu_db"
    buffer_pool_size_mb: int = 256
    max_threads: int = 4


class RyuStore:
    """
    RyuGraph 图存储封装。

    管理节点和边的 CRUD，提供 Cypher 查询接口。
    内置超边辅助节点支持和断路器降级机制。
    """

    def __init__(self, config: Optional[RyuConfig] = None) -> None:
        self.config = config or RyuConfig()
        self.db: Optional[kuzu.Database] = None
        self._connections: list[kuzu.Connection] = []  # 连接池
        self._conn_pool_idx: int = 0
        # [Harness Fix] 断路器实例
        self.circuit_breaker: CircuitBreaker = CircuitBreaker()
        from collections import deque
        self._sensory_buffer: deque = deque(maxlen=1000)

    @property
    def conn(self) -> kuzu.Connection:
        """轮询从连接池中获取下一个连接。"""
        if not self._connections:
            raise RuntimeError("RyuStore not connected (connection pool empty)")
        self._conn_pool_idx = (self._conn_pool_idx + 1) % len(self._connections)
        return self._connections[self._conn_pool_idx]

    def connect(self) -> None:
        """连接/创建 RyuGraph 数据库并初始化 schema + 连接池。"""
        db_path = Path(self.config.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(
            str(db_path),
            auto_checkpoint=True,
            checkpoint_threshold=4_194_304,  # 4MB: frequent checkpointing reduces WAL corruption risk
            max_num_threads=self.config.max_threads,
        )

        # 创建连接池（默认 4 个连接，与 max_threads 一致）
        pool_size = max(2, self.config.max_threads or 4)
        for i in range(pool_size):
            c = kuzu.Connection(self.db)
            self._connections.append(c)
        logger.info("RyuGraph connection pool created", pool_size=pool_size)

        self._init_schema()

    def _init_schema(self) -> None:
        """
        初始化节点和边的 schema。

        Node tables:
          - EpisodeNode (id STRING, content STRING, embedding FLOAT[384],
                         created_at DOUBLE, tau_initial DOUBLE, source STRING)
          - HyperedgeNode (id STRING, type STRING, created_at DOUBLE,
                           gate_value DOUBLE, metadata STRING)
          - CommunityNode (id STRING, name STRING, summary STRING,
                           leiden_score DOUBLE, created_at DOUBLE)
          - **SessionNode** (id STRING, session_id STRING, created_at DOUBLE,
                             metadata STRING)

        Edge tables:
          - HEBBIAN_CONNECTION (from EpisodeNode, to EpisodeNode, weight DOUBLE)
          - HYPEREDGE_MEMBER (from HyperedgeNode, to EpisodeNode|CommunityNode)
          - COMMUNITY_MEMBER (from CommunityNode, to EpisodeNode)
          - TEMPORAL_LINK (from EpisodeNode, to EpisodeNode, time_diff DOUBLE)
          - **SESSION_MEMBER (from SessionNode, to EpisodeNode)**
        """
        self.conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS EpisodeNode ("
            "id STRING, content STRING, embedding FLOAT[384], "
            "created_at DOUBLE, tau_initial DOUBLE, tau_value DOUBLE, "
            "trust_score DOUBLE, ontology_type STRING, source STRING, "
            "visibility STRING, "
            "quarantine BOOLEAN, quarantine_reason STRING, "
            "quarantine_source STRING, quarantined_at DOUBLE, "
            "version INT64, "
            "PRIMARY KEY (id))"
        )
        self.conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS HyperedgeNode ("
            "id STRING, type STRING, created_at DOUBLE, "
            "gate_value DOUBLE, metadata STRING, "
            "PRIMARY KEY (id))"
        )
        self.conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS CommunityNode ("
            "id STRING, name STRING, summary STRING, "
            "leiden_score DOUBLE, created_at DOUBLE, "
            "PRIMARY KEY (id))"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS HEBBIAN_CONNECTION "
            "(FROM EpisodeNode TO EpisodeNode, weight DOUBLE)"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS HYPEREDGE_MEMBER "
            "(FROM HyperedgeNode TO EpisodeNode)"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS COMMUNITY_MEMBER "
            "(FROM CommunityNode TO EpisodeNode)"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS TEMPORAL_LINK "
            "(FROM EpisodeNode TO EpisodeNode, time_diff DOUBLE)"
        )
        # 会话观测节点 — 连接同一交互会话中的多个 EpisodeNode
        self.conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS SessionNode ("
            "id STRING, session_id STRING, created_at DOUBLE, "
            "metadata STRING, "
            "PRIMARY KEY (id))"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS SESSION_MEMBER "
            "(FROM SessionNode TO EpisodeNode)"
        )
        # 多模态：视觉节点 — 存储图像记忆
        self.conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS VisualNode ("
            "id STRING, image_path STRING, caption STRING, "
            "embedding FLOAT[384], source STRING, created_at DOUBLE, "
            "PRIMARY KEY (id))"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS VISUAL_HYPEREDGE_MEMBER "
            "(FROM HyperedgeNode TO VisualNode)"
        )
        # 本体论节点/边
        self.conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS OntologyType ("
            "name STRING, category STRING, "
            "PRIMARY KEY (name))"
        )
        self.conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS OntologyEntity ("
            "name STRING, type STRING, category STRING, "
            "PRIMARY KEY (name))"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS IS_A "
            "(FROM OntologyEntity TO OntologyType)"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS RELATES_TO "
            "(FROM OntologyEntity TO OntologyEntity, relation STRING)"
        )

        # Phase 2: 程序记忆节点 — 存储重复出现的行动模式/工作流模板
        self.conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS ProceduralNode ("
            "id STRING, pattern_name STRING, pattern_type STRING, "
            "trigger_sequence STRING, action_template STRING, "
            "confidence DOUBLE, frequency INT64, created_at DOUBLE, "
            "last_matched_at DOUBLE, "
            "PRIMARY KEY (id))"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS PROCEDURAL_PATTERN "
            "(FROM ProceduralNode TO EpisodeNode, match_count INT64)"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS PROCEDURAL_HYPEREDGE "
            "(FROM HyperedgeNode TO ProceduralNode)"
        )

        # Phase 2: 概念记忆节点 — 最高抽象层，链接多个社区形成统一框架
        self.conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS ConceptualNode ("
            "id STRING, concept_name STRING, description STRING, "
            "abstraction_level STRING, confidence DOUBLE, "
            "created_at DOUBLE, source_communities STRING, "
            "PRIMARY KEY (id))"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS CONCEPTUAL_FRAMEWORK "
            "(FROM ConceptualNode TO CommunityNode, weight DOUBLE)"
        )
        self.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS CONCEPTUAL_HYPEREDGE "
            "(FROM HyperedgeNode TO ConceptualNode)"
        )

        # [Migration] 添加旧版本可能缺失的列
        # RyuGraph 0.11.x does not support "IF NOT EXISTS" in ALTER TABLE
        # RyuGraph 0.11.x also does NOT support ALTER TABLE ADD COLUMN at all
        # We detect this by checking the available ALTER options
        _supports_alter_add = True
        for col_type in [
            ("ontology_type", "STRING"),
            ("trust_score", "DOUBLE"),
            ("tau_value", "DOUBLE"),
        ]:
            try:
                self.conn.execute(
                    f"ALTER TABLE EpisodeNode ADD COLUMN {col_type[0]} {col_type[1]}"
                )
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "already exists" in err_msg:
                    pass  # Column already exists — expected
                elif "add column" in err_msg and "not supported" in err_msg:
                    _supports_alter_add = False
                elif "invalid input" in err_msg and "add column" in err_msg:
                    # RyuGraph 0.11.x: Parser exception — ALTER ADD COLUMN not supported
                    _supports_alter_add = False
                else:
                    # Use stdlib-safe logging (no extra kwargs)
                    logger.warning(f"Schema migration skipped: {col_type[0]} -> {e}")

        # If ALTER ADD COLUMN is not supported, log a one-time warning
        # The database must be recreated from scratch with the correct schema
        if not _supports_alter_add:
            logger.warning(
                "RyuGraph version does not support ALTER TABLE ADD COLUMN. "
                "All required columns must be present at CREATE TABLE time. "
                "If you are migrating from an older database, delete the DB file and restart."
            )

        # 查询索引 — 加速按时间/来源的过滤和排序
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_created_at ON (EpisodeNode.created_at)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_source ON (EpisodeNode.source)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_hyperedge_created_at ON (HyperedgeNode.created_at)")
        except RuntimeError:
            logger.warning("CREATE INDEX not supported by this RyuGraph version; query performance may degrade")

    def _execute_with_circuit_breaker(self, query_func, *args, **kwargs):
        """
        [Harness Fix] 带断路器的查询执行封装。

        1. 检查断路器状态，OPEN 时抛 CircuitBreakerOpen
        2. 执行查询
        3. 成功 → record_success()，失败 → record_failure()
        """
        self.circuit_breaker.is_available()  # 可能抛 CircuitBreakerOpen
        try:
            result = query_func(*args, **kwargs)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise

    def create_episode(self, episode: dict) -> str:
        """创建情节节点（默认 version=1）。"""
        def _do_create():
            params = dict(episode)
            params.setdefault("version", 1)
            # visibility 可选列（向后兼容旧版本 RyuGraph）
            if "visibility" in params:
                self.conn.execute(
                    "CREATE (e:EpisodeNode {id: $id, content: $content, "
                    "created_at: $created_at, tau_initial: $tau_initial, "
                    "source: $source, visibility: $visibility, version: $version})",
                    params
                )
            else:
                self.conn.execute(
                    "CREATE (e:EpisodeNode {id: $id, content: $content, "
                    "created_at: $created_at, tau_initial: $tau_initial, "
                    "source: $source, version: $version})",
                    params
                )
            return episode['id']

        return self._execute_with_circuit_breaker(_do_create)

    def get_episode(self, node_id: str) -> Optional[dict]:
        """按 ID 查询情节节点"""
        def _do_get():
            result = self.conn.execute(
                "MATCH (e:EpisodeNode) WHERE e.id = $id RETURN e.*",
                {"id": node_id}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            if dicts:
                return _clean_kuzu_row(dicts[0])
            return None

        return self._execute_with_circuit_breaker(_do_get)

    def update_with_version(
        self,
        node_id: str,
        data: dict,
        expected_version: Optional[int] = None,
    ) -> dict:
        """
        OCC 安全写入：仅当 version 匹配时才更新节点。

        如果 expected_version 为 None，跳过版本检查（强制写入）。
        否则执行条件 SET，更新后 version 自动递增。

        Args:
            node_id: 目标节点 ID。
            data: 待更新的字段 dict（可包含 content, source, visibility 等）。
            expected_version: 预期版本号。None 表示不检查版本。

        Returns:
            {
                "success": bool,
                "updated": bool,       # True 如果实际更新了数据
                "version_conflict": bool,  # True 如果版本不匹配
                "current_version": Optional[int],
            }
        """
        def _do_update():
            if expected_version is None:
                # 强制写入 — 不检查版本
                set_clauses = ", ".join(
                    f"e.{k} = ${k}" for k in data if k != "id"
                )
                if not set_clauses:
                    return {"success": True, "updated": False,
                            "version_conflict": False, "current_version": None}
                params = {k: v for k, v in data.items() if k != "id"}
                params["id"] = node_id
                # version 递增
                set_clauses += ", e.version = COALESCE(e.version, 1) + 1"
                self.conn.execute(
                    f"MATCH (e:EpisodeNode {{id: $id}}) "
                    f"SET {set_clauses}",
                    params,
                )
                # 回读确认
                node = self.get_episode(node_id)
                return {
                    "success": True,
                    "updated": True,
                    "version_conflict": False,
                    "current_version": node.get("version", 1) if node else None,
                }

            # OCC 条件更新：version 必须匹配
            set_clauses = ", ".join(
                f"e.{k} = ${k}" for k in data if k != "id"
            )
            if not set_clauses:
                return {"success": True, "updated": False,
                        "version_conflict": False, "current_version": None}
            params = {k: v for k, v in data.items() if k != "id"}
            params["id"] = node_id
            params["expected_version"] = expected_version
            # version 递增
            set_clauses += ", e.version = COALESCE(e.version, 1) + 1"

            self.conn.execute(
                f"MATCH (e:EpisodeNode {{id: $id}}) "
                f"WHERE e.version = $expected_version "
                f"SET {set_clauses}",
                params,
            )

            # 回读验证：如果 version 没变，说明 WHERE 条件未匹配
            node = self.get_episode(node_id)
            if node is None:
                return {"success": False, "updated": False,
                        "version_conflict": False, "current_version": None}

            current_version = node.get("version", 1)
            version_changed = current_version != expected_version

            return {
                "success": version_changed,
                "updated": version_changed,
                "version_conflict": not version_changed,
                "current_version": current_version,
            }

        return self._execute_with_circuit_breaker(_do_update)

    def get_episodes_batch(self, node_ids: list[str]) -> list[dict]:
        """批量按ID查询情节节点。一次查询替代N次 get_episode。"""
        if not node_ids:
            return []

        def _do_batch():
            result = self.conn.execute(
                "MATCH (e:EpisodeNode) WHERE e.id IN $ids RETURN e.*",
                {"ids": node_ids}
            )
            # RyuGraph 25.9: get_as_pl 可能失败，用原生遍历兜底
            try:
                pl = result.get_as_pl()
                if pl is not None and hasattr(pl, 'to_dicts'):
                    return [_clean_kuzu_row(d) for d in pl.to_dicts()]
            except Exception:
                pass
            
            # 原生方式遍历
            # 重新执行获取 column names（第一次已被 get_as_pl 消耗）
            r2 = self.conn.execute(
                "MATCH (e:EpisodeNode) WHERE e.id IN $ids RETURN e.*",
                {"ids": node_ids}
            )
            try:
                cols = r2.get_column_names()
            except Exception:
                cols = ['e.id', 'e.content', 'e.embedding', 'e.created_at',
                        'e.tau_initial', 'e.tau_value', 'e.trust_score',
                        'e.ontology_type', 'e.source', 'e.visibility',
                        'e.quarantine', 'e.quarantine_reason',
                        'e.quarantine_source', 'e.quarantined_at', 'e.version']
            
            cleaned = []
            while r2.has_next():
                row = r2.get_next()
                if isinstance(row, dict):
                    cleaned.append(_clean_kuzu_row(row))
                elif isinstance(row, (list, tuple)):
                    d = {}
                    for i, val in enumerate(row):
                        k = cols[i] if i < len(cols) else f"col_{i}"
                        d[k] = val
                    cleaned.append(_clean_kuzu_row(d))
            return cleaned

        return self._execute_with_circuit_breaker(_do_batch)

    def query_cypher(self, query: str, params: Optional[dict] = None) -> list:
        """
        执行任意 Cypher 查询（带断路器保护）。
        返回 list[tuple] — 统一 Polars DataFrame 和原生格式。
        """
        def _do_query():
            result = self.conn.execute(query, params or {})
            rows = result.get_as_pl()
            # query_cypher returns list[tuple] for backward compat (health checks etc.)
            if hasattr(rows, 'rows'):
                return rows.rows()
            if hasattr(rows, 'to_numpy'):
                return [tuple(r) for r in rows.to_numpy()]
            # If rows is a Polars DataFrame, convert to list of tuples
            if hasattr(rows, 'iter_rows'):
                return list(rows.iter_rows())
            return rows

        return self._execute_with_circuit_breaker(_do_query)

    def get_episodes_by_tau_range(self,
                                   min_tau: float,
                                   max_tau: float,
                                   limit: int = 100) -> List[dict]:
        """按 τ 值范围查询情节节点（需先有 tau_current，否则用 tau_initial 兜底）。"""
        def _do_query():
            result = self.conn.execute(
                "MATCH (e:EpisodeNode) "
                "WHERE e.tau_initial >= $min_tau AND e.tau_initial <= $max_tau "
                "RETURN e.* ORDER BY e.tau_initial DESC LIMIT $limit",
                {"min_tau": min_tau, "max_tau": max_tau, "limit": limit}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            if dicts:
                return [_clean_kuzu_row(r) for r in dicts]
            return []

        return self._execute_with_circuit_breaker(_do_query)

    def get_active_episodes(self, time_window_seconds: float = 1800) -> List[dict]:
        """获取时间窗口内的活跃情节节点。"""
        def _do_query():
            now = time.time()
            cutoff = now - time_window_seconds
            result = self.conn.execute(
                "MATCH (e:EpisodeNode) WHERE e.created_at >= $cutoff "
                "RETURN e.* ORDER BY e.created_at DESC",
                {"cutoff": cutoff}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            if dicts:
                return [_clean_kuzu_row(r) for r in dicts]
            return []

        return self._execute_with_circuit_breaker(_do_query)

    def create_hyperedge_node(self, hyperedge: dict) -> str:
        """创建超边辅助节点。"""
        def _do_create():
            self.conn.execute(
                "CREATE (h:HyperedgeNode {id: $id, type: $type, "
                "created_at: $created_at, gate_value: $gate_value, metadata: $metadata})",
                hyperedge
            )
            return hyperedge['id']

        return self._execute_with_circuit_breaker(_do_create)

    def link_hyperedge_member(self, hyperedge_id: str, episode_id: str) -> None:
        """将情节节点连接到超边辅助节点。"""
        def _do_link():
            self.conn.execute(
                "MATCH (h:HyperedgeNode), (e:EpisodeNode) "
                "WHERE h.id = $hid AND e.id = $eid "
                "CREATE (h)-[:HYPEREDGE_MEMBER]->(e)",
                {"hid": hyperedge_id, "eid": episode_id}
            )

        return self._execute_with_circuit_breaker(_do_link)

    def get_hyperedge_members(self, hyperedge_id: str) -> List[dict]:
        """查询超边的所有成员节点。"""
        def _do_query():
            result = self.conn.execute(
                "MATCH (h:HyperedgeNode)-[:HYPEREDGE_MEMBER]->(e:EpisodeNode) "
                "WHERE h.id = $id RETURN e.*",
                {"id": hyperedge_id}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            if dicts:
                return [_clean_kuzu_row(r) for r in dicts]
            return []

        return self._execute_with_circuit_breaker(_do_query)

    def get_hyperedges_by_node(self, node_id: str) -> List[dict]:
        """查询包含指定节点的所有超边。"""
        def _do_query():
            result = self.conn.execute(
                "MATCH (h:HyperedgeNode)-[:HYPEREDGE_MEMBER]->(e:EpisodeNode) "
                "WHERE e.id = $id RETURN h.*",
                {"id": node_id}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            if dicts:
                return [_clean_kuzu_row(r) for r in dicts]
            return []

        return self._execute_with_circuit_breaker(_do_query)

    # ─── Session 操作 ─────────────────────────────────────

    def create_session_node(self, session: dict) -> str:
        """创建会话观测节点。"""
        def _do_create():
            self.conn.execute(
                "CREATE (s:SessionNode {id: $id, session_id: $session_id, "
                "created_at: $created_at, metadata: $metadata})",
                session
            )
            return session['id']
        return self._execute_with_circuit_breaker(_do_create)

    def get_or_create_session(self, session_id: str, metadata: str = "{}") -> str:
        """按 session_id 查找或创建 SessionNode。"""
        def _do_get_or_create():
            result = self.conn.execute(
                "MATCH (s:SessionNode) WHERE s.session_id = $sid "
                "RETURN s.id ORDER BY s.created_at DESC LIMIT 1",
                {"sid": session_id}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            if dicts:
                row = _clean_kuzu_row(dicts[0])
                return row.get("id", "")
            # 不存在则创建
            import uuid
            import time
            node_id = str(uuid.uuid4())
            self.conn.execute(
                "CREATE (s:SessionNode {id: $id, session_id: $session_id, "
                "created_at: $created_at, metadata: $metadata})",
                {"id": node_id, "session_id": session_id,
                 "created_at": time.time(), "metadata": metadata}
            )
            return node_id
        return self._execute_with_circuit_breaker(_do_get_or_create)

    def link_session_member(self, session_node_id: str, episode_id: str) -> None:
        """将 EpisodeNode 连接到 SessionNode。"""
        def _do_link():
            self.conn.execute(
                "MATCH (s:SessionNode), (e:EpisodeNode) "
                "WHERE s.id = $sid AND e.id = $eid "
                "CREATE (s)-[:SESSION_MEMBER]->(e)",
                {"sid": session_node_id, "eid": episode_id}
            )
        return self._execute_with_circuit_breaker(_do_link)

    def get_session_memories(self, session_id: str, limit: int = 100) -> List[dict]:
        """查询某会话的所有关联记忆。"""
        def _do_query():
            result = self.conn.execute(
                "MATCH (s:SessionNode)-[:SESSION_MEMBER]->(e:EpisodeNode) "
                "WHERE s.session_id = $sid "
                "RETURN e.id, e.content, e.created_at, e.source "
                "ORDER BY e.created_at DESC LIMIT $limit",
                {"sid": session_id, "limit": limit}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            if dicts:
                return [_clean_kuzu_row(r) for r in dicts]
            return []
        return self._execute_with_circuit_breaker(_do_query)

    # ─── 多模态视觉操作 ──────────────────────────────────

    def create_visual_node(self, visual: dict) -> str:
        """创建视觉节点。"""
        def _do_create():
            self.conn.execute(
                "CREATE (v:VisualNode {id: $id, image_path: $image_path, "
                "caption: $caption, embedding: $embedding, "
                "source: $source, created_at: $created_at})",
                visual
            )
            return visual['id']
        return self._execute_with_circuit_breaker(_do_create)

    def get_visual_nodes(self, limit: int = 50) -> list[dict]:
        """列出所有视觉节点。"""
        def _do_query():
            result = self.conn.execute(
                "MATCH (v:VisualNode) "
                "RETURN v.id, v.image_path, v.caption, "
                "v.source, v.created_at "
                "ORDER BY v.created_at DESC LIMIT $limit",
                {"limit": limit}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            if dicts:
                return [_clean_kuzu_row(r) for r in dicts]
            return []
        return self._execute_with_circuit_breaker(_do_query)

    def get_visual_node(self, visual_id: str) -> Optional[dict]:
        """查询单个视觉节点。"""
        def _do_query():
            result = self.conn.execute(
                "MATCH (v:VisualNode) WHERE v.id = $id "
                "RETURN v.id, v.image_path, v.caption, "
                "v.embedding, v.source, v.created_at",
                {"id": visual_id}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            if dicts:
                return _clean_kuzu_row(dicts[0])
            return None
        return self._execute_with_circuit_breaker(_do_query)

    def execute_cypher(self, query: str, params: dict) -> list[dict]:
        """Execute a raw CYPHER query and return results as list of dicts."""
        try:
            result = self.conn.execute(query, params or {})
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            if dicts:
                return [_clean_kuzu_row(r) for r in dicts]
            return []
        except Exception:
            logger.exception("execute_cypher failed")
            return []

    # ─── 命名空间/会话隔离 ────────────────────────────────────
    
    def ensure_session(self, namespace: str) -> str:
        """确保 SessionNode 存在，返回 session_id。"""
        def _do_ensure():
            session_id = f"ns_{namespace}"
            self.conn.execute(
                "MERGE (s:SessionNode {id: $id}) "
                "ON CREATE SET s.session_id = $ns, "
                "s.created_at = $t, s.metadata = 'namespace'",
                {"id": session_id, "ns": namespace, "t": time.time()}
            )
            return session_id
        return self._execute_with_circuit_breaker(_do_ensure)
    
    def link_to_session(self, namespace: str, episode_id: str) -> None:
        """通过 SESSION_MEMBER 边将 EpisodeNode 关联到命名空间。"""
        def _do_link():
            session_id = f"ns_{namespace}"
            self.conn.execute(
                "MERGE (s:SessionNode {id: $sid}) "
                "MERGE (e:EpisodeNode {id: $eid}) "
                "MERGE (s)-[:SESSION_MEMBER]->(e)",
                {"sid": session_id, "eid": episode_id}
            )
        self._execute_with_circuit_breaker(_do_link)
    
    def delete_namespace(self, namespace: str) -> int:
        """删除命名空间下所有节点 + 关联边，返回删除的 episode 数。"""
        def _do_delete():
            session_id = f"ns_{namespace}"
            # 收集该空间下所有 episode ID
            result = self.conn.execute(
                "MATCH (s:SessionNode {id: $sid})-[:SESSION_MEMBER]->(e:EpisodeNode) "
                "RETURN collect(e.id) AS ids",
                {"sid": session_id}
            )
            dicts = result.get_as_pl().to_dicts()
            ids = dicts[0]["ids"] if dicts and dicts[0].get("ids") else []
            count = len(ids)
            if not ids:
                return 0
            # 分批 DETACH DELETE（自动处理所有入边/出边）
            for i in range(0, len(ids), 50):
                batch = ids[i:i+50]
                params = {"ids": batch}
                self.conn.execute(
                    "MATCH (e:EpisodeNode) WHERE e.id IN $ids "
                    "DETACH DELETE e",
                    params
                )
            # 删除 SessionNode 自身
            self.conn.execute(
                "MATCH (s:SessionNode {id: $sid}) DELETE s",
                {"sid": session_id}
            )
            return count
        return self._execute_with_circuit_breaker(_do_delete)


    # ─── Phase 2: Hebbian 持久化 ─────────────────────────────

    def update_hebbian_connection(self, src_id: str, dst_id: str, weight: float) -> None:
        """更新 Hebbian 连接权重（MERGE 语义）。"""
        def _do():
            self.conn.execute(
                "MATCH (a:EpisodeNode {id: $sid}), (b:EpisodeNode {id: $did}) "
                "MERGE (a)-[h:HEBBIAN_CONNECTION]->(b) "
                "SET h.weight = $w",
                {"sid": src_id, "did": dst_id, "w": weight}
            )
        self._execute_with_circuit_breaker(_do)

    def get_hebbian_connections(self, node_id: str) -> list[dict]:
        """查询一个节点的所有 Hebbian 输出连接。"""
        def _do():
            result = self.conn.execute(
                "MATCH (a:EpisodeNode {id: $id})-[h:HEBBIAN_CONNECTION]->(b:EpisodeNode) "
                "RETURN b.id AS target_id, h.weight AS weight "
                "ORDER BY h.weight DESC",
                {"id": node_id}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            return [_clean_kuzu_row(r) for r in dicts] if dicts else []
        return self._execute_with_circuit_breaker(_do)

    def get_all_hebbian_connections(self, limit: int = 5000) -> dict[str, dict[str, float]]:
        """批量获取所有 Hebbian 连接，格式兼容 SparseHebbianUpdater。"""
        def _do():
            result = self.conn.execute(
                "MATCH (a:EpisodeNode)-[h:HEBBIAN_CONNECTION]->(b:EpisodeNode) "
                "RETURN a.id AS src, b.id AS dst, h.weight AS w "
                "LIMIT $limit",
                {"limit": limit}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts() if rows else []
            conns: dict[str, dict[str, float]] = {}
            for r in dicts:
                r = _clean_kuzu_row(r)
                src, dst, w = r.get("src", ""), r.get("dst", ""), r.get("w", 0.0)
                if src and dst:
                    conns.setdefault(src, {})[dst] = w
            return conns
        return self._execute_with_circuit_breaker(_do)

    # ─── Phase 2: 程序记忆 ───────────────────────────────

    def create_procedural_node(self, node: dict) -> str:
        """创建程序记忆节点。"""
        def _do():
            self.conn.execute(
                "CREATE (p:ProceduralNode {id: $id, pattern_name: $pattern_name, "
                "pattern_type: $pattern_type, trigger_sequence: $trigger_sequence, "
                "action_template: $action_template, confidence: $confidence, "
                "frequency: $frequency, created_at: $created_at, "
                "last_matched_at: $last_matched_at})",
                node
            )
            return node["id"]
        return self._execute_with_circuit_breaker(_do)

    def link_procedural_pattern(self, proc_id: str, episode_id: str, match_count: int = 1) -> None:
        """将模式节点关联到匹配的条。"""
        def _do():
            self.conn.execute(
                "MATCH (p:ProceduralNode), (e:EpisodeNode) "
                "WHERE p.id = $pid AND e.id = $eid "
                "MERGE (p)-[r:PROCEDURAL_PATTERN]->(e) "
                "SET r.match_count = COALESCE(r.match_count, 0) + $mc",
                {"pid": proc_id, "eid": episode_id, "mc": match_count}
            )
        self._execute_with_circuit_breaker(_do)

    def find_procedural_patterns(self, min_confidence: float = 0.3) -> list[dict]:
        """查询置信度以上的程序模式。"""
        def _do():
            result = self.conn.execute(
                "MATCH (p:ProceduralNode) "
                "WHERE p.confidence >= $min_conf "
                "RETURN p.* ORDER BY p.confidence DESC",
                {"min_conf": min_confidence}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            return [_clean_kuzu_row(r) for r in dicts] if dicts else []
        return self._execute_with_circuit_breaker(_do)

    # ─── Phase 2: 概念记忆 ───────────────────────────────

    def create_conceptual_node(self, node: dict) -> str:
        """创建概念记忆节点。"""
        def _do():
            self.conn.execute(
                "CREATE (c:ConceptualNode {id: $id, concept_name: $concept_name, "
                "description: $description, abstraction_level: $abstraction_level, "
                "confidence: $confidence, created_at: $created_at, "
                "source_communities: $source_communities})",
                node
            )
            return node["id"]
        return self._execute_with_circuit_breaker(_do)

    def link_conceptual_framework(self, concept_id: str, community_id: str, weight: float = 1.0) -> None:
        """将概念连接到其来源社区。"""
        def _do():
            self.conn.execute(
                "MATCH (c:ConceptualNode), (cm:CommunityNode) "
                "WHERE c.id = $cid AND cm.id = $cmid "
                "MERGE (c)-[f:CONCEPTUAL_FRAMEWORK]->(cm) "
                "SET f.weight = $w",
                {"cid": concept_id, "cmid": community_id, "w": weight}
            )
        self._execute_with_circuit_breaker(_do)

    def get_concepts_by_level(self, level: str = "high") -> list[dict]:
        """按抽象层级查询概念。"""
        def _do():
            result = self.conn.execute(
                "MATCH (c:ConceptualNode) "
                "WHERE c.abstraction_level = $level "
                "RETURN c.* ORDER BY c.confidence DESC",
                {"level": level}
            )
            rows = result.get_as_pl()
            dicts = rows.to_dicts()
            return [_clean_kuzu_row(r) for r in dicts] if dicts else []
        return self._execute_with_circuit_breaker(_do)


    def close(self) -> None:
        """关闭所有数据库连接"""
        for c in self._connections:
            try:
                c.close()
            except Exception:
                pass
        self._connections.clear()
        if self.db:
            self.db.close()
