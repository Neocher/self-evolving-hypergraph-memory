"""SHM 图存储层跨后端共享符号（v6.0.0: GraphLite | OverGraph 双后端并存期）。

GraphLiteStore（graph/graphlite_store.py）与 OverGraphStore
（graph/overgraph_store.py）共用：GQL 字面量 helpers（_gql_value /
_dict_to_gql_values / _dict_to_gql_set_values）、熔断器状态机
（CircuitBreaker / CircuitBreakerOpen / CircuitBreakerState）、检索侧
episode 内容缓存（EpisodeCache）、损坏库自动备份（_backup_corrupt_db）。

后端专属逻辑留在各自 store：GraphLite SDK 异常集 / b64 解码 / 行扁平化
（graphlite_store.py），OverGraph SDK 异常集 / GQL 翻译层（overgraph_store.py）。

熔断器 infra 门控设计（AGENTS.md「SDK 异常类型不匹配 → 熔断器死代码」坑）：
基类 CircuitBreaker.record_failure 用类属性 `_infra_exceptions`（默认仅内置
ConnectionError/TimeoutError，兼容测试 mock）；各后端定义子类覆盖为 SDK 真实
异常集——graphlite_store.GraphLiteCircuitBreaker（graphlite_sdk 异常）、
overgraph_store.OverGraphCircuitBreaker（overgraph.OverGraphError）。
"""
import json
import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from enum import Enum
from typing import Optional, Any

logger = logging.getLogger("shm.graph_common")

# 基类熔断器默认 infra 异常集（仅内置类，兼容测试 mock；生产子类覆盖）。
_INFRA_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
)


def _now() -> float:
    return time.time()


def _backup_corrupt_db(db_path: str) -> None:
    """open 失败时自动备份损坏库，保留崩溃现场供恢复；备份失败仅日志，不吞原始异常。"""
    try:
        if not os.path.isdir(db_path):
            logger.error("Graph store open failed; DB path absent, nothing to back up: %s", db_path)
            return
        backup_path = f"{db_path}.corrupt.{time.strftime('%Y%m%d_%H%M%S_%f')}"
        shutil.copytree(db_path, backup_path)
        logger.error("Graph store open failed; corrupted DB backed up: %s", backup_path)
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


def _dict_to_gql_set_values(d: dict, skip_keys: set = None, alias: str = "e") -> str:
    """Convert Python dict to GQL SET clause (alias.key = value, ...).

    逐字段直接构建 (不复用 split), 值含 ', ' (如 content="a, b") 不会拆坏 SQL。
    """
    skip = skip_keys or set()
    parts = []
    for k, v in d.items():
        if k in skip or v is None:
            continue
        lit = _gql_value(v)
        if lit is not None:
            parts.append(f"{alias}.{k} = {lit}")
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
    - infra 门控：record_failure 只对 `_infra_exceptions` 类属性中的异常计数
      （默认内置 ConnectionError/TimeoutError；各后端子类覆盖为 SDK 真实异常集）。
    """

    _infra_exceptions = _INFRA_EXCEPTIONS

    def __init__(self, config: Optional[Any] = None):
        cfg = config or type("cfg", (), {})()
        self.failure_threshold: float = float(getattr(cfg, "failure_threshold", 0.5))
        self.recovery_timeout: float = float(getattr(cfg, "recovery_timeout", 30.0))
        self.half_open_max_requests: int = int(getattr(cfg, "half_open_max_requests", 1))
        self.window_size: int = int(getattr(cfg, "window_size", 10))
        # 并发访问保护: Store 单例被事件循环 + to_thread + ThreadPool 共享
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

        只对 `_infra_exceptions` 中的基础设施错误计数（各后端子类覆盖为 SDK
        真实异常集）；应用错误（RuntimeError 等）不计数，避免坏查询反复调用
        10 次后污染整个窗口导致全图熔断。
        exc=None 视为显式失败信号，计数。
        折中: SDK 把连接失败与坏 GQL 语法统一包装成查询异常（无子类区分），
        两者都计数——比 P0 前（SDK 异常永远匹配不到内置类 → 熔断器永不
        跳闸的死代码）更好。
        """
        if exc is not None and not isinstance(exc, self._infra_exceptions):
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
