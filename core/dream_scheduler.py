"""
梦境调度器
=========
三种触发模式：
1. 空闲触发 — 对话结束空闲 5 分钟后自动启动
2. 累积触发 — 累积 100 个新节点时触发
3. 显式触发 — 通过 API 调用显式触发

优先级计算:
    P = (community_total_degree / community_node_count)
        × (1 + community_freshness_factor)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TriggerMode(Enum):
    IDLE = "idle"  # 空闲触发
    ACCUMULATED = "accum"  # 累积触发
    EXPLICIT = "explicit"  # 显式触发
    CONFLICT_RESOLUTION = "conflict"  # P2: 矛盾驱动触发


@dataclass
class DreamSchedulerConfig:
    """梦境调度配置"""

    idle_timeout_seconds: int = 300  # 空闲 5 分钟后触发
    accum_threshold: int = 100  # 累积 100 个新节点触发
    min_interval_seconds: int = 60  # 最小间隔（防止频繁触发）
    max_dream_duration_seconds: int = 450  # 单次梦境最长 7.5 分钟
    cpu_affinity_low_priority: bool = True  # 低优先级 CPU 亲和性
    memory_limit_mb: int = 256  # 梦境线程内存限制
    conflict_accum_threshold: int = 5  # P2: 累积 5 个未解决冲突触发矛盾驱动梦境
    # P2-1: 写入压力感知 — 最近 write_pressure_window_seconds 内写请求数
    # 达到 write_pressure_threshold 时推迟自动梦境触发 (消除梦境与批量写竞争超时)
    write_pressure_window_seconds: float = 30.0
    write_pressure_threshold: int = 15


class DreamScheduler:
    """
    梦境调度器。

    监控系统状态，在合适的时机触发梦境整合管道。
    支持三种触发模式，带优先级计算和并发控制。
    """

    def __init__(
        self,
        config: Optional[DreamSchedulerConfig] = None,
        pipeline_fn: Optional[Callable[[], None]] = None,
        state_persist_fn: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.config = config or DreamSchedulerConfig()
        self.pipeline_fn = pipeline_fn
        self._state_persist_fn = state_persist_fn  # 【H4】状态持久化回调（由 app.py 注入）
        self._last_run_time: float = 0.0
        self._is_running: bool = False
        self._current_dream_id: Optional[str] = None  # 【H5】当前梦境 ID（崩溃后可 reconcile）
        self._new_node_count: int = 0
        self._last_activity_time: float = time.time()
        # P2-1: 最近写入时间戳滑动窗口 (写压力感知, 不持久化)
        self._recent_write_times: deque[float] = deque(maxlen=4096)
        self._lock = asyncio.Lock()
        self._dream_run_count: int = 0  # 【FIX】梦境执行计数器
        self._dream_fail_count: int = 0  # 【H3】梦境失败/超时统计
        # 【H6】失败退避: 超时失败后冷却 N 秒再允许触发 (防无限空转循环)
        self._dream_fail_cooldown_until: float = 0.0
        self._dream_fail_cooldown_seconds: int = 1800
        # FAISS 增量更新引用（由 app.py 注入）
        self._faiss_index = None
        self._faiss_id_map: dict = {}
        self._incremental_update_fn = None  # incremental_faiss_update 引用
        # P2: 冲突驱动梦境
        self._unresolved_conflict_count: int = 0
        # 候选目录路径（可由 app.py 注入覆盖）
        self._candidate_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "dream_candidates"
        )

    async def on_conflict_detected(self) -> None:
        """P2: 记录一个新冲突（达到阈值时触发矛盾解析梦境）。"""
        self._unresolved_conflict_count += 1
        # 如果冲突积压已达阈值，触发检查
        if self._unresolved_conflict_count >= self.config.conflict_accum_threshold:
            logger.info("Conflict accum %d >= threshold %d, scheduling conflict resolution",
                        self._unresolved_conflict_count, self.config.conflict_accum_threshold)

    async def on_activity(self) -> None:
        """记录活动时间戳（每次有节点创建/更新时调用）。"""
        self._last_activity_time = time.time()

    async def on_node_created(self) -> None:
        """节点创建通知，增加累积计数。"""
        self._new_node_count += 1
        self._recent_write_times.append(time.time())  # P2-1: 写压力信号

    def _under_write_pressure(self) -> bool:
        """P2-1: 最近 write_pressure_window_seconds 内写请求数是否达到阈值。"""
        now = time.time()
        cutoff = now - self.config.write_pressure_window_seconds
        while self._recent_write_times and self._recent_write_times[0] < cutoff:
            self._recent_write_times.popleft()
        return len(self._recent_write_times) >= self.config.write_pressure_threshold

    async def check_and_trigger(self) -> bool:
        """
        检查是否满足触发条件，是则启动梦境。

        Returns:
            True 如果梦境被触发
        """
        async with self._lock:
            if self._is_running:
                return False
            now = time.time()
            if now - self._last_run_time < self.config.min_interval_seconds:
                return False
            # 【H6】失败退避: 超时失败后冷却期内不触发 (防无限空转)
            if now < self._dream_fail_cooldown_until:
                return False

            idle_time = now - self._last_activity_time
            should_run = False
            trigger_mode: Optional[TriggerMode] = None

            if idle_time > self.config.idle_timeout_seconds:
                should_run = True
                trigger_mode = TriggerMode.IDLE
            elif self._new_node_count >= self.config.accum_threshold:
                should_run = True
                trigger_mode = TriggerMode.ACCUMULATED
            elif self._unresolved_conflict_count >= self.config.conflict_accum_threshold:
                should_run = True
                trigger_mode = TriggerMode.CONFLICT_RESOLUTION

            # 兜底：距离上次梦境超过 6 小时，强制触发一次
            # 【FIX】_last_run_time 初值为 0，须特殊处理首次运行
            FORCED_INTERVAL_HOURS = 6
            if not should_run:
                hours_since_last = ((now - self._last_run_time) / 3600.0
                                    if self._last_run_time > 0 else float('inf'))
                if hours_since_last >= FORCED_INTERVAL_HOURS:
                    should_run = True
                    trigger_mode = TriggerMode.IDLE
                    logger.info("Forced dream trigger: %.1f hours since last run >= %d hours",
                                hours_since_last, FORCED_INTERVAL_HOURS)

            # 候选文件触发：data/dream_candidates/ 目录中候选文件 > 10 时触发
            if not should_run:
                candidate_dir = getattr(self, '_candidate_dir', None)
                try:
                    if os.path.exists(candidate_dir):
                        candidate_files = [f for f in os.listdir(candidate_dir)
                                           if os.path.isfile(os.path.join(candidate_dir, f))]
                        if len(candidate_files) > 10:
                            should_run = True
                            trigger_mode = TriggerMode.ACCUMULATED
                            logger.info(
                                "Candidate trigger: %d candidate files > 10 in %s",
                                len(candidate_files), candidate_dir,
                            )
                except OSError:
                    pass

            # P2-1: 写入压力感知 — 持续写入时推迟梦境 (批处理导入与梦境 LLM
            # 串行调用竞争是写路径偶发超时主因)。推迟不丢候选: 节点累积计数
            # 与候选文件保留, 下次 poll 重新评估。
            if should_run and self._under_write_pressure():
                logger.info(
                    "Dream deferred: %d writes in last %.0fs >= threshold %d "
                    "(write pressure)",
                    len(self._recent_write_times),
                    self.config.write_pressure_window_seconds,
                    self.config.write_pressure_threshold,
                )
                return False

            if should_run:
                self._is_running = True
                self._current_dream_id = str(uuid.uuid4())  # 【H5】追踪本次梦境
                asyncio.create_task(self._run_dream(trigger_mode))
                # 【H5】触发即持久化 is_running=true + dream_id（崩溃后可检测中断梦境）
                self._persist_state()
                return True
        return False

    async def trigger_explicit(self) -> bool:
        """显式触发梦境（通过 API 调用）。"""
        async with self._lock:
            if self._is_running:
                return False
            self._is_running = True
            self._current_dream_id = str(uuid.uuid4())  # 【H5】追踪本次梦境
            asyncio.create_task(self._run_dream(TriggerMode.EXPLICIT))
            # 【H5】显式触发同样持久化运行状态
            self._persist_state()
            return True

    async def _run_dream(self, trigger_mode: Optional[TriggerMode]) -> None:
        """执行梦境管道（内部协程）。
        
        在调用 pipeline_fn 之前，从 GraphLite 拉取节点数据和连接图。
        """
        logger.info("Dream triggered by %s mode", trigger_mode.value if trigger_mode else "unknown")
        try:
            if self.pipeline_fn:
                # 【FIX】从GraphLite获取nodes和connections数据
                nodes = []
                connections = {}
                graphlite_store = getattr(self, '_graphlite_store', None)
                if graphlite_store is not None:
                    try:
                        # 单次获取节点（6844节点，一次查询即可）
                        rows = await asyncio.to_thread(
                            graphlite_store.query_cypher,
                            "MATCH (e:EpisodeNode) "
                            "WHERE (e.archived IS NULL OR e.archived = false) "
                            "RETURN e.* "
                            "ORDER BY e.created_at DESC LIMIT 10000"
                        )
                        if rows:
                            for row in rows:
                                if isinstance(row, dict):
                                    # 【FIX v5.18】GraphLite 返回深层嵌套格式，需 flatten
                                    flat = graphlite_store._flatten_row(row, "e")
                                    if flat:
                                        # 【FIX v5.18】GraphLite 所有属性为字符串，需类型转换
                                        try:
                                            flat["created_at"] = float(flat.get("created_at", 0))
                                        except (ValueError, TypeError):
                                            flat["created_at"] = 0.0
                                        try:
                                            flat["tau_initial"] = float(flat.get("tau_initial", 1.0))
                                        except (ValueError, TypeError):
                                            flat["tau_initial"] = 1.0
                                        nodes.append(flat)
                                    else:
                                        nodes.append(row)
                                elif isinstance(row, (list, tuple)) and len(row) > 0:
                                    nodes.append({
                                        "id": str(row[0]),
                                        "content": str(row[1]) if len(row) > 1 else "",
                                        "created_at": float(row[3]) if len(row) > 3 else 0.0,
                                        "tau_initial": float(row[4]) if len(row) > 4 else 1.0,
                                    })
                        # 获取Hebbian连接
                        edge_rows = await asyncio.to_thread(
                            graphlite_store.query_cypher,
                            "MATCH (a)-[r:HEBBIAN_CONNECTION]->(b) "
                            "RETURN a.id AS src, b.id AS dst, r.weight AS w LIMIT 5000"
                        )
                        if edge_rows:
                            for row in edge_rows:
                                if isinstance(row, dict):
                                    s, d, w = row.get("src", ""), row.get("dst", ""), float(row.get("w", 0))
                                elif isinstance(row, (list, tuple)):
                                    s, d, w = str(row[0]), str(row[1]), float(row[2]) if len(row) > 2 else 0.0
                                else:
                                    continue
                                connections.setdefault(s, {})[d] = w
                        logger.info("Dream sourced data: %d nodes, %d connections", len(nodes), len(connections))
                    except Exception as src_exc:
                        logger.warning("Dream data sourcing failed, running with empty data: %s", src_exc)
                
                # 【FIX】正确传递nodes, connections, trigger_mode, graphlite_store, candidate_store
                candidate_store = getattr(self, '_candidate_store', None)
                graphlite_store = getattr(self, '_graphlite_store', None)
                # 【H3】max_dream_duration 强制：超时中止梦境并计入失败统计
                try:
                    report = await asyncio.wait_for(
                        self.pipeline_fn(
                            nodes, connections,
                            trigger_mode.value if trigger_mode else "idle",
                            graphlite_store=graphlite_store,
                            candidate_store=candidate_store,
                        ),
                        timeout=self.config.max_dream_duration_seconds,
                    )
                except asyncio.TimeoutError:
                    self._dream_fail_count += 1
                    # 【H6】失败退避: 超时失败后 30 分钟冷却, 防 poll+idle 双触发无限循环
                    self._dream_fail_cooldown_until = time.time() + self._dream_fail_cooldown_seconds
                    logger.error(
                        "Dream %s timed out after %d s (max_dream_duration_seconds=%d); counted as failed; "
                        "cooldown %ds until next attempt",
                        self._current_dream_id,
                        self.config.max_dream_duration_seconds,
                        self.config.max_dream_duration_seconds,
                        self._dream_fail_cooldown_seconds,
                    )
                    return

                # FAISS 增量更新：移除 PRUNE/RESOLVE 中删除的节点
                if report and hasattr(report, "pruned_node_ids"):
                    removed = report.pruned_node_ids
                    if removed and self._incremental_update_fn:
                        try:
                            # 为增量更新构建临时依赖对象
                            class _FaissDeps:
                                faiss_index = self._faiss_index
                                faiss_id_map = self._faiss_id_map
                            count = self._incremental_update_fn(
                                _FaissDeps(), removed
                            )
                            if count:
                                logger.info(
                                    "Dream FAISS cleanup: %d vectors removed",
                                    count,
                                )
                        except Exception:
                            logger.exception("Dream FAISS cleanup failed")
            self._new_node_count = 0
            self._last_run_time = time.time()
            self._dream_run_count += 1  # 【FIX】计数
            # P2: 矛盾驱动梦境后重置冲突计数
            if trigger_mode == TriggerMode.CONFLICT_RESOLUTION:
                self._unresolved_conflict_count = 0
            # 【v5.25】auto_apply 已移除：v5.24 起由 app.py _dream_poll_loop 承担
            # （qsubmit 整体闭包入队 + 队列深度守卫）。调度器内不再有 loop 线程
            # 同步写（auto_apply 的 _persist_community_nodes 是几十次 execute_cypher
            # 循环）。梦境完成后候选延迟最多一个 poll interval 应用，可接受。
        except Exception:
            self._dream_fail_count += 1
            logger.exception("Dream pipeline failed")
        finally:
            self._is_running = False
            # 【H9】dream 完成视为活动：刷新活动时间戳, 防 idle 链式触发
            # (启动后无节点活动时 idle 恒 >300s → 每 min_interval 立即触发)
            self._last_activity_time = time.time()
            self._current_dream_id = None  # 【H5】梦境结束，清空运行中标记
            # 【H4】完成/失败/超时后保存最新状态（含更新后的计数），
            # 显式触发路径同样走此保存
            self._persist_state()

    def compute_priority(
        self,
        community_total_degree: float,
        community_node_count: int,
        community_freshness: float,
    ) -> float:
        """
        计算社区梦境优先级。

        P = (total_degree / node_count) × (1 + freshness)

        Args:
            community_total_degree: 社区内节点连接度之和
            community_node_count: 社区内节点数量
            community_freshness: 社区新鲜度 (0~1, 1=全新)

        Returns:
            优先级分数（越高越优先处理）
        """
        if community_node_count == 0:
            return 0.0
        density = community_total_degree / community_node_count
        return density * (1.0 + community_freshness)

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def run_count(self) -> int:
        return self._dream_run_count

    @property
    def last_run_time(self) -> float:
        return self._last_run_time

    def force_stop(self) -> None:
        """强制停止当前梦境（设置为空闲状态）"""
        self._is_running = False
        self._current_dream_id = None  # 【H5】清空运行中标记
        self._persist_state()
        logger.info("Dream pipeline force-stopped")

    @property
    def accumulated_count(self) -> int:
        return self._new_node_count

    # ─── 状态持久化 (P1-3) ──────────────────────────────────

    def _persist_state(self) -> None:
        """【H4】将最新状态写入持久层（回调由 app.py 注入，写 GraphLite SystemNode）。"""
        if self._state_persist_fn is not None:
            try:
                self._state_persist_fn(self.save_state())
            except Exception:
                logger.exception("Dream scheduler state persist failed")

    def reconcile_after_restart(self) -> bool:
        """【H5】重启恢复：若上次状态标记梦境运行中（is_running=true 且无完成记录），
        标记为 interrupted 并允许下次触发。

        Returns:
            True 如果检测到并标记了中断的梦境。
        """
        if self._is_running:
            logger.warning(
                "Dream scheduler: previous dream %s interrupted (no completion record); "
                "marked interrupted, next trigger allowed",
                self._current_dream_id or "unknown",
            )
            self._is_running = False
            self._current_dream_id = None
            self._persist_state()
            return True
        return False

    def save_state(self) -> dict:
        """导出调度器运行时状态（供持久化到 GraphLite SystemNode）。"""
        return {
            "last_run_time": self._last_run_time,
            "dream_run_count": self._dream_run_count,
            "dream_fail_count": self._dream_fail_count,
            "new_node_count": self._new_node_count,
            "unresolved_conflict_count": self._unresolved_conflict_count,
            "last_activity_time": self._last_activity_time,
            "is_running": self._is_running,
            "current_dream_id": self._current_dream_id,
            "saved_at": time.time(),
        }

    def load_state(self, state: dict) -> None:
        """从持久化状态恢复调度器运行时状态。"""
        self._last_run_time = state.get("last_run_time", 0.0)
        self._dream_run_count = state.get("dream_run_count", 0)
        self._dream_fail_count = state.get("dream_fail_count", 0)
        self._new_node_count = state.get("new_node_count", 0)
        self._unresolved_conflict_count = state.get("unresolved_conflict_count", 0)
        self._last_activity_time = state.get("last_activity_time", time.time())
        self._is_running = bool(state.get("is_running", False))
        self._current_dream_id = state.get("current_dream_id") or None
