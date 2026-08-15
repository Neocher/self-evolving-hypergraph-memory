"""
检索自演化 (Retrieval Self-Evolution) 模块
=========================================
借鉴 EvolveMem (arXiv:2605.13941) 的核心思想：
「检索基础设施不应冻结——评分函数、融合策略、生成策略也应随使用而演化」

组件：
  - FailureLogger:  记录失败检索的快照
  - DiagnosisEngine: LLM 分析失败根因 → 提出配置调整
  - ConfigEvolver:   应用调整 + 版本追踪 + 回滚保护
  - SelfEvolvingRetrieval: 包装 QueryRouter.retrieve() 的入口

工作流：
  retrieve(req) → QueryRouter.retrieve()
    ↓ 结果质量评估
  [失败/差] → FailureLogger.snapshot()
    ↓ N 次快照后
  DiagnosisEngine.analyze() → 根因 + 建议配置
    ↓
  ConfigEvolver.apply() → 新配置 → 验证 → 回滚/保留
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import random
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

# 可插拔向量存储工厂
from retrieval.vector_store import VectorStoreFactory, BaseVectorStore
from observability.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. 可演化配置空间 (Evolvable Config Space)
# ═══════════════════════════════════════════════════════════

@dataclass
class EvolvableParams:
    """可以从 QueryRouterConfig 和 QueryRouter 中提取的可演化参数"""

    # — 融合权重 —
    weight_fusion_vector: float = 0.35
    weight_fusion_bm25: float = 0.40
    weight_fusion_entity: float = 0.25
    # — 混合策略权重 —
    tau_weight: float = 0.4
    vector_weight: float = 0.6
    # — 检索参数 —
    top_k_fusion: int = 30
    top_k_keyword: int = 20
    top_k_vector: int = 20
    # — BM25 —
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    def validate(self) -> list[str]:
        errs = []
        for w in ["weight_fusion_vector", "weight_fusion_bm25", "weight_fusion_entity"]:
            v = getattr(self, w)
            if not 0.0 <= v <= 1.0:
                errs.append(f"{w}={v} 超出 [0,1]")
        if self.top_k_fusion < 1:
            errs.append(f"top_k_fusion={self.top_k_fusion} < 1")
        if self.bm25_k1 < 0:
            errs.append(f"bm25_k1={self.bm25_k1} < 0")
        return errs

    def snapshot(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════
# 2. 失败快照 (Failure Snapshot)
# ═══════════════════════════════════════════════════════════

@dataclass
class RetrievalSnapshot:
    """一次检索的快照，用于分析"""
    timestamp: float
    query: str
    params_before: dict  # 当时的 EvolvableParams
    num_results: int
    top_scores: list[float]  # 前 k 个分数
    avg_score: float
    top_distinct: int  # 前 10 个结果的去重内容数（低 → 内容单一）
    latency_ms: float
    degraded: bool
    user_feedback: Optional[float] = None  # 人工/隐式反馈（如有）

    def quality(self) -> float:
        """综合质量评分 0~1"""
        if self.num_results == 0:
            return 0.0
        # 数量分
        n_score = min(1.0, self.num_results / 10)
        # 分数分
        s_score = self.avg_score
        # 多样性分
        d_score = min(1.0, self.top_distinct / 5)
        return 0.4 * n_score + 0.4 * s_score + 0.2 * d_score


# ═══════════════════════════════════════════════════════════
# 3. 失败收集器 (Failure Logger)
# ═══════════════════════════════════════════════════════════

class FailureLogger:
    """收集低质量检索快照，触发演化"""

    def __init__(self, min_snapshots_for_diagnosis: int = 5,
                 quality_threshold: float = 0.4):
        self.snapshots: deque[RetrievalSnapshot] = deque(maxlen=100)
        self._min_snapshots = min_snapshots_for_diagnosis
        self._quality_threshold = quality_threshold

    def log(self, snapshot: RetrievalSnapshot) -> bool:
        """记录一次快照，返回是否需要触发诊断"""
        self.snapshots.append(snapshot)
        # 【H1】快照拷贝遍历：避免并发 retrieve 下 deque.append 后遍历时
        # "deque mutated during iteration" RuntimeError
        poor = [s for s in list(self.snapshots) if s.quality() < self._quality_threshold]
        if len(poor) >= self._min_snapshots:
            logger.warning("触发诊断: %d 次低质量检索 (共 %d 条快照)",
                           len(poor), len(self.snapshots))
            return True
        return False

    def recent_poor(self, n: int = 5) -> list[RetrievalSnapshot]:
        return [s for s in list(self.snapshots) if s.quality() < self._quality_threshold][-n:]

    def state(self) -> dict:
        # 【H1】快照拷贝遍历：避免并发修改下迭代 RuntimeError
        snapshots = list(self.snapshots)
        good = sum(1 for s in snapshots if s.quality() >= self._quality_threshold)
        poor = len(snapshots) - good
        return {"total": len(snapshots), "good": good, "poor": poor,
                "avg_quality": (sum(s.quality() for s in snapshots) /
                                max(1, len(snapshots)))}


# ═══════════════════════════════════════════════════════════
# 4. 诊断引擎 (Diagnosis Engine)
# ═══════════════════════════════════════════════════════════

class DiagnosisResult:
    """诊断结果：根因 + 建议配置变化"""

    def __init__(self, root_cause: str, confidence: float,
                 suggested: dict, reasoning: str = ""):
        self.root_cause = root_cause
        self.confidence = confidence  # 0~1
        self.suggested = suggested  # {param: new_value}
        self.reasoning = reasoning

    def __repr__(self):
        return (f"Diagnosis(confidence={self.confidence:.2f}, "
                f"cause={self.root_cause[:40]}, "
                f"changes={len(self.suggested)} params)")


class DiagnosisEngine:
    """基于规则的诊断引擎（轻量级，无需 LLM 调用）
    
    规则示例：
      - 结果太少 + 分数低 → 提权向量权重 / 降低阈值
      - 高分数但内容单一 → 提权 BM25 增加多样性
      - 延迟高 → 降低 top_k
      - 退化模式触发 → 提权实体匹配精确度
    """

    def __init__(self, damping: float = 0.5):
        self._damping = damping  # 阻尼因子 (0~1)，越小越不敏感

    def diagnose(self, snapshots: list[RetrievalSnapshot],
                 current_params: EvolvableParams) -> DiagnosisResult:
        if not snapshots:
            return DiagnosisResult("无快照", 0.0, {})

        avg_q = sum(s.quality() for s in snapshots) / len(snapshots)
        avg_n = sum(s.num_results for s in snapshots) / len(snapshots)
        avg_s = sum(s.avg_score for s in snapshots) / len(snapshots)
        avg_d = sum(s.top_distinct for s in snapshots) / len(snapshots)
        latencies = [s.latency_ms for s in snapshots]
        avg_lat = sum(latencies) / len(latencies)

        suggested = {}
        reasons = []
        bm25_delta = 0.0
        entity_delta = 0.0
        vector_delta = 0.0
        confidence = 0.5  # 基线

        d = self._damping

        # 规则 1: 结果太少 → 提权向量
        if avg_n < 3:
            vector_delta += 0.1 * d
            reasons.append(f"结果太少(avg={avg_n:.1f})，提权向量+{0.1*d:.3f}")
            confidence = min(0.8, confidence + 0.2 * d)

        # 规则 2: 分数低 → 降 BM25（短查询质量差），提权实体匹配
        if avg_s < 0.3:
            bm25_delta -= 0.05 * d
            entity_delta += 0.05 * d
            reasons.append(f"平均分低(avg={avg_s:.2f})，降BM25-{0.05*d:.3f}，提权实体+{0.05*d:.3f}")
            confidence = min(0.8, confidence + 0.2 * d)

        # 规则 3: 内容单一 → 提权 BM25 增加多样性
        if avg_d < 2:
            bm25_delta += 0.10 * d
            reasons.append(f"结果单一(多样={avg_d:.1f})，提权BM25+{0.10*d:.3f}")
            confidence = min(0.75, confidence + 0.15 * d)

        # 合并 delta → suggested（单次演化只改一个维度，防振荡）
        changes = []
        if bm25_delta != 0:
            changes.append(("weight_fusion_bm25", bm25_delta))
        if entity_delta != 0:
            changes.append(("weight_fusion_entity", entity_delta))
        if vector_delta != 0:
            changes.append(("weight_fusion_vector", vector_delta))

        if changes:
            # 只选 delta 绝对值最大的维度修改，单次只改一个
            changes.sort(key=lambda x: abs(x[1]), reverse=True)
            chosen_param, chosen_delta = changes[0]
            suggested[chosen_param] = max(0.0, min(1.0,
                getattr(current_params, chosen_param) + chosen_delta))

        # 规则 4: 延迟高 → 降低 top_k
        if avg_lat > 500:
            new_topk = max(5, int(current_params.top_k_fusion * (1.0 - 0.2 * d)))
            suggested["top_k_fusion"] = new_topk
            reasons.append(f"延迟高({avg_lat:.0f}ms)，降top_k {current_params.top_k_fusion}→{new_topk}")
            confidence = min(0.7, confidence + 0.1 * d)

        # 规则 5: 降级触发 → 提权精确匹配
        if any(s.degraded for s in snapshots):
            suggested["weight_fusion_entity"] = min(
                0.7, current_params.weight_fusion_entity + 0.1 * d)
            suggested["weight_fusion_vector"] = min(
                0.5, current_params.weight_fusion_vector + 0.05 * d)
            reasons.append("降级触发，提权实体+向量")
            confidence = min(0.85, confidence + 0.2 * d)

        if not suggested:
            return DiagnosisResult("检索正常，无需调整", 0.3, {})

        return DiagnosisResult(
            root_cause="; ".join(reasons),
            confidence=round(confidence, 2),
            suggested=suggested,
            reasoning=f"质量=({avg_q:.2f}), 数量=({avg_n:.1f}), 分数=({avg_s:.2f}), 多样=({avg_d:.1f}), 延迟=({avg_lat:.0f}ms)"
        )


# ═══════════════════════════════════════════════════════════
# 5. 配置演化和回滚保护 (Evolution Guard)
# ═══════════════════════════════════════════════════════════

@dataclass
class ConfigVersion:
    version: int
    params: EvolvableParams
    applied_at: float
    avg_quality_after: float = 0.0
    reverted: bool = False


class EvolutionGuard:
    """演化守卫：应用 → 验证 → 回滚 → 停滞探索
    
    阈值:
      - revert_on_regression: 新配置质量低于旧配置超过此比例则回滚
      - explore_on_stagnation: 连续 N 次诊断无变化则小幅探索
    """

    def __init__(self, revert_threshold: float = 0.15,
                 explore_after: int = 6,
                 min_samples: int = 3):
        self.params = EvolvableParams()
        self.history: list[ConfigVersion] = []
        self._version = 0
        self._revert_threshold = revert_threshold
        self._explore_after = explore_after
        self._min_samples = min_samples
        self._no_change_count = 0
        self._pending: Optional[ConfigVersion] = None  # 待验证的版本

    def current(self) -> EvolvableParams:
        return self.params

    def apply(self, suggested: dict) -> tuple[bool, str]:
        """应用配置调整，返回 (是否应用, 消息)

        版本号使用单调递增 int，添加 reset 保护：
        当 history 超过 100 条时强制清理旧记录并重置版本标记。
        """
        errs = []
        new_params = deepcopy(self.params)
        for k, v in suggested.items():
            if hasattr(new_params, k):
                setattr(new_params, k, v)
            else:
                errs.append(f"未知参数: {k}")

        if errs:
            return False, f"未知参数: {', '.join(errs)}"

        verrs = new_params.validate()
        if verrs:
            return False, f"校验失败: {', '.join(verrs)}"

        # 版本保护：history 超过 100 条时归档清理
        if len(self.history) >= 100:
            self.history = self.history[-50:]
            logger.info("Evolution history trimmed to last 50 entries")

        self._version += 1
        self._no_change_count = 0
        old_params = deepcopy(self.params)
        self.params = new_params
        self._pending = ConfigVersion(
            version=self._version,
            params=deepcopy(new_params),
            applied_at=time.time()
        )
        self.history.append(self._pending)

        changes = {k: f"{getattr(old_params, k)}→{v}"
                   for k, v in suggested.items() if hasattr(old_params, k)}
        logger.info("演化 #%d: %s", self._version, changes)
        return True, f"应用演化 #{self._version}: {changes}"

    def report_quality(self, quality: float):
        """报告当前配置下的检索质量，用于验证"""
        # 【H1】快照拷贝遍历：guard.history 可能在并发线程被 append/改写，
        # 直接 reversed(self.history) 会触发 "list changed size during iteration"
        history = list(self.history)
        for v in reversed(history):
            if not v.reverted:
                v.avg_quality_after = (
                    (v.avg_quality_after * 0.7) + (quality * 0.3)
                )
                break
        # 如果没有历史记录，创建初始版本
        if not history:
            self.history.append(ConfigVersion(
                version=0,
                params=deepcopy(self.params),
                applied_at=time.time(),
                avg_quality_after=quality,
            ))

    def check_revert(self) -> tuple[bool, Optional[str]]:
        """检查是否需要回滚。返回 (是否回滚, 原因)"""
        if self._pending is None or len(self.history) < 2:
            return False, None

        current_v = self._pending
        if current_v.reverted:
            return False, None

        # 需要足够样本
        if current_v.avg_quality_after == 0.0:
            return False, None

        # 找上一个非回滚版本的质量
        prev_quality = None
        for v in reversed(self.history[:-1]):
            if not v.reverted and v.avg_quality_after > 0:
                prev_quality = v.avg_quality_after
                break

        if prev_quality is not None and current_v.avg_quality_after > 0:
            drop = (prev_quality - current_v.avg_quality_after) / max(prev_quality, 0.001)
            if drop > self._revert_threshold:
                # 回滚到上一个版本（排除当前版本 self.history[:-1]）
                prev = [v for v in reversed(self.history[:-1]) if not v.reverted]
                if prev:
                    self.params = deepcopy(prev[0].params)
                current_v.reverted = True
                self._pending = None
                logger.warning("回滚 #%d: 质量下降 %.1f%% (阈值 %.0f%%)",
                               current_v.version, drop * 100,
                               self._revert_threshold * 100)
                return True, f"质量下降 {drop:.0%} > {self._revert_threshold:.0%}，回滚"

        return False, None

    def _explore_if_stagnant(self):
        """检查停滞探索条件，满足时随机扰动一个参数"""
        if self._pending is None or len(self.history) < 2:
            return
        if self._pending.reverted:
            return
        if self._pending.avg_quality_after == 0.0:
            return

        self._no_change_count += 1
        if self._no_change_count >= self._explore_after:
            self._no_change_count = 0
            param_to_tweak = random.choice([
                "weight_fusion_vector", "weight_fusion_bm25",
                "weight_fusion_entity", "tau_weight"
            ])
            current_val = getattr(self.params, param_to_tweak)
            delta = random.uniform(-0.1, 0.1)
            new_val = max(0.0, min(1.0, current_val + delta))
            setattr(self.params, param_to_tweak, new_val)
            logger.info("停滞探索: %s %.2f→%.2f", param_to_tweak, current_val, new_val)

    def state(self) -> dict:
        return {
            "version": self._version,
            "params": self.params.snapshot(),
            "history": [
                {"ver": v.version, "quality": round(v.avg_quality_after, 3),
                 "reverted": v.reverted, "at": time.strftime("%H:%M:%S",
                 time.localtime(v.applied_at))}
                for v in self.history[-10:]
            ],
            "no_change_count": self._no_change_count,
            "pending": self._pending is not None,
        }


# ═══════════════════════════════════════════════════════════
# 6. 综合入口 (SelfEvolvingRetrieval)
# ═══════════════════════════════════════════════════════════

class SelfEvolvingRetrieval:
    """包装 QueryRouter 的自演化检索层
    
    使用方式：
      router = QueryRouter(...)
      se = SelfEvolvingRetrieval(router)
      result = se.retrieve(query)  # 自动日志 + 演化
    """

    def __init__(self, query_router,
                 evolve_interval: int = 10,  # 每 N 次低质量触发诊断
                 quality_threshold: float = 0.4):
        self._qr = query_router
        self.logger = FailureLogger(
            min_snapshots_for_diagnosis=evolve_interval,
            quality_threshold=quality_threshold)
        self.diagnoser = DiagnosisEngine()
        self.guard = EvolutionGuard()
        self._total_calls = 0
        self._last_diagnosis = 0.0
        self._lock = threading.Lock()
        # 可插拔向量存储实例（设为 None，由外部共享 Services.faiss_index 代替）
        self._vector_store: Optional[BaseVectorStore] = None

    def retrieve(self, query: str):
        """执行检索 + 质量评估 + 自演化（线程安全）。

        【H1-a】锁粒度收窄：不再用方法级大锁包裹整个 _qr.retrieve
        （FAISS + GraphLite + encoder 100-500ms，全部检索被串行化；
        且 asyncio.wait_for 超时无法取消 to_thread 线程，超时后 zombie
        线程仍持有大锁 → 后续请求阻塞等锁 → 默认池耗尽 → 永久楔死）。
        改为：
          1. 无锁调用底层 _qr.retrieve(query)（只读检索，不触碰共享状态）
          2. 仅在短锁段内更新 _total_calls/logger/guard（共享状态变更段）
        共享 QueryRouterConfig 的 setattr 在 GIL 下原子，检索读操作无需互斥。
        """
        params_before = self.guard.current().snapshot()
        start = time.perf_counter()

        # 应用当前演化参数（GIL 下 setattr 原子；guard 变更在下方短锁段内完成）
        self._sync_params()

        # 无锁执行检索（读操作，可并发执行，不再串行化）
        try:
            raw = self._qr.retrieve(query, include_archived=False)
        except Exception as e:
            logger.error("检索失败: %s", e)
            return []

        elapsed = (time.perf_counter() - start) * 1000

        # 构建结果快照
        results = raw if isinstance(raw, list) else raw.get("results", [])
        scores = [r.get("score", 0.5) for r in results[:10]] if results else [0.0]
        contents = [r.get("content", "")[:100] for r in results[:10]]
        distinct = len(set(contents))

        snapshot = RetrievalSnapshot(
            timestamp=time.time(),
            query=query,
            params_before=params_before,
            num_results=len(results),
            top_scores=scores[:5],
            avg_score=sum(scores) / max(1, len(scores)),
            top_distinct=distinct,
            latency_ms=elapsed,
            degraded=len(results) == 0,
        )

        # 【H1-a】短锁段：仅保护共享状态变更（_total_calls/logger/guard），
        # 不包裹检索本身 → 并发检索不再串行化；锁持有时间微秒级
        with self._lock:
            # 日志 + 演化触发
            self._total_calls += 1
            needs_diagnosis = self.logger.log(snapshot)

            # 报告质量给回滚守卫
            self.guard.report_quality(snapshot.quality())

            # 检查回滚
            reverted, reason = self.guard.check_revert()
            if reverted:
                self._sync_params()
                logger.info("回滚生效，参数已重置")

            # 仅在诊断周期检查停滞，而非每次 retrieve
            if needs_diagnosis:
                self.guard._explore_if_stagnant()

            # 触发诊断
            if needs_diagnosis and (time.time() - self._last_diagnosis > 60):
                self._evolve()

        # 返回结果 + 演化元数据
        if isinstance(raw, dict):
            raw["_evolved"] = {
                "config": params_before,
                "snapshot_quality": round(snapshot.quality(), 3),
            }
            return raw
        return raw

    def _sync_params(self):
        """将演化参数同步到 QueryRouter 的 config"""
        p = self.guard.current()
        cfg = self._qr.config
        cfg.weight_fusion_vector = p.weight_fusion_vector
        cfg.weight_fusion_bm25 = p.weight_fusion_bm25
        cfg.weight_fusion_entity = p.weight_fusion_entity
        cfg.tau_weight = p.tau_weight
        cfg.vector_weight = p.vector_weight
        cfg.top_k_fusion = p.top_k_fusion
        cfg.top_k_keyword = p.top_k_keyword
        cfg.top_k_vector = p.top_k_vector
        cfg.bm25_k1 = p.bm25_k1
        cfg.bm25_b = p.bm25_b

    def _evolve(self):
        """执行一轮演化：诊断 → 应用 → 记录"""
        self._last_diagnosis = time.time()
        poor = self.logger.recent_poor(10)
        if not poor:
            return

        result = self.diagnoser.diagnose(poor, self.guard.current())
        if not result.suggested:
            logger.info("诊断无建议: %s", result.root_cause)
            return

        ok, msg = self.guard.apply(result.suggested)
        if ok:
            self._sync_params()
            logger.info("演化完成: confidence=%.2f, changes=%d",
                        result.confidence, len(result.suggested))

    def state(self) -> dict:
        # 【H1】读侧同样加锁：避免与并发 retrieve 的写操作竞争
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "logger": self.logger.state(),
                "guard": self.guard.state(),
                "params": self.guard.current().snapshot(),
            }
