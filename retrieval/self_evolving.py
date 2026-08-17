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
  [周期强制 _total_calls % probe_every == 0 | 时间兜底 > probe_interval_s 未评估
   | 即时硬失败 degraded/延迟超时（快照显式传诊断，不经质量过滤）]
  DiagnosisEngine.analyze() → 根因 + 建议配置（生效旋钮）
    ↓
  ConfigEvolver.apply() → 新配置 → 同步 cfg → 验证 → 回滚/保留 → 持久化
    ↓
  梦境侧 retrieval_health_probe() 离线探针 → report_probe() 低召回触发
"""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
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

# 【P2-1】mesa_boost 演化上界：严格 < community_expansion.boost=0.6（query_router.py
# _mesa_synthesis 数学保证「合成节点低于本社区原始成员」）。用 0.59（非 0.6 开区间）
# 避免 float 精度下边界值破坏分数契约；validate 与规则 6 共用此常量保证一致。
_MESA_BOOST_MAX = 0.59


# ═══════════════════════════════════════════════════════════
# 1. 可演化配置空间 (Evolvable Config Space)
# ═══════════════════════════════════════════════════════════

@dataclass
class EvolvableParams:
    """可以从 QueryRouterConfig 和 QueryRouter 中提取的可演化参数

    weight_fusion_* 被 _fuse_results（query_router.py:703-705）消费，保留给
    停滞探索；tau_weight/vector_weight 保留字段（cfg 存在 + 持久化兼容），
    但不再作为演化目标——它们仅被全仓零调用的 hybrid_score（query_router.
    py:1404）消费，生产默认路径不消费（死旋钮，P1-1 修复）。
    失败信号对应的直接旋钮是 top_k_*：top_k_l1 为生产默认 HYPERGRAPH 路径
    旋钮（query_router.py:950），top_k_vector/keyword 为 VECTOR/FUSION/级联
    路径旋钮（query_router.py:893/896/1214/1270/1349）。
    """

    # — 融合权重（被 fusion 消费；规则不再调，保留给停滞探索/未来规则）—
    weight_fusion_vector: float = 0.35
    weight_fusion_bm25: float = 0.40
    weight_fusion_entity: float = 0.25
    # — 混合策略权重 —
    tau_weight: float = 0.4
    vector_weight: float = 0.6
    # — 检索参数 —
    top_k_l1: int = 5  # L1 FAISS 检索 top-K（生产默认 HYPERGRAPH 路径的生效旋钮）
    top_k_fusion: int = 30
    top_k_keyword: int = 20
    top_k_vector: int = 20
    # — BM25 —
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # — MESA 记忆增强检索（v5.49.0）—
    mesa_boost: float = 0.4  # 合成分 = relevance × min(种子分) × mesa_boost

    @classmethod
    def from_config(cls, cfg) -> "EvolvableParams":
        """从 QueryRouterConfig 读取当前生效值（不硬编码覆盖用户配置）。

        cfg 缺字段（如测试 mock）时按默认值兜底，不抛错。
        """
        fields = set(cls.__dataclass_fields__)
        return cls(**{f: getattr(cfg, f) for f in fields if hasattr(cfg, f)})

    def validate(self) -> list[str]:
        errs = []
        for w in ["weight_fusion_vector", "weight_fusion_bm25", "weight_fusion_entity",
                  "tau_weight", "vector_weight"]:
            v = getattr(self, w)
            if not 0.0 <= v <= 1.0:
                errs.append(f"{w}={v} 超出 [0,1]")
        if not 0.0 <= self.mesa_boost <= _MESA_BOOST_MAX:
            errs.append(f"mesa_boost={self.mesa_boost} 超出 [0,{_MESA_BOOST_MAX}]")
        for k in ["top_k_l1", "top_k_fusion", "top_k_vector", "top_k_keyword"]:
            if getattr(self, k) < 1:
                errs.append(f"{k}={getattr(self, k)} < 1")
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
    source: str = "retrieve"  # 快照来源: retrieve=生产检索 / probe=梦境探针合成
    mesa_hit_count: int = 0  # 【v5.49.0 MESA】最终结果中 level=="mesa_synthesis" 条数
    mesa_avg_score: float = 0.0  # 【v5.49.0 MESA】合成节点平均分（score 非原始相关度，命中强度信号）

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
    """收集检索快照（触发判定由 SelfEvolvingRetrieval 周期/硬失败负责）。"""

    def __init__(self, quality_threshold: float = 0.4):
        self.snapshots: deque[RetrievalSnapshot] = deque(maxlen=100)
        # quality_threshold 仅用于 state() 统计与 recent_poor 筛选，
        # 不再作为诊断触发门槛（触发改周期强制 + 即时硬失败）
        self._quality_threshold = quality_threshold

    def log(self, snapshot: RetrievalSnapshot) -> None:
        """记录一次快照（不再返回触发信号）。"""
        self.snapshots.append(snapshot)

    def recent_poor(self, n: int = 10,
                    source: Optional[str] = None) -> list[RetrievalSnapshot]:
        """低质量快照（可限定来源）。

        【P2】source 过滤：probe 合成快照与生产快照分离——probe 触发诊断
        只合并 probe 快照，生产诊断不受合成快照污染（反之亦然）。
        """
        snaps = [s for s in list(self.snapshots) if s.quality() < self._quality_threshold]
        if source is not None:
            snaps = [s for s in snaps if s.source == source]
        return snaps[-n:]

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

    规则目标为生效旋钮（weight_fusion_* 留给停滞探索；tau_weight/vector_weight
    仅被零调用的 hybrid_score 消费，不再作为演化目标）：
      - 结果太少 → 扩大候选集（top_k_l1 + top_k_vector/keyword）
      - 分数低/噪声 → 扩 top_k_vector
      - 内容单一 → 提 top_k_keyword + 收窄 top_k_vector（结果太少时不收窄）
      - 延迟高 → 降 top_k_vector/keyword（结果足够时；不降 top_k_l1，增量合并不覆写他规则）
      - 降级触发 → 扩 top_k_l1
    """

    def __init__(self, damping: float = 0.5):
        self._damping = damping  # 阻尼因子 (0~1)，越小越不敏感

    def diagnose(self, snapshots: list[RetrievalSnapshot],
                 current_params: EvolvableParams,
                 mesa_enabled: bool = False) -> DiagnosisResult:
        if not snapshots:
            return DiagnosisResult("无快照", 0.0, {})

        avg_q = sum(s.quality() for s in snapshots) / len(snapshots)
        avg_n = sum(s.num_results for s in snapshots) / len(snapshots)
        avg_s = sum(s.avg_score for s in snapshots) / len(snapshots)
        avg_d = sum(s.top_distinct for s in snapshots) / len(snapshots)
        avg_lat = sum(s.latency_ms for s in snapshots) / len(snapshots)
        # 【P1-3】规则 6 只基于 source=="retrieve" 快照计算/调整：probe 合成快照
        # 固定 mesa_hit_count=0，与 MESA 检索质量无关，混入会持续误降 mesa_boost。
        retrieve_snaps = [s for s in snapshots
                          if getattr(s, "source", "retrieve") == "retrieve"]
        avg_mesa_hit = sum(getattr(s, "mesa_hit_count", 0) for s in retrieve_snaps) / max(1, len(retrieve_snaps))
        avg_mesa_score = sum(getattr(s, "mesa_avg_score", 0.0) for s in retrieve_snaps) / max(1, len(retrieve_snaps))

        suggested: dict[str, float | int] = {}
        reasons: list[str] = []
        confidence = 0.5  # 基线
        d = self._damping

        # 【P1-1】同旋钮多规则增量合并：每条规则对 top_k_vector/top_k_keyword
        # 产生乘性 delta（+/-），累加后统一应用——后写规则不再覆盖先写规则
        #（规则 4 曾无条件覆写规则 2/3 的改动，导致「诊断理由写扩候选、实际
        # 值反而被降」的矛盾，规则 2/4、3/4 互覆同源）。top_k_l1 用固定 +1
        #（乘性会 int 截断为 0），不参与 delta。
        vec_delta = 0.0  # top_k_vector 增量
        kw_delta = 0.0   # top_k_keyword 增量

        # 规则 1: 结果太少 → 扩大候选集
        # top_k_l1 是生产默认 HYPERGRAPH 路径（search.py/gateway_api.py）的
        # 直接旋钮；固定 +1 而非乘性增幅——5×1.25=6.25 经 int() 截断仍为 5。
        if avg_n < 3:
            suggested["top_k_l1"] = min(100, current_params.top_k_l1 + 1)
            vec_delta += 0.25 * d
            kw_delta += 0.25 * d
            reasons.append(f"结果太少(avg={avg_n:.1f})，扩大候选集")
            confidence = min(0.8, confidence + 0.2 * d)

        # 规则 2: 分数低/噪声 → 扩 top_k_vector（向量通道候选集）
        if avg_s < 0.3:
            vec_delta += 0.25 * d
            reasons.append(f"平均分低(avg={avg_s:.2f})，扩向量候选+{0.25*d:.3f}")
            confidence = min(0.8, confidence + 0.2 * d)

        # 规则 3: 内容单一 → 提 top_k_keyword + 收窄 top_k_vector
        # （结果太少时只扩不窄——P1-2 同原则：扩大优先于多样性收窄）
        if avg_d < 2:
            kw_delta += 0.25 * d
            if avg_n >= 3:
                vec_delta -= 0.125 * d
                reasons.append(f"结果单一(多样={avg_d:.1f})，提关键词+收窄向量")
            else:
                reasons.append(f"结果单一(多样={avg_d:.1f})，提关键词（结果太少不收窄向量）")
            confidence = min(0.75, confidence + 0.15 * d)

        # 规则 4: 延迟高 → 降 top_k_vector/top_k_keyword
        # 【P1-1】以 delta 累加（-0.2*d）而非覆写规则 2/3 的同旋钮改动；
        # 【P1-2】不再降 top_k_l1（生产默认路径旋钮，k=5 已小且 FAISS 非延迟
        # 主因）；且结果太少（avg_n<3）时规则 1 的扩大优先——防「0 结果+慢」
        # 同旋钮互覆（扩大被延迟降覆盖，与「结果太少→扩大候选集」矛盾）
        if avg_lat > 500 and avg_n >= 3:
            vec_delta -= 0.2 * d
            kw_delta -= 0.2 * d
            reasons.append(f"延迟高({avg_lat:.0f}ms)，降候选集")
            confidence = min(0.7, confidence + 0.1 * d)

        # 统一应用 delta：净 delta 合并后可能仍是当前值（int 截断，如规则 2
        # +0.125 与规则 4 -0.1 合并为 +0.025 → 20*1.025=20.5 截断为 20，值不上浮）
        # ——由 apply 空转保护跳过演化（P3-1 注释修正）
        for knob, delta in (("top_k_vector", vec_delta),
                            ("top_k_keyword", kw_delta)):
            if delta:
                cur = getattr(current_params, knob)
                new_val = int(cur * (1.0 + delta))
                # 【P2】方向语义保护：clamp 不得反转变化方向——
                #   下界 1（validate 允许 >=1；cur=3 延迟降 → 2 而非被抬到 5）
                #   上界 max(100, cur+1)（cur 已超 100 时允许继续上浮 → 121 而非压回 100）
                suggested[knob] = min(max(100, cur + 1), max(1, new_val))

        # 规则 5: 降级触发 → 扩 top_k_l1（生产默认 HYPERGRAPH 路径的直接旋钮；
        # 原调 tau_weight 仅被零调用的 hybrid_score 消费，死旋钮，P1-1 修复）
        if any(s.degraded for s in snapshots):
            suggested["top_k_l1"] = min(100, current_params.top_k_l1 + 1)
            reasons.append("降级触发，扩大L1候选集")
            confidence = min(0.85, confidence + 0.2 * d)

        # 规则 6: MESA 合成命中信号（v5.49.0）— 命中多且强（avg_hit≥1 且
        # avg_score≥0.3）→ 升 mesa_boost；零命中（avg_hit==0）→ 降；
        # 中间值/弱命中（0<hit<1 或 avg_score<0.3）→ 维持（不调整）
        # 【P2-1】上界 _MESA_BOOST_MAX（< community boost 0.6），保持合成节点低于
        # 社区成员契约；【P3-1】mesa_enabled=False 跳过（不调 mesa_boost）——MESA
        # 关闭时 avg_mesa_hit 恒 0，否则每轮持续降 mesa_boost 造成参数漂移。
        # 【P3-2】mesa_avg_score 作为命中质量信号消费：命中多但合成分弱
        # （avg_score<0.3，合成节点相关性差）→ 不升（噪声命中不值得提 boost）。
        # 【P1-3】retrieve_snaps 非空才调整——纯 probe 触发（无 retrieve 快照）跳过
        # MESA 调整，探针固定零命中不误降 mesa_boost。
        if mesa_enabled and retrieve_snaps and avg_mesa_hit >= 1 and avg_mesa_score >= 0.3:
            suggested["mesa_boost"] = min(_MESA_BOOST_MAX, round(current_params.mesa_boost + 0.05 * d, 4))
            reasons.append(f"MESA 合成命中多且强(hit={avg_mesa_hit:.1f},score={avg_mesa_score:.2f})，提升 mesa_boost")
            confidence = min(0.7, confidence + 0.1 * d)
        elif mesa_enabled and retrieve_snaps and avg_mesa_hit == 0:
            suggested["mesa_boost"] = max(0.0, round(current_params.mesa_boost - 0.05 * d, 4))
            reasons.append(f"MESA 合成命中少(hit={avg_mesa_hit:.1f})，降低 mesa_boost")
            confidence = min(0.7, confidence + 0.1 * d)

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
    samples: int = 0  # 【P2】质量样本计数：回滚需 min_samples 个样本（防单样本误回滚）


class EvolutionGuard:
    """演化守卫：应用 → 验证 → 回滚 → 停滞探索
    
    阈值:
      - revert_on_regression: 新配置质量低于旧配置超过此比例则回滚
      - explore_on_stagnation: 连续 N 次诊断无变化则小幅探索
    """

    def __init__(self, revert_threshold: float = 0.15,
                 explore_after: int = 6,
                 min_samples: int = 3,
                 initial_params: Optional[EvolvableParams] = None):
        self.params = initial_params or EvolvableParams()
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

        # 空转保护：建议值全部等于当前值（如旋钮已到边界）→ 跳过，
        # 防退化场景（如空库）每次检索都空转一次版本号
        if new_params == self.params:
            return False, "参数无变化，跳过演化"

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
                v.samples += 1
                break
        # 如果没有历史记录，创建初始版本
        if not history:
            self.history.append(ConfigVersion(
                version=0,
                params=deepcopy(self.params),
                applied_at=time.time(),
                avg_quality_after=quality,
                samples=1,
            ))

    def check_revert(self) -> tuple[bool, Optional[str]]:
        """检查是否需要回滚。返回 (是否回滚, 原因)"""
        if self._pending is None or len(self.history) < 2:
            return False, None

        current_v = self._pending
        if current_v.reverted:
            return False, None

        # 【P2】需要足够样本：_min_samples 此前从未生效，单样本误回滚
        #（degraded 触发演化后首次恢复检索仅 1 样本就被判质量下降回滚）
        if current_v.avg_quality_after == 0.0 or current_v.samples < self._min_samples:
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

    def _explore_if_stagnant(self) -> bool:
        """检查停滞探索条件，满足时经 apply() 扰动一个参数。返回是否扰动。

        【P2】不再裸 setattr：走 apply() 保证校验 + 版本号 + history +
        回滚保护（探索改动可被 check_revert 撤销），由调用方持久化。
        """
        if self._pending is None or len(self.history) < 2:
            return False
        if self._pending.reverted:
            return False
        if self._pending.avg_quality_after == 0.0:
            return False

        self._no_change_count += 1
        if self._no_change_count < self._explore_after:
            return False

        # 【P1-1】tau_weight 仅被零调用的 hybrid_score 消费（死旋钮），
        # 不再纳入停滞探索候选；只扰动 fusion 消费的 weight_fusion_*
        param_to_tweak = random.choice([
            "weight_fusion_vector", "weight_fusion_bm25",
            "weight_fusion_entity"
        ])
        current_val = getattr(self.params, param_to_tweak)
        delta = random.uniform(-0.1, 0.1)
        new_val = max(0.0, min(1.0, current_val + delta))
        ok, _msg = self.apply({param_to_tweak: round(new_val, 4)})
        if ok:
            self._no_change_count = 0
            logger.info("停滞探索: %s %.2f→%.2f", param_to_tweak, current_val, new_val)
        return ok

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
                 probe_every: int = 40,  # 每 N 次检索周期强制评估（低流量下调至 40）
                 quality_threshold: float = 0.4,  # 仅 state() 统计用，非触发门槛
                 latency_threshold_ms: float = 500.0,  # 即时硬失败延迟阈值
                 probe_interval_s: float = 6 * 3600.0,  # 时间兜底：超 N 秒未评估强制一次（低流量防饿死）
                 persist_path: Optional[str] = None):
        self._qr = query_router
        self.logger = FailureLogger(quality_threshold=quality_threshold)
        self.diagnoser = DiagnosisEngine()
        # 初始值从 QueryRouterConfig 读（不硬编码覆盖用户配置）
        self.guard = EvolutionGuard(
            initial_params=EvolvableParams.from_config(query_router.config))
        self._total_calls = 0
        self._probe_every = max(1, probe_every)
        self._latency_threshold_ms = latency_threshold_ms
        self._probe_interval_s = float(probe_interval_s)
        self._last_probe_time = time.time()
        self._lock = threading.Lock()
        self._persist_path = persist_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "retrieval_evolved.json",
        )
        # 可插拔向量存储实例（设为 None，由外部共享 Services.faiss_index 代替）
        self._vector_store: Optional[BaseVectorStore] = None

    def retrieve(self, query: str, include_archived: bool = False,
                 session_ts: Optional[float] = None):
        """执行检索 + 质量评估 + 自演化（线程安全）。

        【H1-a】锁粒度收窄：不再用方法级大锁包裹整个 _qr.retrieve
        （FAISS + GraphLite + encoder 100-500ms，全部检索被串行化；
        且 asyncio.wait_for 超时无法取消 to_thread 线程，超时后 zombie
        线程仍持有大锁 → 后续请求阻塞等锁 → 默认池耗尽 → 永久楔死）。
        改为：
         1. 无锁调用底层 _qr.retrieve(query)（只读检索，不触碰共享状态）
         2. 仅在短锁段内更新 _total_calls/logger/guard（共享状态变更段）
        共享 QueryRouterConfig 的 setattr 在 GIL 下原子，检索读操作无需互斥。

        演化触发（不再依赖质量阈值累积）：
          - 周期强制：_total_calls % probe_every == 0 时评估
          - 即时硬失败：degraded（0 结果）或 latency 超阈值

        include_archived: 透传给底层 QueryRouter（默认 False 排除归档节点）。
        session_ts: session 时间锚（P0-2；透传给 QueryRouter，None 回落墙钟）。
        """
        params_before = self.guard.current().snapshot()
        start = time.perf_counter()

        # 无锁执行检索（读操作，可并发执行；演化参数仅在 apply 后同步到 cfg）
        try:
            raw = self._qr.retrieve(query, include_archived=include_archived,
                                    session_ts=session_ts)
        except Exception as e:
            logger.error("检索失败: %s", e)
            return []

        elapsed = (time.perf_counter() - start) * 1000

        # 构建结果快照
        results = raw if isinstance(raw, list) else raw.get("results", [])
        scores = [r.get("score", 0.5) for r in results[:10]] if results else [0.0]
        contents = [r.get("content", "")[:100] for r in results[:10]]
        distinct = len(set(contents))
        # 【v5.49.0 MESA】统计合成节点命中：条数 + 平均分（命中率/强度信号）
        mesa_nodes = [r for r in results if r.get("level") == "mesa_synthesis"]
        mesa_hit_count = len(mesa_nodes)
        mesa_avg_score = (
            sum(float(r.get("score") or 0.0) for r in mesa_nodes) / mesa_hit_count
            if mesa_hit_count else 0.0
        )

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
            mesa_hit_count=mesa_hit_count,
            mesa_avg_score=round(mesa_avg_score, 6),
        )

        # 【H1-a】短锁段：仅保护共享状态变更（_total_calls/logger/guard），
        # 不包裹检索本身 → 并发检索不再串行化；锁持有时间微秒级
        with self._lock:
            self._total_calls += 1
            self.logger.log(snapshot)

            # 报告质量给回滚守卫
            self.guard.report_quality(snapshot.quality())

            # 检查回滚
            reverted, reason = self.guard.check_revert()
            if reverted:
                self._sync_params()
                logger.info("回滚生效，参数已重置")

            # 触发条件：即时硬失败（降级/延迟超时）、周期强制评估、
            # 或时间兜底（低流量下超过 probe_interval_s 未评估 → 强制一次）
            hard_fail = snapshot.degraded or elapsed > self._latency_threshold_ms
            periodic = (self._total_calls % self._probe_every) == 0
            now = time.time()
            stale = (now - self._last_probe_time) >= self._probe_interval_s
            if hard_fail or periodic or stale:
                self._last_probe_time = now
                # 【P1】硬失败快照显式传给诊断（不经 recent_poor 质量过滤）：
                # 延迟>500ms 但结果高质量、degraded 等信号不被质量门槛吞掉
                self._evolve(trigger_snapshot=snapshot if hard_fail else None)

        return raw

    def report_probe(self, recall: float, sample_size: int) -> None:
        """梦境探针信号入口：不经 retrieve()，不污染 _total_calls。

        低召回（< 0.5）→ 记录合成降级快照 → 立即触发演化。
        """
        with self._lock:
            if recall >= 0.5:
                return
            hits = int(round(recall * sample_size))
            snap = RetrievalSnapshot(
                timestamp=time.time(),
                query="<health-probe>",
                params_before=self.guard.current().snapshot(),
                num_results=hits,
                top_scores=[],
                avg_score=recall,
                top_distinct=min(hits, 5),
                latency_ms=0.0,
                degraded=True,
                source="probe",
                mesa_hit_count=0,  # 【P3-1】探针结果不统计 MESA 命中，显式 0
            )
            self.logger.log(snap)
            self._last_probe_time = time.time()
            # 触发快照显式传给诊断：recall 0.16-0.49 时合成快照 quality()≥0.4，
            # 不经 recent_poor 质量过滤会被吞掉（P1 修复）
            self._evolve(trigger_snapshot=snap)

    def _sync_params(self):
        """将演化参数同步到 QueryRouter 的 config（仅演化 apply/回滚后调用）"""
        p = self.guard.current()
        cfg = self._qr.config
        cfg.weight_fusion_vector = p.weight_fusion_vector
        cfg.weight_fusion_bm25 = p.weight_fusion_bm25
        cfg.weight_fusion_entity = p.weight_fusion_entity
        cfg.tau_weight = p.tau_weight
        cfg.vector_weight = p.vector_weight
        cfg.top_k_l1 = p.top_k_l1
        cfg.top_k_fusion = p.top_k_fusion
        cfg.top_k_keyword = p.top_k_keyword
        cfg.top_k_vector = p.top_k_vector
        cfg.bm25_k1 = p.bm25_k1
        cfg.bm25_b = p.bm25_b
        cfg.mesa_boost = p.mesa_boost

    def _evolve(self, trigger_snapshot: Optional[RetrievalSnapshot] = None):
        """执行一轮演化：诊断 → 应用 → 同步 → 持久化

        trigger_snapshot: 触发本次演化的快照（硬失败/探针）。显式传入时
        不经 recent_poor 质量过滤——硬失败信号（延迟高但高质量、探针低召回）
        不被质量门槛吞掉；诊断集按 source 与同源历史低质量快照合并，
        探针合成快照与生产快照互不污染。周期/时间兜底路径（trigger_snapshot
        为 None）只合并 retrieve 源——probe 源只影响 probe 触发路径（P2）。
        """
        if trigger_snapshot is not None:
            hist = [
                s for s in self.logger.recent_poor(10, source=trigger_snapshot.source)
                if s is not trigger_snapshot
            ]
            # 【P1-2】触发快照加权：硬失败触发是当前即时状态，聚合中权重须
            # 显著高于历史——复制份数随历史量增长（≥ 历史 +1 份），零结果
            # 触发 + 多条 num=4 历史时 avg_n 不被历史平均稀释到 ≥3（修复前
            # avg_n=(0+5*4)/6=3.3 → 规则 1 跳过、规则 4 反而降候选集，与
            # 「结果太少→扩大候选集」相反）。
            poor = [trigger_snapshot] * max(3, len(hist) + 1) + hist
        else:
            poor = self.logger.recent_poor(10, source="retrieve")
        if not poor:
            return

        result = self.diagnoser.diagnose(
            poor, self.guard.current(),
            mesa_enabled=bool(getattr(self._qr.config, "mesa_enabled", False)),
        )
        if result.suggested:
            ok, msg = self.guard.apply(result.suggested)
            if ok:
                self._sync_params()
                self.save_state()
                logger.info("演化完成: confidence=%.2f, changes=%d, %s",
                            result.confidence, len(result.suggested), msg)
            else:
                logger.info("演化跳过: %s", msg)
        else:
            logger.info("诊断无建议: %s", result.root_cause)

        # 停滞探索：N 轮无变更 → 小幅扰动（走 apply → 校验/版本/history 同步）
        if self.guard._explore_if_stagnant():
            self._sync_params()
            self.save_state()

    # ─── 持久化（tempfile.mkstemp + os.replace 原子写，同 user_profile 模式）───

    def save_state(self, path: Optional[str] = None) -> bool:
        """原子写持久化 {version, params, history(最近50), applied_at}；失败仅告警。"""
        path = path or self._persist_path
        payload = {
            "version": self.guard._version,
            "params": self.guard.current().snapshot(),
            "history": [
                {"version": v.version, "params": v.params.snapshot(),
                 "applied_at": v.applied_at,
                 "avg_quality_after": v.avg_quality_after, "reverted": v.reverted}
                for v in self.guard.history[-50:]
            ],
            "applied_at": time.time(),
        }
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)
                return True
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning("Retrieval evolution persist failed: %s", e)
            return False

    def load_state(self, path: Optional[str] = None) -> Optional[dict]:
        """读持久化；缺失/损坏/非 dict → None（降级不抛错）。"""
        path = path or self._persist_path
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def restore_state(self, path: Optional[str] = None) -> bool:
        """从持久化恢复 guard.params/version/history，并同步到 cfg。

        启动接线用：无文件/损坏 → False，保持默认（从 config 读的初始值）。
        """
        data = self.load_state(path)
        if not data or not data.get("params"):
            return False
        try:
            params = EvolvableParams(**data["params"])
            # 【P2】读回参数 validate()：非法值拒绝恢复（防损坏/篡改持久化污染生产 cfg）
            verrs = params.validate()
            if verrs:
                logger.warning(
                    "Retrieval evolution restore rejected: invalid params %s", verrs)
                return False
            self.guard.params = params
            self.guard._version = int(data.get("version", 0))
            self.guard.history = [
                ConfigVersion(
                    version=h["version"],
                    params=EvolvableParams(**h["params"]),
                    applied_at=h["applied_at"],
                    avg_quality_after=h.get("avg_quality_after", 0.0),
                    reverted=h.get("reverted", False),
                )
                for h in (data.get("history") or [])[-50:]
            ]
            # 确保首次检索前同步到 cfg
            self._sync_params()
            logger.info("Retrieval evolution restored: v%d from %s",
                        self.guard._version, path or self._persist_path)
            return True
        except Exception as e:
            logger.warning("Retrieval evolution restore failed: %s", e)
            return False

    def state(self) -> dict:
        # 【H1】读侧同样加锁：避免与并发 retrieve 的写操作竞争
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "logger": self.logger.state(),
                "guard": self.guard.state(),
                "params": self.guard.current().snapshot(),
            }
