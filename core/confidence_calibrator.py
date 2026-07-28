"""
信心校准器 (Confidence Calibrator)
======================================
基于「Manufactured Confidence: How Memory Consolidation Turns Hearsay
into Confident Facts」(arXiv:2606.29279, Jun 2026)。

核心发现：记忆整合过程会反复巩固同一信息，导致信心膨胀——
「听说的」经过几次梦境循环变成「确信的」。

校准策略：
  1. 每次梦境整合后，对信息做 confidence 指数衰减
  2. 跟踪信息源类型：直接观察 > 推理 > 传闻
  3. 对过度巩固的信息标记审查（不直接删除，留人工/LLM判定）
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

@dataclass
class CalibratorConfig:
    # 指数衰减参数
    decay_rate: float = 0.15  # γ — 每次整合信心衰减率
    min_confidence: float = 0.1  # 最低信心值（< 此值标记审查）
    inspect_threshold: float = 0.3  # 低于此值但高于 min → 标记审查

    # 源类型权重（校准基础信心）
    source_weight_direct: float = 1.0  # 直接观察（用户输入/传感器）
    source_weight_inferred: float = 0.7  # 推理/推断
    source_weight_hearsay: float = 0.4  # 传闻/间接

    # 审查水位
    max_consolidations: int = 10  # 超过此次数自动标记审查
    flag_cooldown: float = 3600  # 同一信息两次标记的最小间隔(s)


# ═══════════════════════════════════════════════════════════
# 信心校准器
# ═══════════════════════════════════════════════════════════

@dataclass
class ConsolidationRecord:
    """一条信息的整合历史"""
    content_hash: str
    consolidation_count: int = 0
    last_consolidated: float = 0.0
    source_type: str = "direct"  # direct | inferred | hearsay
    base_confidence: float = 1.0
    flagged: bool = False
    last_flagged: float = 0.0


class ConfidenceCalibrator:
    """信心校准器 — 防止过度巩固导致的信心膨胀"""

    def __init__(self, config: Optional[CalibratorConfig] = None):
        self.config = config or CalibratorConfig()
        self._records: dict[str, ConsolidationRecord] = {}  # content_hash → record

    # ── 核心方法 ──

    def calibrate(self, content: str, confidence: float,
                  source_type: str = "direct") -> tuple[float, bool]:
        """校准一条信息的信心值。
        
        Args:
            content: 信息内容（用于生成 hash）
            confidence: 原始信心值 0~1
            source_type: 'direct' | 'inferred' | 'hearsay'
        
        Returns:
            (calibrated_confidence, flagged)
        """
        h = self._hash(content)
        rec = self._get_or_create(h, source_type, confidence)

        # 复合信心 = 源权重 × 指数衰减
        source_weight = self._source_weight(source_type)
        decay = math.exp(-self.config.decay_rate * rec.consolidation_count)
        calibrated = confidence * source_weight * decay

        # 硬下限
        calibrated = max(0.01, min(1.0, calibrated))

        # 标记审查
        flagged = self._should_flag(rec, calibrated)
        if flagged:
            rec.flagged = True
            rec.last_flagged = time.time()

        return calibrated, flagged

    def record_consolidation(self, content: str, source_type: str = "inferred"):
        """记录一次整合。每次梦境管道 SYNTHESIZE 后调用。"""
        h = self._hash(content)
        rec = self._get_or_create(h, source_type)
        rec.consolidation_count += 1
        rec.last_consolidated = time.time()
        # 源类型仅首次设置，后续不降级
        if rec.source_type != "direct":
            rec.source_type = source_type

    def set_source_type(self, content: str, source_type: str):
        """覆盖源类型（例如 LLM 判定发现某条信息的真正来源）"""
        h = self._hash(content)
        rec = self._get_or_create(h, source_type)
        rec.source_type = source_type

    def flagged_items(self) -> list[dict]:
        """返回所有标记为审查的信息"""
        return [
            {
                "content_hash": r.content_hash,
                "consolidation_count": r.consolidation_count,
                "source_type": r.source_type,
                "calibrated_confidence": self._current_confidence(r),
            }
            for r in self._records.values()
            if r.flagged
        ]

    def state(self) -> dict:
        """校准器状态摘要"""
        total = len(self._records)
        flagged = sum(1 for r in self._records.values() if r.flagged)
        high_consolidation = sum(
            1 for r in self._records.values()
            if r.consolidation_count >= self.config.max_consolidations
        )
        return {
            "total_tracked": total,
            "flagged": flagged,
            "high_consolidation": high_consolidation,
            "flagged_pct": round(flagged / max(total, 1) * 100, 1),
        }

    # ── 内部方法 ──

    def _hash(self, content: str) -> str:
        return hashlib.blake2s(content.encode(), digest_size=8).hexdigest()

    def _get_or_create(self, h: str, source_type: str,
                       confidence: float = 1.0) -> ConsolidationRecord:
        if h not in self._records:
            self._records[h] = ConsolidationRecord(
                content_hash=h,
                consolidation_count=0,
                source_type=source_type,
                base_confidence=confidence,
            )
        return self._records[h]

    def _source_weight(self, source_type: str) -> float:
        return {
            "direct": self.config.source_weight_direct,
            "inferred": self.config.source_weight_inferred,
            "hearsay": self.config.source_weight_hearsay,
        }.get(source_type, self.config.source_weight_inferred)

    def _current_confidence(self, rec: ConsolidationRecord) -> float:
        val = rec.base_confidence * self._source_weight(rec.source_type) * \
            math.exp(-self.config.decay_rate * rec.consolidation_count)
        return max(0.01, min(1.0, val))

    def _should_flag(self, rec: ConsolidationRecord,
                     calibrated: float) -> bool:
        # 检查是否需要审查
        if calibrated < self.config.min_confidence:
            return True
        if calibrated < self.config.inspect_threshold and \
           rec.consolidation_count >= 3:
            return True
        if rec.consolidation_count >= self.config.max_consolidations:
            return True
        # 冷却中的不重复标记
        if rec.flagged and (time.time() - rec.last_flagged <
                            self.config.flag_cooldown):
            return False
        return False


