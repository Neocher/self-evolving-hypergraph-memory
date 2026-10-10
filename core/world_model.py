"""SHM 世界模型最小可行预测器（P1/Phase-1）。

零训练：只读超图已有统计（AtomicFact subject→(predicate, object) 频率），
预测给定 subject 的下一个 (关系, 实体)，并计算预测残差 surprise_Σ ∈ [0,1]。
不更新任何参数、不写库，纯只读统计；α 默认 0 → 与既有惊奇度融合零回归。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SigmaPrediction:
    """PredictSigma 的一次下一步预测结果。"""

    subject: str
    relation: str = ""
    entity: str = ""
    probability: float = 0.0


def fuse_surprise(sim_surprise: float, sigma_surprise: float, alpha: float = 0.0) -> float:
    """α 融合既有 1-max_sim 惊奇度与世界模型预测残差 surprise_Σ。

    α=0 → 恒等于 sim_surprise（零回归）；α∈[0,1] 凸组合。两个输入同方向
    （越新奇/越难预测 → 越大），故融合结果仍 ∈ [0,1]。
    """
    a = max(0.0, min(1.0, float(alpha)))
    return (1.0 - a) * float(sim_surprise) + a * float(sigma_surprise)


class PredictSigma:
    """基于超图统计的零训练预测器。

    surprise_Σ 语义（预测残差）：
    * 无实际观测 → 1 - P(最可能的 (关系,实体) | subject)（模型不确定性）；
    * 有实际观测 → 1 - P(实际 (关系,实体) | subject)（实体未观测时退化为
      关系边缘概率）。
    """

    def __init__(self, store: Any = None, alpha: float = 0.0,
                 max_facts: int = 200, max_entities: int = 200) -> None:
        self.store = store
        self.alpha = float(alpha)
        self.max_facts = int(max_facts)
        self.max_entities = int(max_entities)

    # ── 统计 ─────────────────────────────────────────────────────────
    def _pair_counts(self, subject: str) -> dict[tuple[str, str], int]:
        """subject 的历史 (predicate, object) 频率表。"""
        counts: dict[tuple[str, str], int] = {}
        if self.store is None:
            return counts
        try:
            facts = self.store.get_atomic_facts_by_subject(subject, limit=self.max_facts) or []
        except Exception:
            return counts
        for f in facts:
            if not isinstance(f, dict):
                continue
            pred = str(f.get("predicate", "") or "").strip()
            if not pred:
                continue
            obj = str(f.get("object", "") or "").strip()
            counts[(pred, obj)] = counts.get((pred, obj), 0) + 1
        return counts

    @staticmethod
    def _prob(counts: dict[tuple[str, str], int], key: tuple[str, str]) -> float:
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        return counts.get(key, 0) / total

    # ── 预测 ─────────────────────────────────────────────────────────
    def predict(self, subject: str) -> SigmaPrediction:
        """预测 subject 的下一个 (关系, 实体)（历史频率众数）。"""
        counts = self._pair_counts(subject)
        if not counts:
            return SigmaPrediction(subject=subject)
        (pred, obj), _ = max(counts.items(), key=lambda kv: kv[1])
        return SigmaPrediction(
            subject=subject, relation=pred, entity=obj,
            probability=self._prob(counts, (pred, obj)),
        )

    # ── 预测残差 ─────────────────────────────────────────────────────
    def surprise(self, subject: str, actual_relation: str | None = None,
                 actual_entity: str | None = None) -> float:
        """预测残差 surprise_Σ ∈ [0,1]（越难预测/越新奇 → 越大）。"""
        counts = self._pair_counts(subject)
        total = sum(counts.values())
        if total <= 0:
            return 1.0  # 无历史 → 完全新颖
        if actual_relation is not None:
            rel = actual_relation.strip()
            if actual_entity is not None:
                n = counts.get((rel, actual_entity.strip()), 0)
                if n == 0:  # 实体未观测 → 退化为关系边缘频率
                    n = sum(c for (pr, _), c in counts.items() if pr == rel)
            else:
                n = sum(c for (pr, _), c in counts.items() if pr == rel)
            return 1.0 - (n / total)
        return 1.0 - self.predict(subject).probability

    def distribution_entropy(self, subject: str) -> float:
        """预测分布熵 H(P) = −Σ p·ln p（nats；空分布 → 0.0）。

        对 subject 的历史 (predicate, object) 频率归一化后计算 Shannon 熵，
        度量"下一步预测有多不确定"。零训练只读，不写库。
        """
        counts = self._pair_counts(subject)
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for c in counts.values():
            p = c / total
            if p > 0.0:
                entropy -= p * math.log(p)
        return entropy

    # ── 内容级零训练 subject 识别 ────────────────────────────────────
    def surprise_from_content(self, content: str) -> float:
        """从原始内容识别 subject 后计算 surprise_Σ（写路径融合用）。"""
        subject = self._match_subject(content)
        if not subject:
            return 1.0
        return self.surprise(subject)

    def sigma_from_content(self, content: str) -> tuple[float, float]:
        """从原始内容返回 (surprise_Σ, 预测分布熵)；无 subject 匹配 → (1.0, 0.0)。

        写路径一次匹配 subject 同时取预测残差与分布熵，避免重复实体检索。
        """
        subject = self._match_subject(content)
        if not subject:
            return 1.0, 0.0
        return self.surprise(subject), self.distribution_entropy(subject)

    def _match_subject(self, content: str) -> str:
        """零训练 subject 识别：已知实体名在内容中的最长子串匹配。"""
        text = (content or "").strip()
        if not text or self.store is None:
            return ""
        try:
            entities = self.store.get_entities(limit=self.max_entities) or []
        except Exception:
            return ""
        low = text.lower()
        best = ""
        for e in entities:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name", "") or "").strip()
            if not name:
                continue
            if len(name) > len(best) and name.lower() in low:
                best = name
        return best
