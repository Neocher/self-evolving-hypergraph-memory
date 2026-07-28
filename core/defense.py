"""
记忆投毒防御系统
================
5 条规则检测并防御记忆投毒攻击。

使用方式:
    engine = MemoryDefenseEngine(config=DefenseConfig(), encoder=svc.encoder)
    verdict, reason = engine.pre_check(content=text, source="agent_x")
    if verdict == MemoryDefenseVerdict.BLOCK:
        ...  # 拒绝写入
    elif verdict == MemoryDefenseVerdict.QUARANTINE:
        ...  # 写入后隔离

默认 silent=True: BLOCK 降级为 QUARANTINE（只隔离不阻断）。
非静默模式: BLOCK 直接阻断写入。
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── 枚举 ──────────────────────────────────────────────────


class MemoryDefenseVerdict(Enum):
    """防御系统判定结果"""
    ALLOW = "allow"
    QUARANTINE = "quarantine"
    BLOCK = "block"


# ─── 配置 ──────────────────────────────────────────────────


@dataclass
class DefenseConfig:
    """记忆投毒防御配置 —— 所有阈值均可调"""

    # 全局开关
    enabled: bool = True
    # 静默模式：True 时 BLOCK 降级为 QUARANTINE，False 时直接 403 阻断
    silent: bool = True

    # R1 — 写入频率尖峰
    max_writes_per_window: int = 20          # 时间窗口内最大写入次数
    write_window_seconds: float = 60.0       # 时间窗口（秒）

    # R2 — 语义漂移
    drift_cosine_threshold: float = 0.65     # 余弦相似度阈值（低于此值视为漂移）
    drift_reference_window: int = 10         # 参考历史窗口数

    # R3 — 实体共现异常
    max_entity_cooccurrence: int = 15        # 单次写入最大实体数

    # R4 — 重复洪泛
    max_repeat_exact: int = 5                # 去重窗口内最大精确重复数
    repeat_dedup_window: float = 300.0       # 去重时间窗口（秒）

    # R5 — 信任衰减
    trust_decay_per_block: float = 0.2       # 每次阻断/隔离信任衰减量
    trust_recovery_writes: int = 20          # 完全恢复需要的正常写入数
    initial_trust: float = 1.0               # 新来源初始信任值
    block_trust_threshold: float = 0.3       # 信任值低于此值时阻断
    quarantine_trust_threshold: float = 0.5  # 信任值低于此值时隔离


# ─── 写入历史（滑动窗口） ──────────────────────────────────


class AgentWriteHistory:
    """按来源维护滑动窗口写入记录。"""

    def __init__(self, window_seconds: float = 300.0):
        self._window = window_seconds
        self._records: dict[str, list[dict]] = defaultdict(list)

    def record(self, source: str, content: str, timestamp: Optional[float] = None) -> None:
        ts = timestamp or time.time()
        self._records[source].append({
            "content": content,
            "timestamp": ts,
        })
        self._trim(source)

    def _trim(self, source: str) -> None:
        """清理窗口外的旧记录。"""
        now = time.time()
        cutoff = now - self._window
        records = self._records[source]
        while records and records[0]["timestamp"] < cutoff:
            records.pop(0)

    def count_in_window(self, source: str, window: float) -> int:
        """统计指定时间窗口内某个来源的写入次数。"""
        now = time.time()
        cutoff = now - window
        return sum(1 for r in self._records.get(source, []) if r["timestamp"] >= cutoff)

    def recent_contents(self, source: str, n: int) -> list[str]:
        """获取某个来源最近 n 条写入内容。"""
        return [r["content"] for r in self._records.get(source, [])[-n:]]

    def clear_source(self, source: str) -> None:
        """清除某个来源的所有记录。"""
        self._records.pop(source, None)

    def all_sources(self) -> list[str]:
        """返回所有有记录的来源。"""
        return list(self._records.keys())


# ─── 工具函数 ──────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度。"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _extract_entities(text: str) -> set[str]:
    """从文本中提取命名实体（大写词组 + 引号内文本）。"""
    entities: set[str] = set()
    # 首字母大写的连续英文词（2-4个连续词）
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', text):
        name = match.group(1).strip()
        if name and len(name) >= 2 and name.lower() not in {
            "this", "that", "the", "what", "when", "where", "which",
            "there", "these", "those", "then", "than", "also", "with", "from",
        }:
            entities.add(name)
    # 大写缩写（2-6个大写字母）
    for match in re.finditer(r'\b([A-Z]{2,6})\b', text):
        name = match.group(1)
        if name not in {"AI", "API", "I", "II", "III", "IV", "VI"}:
            entities.add(name)
    # 引号内的内容
    for match in re.finditer(r'["「『""]([^"「』""]{2,50})["」』""]', text):
        name = match.group(1).strip()
        if name and len(name) >= 2:
            entities.add(name)
    return entities


# ─── 主引擎 ────────────────────────────────────────────────


class MemoryDefenseEngine:
    """
    记忆投毒防御引擎。

    在记忆写入前调用 pre_check() 执行 5 条规则判定，
    返回 Verdict 指导调用方处理。
    """

    def __init__(
        self,
        config: Optional[DefenseConfig] = None,
        encoder: Any = None,
        llm_client: Any = None,
    ):
        """
        Args:
            config: 防御配置（使用默认值当 None）
            encoder: 编码器实例（用于 R2 语义漂移检测，为 None 时尝试 LLM 降级）
            llm_client: LLM 客户端实例（用于 R2 降级编码）
        """
        self.config = config or DefenseConfig()
        self.encoder = encoder
        self.llm_client = llm_client
        self._history = AgentWriteHistory(window_seconds=self.config.write_window_seconds * 2)
        self._trust_scores: dict[str, float] = defaultdict(lambda: self.config.initial_trust)
        self._exact_contents: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._recovery_counter: dict[str, int] = defaultdict(int)

    def pre_check(
        self,
        content: str,
        source: str,
        created_at: Optional[float] = None,
    ) -> tuple[MemoryDefenseVerdict, str]:
        """
        执行 5 条规则检测，返回判定结果和原因。

        Args:
            content: 写入内容
            source: 来源标识
            created_at: 创建时间戳（None 时为当前时间）

        Returns:
            (verdict, reason): 判定结果 + 描述字符串
        """
        ts = created_at or time.time()
        reasons: list[str] = []
        verdict = MemoryDefenseVerdict.ALLOW

        # 记录本次写入（不分 verdict，所有写入都记录）
        self._history.record(source, content, ts)
        self._exact_contents[source].append((content, ts))
        self._trim_exact(source)

        # R1 — 写入频率尖峰
        r1_pass, r1_reason = self._check_r1(source)
        if not r1_pass:
            reasons.append(r1_reason)
            verdict = self._escalate(verdict)

        # R2 — 语义漂移（需要编码器或 LLM）
        r2_pass, r2_reason = self._check_r2(source, content)
        if not r2_pass:
            reasons.append(r2_reason)
            verdict = self._escalate(verdict)

        # R3 — 实体共现异常
        r3_pass, r3_reason = self._check_r3(content)
        if not r3_pass:
            reasons.append(r3_reason)
            verdict = self._escalate(verdict)

        # R4 — 重复洪泛
        r4_pass, r4_reason = self._check_r4(source, content)
        if not r4_pass:
            reasons.append(r4_reason)
            verdict = self._escalate(verdict)

        # R5 — 信任衰减
        r5_pass, r5_reason = self._check_r5(source)
        if not r5_pass:
            reasons.append(r5_reason)
            verdict = self._escalate(verdict, r5_reason)

        # 信任分更新
        if verdict in (MemoryDefenseVerdict.BLOCK, MemoryDefenseVerdict.QUARANTINE):
            self._trust_scores[source] = max(
                0.0,
                self._trust_scores[source] - self.config.trust_decay_per_block,
            )
            self._recovery_counter[source] = 0
            logger.warning(
                "Defense %s: source=%s, reasons=%s",
                verdict.value, source, reasons,
            )
        elif verdict == MemoryDefenseVerdict.ALLOW:
            # 正常写入逐步恢复信任
            self._recovery_counter[source] += 1
            recovery_rate = 1.0 / max(self.config.trust_recovery_writes, 1)
            self._trust_scores[source] = min(
                self.config.initial_trust,
                self._trust_scores[source] + recovery_rate,
            )

        # 静默模式：BLOCK 降级为 QUARANTINE
        if verdict == MemoryDefenseVerdict.BLOCK and self.config.silent:
            verdict = MemoryDefenseVerdict.QUARANTINE
            reasons = [f"[silent] {r}" for r in reasons]

        reason_str = "; ".join(reasons) if reasons else "all rules passed"
        return verdict, reason_str

    # ── Verdict 升级 ─────────────────────────────────────

    @staticmethod
    def _escalate(current: MemoryDefenseVerdict, detail: str = "") -> MemoryDefenseVerdict:
        """ALLOW → QUARANTINE → BLOCK。含 block 关键字则直接 BLOCK。"""
        if "block" in detail.lower():
            return MemoryDefenseVerdict.BLOCK
        if current == MemoryDefenseVerdict.ALLOW:
            return MemoryDefenseVerdict.QUARANTINE
        elif current == MemoryDefenseVerdict.QUARANTINE:
            return MemoryDefenseVerdict.BLOCK
        return current

    # ── R1: 写入频率尖峰 ─────────────────────────────────

    def _check_r1(self, source: str) -> tuple[bool, str]:
        count = self._history.count_in_window(source, self.config.write_window_seconds)
        if count > self.config.max_writes_per_window:
            return False, (
                f"R1: write frequency spike — {count} writes in "
                f"{self.config.write_window_seconds:.0f}s from '{source}' "
                f"(threshold: {self.config.max_writes_per_window})"
            )
        return True, ""

    # ── R2: 语义漂移 ─────────────────────────────────────

    def _check_r2(self, source: str, content: str) -> tuple[bool, str]:
        recent = self._history.recent_contents(source, self.config.drift_reference_window)
        if len(recent) < 3:
            return True, ""  # 历史不足，无法检测漂移

        content_emb = self._get_embedding(content)
        if content_emb is None:
            return True, ""  # 无编码器可用，跳过 R2

        # 取除当前外最近的 N 条作为参考
        ref_contents = recent[:-1]
        ref_embs: list[np.ndarray] = []
        for prev in ref_contents:
            emb = self._get_embedding(prev)
            if emb is not None:
                ref_embs.append(emb)

        if not ref_embs:
            return True, ""

        avg_ref = np.mean(ref_embs, axis=0)
        similarity = _cosine_similarity(content_emb, avg_ref)

        if similarity < self.config.drift_cosine_threshold:
            return False, (
                f"R2: semantic drift — cosine similarity {similarity:.3f} "
                f"vs reference (threshold: {self.config.drift_cosine_threshold}) "
                f"for source '{source}'"
            )
        return True, ""

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取文本嵌入向量。

        优先级: encoder.embed() → cloud embedding API → None（跳过 R2）
        """
        if self.encoder is not None:
            try:
                return self.encoder.embed(text)
            except Exception:
                logger.debug("encoder.embed() failed, trying cloud embedding fallback")

        # 降级：尝试云端 Embedding API（复用 encoder 模块中的函数）
        try:
            from embedding.encoder import _cloud_embed
            result = _cloud_embed([text])
            if result:
                return np.array(result[0], dtype=np.float32)
        except Exception:
            logger.debug("Cloud embedding fallback failed, R2 will be skipped")

        return None

    # ── R3: 实体共现异常 ─────────────────────────────────

    def _check_r3(self, content: str) -> tuple[bool, str]:
        entities = _extract_entities(content)
        if len(entities) > self.config.max_entity_cooccurrence:
            return False, (
                f"R3: entity co-occurrence anomaly — {len(entities)} entities detected "
                f"(threshold: {self.config.max_entity_cooccurrence})"
            )
        return True, ""

    # ── R4: 重复洪泛 ─────────────────────────────────────

    def _check_r4(self, source: str, content: str) -> tuple[bool, str]:
        cutoff = time.time() - self.config.repeat_dedup_window
        recent = [c for c, ts in self._exact_contents[source] if ts >= cutoff]
        repeat_count = sum(1 for c in recent if c == content)
        if repeat_count > self.config.max_repeat_exact:
            return False, (
                f"R4: repeat flooding — {repeat_count} exact duplicates "
                f"in {self.config.repeat_dedup_window:.0f}s from '{source}' "
                f"(threshold: {self.config.max_repeat_exact})"
            )
        return True, ""

    # ── R5: 信任衰减 ─────────────────────────────────────

    def _check_r5(self, source: str) -> tuple[bool, str]:
        trust = self._trust_scores[source]
        if trust < self.config.block_trust_threshold:
            return False, (
                f"R5: trust decay — trust score {trust:.2f} for '{source}' "
                f"below block threshold {self.config.block_trust_threshold}"
            )
        if trust < self.config.quarantine_trust_threshold:
            return False, (
                f"R5: trust decay — trust score {trust:.2f} for '{source}' "
                f"below quarantine threshold {self.config.quarantine_trust_threshold}"
            )
        return True, ""

    # ── 内部辅助 ─────────────────────────────────────────

    def _trim_exact(self, source: str) -> None:
        cutoff = time.time() - self.config.repeat_dedup_window
        records = self._exact_contents[source]
        while records and records[0][1] < cutoff:
            records.pop(0)

    @property
    def history(self) -> AgentWriteHistory:
        return self._history

    @property
    def trust_scores(self) -> dict[str, float]:
        return dict(self._trust_scores)
