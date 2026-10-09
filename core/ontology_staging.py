"""本体暂存台账 —— 未匹配概念的显式通道
================================================

问题（研究依据）
----------------
`ontology_validator._classify_ontology_type` 与
`ontology_evolution.classify_with_extended` 在无法匹配任何本体类型时
**静默返回 ``generic_fact``**：既没有记录「什么没匹配上」，也没有晋升路径。
后果是概念空间永远追不上真实语料 —— 同一类新概念被反复降级，
且无人知道本体缺什么。这正是 OaK（动态本体内核）与 MOOSEDev
（本体接地记忆）指出的失效点：*语义上看似合理的本体，未必包含
实际决策所需的关系结构*，而静默降级让这个缺口不可观测。

本模块做的事
------------
给未匹配候选一个**可审计的落点 + 晋升路径**，而不是静默降级：

1. **记录**：未匹配文本 → 归一化键去重 → 计数 / 样本 / 证据 episode 链；
2. **度量**：匹配率（unmatched_rate）= 一个廉价的**漂移代理指标**
   （对齐 OntoDrift「度量新数据落在本体之外的比例」的思路）；
3. **晋升**：`promote()` 记录裁决，`build_extended_patch()` 产出可合并进
   ``data/ontology_extended.json`` 的片段 —— **默认不改动活本体**，
   避免自动演化污染全局（沿用 v5.38.0 `merge 原生类型 skip` 的保守原则）。

设计约束（对齐仓库既有约定）
----------------------------
- 纯新增旁挂，**不改主写入语义**；开关 ``SHM_ONTOLOGY_STAGING``
  （"0"/"false"/"" → False，默认 True）。
- 落盘走 ``tempfile.mkstemp`` + ``os.replace`` 原子写（同 ontology_evolution）。
- **fail-open**：热路径调用只许降级不许失败 —— 所有异常内部吞掉并 debug 日志。
- 热路径不落盘：内存累加，**计数阈值（每 10 次变更）或时间阈值（60s）
  任一触发**才 flush —— 时间阈值是必需的，本服务常被 SIGKILL（无优雅
  退出钩子），只按计数触发会让低写入量下的记账在重启时整段丢失。

已知边界（有意为之，非遗漏）
----------------------------
1. **提取不到实体的内容不进台账**：``write_validate`` 在 ``_extract_entities``
   为空时提前 return，根本走不到分类（见 ontology_validator.py:1135）。
   这是**正确**的取舍 —— 会话里绝大多数无专名消息都属于这一类，记进来会把
   台账淹成噪声，反而盖住真正的本体缺口。本通道只度量「**分类器被调用但
   全不命中**」这一口径。
2. **不自动改活本体**：``promote()`` 只记裁决，改本体要走
   ``build_extended_patch()`` 产出片段 + 人工/演化通道合入，避免自动演化
   污染全局（沿用 v5.38.0 ``merge 原生类型 skip`` 的保守原则）。

数据格式（``data/ontology_staging.json``）
-----------------------------------------
``{"version": 1, "updated_at": ts, "counters": {...}, "candidates": {key: {...}}}``
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PATH = os.path.join(_REPO_ROOT, "data", "ontology_staging.json")

_SCHEMA_VERSION = 1
_FLUSH_EVERY = 10          # 每 N 次变更落盘一次（热路径不写盘）
_FLUSH_INTERVAL_S = 60.0   # 或距上次落盘超过 N 秒且有变更 → 落盘（防进程重启丢账）
_TEXT_CLIP = 200           # 样本文本截断长度
_MAX_SAMPLES = 3           # 每个候选保留的样本文本数
_MAX_EVIDENCE = 20         # 每个候选保留的证据 episode id 数
_LIST_LIMIT = 200          # 单次列出上限，防大台账刷屏

_WS_RE = re.compile(r"\s+")


# ─────────────────────────── 开关与工具 ───────────────────────────

def enabled() -> bool:
    """读环境变量 ``SHM_ONTOLOGY_STAGING``："0"/"false"/"" → False，默认 True。"""
    return os.environ.get("SHM_ONTOLOGY_STAGING", "1").strip().lower() not in ("0", "false", "")


def _normalize(text: str) -> str:
    """归一键的输入归一化：小写 + 折叠空白 + 去首尾。"""
    return _WS_RE.sub(" ", (text or "").strip().lower())


def candidate_key(text: str) -> str:
    """未匹配文本 → 稳定去重键（sha1 前 16 位十六进制）。

    同一句被反复降级只累加计数，不会灌爆台账。
    """
    return hashlib.sha1(_normalize(text).encode("utf-8")).hexdigest()[:16]


def _atomic_write(path: str, payload: dict) -> bool:
    """原子写 JSON（tempfile + os.replace），与 ontology_evolution 同约定。"""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as exc:  # 落盘失败绝不影响主流程
        logger.debug("ontology_staging: atomic write failed: %s", exc)
        return False


# ─────────────────────────── 数据模型 ───────────────────────────

@dataclass
class StagedCandidate:
    """一个未匹配概念的台账条目。"""

    key: str
    sample_text: str
    samples: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    evidence: list[str] = field(default_factory=list)   # 证据 episode id（溯源）
    status: str = "pending"                             # pending | promoted | rejected
    promoted_to: Optional[str] = None
    decided_at: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StagedCandidate":
        known = {f for f in cls.__dataclass_fields__}          # 容忍前向字段
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# ─────────────────────────── 台账主体 ───────────────────────────

class OntologyStaging:
    """未匹配概念台账（线程安全；热路径内存累加，批量落盘）。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or _DEFAULT_PATH
        self._lock = threading.Lock()
        self._candidates: dict[str, StagedCandidate] = {}
        self._counters: dict[str, int] = {"matched": 0, "unmatched": 0}
        self._dirty = 0
        self._last_flush = time.time()
        self._loaded = False

    # ---- 读盘 / 落盘 ----

    def _load_locked(self) -> None:
        """惰性加载（持锁调用）。文件缺失/损坏 → 空台账，不抛。"""
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
            for k, v in (raw.get("candidates") or {}).items():
                try:
                    self._candidates[k] = StagedCandidate.from_dict(v)
                except Exception:
                    continue
            for k, v in (raw.get("counters") or {}).items():
                if isinstance(v, int):
                    self._counters[k] = v
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.debug("ontology_staging: load failed (%s) — starting empty", exc)

    def _payload_locked(self) -> dict:
        return {
            "version": _SCHEMA_VERSION,
            "updated_at": time.time(),
            "counters": dict(self._counters),
            "candidates": {k: c.to_dict() for k, c in self._candidates.items()},
        }

    def flush(self) -> bool:
        """强制落盘（持锁快照 → 锁外写盘，避免长临界区）。"""
        with self._lock:
            self._load_locked()
            payload = self._payload_locked()
            self._dirty = 0
        return _atomic_write(self.path, payload)

    def _maybe_flush_locked(self) -> None:
        """计数阈值 **或** 时间阈值触发落盘。

        时间阈值是必需的：只按计数触发时，低写入量下台账长期不落盘，
        进程一旦重启（本服务常被 SIGKILL，无优雅退出钩子）整段记账丢失。
        """
        now = time.time()
        by_count = self._dirty >= _FLUSH_EVERY
        by_time = self._dirty > 0 and (now - self._last_flush) >= _FLUSH_INTERVAL_S
        if not (by_count or by_time):
            return
        self._dirty = 0
        self._last_flush = now
        payload = self._payload_locked()
        # 落盘放锁外（复制线程不安全，用线程池执行避免阻塞热路径过长）
        threading.Thread(target=_atomic_write, args=(self.path, payload), daemon=True).start()

    # ---- 写入侧 ----

    def note_classification(self, matched: bool) -> None:
        """每次分类都计一笔；matched=False 即「落在本体之外」。绝不抛。"""
        if not enabled():
            return
        try:
            with self._lock:
                self._load_locked()
                self._counters["matched" if matched else "unmatched"] += 1
                self._dirty += 1
                self._maybe_flush_locked()
        except Exception as exc:
            logger.debug("ontology_staging: note_classification failed: %s", exc)

    def record(
        self,
        text: str,
        entities: Optional[Sequence[str]] = None,
        episode_id: Optional[str] = None,
    ) -> Optional[str]:
        """记录一个未匹配候选。返回 key；失败或关闭时返回 None。绝不抛。"""
        if not enabled() or not (text or "").strip():
            return None
        try:
            key = candidate_key(text)
            now = time.time()
            clip = (text or "").strip()[:_TEXT_CLIP]
            with self._lock:
                self._load_locked()
                c = self._candidates.get(key)
                if c is None:
                    c = StagedCandidate(key=key, sample_text=clip, first_seen=now)
                    self._candidates[key] = c
                c.count += 1
                c.last_seen = now
                if len(c.samples) < _MAX_SAMPLES and clip not in c.samples:
                    c.samples.append(clip)
                for e in (entities or []):
                    if isinstance(e, str) and e and e not in c.entities and len(c.entities) < 12:
                        c.entities.append(e)
                if episode_id and episode_id not in c.evidence and len(c.evidence) < _MAX_EVIDENCE:
                    c.evidence.append(episode_id)
                self._dirty += 1
                self._maybe_flush_locked()
            return key
        except Exception as exc:
            logger.debug("ontology_staging: record failed: %s", exc)
            return None

    # ---- 读取 / 裁决 ----

    def list_candidates(
        self,
        min_count: int = 1,
        status: Optional[str] = "pending",
        limit: int = _LIST_LIMIT,
    ) -> list[dict]:
        """按出现次数降序列出候选（最常被降级的概念 = 最该补的本体缺口）。"""
        with self._lock:
            self._load_locked()
            items = [
                c.to_dict() for c in self._candidates.values()
                if c.count >= min_count and (status is None or c.status == status)
            ]
        items.sort(key=lambda d: (-d.get("count", 0), d.get("last_seen", 0)))
        return items[: max(1, min(limit, _LIST_LIMIT))]

    def _decide(self, key: str, status: str, ontology_type: Optional[str] = None) -> bool:
        with self._lock:
            self._load_locked()
            c = self._candidates.get(key)
            if c is None:
                return False
            c.status = status
            c.promoted_to = ontology_type
            c.decided_at = time.time()
            self._dirty += 1
        return self.flush()

    def promote(self, key: str, ontology_type: str) -> bool:
        """裁决：该候选晋升为 ``ontology_type``（记录裁决 + 落盘）。

        注意：**不直接改动活本体** —— 用 ``build_extended_patch()`` 取片段，
        再由人/演化通道合入 ``ontology_extended.json``。
        """
        return self._decide(key, "promoted", ontology_type)

    def reject(self, key: str) -> bool:
        """裁决：该候选不构成本体缺口（噪声/一次性）。"""
        return self._decide(key, "rejected")

    def build_extended_patch(self, keys: Optional[Sequence[str]] = None) -> dict:
        """产出可合入 ``ontology_extended.json`` 的片段 ``{type: {conflict_keys: [...]}}``。

        只含已 ``promote`` 的候选；同名类型聚合其样本关键词（去重、限长）。
        合入动作由调用方决定 —— 本模块不写活本体。
        """
        patch: dict[str, dict] = {}
        with self._lock:
            self._load_locked()
            chosen = [
                c for c in self._candidates.values()
                if c.status == "promoted" and c.promoted_to
                and (keys is None or c.key in set(keys))
            ]
        for c in chosen:
            target = c.promoted_to
            if not target:
                continue
            slot = patch.setdefault(target, {"conflict_keys": []})
            for s in [c.sample_text, *c.samples]:
                if s and s not in slot["conflict_keys"] and len(slot["conflict_keys"]) < 32:
                    slot["conflict_keys"].append(s)
        return patch

    def stats(self) -> dict:
        """台账统计 + 漂移代理指标 ``unmatched_rate``。"""
        with self._lock:
            self._load_locked()
            counters = dict(self._counters)
            by_status: dict[str, int] = {}
            for c in self._candidates.values():
                by_status[c.status] = by_status.get(c.status, 0) + 1
            total_candidates = len(self._candidates)
            ranked = sorted(
                ((c.count, c.key, c.sample_text)
                 for c in self._candidates.values() if c.status == "pending"),
                key=lambda t: -t[0],
            )[:5]
            top = [{"key": k, "count": n, "sample_text": s} for n, k, s in ranked]
        total = counters.get("matched", 0) + counters.get("unmatched", 0)
        return {
            "enabled": enabled(),
            "path": self.path,
            "counters": counters,
            "unmatched_rate": round(counters.get("unmatched", 0) / total, 4) if total else 0.0,
            "total_candidates": total_candidates,
            "by_status": by_status,
            "top_pending": top,
        }


# ─────────────────────────── 模块级单例（热路径入口） ───────────────────────────

_STAGING: Optional[OntologyStaging] = None
_SINGLETON_LOCK = threading.Lock()


def get_staging(path: Optional[str] = None) -> OntologyStaging:
    """进程级单例（热路径不重复建对象 / 不重复读盘）。"""
    global _STAGING
    with _SINGLETON_LOCK:
        if _STAGING is None or (path and path != _STAGING.path):
            _STAGING = OntologyStaging(path)
        return _STAGING


def reset_staging(path: Optional[str] = None) -> OntologyStaging:
    """测试/运维用：丢弃单例并返回新实例。"""
    global _STAGING
    with _SINGLETON_LOCK:
        _STAGING = OntologyStaging(path)
        return _STAGING


def note_classification(matched: bool) -> None:
    """热路径钩子：记录一次分类结果。绝不抛。"""
    try:
        get_staging().note_classification(matched)
    except Exception:  # 热路径只许降级
        pass


def record_unmatched(
    text: str,
    entities: Optional[Sequence[str]] = None,
    episode_id: Optional[str] = None,
) -> Optional[str]:
    """热路径钩子：记录未匹配候选。绝不抛。"""
    try:
        return get_staging().record(text, entities, episode_id)
    except Exception:
        return None
