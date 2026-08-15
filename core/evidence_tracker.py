"""
Evidence Tracker — 置信度累积引擎
====================================
追踪同一事实被不同来源确认的次数，在检索时用置信度加分。

原理:
  τ衰减管理「时间老化」
  evidence_count管理「证据强度」
  检索得分 = τ_score × (1 + evidence_boost)

用法:
    tracker = EvidenceTracker(data_dir="/app/data")
    tracker.record("Elon Musk founded SpaceX in 2002.", source="user")
    score = tracker.get_boost("Elon Musk founded SpaceX in 2002.")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# BLAKE3 后备（如果库不可用）
try:
    import blake3
    _has_blake3 = True
except ImportError:
    _has_blake3 = False


def _hash_content(content: str) -> str:
    """对内容做归一化哈希（用于精确/近似去重）"""
    # 归一化：去除多余空格、标点、大小写
    norm = content.lower().strip()
    norm = re.sub(r'[^\w\s\u4e00-\u9fff]', '', norm)
    norm = re.sub(r'\s+', ' ', norm)
    # BLAKE3 或 SHA256
    if _has_blake3:
        return blake3.blake3(norm.encode()).hexdigest()[:16]
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def _extract_factual_core(content: str) -> str:
    """提取事实核心（去除修饰成分），用于近似匹配"""
    text = content.lower().strip()
    # 去除引语标记
    text = re.sub(r'"(?:[^"]*?)"(?:,|\.|$)?', '', text)
    text = re.sub(r'表示|认为|指出|说[：:]\s*', '', text)
    # 取主句（句号/分号分割的第一句）
    text = text.split('.')[0].split(';')[0].split('，')[0]
    return text.strip()


class EvidenceTracker:
    """置信度追踪器"""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.environ.get("SHM_DATA_DIR", "./data")
        self._evidence: Dict[str, Dict[str, Any]] = {}  # hash -> record
        self._source_hash: Dict[str, str] = {}           # source_text_hash -> evidence_hash
        self._dirty = False
        self._load()

    # ─── 持久化 ─────────────────────────────────────────────

    def _load(self) -> None:
        """从磁盘加载证据数据"""
        path = os.path.join(self.data_dir, "evidence_tracker.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self._evidence = data.get("evidence", {})
                self._source_hash = data.get("source_hash", {})
                logger.info("EvidenceTracker: loaded %d evidence records", len(self._evidence))
            except Exception as e:
                logger.warning("EvidenceTracker: load failed (%s), starting fresh", e)

    def _save(self) -> None:
        """标记数据为脏，延迟写入"""
        self._dirty = True

    def flush(self) -> None:
        """立即将脏数据写入磁盘"""
        if not self._dirty:
            return
        path = os.path.join(self.data_dir, "evidence_tracker.json")
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump({
                    "evidence": self._evidence,
                    "source_hash": self._source_hash,
                    "updated_at": time.time(),
                }, f, ensure_ascii=False, indent=2)
            self._dirty = False
            logger.debug("EvidenceTracker: saved %d records", len(self._evidence))
        except Exception as e:
            logger.warning("EvidenceTracker: save failed (%s)", e)

    # ─── 核心操作 ───────────────────────────────────────────

    def record(self, content: str, source: str = "user",
               metadata: Optional[Dict[str, Any]] = None) -> int:
        """记录一条内容，返回累计的 evidence_count

        如果内容与已有记录高度相似，视为同一事实的证据累加。
        """
        content_hash = _hash_content(content)

        # 检查是否有完全相同的来源文本
        if content_hash in self._source_hash:
            evidence_key = self._source_hash[content_hash]
            record = self._evidence.get(evidence_key)
            if record:
                record["count"] += 1
                record["sources"].append(source)
                record["last_seen"] = time.time()
                self._dirty = True
                logger.debug("EvidenceTracker: increment %s → count=%d (exact match)",
                             evidence_key[:8], record["count"])
                return record["count"]

        # 检查是否有相似的事实核心
        factual_core = _extract_factual_core(content)
        for ekey, erecord in self._evidence.items():
            existing_core = erecord.get("factual_core", "")
            if existing_core and self._similarity(factual_core, existing_core) > 0.75:
                # 相似事实→累加
                erecord["count"] += 1
                erecord["sources"].append(source)
                erecord["aliases"].append(content_hash)
                erecord["last_seen"] = time.time()
                self._source_hash[content_hash] = ekey
                self._dirty = True
                logger.debug("EvidenceTracker: increment %s → count=%d (semantic match %.2f)",
                             ekey[:8], erecord["count"],
                             self._similarity(factual_core, existing_core))
                return erecord["count"]

        # 全新事实→创建记录
        evidence_key = f"ev_{content_hash}_{int(time.time())}"
        evidence = {
            "key": evidence_key,
            "factual_core": factual_core,
            "content_sample": content[:200],
            "count": 1,
            "sources": [source],
            "aliases": [content_hash],
            "first_seen": time.time(),
            "last_seen": time.time(),
            "metadata": metadata or {},
        }
        self._evidence[evidence_key] = evidence
        self._source_hash[content_hash] = evidence_key
        self._dirty = True
        logger.debug("EvidenceTracker: new record %s (total now %d)",
                     evidence_key[:8], len(self._evidence))
        return 1

    def get_count(self, content: str) -> int:
        """获取某条内容的累计 evidence_count"""
        content_hash = _hash_content(content)
        if content_hash in self._source_hash:
            ekey = self._source_hash[content_hash]
            return self._evidence.get(ekey, {}).get("count", 1)
        return 1

    def get_boost(self, content: str) -> float:
        """计算置信度加分系数

        boost = log2(evidence_count + 1)
        count=1 → boost=1.0  (单人确认，无加分)
        count=2 → boost=1.58
        count=5 → boost=2.58
        count=10→ boost=3.46 (多人确认，显著加分)
        """
        count = self.get_count(content)
        import math
        return min(5.0, math.log2(count + 1))

    def is_multi_source(self, content: str) -> bool:
        """判断一条内容是否被 ≥2 个不同来源确认（多源交叉验证）。"""
        content_hash = _hash_content(content)
        ekey = self._source_hash.get(content_hash)
        if not ekey:
            return False
        record = self._evidence.get(ekey)
        if not record:
            return False
        return len(set(record.get("sources", []))) >= 2

    # ─── 工具函数 ───────────────────────────────────────────

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """计算两个事实核心的相似度（基于词重叠）"""
        if not a or not b:
            return 0.0
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / max(len(union), 1)

    # ─── 查询 ────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """获取置信度统计"""
        if not self._evidence:
            return {"total_records": 0, "top_evidence": []}

        sorted_by_count = sorted(
            self._evidence.values(),
            key=lambda x: -x["count"]
        )

        result = {
            "total_records": len(self._evidence),
            "total_sources": len(self._source_hash),
            "top_evidence": [
                {
                    "content": e["content_sample"][:80],
                    "count": e["count"],
                    "boost": round(self.get_boost(e["content_sample"]), 2),
                    "sources_agg": len(set(e["sources"])),
                }
                for e in sorted_by_count[:10]
            ],
        }
        self.flush()
        return result
