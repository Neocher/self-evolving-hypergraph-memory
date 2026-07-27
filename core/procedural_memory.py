"""
程序记忆引擎（Procedural Memory）
==============================
对应四层记忆架构中的「程序记忆」层。

检测Agent行动中重复出现的模式（pattern），
将高频模式抽象为可复用的「程序模板」，
存储在 RyuGraph 的 ProceduralNode 中。

核心思路：
- Agent 在模拟中重复执行类似的行动序列
- 如果同一模式出现 >= min_occurrences 次，抽象为 ProceduralNode
- 下次遇到类似情境时，匹配的程序可以直接指导行动
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProceduralMemoryConfig:
    """程序记忆配置"""

    min_occurrences: int = 3  # 同一模式出现多少次后抽象为程序
    min_confidence: float = 0.3  # 模式置信度阈值
    window_size: int = 3  # 模式检测的滑动窗口大小
    max_patterns: int = 100  # 最大保留模式数


@dataclass
class ActionEvent:
    """一个行动事件（来自Agent的action日志）"""
    agent_name: str
    action_type: str
    platform: str
    content_snippet: str
    timestamp: float = 0.0


class ProceduralMemoryEngine:
    """
    程序记忆引擎。
    
    实时检测重复行动模式，将高频模式提升为程序记忆。
    """

    def __init__(self, config: Optional[ProceduralMemoryConfig] = None,
                 kuzu_store=None):
        self.config = config or ProceduralMemoryConfig()
        self._kuzu_store = kuzu_store
        # 滑动窗口缓存：{(agent, platform): [ActionEvent, ...]}
        self._windows: dict[tuple[str, str], list[ActionEvent]] = {}
        # 模式计数：{pattern_signature: count}
        self._pattern_counts: dict[str, dict] = {}

    def set_kuzu_store(self, store) -> None:
        self._kuzu_store = store

    def observe(self, event: ActionEvent) -> Optional[dict]:
        """观察一个行动事件，检测模式。
        
        Args:
            event: 行动事件
            
        Returns:
            如果检测到新模式则返回模式信息 dict，否则 None
        """
        key = (event.agent_name, event.platform)
        if key not in self._windows:
            self._windows[key] = []
        
        window = self._windows[key]
        window.append(event)
        if len(window) > self.config.window_size:
            window.pop(0)
        
        # 窗口满时检测模式
        if len(window) == self.config.window_size:
            return self._detect_pattern(window, key)
        
        return None

    def _detect_pattern(self, window: list[ActionEvent],
                        key: tuple[str, str]) -> Optional[dict]:
        """在滑动窗口中检测重复模式。"""
        # 用 action_type 序列作为模式签名
        seq = tuple(e.action_type for e in window)
        
        # 全相同类型才有意义（如连续3次 CREATE_POST）
        if len(set(seq)) != 1:
            return None
        
        pattern_type = seq[0]
        sig = f"{key[0]}:{key[1]}:{pattern_type}"
        
        if sig not in self._pattern_counts:
            self._pattern_counts[sig] = {
                "count": 0,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "sample_content": window[-1].content_snippet[:100],
            }
        
        info = self._pattern_counts[sig]
        info["count"] += 1
        info["last_seen"] = time.time()
        
        # 达到阈值 → 提升为程序记忆
        if info["count"] >= self.config.min_occurrences \
           and info.get("promoted") is None:
            return self._promote_to_procedural(sig, pattern_type, info, key)
        
        return None

    def _promote_to_procedural(self, signature: str, pattern_type: str,
                               info: dict,
                               key: tuple[str, str]) -> dict:
        """将高频模式提升为程序记忆节点。"""
        info["promoted"] = time.time()
        
        pattern_name = f"{key[0]}_{pattern_type}_pattern"
        trigger_seq = json.dumps([pattern_type] * self.config.window_size, ensure_ascii=False)
        
        node = {
            "id": str(uuid.uuid4()),
            "pattern_name": pattern_name,
            "pattern_type": pattern_type,
            "trigger_sequence": trigger_seq,
            "action_template": info.get("sample_content", ""),
            "confidence": min(1.0, info["count"] / (self.config.min_occurrences + 3)),
            "frequency": info["count"],
            "created_at": info["first_seen"],
            "last_matched_at": info["last_seen"],
        }
        
        if self._kuzu_store is not None:
            try:
                node_id = self._kuzu_store.create_procedural_node(node)
                node["id"] = node_id
                logger.info("程序记忆提升: %s (type=%s, freq=%d, conf=%.2f)",
                           pattern_name, pattern_type, info["count"], node["confidence"])
            except Exception as e:
                logger.warning("程序记忆持久化失败: %s", e)
        
        return node

    def query_patterns(self, min_confidence: float = None) -> list[dict]:
        """查询已知程序模式。"""
        min_c = min_confidence or self.config.min_confidence
        if self._kuzu_store is not None:
            try:
                return self._kuzu_store.find_procedural_patterns(min_c)
            except Exception:
                pass
        # 回退到内存模式
        results = []
        for sig, info in self._pattern_counts.items():
            conf = min(1.0, info["count"] / (self.config.min_occurrences + 3))
            if conf >= min_c:
                results.append({
                    "signature": sig,
                    "confidence": conf,
                    "frequency": info["count"],
                })
        return results
