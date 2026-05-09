"""
社区报告生成
==========
[Harness Fix] 决策标准化：
- 节点数 ≤ 5：模板生成（确定性、快速）
- 节点数 > 5：LLM 增强生成（更抽象、更丰富）

模板方案使用 TF-IDF 关键词 + 代表性节点摘要。
"""

from __future__ import annotations

import logging
from typing import Optional, List

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

_TEMPLATE_THRESHOLD = 5
_MAX_LLM_NODES = 20


class CommunityReportGenerator:
    """
    社区报告生成器。

    根据社区规模自动选择生成策略：
    - 小社区（≤5 节点）：模板生成 — TF-IDF 关键词 + 结构化摘要
    - 大社区（>5 节点）：LLM 增强生成 — 更抽象、更丰富的摘要
    """

    def __init__(self, llm_client=None) -> None:
        """
        Args:
            llm_client: LLM 客户端实例（可选，用于大社区增强模式）。
                        需有 generate(prompt: str) -> str 方法。
        """
        self._llm_client = llm_client

    def generate_report(
        self, nodes: list[dict], use_llm: Optional[bool] = None
    ) -> str:
        """
        生成社区报告。

        Args:
            nodes: 社区节点列表，每个节点至少含 'content' 字段
            use_llm: 是否强制使用 LLM（None=自动判断，≤5 节点用模板，>5 用 LLM）

        Returns:
            社区报告文本
        """
        if not nodes:
            return "Community is empty."

        if use_llm is None:
            use_llm = len(nodes) > _TEMPLATE_THRESHOLD

        if use_llm:
            return self._llm_report(nodes)
        return self._template_report(nodes)

    def _template_report(self, nodes: list[dict]) -> str:
        """
        模板化社区报告生成。

        1. TF-IDF 提取前 10 个关键词
        2. 列出代表性节点的摘要
        3. 统计社区规模
        """
        contents = [node.get("content", "") for node in nodes]
        keywords = self._extract_keywords(contents, max_features=10)

        parts: list[str] = [
            f"Community Size: {len(nodes)} nodes",
        ]
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords)}")
        parts.append("Member Nodes Summary:")

        for node in nodes[:_TEMPLATE_THRESHOLD]:
            content = node.get("content", "")
            parts.append(f"- {content[:100]}")

        return "\n".join(parts)

    def _llm_report(self, nodes: list[dict]) -> str:
        """
        LLM 增强的社区报告生成。

        使用 LLM 生成更抽象的社区摘要，LLM 不可用时回退到模板。
        """
        if self._llm_client is None:
            logger.info("No LLM client available, falling back to template report")
            return self._template_report(nodes)

        contents = [node.get("content", "") for node in nodes]
        keywords = self._extract_keywords(contents, max_features=10)
        keyword_hint = ", ".join(keywords) if keywords else "N/A"

        prompt = (
            f"Generate a concise community summary (max 500 tokens) "
            f"based on the following {len(nodes)} related memory nodes.\n"
            f"Key themes detected: {keyword_hint}\n\n"
            + "\n".join(f"- {c[:200]}" for c in contents[:_MAX_LLM_NODES])
        )

        try:
            response = self._llm_client.generate(prompt)
            return response[:2000]
        except Exception:
            logger.exception("LLM report generation failed, falling back to template")
            return self._template_report(nodes)

    @staticmethod
    def _extract_keywords(
        contents: list[str], max_features: int = 10
    ) -> list[str]:
        """
        使用 TF-IDF 从节点内容中提取关键词。

        Args:
            contents: 文本内容列表
            max_features: 最大关键词数

        Returns:
            按 TF-IDF 权重排序的关键词列表
        """
        non_empty = [c for c in contents if c.strip()]
        if len(non_empty) < 1:
            return []
        if len(non_empty) < 2:
            words = non_empty[0].split()
            return words[:max_features]

        try:
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
                token_pattern=r"(?u)\b\w+\b",
            )
            vectorizer.fit_transform(non_empty)
            feature_names = vectorizer.get_feature_names_out()
            return list(feature_names[:max_features])
        except Exception:
            logger.exception("TF-IDF keyword extraction failed")
            return []
