"""
LLM 客户端
==========
为 SHM 梦境管道提供 LLM 调用能力。
通过 OpenAI 兼容 API 调用外部模型，用于语义摘要、模式提取等任务。

依赖: httpx (已在 venv 中)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "deepseek-v4-flash"
_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_TIMEOUT = 30.0
_MAX_RETRIES = 2


class LLMClient:
    """轻量 LLM 客户端，支持 OpenAI 兼容 API。

    支持从环境变量读取 API Key，也可在构造时传入。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # API Key 优先级：构造参数 > 环境变量
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.warning("LLMClient: No API key found (set DEEPSEEK_API_KEY or OPENAI_API_KEY)")

        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        """调用 chat completion API。

        Args:
            messages: OpenAI 格式的消息列表
            temperature: 温度参数（语义摘要用 0.3 保持稳定）
            max_tokens: 最大输出 token
            response_format: 可选，如 {"type": "json_object"}

        Returns:
            模型回复文本，或 None（失败时）
        """
        if not self.api_key:
            logger.error("LLMClient: Cannot call API — no API key configured")
            return None

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        last_error = None
        for attempt in range(1 + _MAX_RETRIES):
            try:
                resp = self._client.post("/chat/completions", json=body)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                if content:
                    return content.strip()
                logger.warning("LLMClient: empty response (attempt %d)", attempt + 1)
                return None
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                logger.warning("LLMClient API error (attempt %d): %s", attempt + 1, last_error)
                if e.response.status_code in (401, 403):
                    break  # 认证错误不重试
            except httpx.TimeoutException:
                last_error = f"timeout after {self.timeout}s"
                logger.warning("LLMClient timeout (attempt %d)", attempt + 1)
            except Exception as e:
                last_error = str(e)
                logger.warning("LLMClient error (attempt %d): %s", attempt + 1, last_error)

        logger.error("LLMClient: all %d attempts failed: %s", 1 + _MAX_RETRIES, last_error)
        return None

    def summarize_community(
        self,
        node_contents: list[str],
        max_nodes: int = 20,
    ) -> dict[str, Any]:
        """为社区节点生成语义摘要。

        Args:
            node_contents: 社区内节点内容列表
            max_nodes: 传给 LLM 的最大节点数

        Returns:
            {"summary": str, "keywords": list[str], "patterns": list[str], "contradictions": list[str]}
            失败时各字段为空列表/空字符串
        """
        if not node_contents:
            return {"summary": "Empty community", "keywords": [], "patterns": [], "contradictions": []}

        # 截断过长内容
        combined = "\n".join(
            f"- {c[:300]}" for c in node_contents[:max_nodes] if c.strip()
        )
        if not combined:
            return {"summary": "No meaningful content", "keywords": [], "patterns": [], "contradictions": []}

        prompt = f"""You are a memory curator for a self-evolving hypergraph memory system.
Analyze the following cluster of related memory entries and produce a structured summary.

Memory entries:
{combined}

Respond in JSON format with exactly these fields:
{{
  "summary": "A concise 2-3 sentence summary of what this cluster is about",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "patterns": ["Any recurring themes or patterns observed across entries"],
  "contradictions": ["Any contradictions or inconsistencies between entries"]
}}

Focus on factual content and meaningful connections. Be concise."""

        result = self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        if not result:
            # fallback: TF-IDF 关键词
            return {
                "summary": f"Community with {len(node_contents)} entries",
                "keywords": self._fallback_keywords(node_contents),
                "patterns": [],
                "contradictions": [],
            }

        try:
            parsed = json.loads(result)
            return {
                "summary": parsed.get("summary", "")[:1000],
                "keywords": parsed.get("keywords", [])[:10],
                "patterns": parsed.get("patterns", []),
                "contradictions": parsed.get("contradictions", []),
            }
        except (json.JSONDecodeError, TypeError):
            logger.warning("LLMClient: failed to parse JSON response, using fallback")
            return {
                "summary": result[:500],
                "keywords": self._fallback_keywords(node_contents),
                "patterns": [],
                "contradictions": [],
            }

    def _fallback_keywords(self, texts: list[str], max_features: int = 5) -> list[str]:
        """TF-IDF 关键词回退。"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            non_empty = [t for t in texts if t.strip()]
            if not non_empty:
                return []
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
            vectorizer.fit_transform(non_empty)
            return list(vectorizer.get_feature_names_out())
        except (ImportError, ValueError):
            pass
        # 词频回退
        word_freq: dict[str, int] = {}
        for text in texts:
            for word in text.lower().split():
                word = "".join(c for c in word if c.isalpha())
                if len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1
        return [w for w, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_features]]

    def close(self) -> None:
        """释放 httpx 连接。"""
        try:
            self._client.close()
        except Exception:
            pass
