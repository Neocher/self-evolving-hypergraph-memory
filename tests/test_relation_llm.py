"""
Relation Extractor LLM 增强单元测试
===================================
覆盖 design_ontology_gaps.md v2 模块3：
  · extract() 保持纯同步正则（向后兼容）
  · extract_async() 正则 + LLM 混合（mock LLM）
  · extract_hybrid() 同步包装器（缓存动态关系）
  · 非法 confidence 回退 0.75（修复 #5）
  · 新关系不写入 OntologyService（防污染）
"""
from __future__ import annotations

import asyncio
import json
import pytest

from core.ontology_v2 import OntologyService
from core.relation_extractor import RelationExtractor, RelationTriple


# ─── Mock LLM ────────────────────────────────────────────

class MockLLMClient:
    """模拟 LLMClient.chat()，返回预置 JSON。"""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or [
            '[{"subject": "OpenAI", "relation": "ACQUIRED", "object": "Anthropic", "confidence": 0.9}]'
        ]
        self._idx = 0
        self.calls = 0

    async def chat(self, messages, *args, **kwargs) -> str:
        self.calls += 1
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return resp


# ─── 工具函数 ────────────────────────────────────────────


def run(coro):
    """同步运行异步协程（测试用）。"""
    return asyncio.run(coro)


def _llm_extractor(responses: list[str] | None = None) -> RelationExtractor:
    return RelationExtractor(llm_client=MockLLMClient(responses))


# ─── extract() 保持向后兼容 ──────────────────────────────


class TestExtractBackCompat:
    def test_extract_still_sync_regex(self):
        """extract() 纯同步正则，不依赖 LLM（签名不变）。"""
        ex = RelationExtractor()
        triples = ex.extract("OpenAI acquired Anthropic for 7.5B")
        assert any(t.relation == "ACQUIRED" for t in triples)

    def test_extract_no_llm_when_not_injected(self):
        """未注入 llm_client 时 extract() 不报错。"""
        ex = RelationExtractor()
        assert ex.extract("随便一句话") == []


# ─── extract_async 正则 + LLM 混合 ───────────────────────


class TestExtractAsync:
    def test_regex_hits_do_not_need_llm(self):
        """正则命中时不调用 LLM。"""
        ex = _llm_extractor()
        triples = run(ex.extract_async("OpenAI founded OpenAI Lab"))
        assert any(t.relation == "FOUNDED" for t in triples)
        assert ex._llm_client.calls == 0  # 未调用 LLM

    def test_uncovered_text_sent_to_llm(self):
        """正则未命中片段送 LLM，解析出三元组。"""
        ex = _llm_extractor()
        triples = run(ex.extract_async("OpenAI 与 Anthropic 达成战略合作"))
        # 中文 PARTNERED_WITH 无正则模式 → 送 LLM → mock 返回 OpenAI ACQUIRED Anthropic
        assert any(t.relation == "ACQUIRED" and t.subject == "OpenAI" for t in triples)
        assert ex._llm_client.calls >= 1

    def test_llm_confidence_used_when_valid(self):
        """LLM 输出合法 confidence 时采用。"""
        ex = _llm_extractor()
        triples = run(ex.extract_async("OpenAI acquired Anthropic, according to sources"))
        llm_triples = [t for t in triples if t.confidence != 0.85]  # 非正则固定值
        if llm_triples:
            assert all(0.7 <= t.confidence <= 0.95 for t in llm_triples)

    def test_invalid_confidence_falls_back_075(self):
        """LLM 输出非法/缺失 confidence 时回退 0.75（修复 #5）。"""
        ex = _llm_extractor(['[{"subject":"X","relation":"WORKS_WITH","object":"Y","confidence":"high"}]'])
        triples = run(ex.extract_async("X collaborates with Y on a project"))
        # 正则固定值 0.85；LLM 非法 confidence 应回退 0.75（不会等于 0.85）
        assert any(t.relation == "WORKS_WITH" and t.confidence == 0.75 for t in triples)

    def test_dynamic_relation_not_written_to_ontology(self):
        """LLM 发现的新关系不写入 OntologyService（防污染）。"""
        ontology = OntologyService()
        before = {t.name for t in ontology.list_entity_types()}
        ex = _llm_extractor()
        run(ex.extract_async("Some company acquired another firm"))
        after = {t.name for t in ontology.list_entity_types()}
        assert before == after


# ─── extract_hybrid 同步包装器 ───────────────────────────


class TestExtractHybrid:
    def test_hybrid_reuses_cached_dynamic_relations(self):
        """extract_hybrid() 复用 _dynamic_relations 缓存，不发起 LLM 调用。"""
        ex = _llm_extractor()
        # 先异步积累一条动态关系（该句子无正则命中 → 触发 LLM）
        run(ex.extract_async("OpenAI 与 Anthropic 达成战略合作"))
        assert ex._llm_client.calls >= 1
        # 再同步 hybrid（无 LLM 调用）
        triples = ex.extract_hybrid("OpenAI 和 Anthropic 的合并案")
        # hybrid 内不再新增调用
        calls_after = ex._llm_client.calls
        ex.extract_hybrid("another text without entities")
        assert ex._llm_client.calls == calls_after

    def test_hybrid_without_dynamic_cache_returns_regex_only(self):
        """无缓存时 hybrid 退化为纯正则。"""
        ex = RelationExtractor()  # 无 LLM
        triples = ex.extract_hybrid("OpenAI founded OpenAI Lab")
        assert any(t.relation == "FOUNDED" for t in triples)


# ─── Integration 标记（真实 LLM，需 DEEPSEEK_API_KEY）────


@pytest.mark.integration
@pytest.mark.skipif(
    __import__("os").environ.get("DEEPSEEK_API_KEY") is None,
    reason="DEEPSEEK_API_KEY not set, skipping integration test",
)
class TestLLMIntegration:
    def test_real_llm_extracts_relation(self):
        """真实 LLM 端到端：从英文句子抽取关系。"""
        from core.llm_client import LLMClient

        ex = RelationExtractor(llm_client=LLMClient())
        triples = run(ex.extract_async("Elon Musk founded SpaceX in 2002"))
        assert any(t.relation == "FOUNDED" for t in triples)
