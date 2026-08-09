"""
梦境 LLM-NER 超时死循环修复测试
===============================
覆盖:
  · _ner_with_llm 并行性 — 6 节点 mock 0.2s 延迟，总耗时 < 串行 1.2s 的 60%
  · _ner_with_llm 节点上限 — 8 节点最多 5 次 LLM 调用
  · 单节点失败不中断其他节点
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.dream_pipeline import (
    DreamPipeline,
    _NER_MAX_NODES_PER_COMMUNITY,
    _MAX_LLM_NER_TOTAL,
    _NER_FAIL_FAST_THRESHOLD,
)


def run(coro):
    return asyncio.run(coro)


def _make_llm_client(mock_chat: AsyncMock) -> MagicMock:
    client = MagicMock()
    client.api_key = "test-key"
    client.chat = mock_chat
    return client


def _make_nodes(count: int, prefix: str = "n") -> list[dict]:
    return [
        {"id": f"{prefix}{i}", "content": f"Node {i} discusses machine learning and deep neural networks"}
        for i in range(count)
    ]


class TestNerParallelism:
    def test_six_nodes_total_under_serial_threshold(self):
        """6 节点 mock 0.2s/次 → 并行总耗时 < 串行 1.2s 的 60%（< 0.72s）。

        Semaphore(5) → 第一批 5 并发，第二批 1 个 → 预期 ~0.4s。
        """

        async def _delayed_chat(messages=None, temperature=0.1, max_tokens=256,
                                 response_format=None, **kwargs):
            await asyncio.sleep(0.2)
            return json.dumps({"entities": ["AI", "ML"]})

        pipe = DreamPipeline()
        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_delayed_chat))

        nodes = _make_nodes(6)
        t0 = time.monotonic()
        result = run(pipe._ner_with_llm(nodes))
        elapsed = time.monotonic() - t0

        assert elapsed < 0.72, (
            f"Parallel NER took {elapsed:.2f}s, expected < 0.72s "
            f"(60% of serial 1.2s)"
        )
        assert len(result) >= 5  # at least 5 nodes should succeed

    def test_individual_failure_does_not_block_others(self):
        """节点 2 抛异常 → 其他节点仍成功返回实体。"""
        call_count = [0]

        async def _one_fail(messages=None, temperature=0.1, max_tokens=256,
                             response_format=None, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("node 2 boom")
            return json.dumps({"entities": ["Entity"]})

        pipe = DreamPipeline()
        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_one_fail))

        nodes = _make_nodes(5)
        result = run(pipe._ner_with_llm(nodes))

        assert len(result) >= 3, (
            f"Expected >= 3 nodes to succeed, got {len(result)}"
        )


class TestNerNodeCap:
    def test_eight_nodes_max_five_llm_calls(self):
        """8 节点 → 最多 _NER_MAX_NODES_PER_COMMUNITY (5) 次 LLM 调用。"""
        call_count = [0]

        async def _counting_chat(messages=None, temperature=0.1, max_tokens=256,
                                  response_format=None, **kwargs):
            call_count[0] += 1
            return json.dumps({"entities": ["X"]})

        pipe = DreamPipeline()
        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_counting_chat))

        nodes = _make_nodes(8)
        result = run(pipe._ner_with_llm(nodes))

        assert call_count[0] == _NER_MAX_NODES_PER_COMMUNITY, (
            f"Expected {_NER_MAX_NODES_PER_COMMUNITY} LLM calls, got {call_count[0]}"
        )
        assert len(result) <= _NER_MAX_NODES_PER_COMMUNITY

    def test_fewer_than_cap_all_processed(self):
        """3 节点（< cap）→ 全部 3 次 LLM 调用。"""
        call_count = [0]

        async def _counting_chat(messages=None, temperature=0.1, max_tokens=256,
                                  response_format=None, **kwargs):
            call_count[0] += 1
            return json.dumps({"entities": ["X"]})

        pipe = DreamPipeline()
        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_counting_chat))

        nodes = _make_nodes(3)
        result = run(pipe._ner_with_llm(nodes))

        assert call_count[0] == 3, f"Expected 3 LLM calls, got {call_count[0]}"
        assert len(result) == 3

    def test_zero_nodes_returns_empty(self):
        """空节点列表 → 0 次 LLM 调用，返回 {}。"""
        pipe = DreamPipeline()
        pipe.llm_client = _make_llm_client(AsyncMock())
        result = run(pipe._ner_with_llm([]))
        assert result == {}
        pipe.llm_client.chat.assert_not_called()

    def test_longest_content_gets_priority(self):
        """8 节点中内容最长的 5 个走 LLM-NER（按 content 长度排序优先）。"""
        processed_ids = []

        async def _track_chat(messages=None, temperature=0.1, max_tokens=256,
                               response_format=None, **kwargs):
            prompt_text = messages[0]["content"] if messages else ""
            for line in prompt_text.split("\n"):
                if line.strip().startswith("Node "):
                    processed_ids.append(line.strip().split(" ")[1])
                    break
            return json.dumps({"entities": ["X"]})

        pipe = DreamPipeline()
        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_track_chat))

        # 创建 8 节点，内容长度递增：n0 最短，n7 最长
        nodes = []
        for i in range(8):
            filler = "x" * (i * 10)
            nodes.append({
                "id": f"n{i}",
                "content": f"Node {i} has content. {filler}"
            })

        result = run(pipe._ner_with_llm(nodes))

        assert len(result) == _NER_MAX_NODES_PER_COMMUNITY
        # 最长的 5 个应该是 n3, n4, n5, n6, n7 (按长度排序)
        # n0-n2 因内容较短被排除
        assert "n0" not in result, "Shortest content node should be skipped"
        assert "n1" not in result, "Short content node should be skipped"
        assert "n2" not in result, "Short content node should be skipped"
        assert "n7" in result, "Longest content node should be processed"


class TestNerNoLlm:
    def test_no_llm_client_returns_empty(self):
        """无 llm_client 时返回 {}（触发调用方正则降级）。"""
        pipe = DreamPipeline()
        pipe.llm_client = None
        result = run(pipe._ner_with_llm(_make_nodes(3)))
        assert result == {}

    def test_no_api_key_returns_empty(self):
        """llm_client 无 api_key 时返回 {}。"""
        pipe = DreamPipeline()
        client = MagicMock()
        client.api_key = None
        pipe.llm_client = client
        result = run(pipe._ner_with_llm(_make_nodes(3)))
        assert result == {}


# ─── P1-1: 正则降级承诺实现 ─────────────────────────────────


class TestP1OneRegexFallback:
    """P1-1: LLM 结果 + 正则结果合并，cap 外节点 & LLM 失败节点有正则兜底。"""

    def test_cap_out_nodes_get_regex_entities(self):
        """8 节点，cap=5 → 最短 3 个节点的实体来自正则提取。

        通过 _extract_entities_from_nodes 验证：LLM 处理最长的 5 个节点，
        其余 3 个由正则兜底，且 LLM 结果优先保留。
        """
        pipe = DreamPipeline()
        call_count = [0]

        async def _mock_chat(messages=None, **kwargs):
            call_count[0] += 1
            return json.dumps({"entities": ["LLM_Entity"]})

        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_mock_chat))

        # 8 节点：n0-n2 最短（正则兜底），n3-n7 最长（LLM 处理）
        nodes = [
            {"id": "n0", "content": 'Short "Alpha"'},
            {"id": "n1", "content": 'Short "Beta" HELLO'},
            {"id": "n2", "content": 'Short "GammaLab"'},
            {"id": "n3", "content": "Node 3 has a very long discussion about machine learning " + "x" * 60},
            {"id": "n4", "content": "Node 4 covers deep neural networks and transformers " + "x" * 55},
            {"id": "n5", "content": "Node 5 explains large language models and attention " + "x" * 50},
            {"id": "n6", "content": "Node 6 details reinforcement learning systems " + "x" * 45},
            {"id": "n7", "content": "Node 7 reviews computer vision architectures " + "x" * 40},
        ]

        result = run(pipe._extract_entities_from_nodes(nodes))

        # LLM 处理了 5 个最长的节点
        assert call_count[0] == _NER_MAX_NODES_PER_COMMUNITY

        # 最短的 3 个节点应有正则实体
        assert "n0" in result, "n0 should have regex entities"
        assert "Alpha" in result["n0"]
        assert "n1" in result, "n1 should have regex entities"
        assert any(e in result["n1"] for e in ["Beta", "HELLO"])
        assert "n2" in result, "n2 should have regex entities"
        assert "GammaLab" in result["n2"]

        # LLM 处理的节点应保留 LLM 实体
        for nid in ["n3", "n4", "n5", "n6", "n7"]:
            assert nid in result
            assert "LLM_Entity" in result[nid]

    def test_llm_failed_node_gets_regex_fallback(self):
        """一个节点 LLM 失败 → 该节点实体来自正则，其余节点来自 LLM。"""
        pipe = DreamPipeline()

        async def _mock_chat(messages=None, **kwargs):
            prompt = messages[0]["content"]
            if "OmegaInc" in prompt:  # 确定性：含 OmegaInc 的节点（n2）失败
                raise RuntimeError("n2 failure")
            return json.dumps({"entities": ["LLM_Entity"]})

        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_mock_chat))

        nodes = [
            {"id": "n0", "content": "Node 0 discusses ProjectAlpha and AI technology " + "x" * 20},
            {"id": "n1", "content": "Node 1 discusses BetaCorp systems research " + "x" * 15},
            {"id": "n2", "content": 'Node 2 "GammaLabs" research HELLO "OmegaInc" extra' + "x" * 10},
            {"id": "n3", "content": "Node 3 discusses DeltaCorp transformer models " + "x" * 5},
            {"id": "n4", "content": "Node 4 discusses Epsilon LLM architecture "},
        ]

        result = run(pipe._extract_entities_from_nodes(nodes))

        # n0, n1, n3, n4: LLM 成功 → LLM_Entity
        for nid in ["n0", "n1", "n3", "n4"]:
            assert nid in result, f"{nid} should be in result"
            assert "LLM_Entity" in result[nid], (
                f"{nid} should have LLM_Entity, got {result[nid]}"
            )

        # n2: LLM 失败 → 正则兜底（引号内 + 大写缩写）
        assert "n2" in result
        assert "LLM_Entity" not in result["n2"], "n2 should NOT have LLM_Entity (LLM failed)"
        assert any(e in result["n2"] for e in ["GammaLabs", "HELLO", "OmegaInc"]), (
            f"n2 should have regex entities, got: {result.get('n2', [])}"
        )

    def test_all_llm_fail_still_gets_regex_for_all(self):
        """所有 LLM 调用都失败 → 全部节点从正则提取实体。"""
        pipe = DreamPipeline()

        async def _always_fail(messages=None, **kwargs):
            raise RuntimeError("all fail")

        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_always_fail))

        nodes = [
            {"id": "n0", "content": 'Node 0 "AlphaProject" HELLO'},
            {"id": "n1", "content": 'Node 1 "BetaCorp" WORLD'},
            {"id": "n2", "content": 'Node 2 "GammaLabs" TEST'},
        ]

        result = run(pipe._extract_entities_from_nodes(nodes))

        assert len(result) == 3
        assert "AlphaProject" in result["n0"]
        assert "BetaCorp" in result["n1"]
        assert "GammaLabs" in result["n2"]


# ─── P1-2a: NER 超时不中断 ────────────────────────────────────


class TestP1TwoTimeout:
    """P1-2a: _ner_single_node 包 asyncio.wait_for(timeout=15s)，超时记 fail 不中断。"""

    def test_ner_single_node_timeout_returns_empty(self):
        """mock llm.chat 挂起 → _ner_with_llm 中该节点被跳过，返回空 dict。"""
        pipe = DreamPipeline()
        pipe.llm_client = _make_llm_client(AsyncMock())

        async def _timeout_wait(coro, timeout=None):
            raise asyncio.TimeoutError()

        with patch("core.dream_pipeline.asyncio.wait_for", _timeout_wait):
            result = run(pipe._ner_with_llm(_make_nodes(1)))

        assert result == {}, "Timeout → no entities returned from LLM"

    def test_timeout_does_not_block_other_nodes(self):
        """1 个节点超时 + 4 个正常 → 其余 4 个正常返回实体。"""
        timeout_called = [False]

        async def _chat_with_one_timeout(messages=None, **kwargs):
            await asyncio.sleep(0.01)
            return json.dumps({"entities": ["Entity"]})

        pipe = DreamPipeline()
        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_chat_with_one_timeout))

        original_wait_for = asyncio.wait_for
        call_idx = [0]

        async def _selective_timeout(coro, timeout=None):
            call_idx[0] += 1
            if call_idx[0] == 1:
                timeout_called[0] = True
                raise asyncio.TimeoutError()
            return await original_wait_for(coro, timeout=timeout)

        with patch("core.dream_pipeline.asyncio.wait_for", _selective_timeout):
            result = run(pipe._ner_with_llm(_make_nodes(5)))

        assert timeout_called[0], "First node should have timed out"
        assert len(result) == 4, f"Expected 4 successful nodes, got {len(result)}"

    def test_ner_with_llm_timeout_fail_fast_counter(self):
        """全部超时 → ner_fails 递增，触达阈值后后续 skip。"""
        pipe = DreamPipeline()
        pipe.llm_client = _make_llm_client(AsyncMock())

        async def _always_timeout(coro, timeout=None):
            raise asyncio.TimeoutError()

        with patch("core.dream_pipeline.asyncio.wait_for", _always_timeout):
            ner_fails = [0]
            for _ in range(_NER_FAIL_FAST_THRESHOLD):
                run(pipe._ner_with_llm(
                    _make_nodes(1), ner_budget=[999], ner_fails=ner_fails
                ))

        assert ner_fails[0] >= _NER_FAIL_FAST_THRESHOLD

        call_happened = [False]

        async def _should_not_be_called(messages=None, **kwargs):
            call_happened[0] = True
            return json.dumps({"entities": []})

        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_should_not_be_called))
        result = run(pipe._ner_with_llm(
            _make_nodes(1), ner_budget=[999], ner_fails=ner_fails
        ))
        assert result == {}
        assert not call_happened[0], "LLM should be skipped after fail-fast"


# ─── P1-2b: 全局 NER 预算 ──────────────────────────────────────


class TestP1TwoGlobalBudget:
    """P1-2b: _MAX_LLM_NER_TOTAL 全局预算，超限后剩余社区走正则。"""

    def test_budget_exhausted_stops_llm_calls(self):
        """20 社区 × 5 节点 = 100 次 → 预算耗尽 → 之后 0 次 LLM 调用。"""
        pipe = DreamPipeline()
        call_count = [0]

        async def _counting_chat(messages=None, **kwargs):
            call_count[0] += 1
            return json.dumps({"entities": ["Entity"]})

        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_counting_chat))

        nodes_template = _make_nodes(5)
        ner_budget = [_MAX_LLM_NER_TOTAL]
        ner_fails = [0]

        # 20 社区 = 100 次调用 → 正好耗尽
        for _ in range(20):
            run(pipe._extract_entities_from_nodes(
                nodes_template, ner_budget=ner_budget, ner_fails=ner_fails
            ))

        assert ner_budget[0] == 0
        assert call_count[0] == _MAX_LLM_NER_TOTAL

        # 额外 3 社区 → 0 次额外 LLM 调用
        for _ in range(3):
            run(pipe._extract_entities_from_nodes(
                nodes_template, ner_budget=ner_budget, ner_fails=ner_fails
            ))

        assert call_count[0] == _MAX_LLM_NER_TOTAL, (
            f"Expected {_MAX_LLM_NER_TOTAL} total calls, got {call_count[0]}"
        )

    def test_budget_partial_community_consumes_remaining(self):
        """预算只剩 3 → 该社区最多调 3 次 LLM，之后预算归零。"""
        pipe = DreamPipeline()
        call_count = [0]

        async def _counting_chat(messages=None, **kwargs):
            call_count[0] += 1
            return json.dumps({"entities": ["Entity"]})

        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_counting_chat))

        nodes_template = _make_nodes(5)
        ner_budget = [3]
        ner_fails = [0]

        run(pipe._extract_entities_from_nodes(
            nodes_template, ner_budget=ner_budget, ner_fails=ner_fails
        ))

        assert call_count[0] == 3, f"Expected 3 calls, got {call_count[0]}"
        assert ner_budget[0] == 0

    def test_no_budget_passed_means_unlimited(self):
        """不传 ner_budget → 无限制（向后兼容）。"""
        pipe = DreamPipeline()
        call_count = [0]

        async def _counting_chat(messages=None, **kwargs):
            call_count[0] += 1
            return json.dumps({"entities": ["Entity"]})

        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_counting_chat))

        run(pipe._extract_entities_from_nodes(_make_nodes(8)))

        assert call_count[0] == _NER_MAX_NODES_PER_COMMUNITY


# ─── P1-2c: NER fail-fast ──────────────────────────────────────


class TestP1TwoFailFast:
    """P1-2c: 连续 NER 失败 → 后续 skip LLM，走正则。"""

    def test_consecutive_failures_trigger_fail_fast(self):
        """连续 3 次 NER 失败 → ner_fails >= 阈值 → 后续 skip LLM。"""
        pipe = DreamPipeline()

        async def _always_fail(messages=None, **kwargs):
            raise RuntimeError("simulated failure")

        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_always_fail))

        ner_budget = [999]
        ner_fails = [0]

        for _ in range(_NER_FAIL_FAST_THRESHOLD):
            run(pipe._ner_with_llm(_make_nodes(1), ner_budget=ner_budget, ner_fails=ner_fails))

        assert ner_fails[0] == _NER_FAIL_FAST_THRESHOLD

        call_made = [False]

        async def _should_not_call(messages=None, **kwargs):
            call_made[0] = True
            return json.dumps({"entities": []})

        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_should_not_call))
        result = run(pipe._ner_with_llm(_make_nodes(1), ner_budget=ner_budget, ner_fails=ner_fails))

        assert result == {}
        assert not call_made[0]

    def test_success_resets_fail_counter(self):
        """一次成功后 ner_fails 归零 → 不触发 fail-fast。"""
        pipe = DreamPipeline()
        fail_then_succeed = [0]

        async def _mixed_chat(messages=None, **kwargs):
            fail_then_succeed[0] += 1
            if fail_then_succeed[0] <= 2:
                raise RuntimeError("fail")
            return json.dumps({"entities": ["Entity"]})

        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_mixed_chat))

        ner_budget = [999]
        ner_fails = [0]

        run(pipe._ner_with_llm(_make_nodes(1), ner_budget=ner_budget, ner_fails=ner_fails))
        assert ner_fails[0] == 1

        run(pipe._ner_with_llm(_make_nodes(1), ner_budget=ner_budget, ner_fails=ner_fails))
        assert ner_fails[0] == 2

        run(pipe._ner_with_llm(_make_nodes(1), ner_budget=ner_budget, ner_fails=ner_fails))
        assert ner_fails[0] == 0, f"Should reset to 0 after success, got {ner_fails[0]}"

    def test_partial_failure_increments_but_not_all_zeros(self):
        """部分节点失败（非全失败）→ ner_fails 递增，ner_budget 仍扣减。"""
        pipe = DreamPipeline()
        call_idx = [0]

        async def _two_fail_three_ok(messages=None, **kwargs):
            call_idx[0] += 1
            if call_idx[0] <= 2:
                raise RuntimeError("fail")
            return json.dumps({"entities": ["Entity"]})

        pipe.llm_client = _make_llm_client(AsyncMock(side_effect=_two_fail_three_ok))

        ner_budget = [999]
        ner_fails = [0]

        result = run(pipe._ner_with_llm(_make_nodes(5), ner_budget=ner_budget, ner_fails=ner_fails))

        assert len(result) == 3
        assert ner_fails[0] == 2
        assert ner_budget[0] == 994
