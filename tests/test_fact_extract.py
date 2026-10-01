"""P1a CJK 事实抽取模块单元测试。

覆盖：
- extract_facts_cjk_rules：4 类中文规则 + 5 条设计基准语料全命中 + 幂等稳定
  + 上限 10 + 空输入
- extract_facts_llm：None / 短内容跳过（不调 LLM）/ 合法 JSON / 非法 JSON /
  超时 / chat 返回 None / 字段缺失 → 全降级 []
- extract_facts：规则路 ∪ LLM 路合并去重（规则路优先，保留其 valid_time=""）
- _enabled：默认 True，"0"/"false" → False

所有用例走公共入口，不 mock 内部方法；LLM 路用 AsyncMock 作鸭子类型。
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from core.fact_extract import (
    _enabled,
    extract_facts,
    extract_facts_cjk_rules,
    extract_facts_llm,
)


# ─── 规则路：5 条设计基准语料 ──────────────────────────────

def test_rule_port():
    facts = extract_facts_cjk_rules("海外代理固化 mihomo SOCKS5 127.0.0.1:1080")
    assert any(
        f["subject"] == "代理端口"
        and f["predicate"] == "是"
        and f["object"] == "127.0.0.1:1080"
        for f in facts
    )


def test_rule_version():
    facts = extract_facts_cjk_rules("版本 v6.22.8 已推")
    assert any(
        f["subject"] == "版本" and f["predicate"] == "是" and f["object"] == "6.22.8"
        for f in facts
    )


def test_rule_path():
    facts = extract_facts_cjk_rules("备份在 ~/backups/search.py.p2b.20261001.bak")
    assert any(
        f["subject"] == "路径"
        and f["predicate"] == "是"
        and f["object"] == "~/backups/search.py.p2b.20261001.bak"
        for f in facts
    )


def test_rule_state_down():
    facts = extract_facts_cjk_rules("旧 WARP 链 :8083/:1081 已下线")
    assert any(f["predicate"] == "状态" for f in facts)


def test_rule_state_pollute():
    facts = extract_facts_cjk_rules("FEISHU secret 污染为 *** 后恢复")
    assert any(
        f["predicate"] == "状态" and "FEISHU secret" in f["subject"] for f in facts
    )


# ─── 规则路：边界 ──────────────────────────────────────────

def test_rule_empty_input():
    assert extract_facts_cjk_rules("") == []


def test_rule_cap_ten():
    content = " ".join(f"127.0.0.1:{1080 + i}" for i in range(12))
    facts = extract_facts_cjk_rules(content)
    assert len(facts) == 10


def test_rule_deterministic():
    content = " ".join([
        "海外代理固化 mihomo SOCKS5 127.0.0.1:1080",
        "版本 v6.22.8 已推",
        "备份在 ~/backups/search.py.p2b.20261001.bak",
        "旧 WARP 链 :8083/:1081 已下线",
        "FEISHU secret 污染为 *** 后恢复",
    ])
    runs = [extract_facts_cjk_rules(content) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


# ─── LLM 路：全降级 ───────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_none():
    assert await extract_facts_llm(None, "x" * 100) == []


@pytest.mark.asyncio
async def test_llm_short_content_skips():
    llm = AsyncMock()
    assert await extract_facts_llm(llm, "短内容") == []
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_llm_valid_json():
    llm = AsyncMock()
    llm.chat.return_value = json.dumps([
        {"subject": "A", "predicate": "是", "object": "B"},
        {"subject": "C", "predicate": "状态", "object": "D", "valid_time": "2024"},
    ])
    facts = await extract_facts_llm(llm, "x" * 100)
    assert len(facts) == 2
    assert facts[0] == {"subject": "A", "predicate": "是", "object": "B", "valid_time": ""}
    assert facts[1]["valid_time"] == "2024"


@pytest.mark.asyncio
async def test_llm_invalid_json():
    llm = AsyncMock()
    llm.chat.return_value = "not json {{{"
    assert await extract_facts_llm(llm, "x" * 100) == []


@pytest.mark.asyncio
async def test_llm_timeout():
    llm = AsyncMock()
    llm.chat.side_effect = asyncio.TimeoutError
    assert await extract_facts_llm(llm, "x" * 100) == []


@pytest.mark.asyncio
async def test_llm_chat_returns_none():
    llm = AsyncMock()
    llm.chat.return_value = None
    assert await extract_facts_llm(llm, "x" * 100) == []


@pytest.mark.asyncio
async def test_llm_missing_fields():
    llm = AsyncMock()
    llm.chat.return_value = json.dumps([{"subject": "只有主语"}])
    assert await extract_facts_llm(llm, "x" * 100) == []


# ─── 合并去重 + 开关 ──────────────────────────────────────

@pytest.mark.asyncio
async def test_merge_dedup_rules_priority():
    content = "127.0.0.1:1080 " + "长" * 80  # ≥80 字符，确保 LLM 路被调用
    llm = AsyncMock()
    llm.chat.return_value = json.dumps([
        {"subject": "代理端口", "predicate": "是", "object": "127.0.0.1:1080",
         "valid_time": "2024-01-01"},
    ])
    facts = await extract_facts(llm, content)
    matched = [
        f for f in facts
        if f["subject"] == "代理端口" and f["object"] == "127.0.0.1:1080"
    ]
    assert len(matched) == 1
    assert matched[0]["valid_time"] == ""


def test_enabled_default(monkeypatch):
    monkeypatch.delenv("SHM_FACT_EXTRACT", raising=False)
    assert _enabled() is True


def test_enabled_false_values(monkeypatch):
    for v in ("0", "false", "False"):
        monkeypatch.setenv("SHM_FACT_EXTRACT", v)
        assert _enabled() is False
    monkeypatch.setenv("SHM_FACT_EXTRACT", "1")
    assert _enabled() is True
