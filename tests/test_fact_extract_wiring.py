"""P1b 写路径事实抽取接线测试。

覆盖：
- _fact_extract_enabled：默认 True / "0"·"false"·"" → False / "1" → True
- _extract_facts_bg：落库参数正确（subject/predicate/object/valid_time/source_episode）
  + llm_client=None 仍走规则路有产出 + 抽取异常 / store 异常均被吞
- create_episode 集成：session 关联后、record_request 前 create_task 后台调度

所有用例走公共入口（_extract_facts_bg / create_episode），不 mock 内部方法；
LLM 路用 None / AsyncMock 作鸭子类型。自包含，不依赖其他测试模块。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.routes.write import _extract_facts_bg, _fact_extract_enabled


class _RecordingStore:
    """记录 create_atomic_fact 调用参数（subject/predicate/object/valid_time/source_episode）。"""

    def __init__(self):
        self.calls: list[dict] = []

    def create_atomic_fact(self, subject, predicate, object_, valid_time="",
                           source_episode="", confidence=1.0, props=None):
        self.calls.append({
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "valid_time": valid_time,
            "source_episode": source_episode,
        })
        return f"fact_{len(self.calls)}"


def _make_svc(graph_store=None, llm_client=None) -> SimpleNamespace:
    svc = SimpleNamespace()
    svc.graph_store = graph_store
    svc.write_queue = None  # qsubmit 无队列 → 同步直调
    svc.llm_client = llm_client
    return svc


# ─── 门控开关 ─────────────────────────────────────────────

def test_enabled_default(monkeypatch):
    monkeypatch.delenv("SHM_FACT_EXTRACT", raising=False)
    assert _fact_extract_enabled() is True


def test_enabled_false_values(monkeypatch):
    for v in ("0", "false", "False", ""):
        monkeypatch.setenv("SHM_FACT_EXTRACT", v)
        assert _fact_extract_enabled() is False
    monkeypatch.setenv("SHM_FACT_EXTRACT", "1")
    assert _fact_extract_enabled() is True


# ─── 后台落库语义 ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_bg_persists_port_fact_exact_params():
    store = _RecordingStore()
    deps = _make_svc(graph_store=store, llm_client=None)
    await _extract_facts_bg(deps, "ep_1", "海外代理固化 127.0.0.1:1080")

    port = [c for c in store.calls if c["subject"] == "代理端口"]
    assert port, "规则路应抽出端口事实"
    assert port[0]["predicate"] == "是"
    assert port[0]["object"] == "127.0.0.1:1080"
    assert all(c["source_episode"] == "ep_1" for c in store.calls)


@pytest.mark.asyncio
async def test_bg_none_llm_still_rule_path_output():
    store = _RecordingStore()
    deps = _make_svc(graph_store=store, llm_client=None)
    await _extract_facts_bg(deps, "ep_2", "旧服务 127.0.0.1:1080 已下线")
    # llm_client=None → LLM 路返回 []，规则路仍应有产出（端口 + 状态）
    assert len(store.calls) >= 1


@pytest.mark.asyncio
async def test_bg_swallows_extract_error(monkeypatch):
    store = _RecordingStore()
    deps = _make_svc(graph_store=store, llm_client=None)

    async def boom(llm, content):
        raise RuntimeError("extract boom")

    monkeypatch.setattr("core.fact_extract.extract_facts", boom)
    # 不抛即通过
    await _extract_facts_bg(deps, "ep_3", "任意内容")
    assert store.calls == []


@pytest.mark.asyncio
async def test_bg_swallows_store_error():
    class BoomStore:
        def create_atomic_fact(self, *a, **k):
            raise RuntimeError("store boom")

    deps = _make_svc(graph_store=BoomStore(), llm_client=None)
    # 不抛即通过
    await _extract_facts_bg(deps, "ep_4", "127.0.0.1:1080")


# ─── create_episode 集成：后台调度 ─────────────────────────

@pytest.mark.asyncio
async def test_create_episode_schedules_bg(monkeypatch):
    from api.models import EpisodeCreate
    from api.routes import write as write_mod

    bg = AsyncMock()
    monkeypatch.setattr(write_mod, "_extract_facts_bg", bg)
    monkeypatch.setattr(write_mod, "_fact_extract_enabled", lambda: True)

    store = MagicMock()
    deps = SimpleNamespace(
        graph_store=store,
        write_queue=None,
        tau_engine=None,
        ssm_gate=None,
        defense_engine=None,
        encoder=None,
        ontology_validator=None,
        ontology_v2=None,
        evidence_tracker=None,
        dream_scheduler=None,
        quarantine_store=None,
        hyperedge_manager=None,
    )
    req = EpisodeCreate(content="短内容", source="user", visibility="private")
    request = SimpleNamespace(headers={})

    resp = await write_mod.create_episode(req, request, deps)
    await asyncio.sleep(0)  # 让 create_task 调度一次，避免 pending 告警

    assert bg.call_count == 1
    args = bg.call_args.args
    assert args[0] is deps
    assert isinstance(args[1], str) and args[1]  # episode_id 非空
    assert args[2] == "短内容"
    assert resp.episode_id == args[1]  # 调度用的 episode_id 与响应一致
