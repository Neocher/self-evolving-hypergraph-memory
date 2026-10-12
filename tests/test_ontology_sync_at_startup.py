# -*- coding: utf-8 -*-
"""本体同步启动化（v6.35）验收测试。

对齐任务书验收点：
  AC2  启动期、就绪（yield）前完成本体一次性同步 → 服务对外服务时
       _ontology_synced 已为 True（fail-soft：异常不置位、不阻断启动）。
  AC4  幂等兜底仍在：search.py 的 _ontology_synced 守卫仍存在且能被触发
       （模拟 _ontology_synced=False 时请求路径仍会执行同步）。
  AC5  回退：load_settings().api.startup_background_stagger_s == 0.0。

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_ontology_sync_at_startup.py -q
"""
from __future__ import annotations

import asyncio
import inspect
import types

from fastapi import HTTPException

import api.app as app_module
import api.routes.search as search_module
from config.settings import load_settings


class _FakeOntologyValidator:
    """最小本体验证器替身：记录两个同步动作的调用次数。"""

    def __init__(self, synced: bool = False):
        self._ontology_synced = synced
        self.sync_calls = 0
        self.relate_calls = 0

    def sync_entity_types_to_graphlite(self):
        self.sync_calls += 1
        return 0

    def _populate_relationships(self):
        self.relate_calls += 1
        return 0


def _fake_svc(ov=None, write_queue=None):
    return types.SimpleNamespace(ontology_validator=ov, write_queue=write_queue)


class TestStartupOntologySync:
    def test_syncs_both_actions_and_sets_flag(self):
        """启动期同步：两个动作各执行一次，完成后 _ontology_synced=True。"""
        ov = _FakeOntologyValidator(synced=False)
        svc = _fake_svc(ov=ov)

        asyncio.run(app_module._startup_ontology_sync(svc))

        assert ov.sync_calls == 1, "sync_entity_types_to_graphlite 必须执行一次"
        assert ov.relate_calls == 1, "_populate_relationships 必须执行一次"
        assert ov._ontology_synced is True, "完成后必须置 _ontology_synced=True"

    def test_skips_when_already_synced(self):
        """已同步 → 两个动作都不得重复执行（幂等快路径）。"""
        ov = _FakeOntologyValidator(synced=True)
        svc = _fake_svc(ov=ov)

        asyncio.run(app_module._startup_ontology_sync(svc))

        assert ov.sync_calls == 0
        assert ov.relate_calls == 0

    def test_no_validator_is_noop(self):
        """无本体验证器 → 直接返回，不抛异常。"""
        asyncio.run(app_module._startup_ontology_sync(_fake_svc(ov=None)))

    def test_fail_soft_on_sync_error_keeps_flag_false(self):
        """同步异常 → 不阻断启动、不置位（保留请求期兜底）。"""
        ov = _FakeOntologyValidator(synced=False)

        def boom():
            raise RuntimeError("graph down")

        ov._populate_relationships = boom
        svc = _fake_svc(ov=ov)

        # 不抛异常（fail-soft），且 _ontology_synced 保持 False
        asyncio.run(app_module._startup_ontology_sync(svc))
        assert ov._ontology_synced is False, "异常时不得置 _ontology_synced=True"

    def test_fail_soft_on_write_queue_busy_keeps_flag_false(self, monkeypatch):
        """写队列忙（qsubmit → HTTPException）→ 不阻断启动、不置位。"""
        ov = _FakeOntologyValidator(synced=False)

        def busy_qsubmit(deps, fn, *args, **kwargs):
            raise HTTPException(status_code=503, detail="write queue busy")

        monkeypatch.setattr(app_module, "qsubmit", busy_qsubmit)

        asyncio.run(app_module._startup_ontology_sync(_fake_svc(ov=ov)))
        assert ov._ontology_synced is False, "写队列忙时不得置 _ontology_synced=True"


class TestLifespanOrdering:
    def test_ontology_sync_awaited_before_yield(self):
        """lifespan 在 yield 前 await _startup_ontology_sync（就绪前完成同步）。"""
        src = inspect.getsource(app_module.lifespan.__wrapped__)
        sync_idx = src.index("await _startup_ontology_sync(")
        yield_idx = src.index("\n    yield\n")
        assert sync_idx < yield_idx, "本体同步必须在 yield 之前 await"


class TestSearchIdempotentFallback:
    def test_guard_still_present_in_search(self):
        """AC4：search.py 的 _ontology_synced 守卫仍存在（幂等兜底不删除）。"""
        src = inspect.getsource(search_module._ensure_ontology_synced)
        assert "if not deps.ontology_validator._ontology_synced:" in src, (
            "幂等守卫必须保留"
        )
        assert "sync_entity_types_to_graphlite" in src
        assert "_populate_relationships" in src
        assert "deps.ontology_validator._ontology_synced = True" in src

    def test_retrieve_calls_fallback_guard(self):
        """retrieve 读验证路径仍调用 _ensure_ontology_synced（快路径）。"""
        src = inspect.getsource(search_module.retrieve)
        assert "await _ensure_ontology_synced(deps)" in src, (
            "检索请求必须仍调用请求期幂等兜底"
        )

    def test_fallback_triggers_sync_when_not_synced(self):
        """AC4：模拟 _ontology_synced=False → 请求期兜底仍会执行同步并置位。"""
        ov = _FakeOntologyValidator(synced=False)
        deps = types.SimpleNamespace(ontology_validator=ov, write_queue=None)

        asyncio.run(search_module._ensure_ontology_synced(deps))

        assert ov.sync_calls == 1
        assert ov.relate_calls == 1
        assert ov._ontology_synced is True

    def test_fallback_is_fast_path_when_synced(self):
        """已同步 → 请求期兜底不执行任何写（正常路径为快路径）。"""
        ov = _FakeOntologyValidator(synced=True)
        deps = types.SimpleNamespace(ontology_validator=ov, write_queue=None)

        asyncio.run(search_module._ensure_ontology_synced(deps))

        assert ov.sync_calls == 0
        assert ov.relate_calls == 0


class TestStaggerRevert:
    def test_startup_background_stagger_default_is_zero(self):
        """AC5：错峰默认回退为 0.0（关闭）。"""
        assert load_settings().api.startup_background_stagger_s == 0.0
