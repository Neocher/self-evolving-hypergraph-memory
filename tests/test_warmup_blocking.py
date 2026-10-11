# -*- coding: utf-8 -*-
"""预热阻塞就绪（v6.33）验收测试。

对齐任务书 AC8（新增针对「预热必须阻塞就绪」的测试）：
  服务在 lifespan yield 前 await 检索就绪准备（FAISS 加载 + prewarm 三段 + FUSION 预热），
  不存在「预热作为后台 create_task 与首条查询并发」的竞态。

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_warmup_blocking.py -q
"""
from __future__ import annotations

import asyncio
import inspect
import types

import api.app as app_module


def _fake_settings(enabled: bool = True):
    return types.SimpleNamespace(
        retrieval=types.SimpleNamespace(warmup_retrieval_enabled=enabled),
    )


class _FakeRouter:
    """带三段 prewarm + retrieve 的最小 QueryRouter 替身。"""

    async def prewarm_bm25(self):
        return None

    async def prewarm_visual(self):
        return None

    async def prewarm_reranker(self):
        return None

    def retrieve(self, query: str, **kwargs):
        return []


def _fake_svc():
    return types.SimpleNamespace(query_router=_FakeRouter())


def _fake_rebuild_index():
    async def _rebuild(svc):
        return {"indexed_count": 0}

    return _rebuild


class TestWarmupBlocksReadiness:
    def test_warmup_flag_set_after_run(self, monkeypatch):
        """预热步骤结束（含失败路径）后 _retrieval_warmup_done 置位。"""
        app_module._retrieval_warmup_done = False
        monkeypatch.setattr(app_module, "get_settings", lambda: _fake_settings(True))

        app_module._run_retrieval_warmup(_fake_svc())

        assert app_module._retrieval_warmup_done is True, (
            "预热步骤结束后就绪标志必须置位"
        )

    def test_startup_prep_runs_warmup_synchronously(self, monkeypatch):
        """_startup_retrieval_prep 返回时预热已执行完毕（非后台 create_task）。"""
        warmed: list = []
        monkeypatch.setattr(
            app_module, "_run_retrieval_warmup", lambda svc: warmed.append(svc)
        )
        # 屏蔽 FAISS 重建（api.routes.rebuild_index 在函数内 import）
        import api.routes as routes

        monkeypatch.setattr(routes, "rebuild_index", _fake_rebuild_index())

        svc = _fake_svc()
        asyncio.run(app_module._startup_retrieval_prep(svc))

        assert warmed == [svc], "预热必须在 startup prep 返回之前完成"

    def test_lifespan_awaits_prep_before_yield(self):
        """lifespan 在 yield 前 await 检索就绪准备，且不再把预热放入后台任务。"""
        src = inspect.getsource(app_module.lifespan.__wrapped__)

        prep_idx = src.index("await _startup_retrieval_prep(")
        yield_idx = src.index("\n    yield\n")
        assert prep_idx < yield_idx, "检索就绪准备必须在 yield 之前 await"

        # 预热调用（_run_retrieval_warmup）不得再出现在 lifespan 内（后台任务承载）
        assert "_run_retrieval_warmup" not in src, (
            "lifespan 不得再直接/后台承载预热调用，应由 yield 前 await 的 prep 完成"
        )
