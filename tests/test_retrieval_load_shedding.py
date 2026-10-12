"""
高并发显式卸载（retrieval load shedding）验收测试
==================================================
覆盖任务书 AC1..AC6（单元级可判定部分）：
  AC1 零回归：in-flight < 阈值时与开关关闭（改动前）逐元素一致
  AC4 可回退：retrieval_load_shedding_enabled=false → 行为与改动前一致（无卸载）
  AC5 无泄漏：计数器异常安全（所有返回路径递减）；并发 12 后 in-flight 回到 0
  AC6 可观测：卸载日志含 in-flight 数与跳过的步骤名
  附加：in-flight 计数器语义（嵌套不重复计数）、fail-soft（计数异常不卸载）、
        配置三处一致 + >=0 校验

AC2/AC3（并发 8 / 并发 12 真实延迟达标）依赖真实 store + 真实并发压测，由 Hermes
在评测环境复验；本文件用进程级 in-flight 计数 + 卸载闸口做确定性单元验证。

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_retrieval_load_shedding.py -q
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import retrieval.query_router as qr
from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel


def _reset_inflight() -> None:
    with qr._RETRIEVE_INFLIGHT_LOCK:
        qr._RETRIEVE_INFLIGHT = 0
    qr._RETRIEVE_INFLIGHT_TLS.entered = False


@pytest.fixture(autouse=True)
def _clean_inflight():
    """进程级计数器跨测试持久：每用例前后归零，保证测试隔离。"""
    _reset_inflight()
    qr.set_user_profile({})
    yield
    _reset_inflight()
    qr.set_user_profile({})


def _make_router(threshold: int = 4, enabled: bool = True) -> QueryRouter:
    """构造零依赖 QueryRouter（__new__ 绕过 __init__，不触发真实模型/索引加载）。"""
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig(
        retrieval_load_shedding_enabled=enabled,
        retrieval_load_shedding_threshold=threshold,
    )
    router._zh_en_tech_map = {}
    router._time_keywords = set()
    router.graph_store = MagicMock()
    router.encoder = None
    return router


def _passthrough(results, *args, **kwargs):
    return results


def _make_fusion_router(threshold: int = 4, enabled: bool = True,
                        expand: bool = False) -> QueryRouter:
    """FUSION 零依赖路由：_fusion_retrieve 返回 2 个种子；各扩展步 passthrough。

    expand=True 时 _community_expansion 追加一个标记节点（"extra"），用于区分
    「社区扩展已执行」与「被卸载跳过」；调用次数记入 router._community_calls。
    """
    router = _make_router(threshold=threshold, enabled=enabled)
    router.config.rerank_enabled = False  # 不加载 bge-reranker 模型（纯闸口验证）
    seeds = [
        {"node_id": "n0", "content": "alpha beta gamma", "score": 0.9,
         "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss"},
        {"node_id": "n1", "content": "delta epsilon zeta", "score": 0.8,
         "fact_track": "active", "tau_value": 1.0, "level": "l1_faiss"},
    ]

    def _fusion(q, emb=None, rq=None, now_ts=None, pool_k=None):
        return [dict(s) for s in seeds]

    router._fusion_retrieve = _fusion
    router._community_calls = []

    def _community(results, query, query_embedding, raw_query):
        router._community_calls.append(1)
        out = list(results)
        if expand:
            out.append({
                "node_id": "extra", "content": "community expansion member",
                "score": 0.5, "tau_value": 1.0, "fact_track": "active",
                "level": "community_expansion",
            })
        return out

    router._community_expansion = _community
    for name in (
        "_mesa_synthesis", "_visual_recall", "_property_temporal_retrieve",
        "_entity_expansion", "_attribute_expansion", "_scope_retrieve",
        "_schema_recall", "_fact_retrieve",
    ):
        setattr(router, name, _passthrough)
    return router


class _RecordingLogger:
    """记录 info 调用（structlog 不经 stdlib，caplog 抓不到 → 直接打桩）。"""

    def __init__(self):
        self.infos: list[tuple] = []

    def info(self, msg, *args, **kwargs):
        self.infos.append((msg, kwargs))

    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class TestLoadSheddingConfig:
    """配置三处一致 + >=0 校验。"""

    def test_query_router_config_defaults(self):
        cfg = QueryRouterConfig()
        assert cfg.retrieval_load_shedding_enabled is True
        assert cfg.retrieval_load_shedding_threshold == 4

    def test_settings_defaults(self):
        from config.settings import load_settings
        rc = load_settings().retrieval
        assert rc.retrieval_load_shedding_enabled is True
        assert rc.retrieval_load_shedding_threshold == 4

    def test_yaml_defaults(self):
        yaml_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert raw["retrieval"]["retrieval_load_shedding_enabled"] is True
        assert raw["retrieval"]["retrieval_load_shedding_threshold"] == 4

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match="retrieval_load_shedding_threshold"):
            QueryRouterConfig(retrieval_load_shedding_threshold=-1)
        from config.settings import RetrievalConfig
        with pytest.raises(ValueError, match="retrieval_load_shedding_threshold"):
            RetrievalConfig(retrieval_load_shedding_threshold=-1)


class TestInflightCounter:
    """进程级 in-flight 计数器语义（嵌套不重复计数）。"""

    def test_top_level_enter_exit(self):
        assert qr._inflight_now() == 0
        assert qr._inflight_enter() is True
        assert qr._inflight_now() == 1
        qr._inflight_exit()
        assert qr._inflight_now() == 0

    def test_nested_enter_not_counted(self):
        assert qr._inflight_enter() is True
        assert qr._inflight_enter() is False  # 同线程嵌套 → 不重复计数
        assert qr._inflight_now() == 1
        qr._inflight_exit()  # 只有顶层调用方递减
        assert qr._inflight_now() == 0


class TestLoadSheddingBehavior:
    """卸载闸口：in-flight >= 阈值才跳过社区扩展；否则逐字节一致。"""

    def test_low_concurrency_runs_community_expansion(self):
        """AC1 核心：in-flight(1) < 阈值(4) → 社区扩展正常执行（结果含 extra）。"""
        router = _make_fusion_router(threshold=4, expand=True)
        out = router.retrieve("alpha beta", level=RetrievalLevel.FUSION)
        assert len(router._community_calls) == 1
        assert [r["node_id"] for r in out] == ["n0", "n1", "extra"]
        assert qr._inflight_now() == 0

    def test_off_switch_byte_identical_to_low_concurrency(self):
        """AC1/AC4：开关关闭（改动前）与低并发（in-flight < 阈值）逐元素一致。"""
        off = _make_fusion_router(enabled=False, expand=True)
        on_low = _make_fusion_router(enabled=True, threshold=4, expand=True)
        out_off = off.retrieve("alpha beta", level=RetrievalLevel.FUSION)
        out_low = on_low.retrieve("alpha beta", level=RetrievalLevel.FUSION)
        # 逐元素比对（含顺位与分数），不是只比 id 列表
        assert out_off == out_low
        assert [(r["node_id"], r["score"]) for r in out_off] == \
            [(r["node_id"], r["score"]) for r in out_low]
        assert len(off._community_calls) == 1
        assert len(on_low._community_calls) == 1

    def test_high_concurrency_skips_community_expansion_and_logs(self, monkeypatch):
        """AC6：in-flight >= 阈值 → 跳过社区扩展，INFO 日志含 in-flight/阈值/步骤名。"""
        router = _make_fusion_router(threshold=4, expand=True)
        rec = _RecordingLogger()
        monkeypatch.setattr(qr, "logger", rec)
        with qr._RETRIEVE_INFLIGHT_LOCK:
            qr._RETRIEVE_INFLIGHT = 4  # 模拟已有 4 个并发请求在进行中
        out = router.retrieve("alpha beta", level=RetrievalLevel.FUSION)
        # 卸载后仍是完整融合结果（种子原样返回，不空、不掉到兜底）
        assert [r["node_id"] for r in out] == ["n0", "n1"]
        assert len(router._community_calls) == 0
        shed_logs = [
            (m, kw) for m, kw in rec.infos
            if m == "Retrieval load shedding: skipping community expansion"
        ]
        assert len(shed_logs) == 1, f"无卸载 INFO 日志: {rec.infos}"
        _, kw = shed_logs[0]
        assert kw["in_flight"] == 5  # 含本次请求（4 + 1）
        assert kw["threshold"] == 4
        assert kw["skipped_step"] == "community_expansion"

    def test_disabled_switch_no_shed_even_over_threshold(self, monkeypatch):
        """AC4：开关关闭 → 即使 in-flight 超过阈值也不卸载（社区扩展照常执行）。"""
        router = _make_fusion_router(enabled=False, expand=True)
        rec = _RecordingLogger()
        monkeypatch.setattr(qr, "logger", rec)
        with qr._RETRIEVE_INFLIGHT_LOCK:
            qr._RETRIEVE_INFLIGHT = 10
        out = router.retrieve("alpha beta", level=RetrievalLevel.FUSION)
        assert [r["node_id"] for r in out] == ["n0", "n1", "extra"]
        assert len(router._community_calls) == 1
        assert not any(
            m == "Retrieval load shedding: skipping community expansion"
            for m, _ in rec.infos
        ), "开关关闭时不得有卸载日志"

    def test_threshold_zero_disables_shedding(self):
        """阈值 <= 0 视为关闭卸载（镜像 budget_s<=0 关闭熔断的既有约定）。"""
        router = _make_fusion_router(threshold=0, expand=True)
        with qr._RETRIEVE_INFLIGHT_LOCK:
            qr._RETRIEVE_INFLIGHT = 10
        out = router.retrieve("alpha beta", level=RetrievalLevel.FUSION)
        assert [r["node_id"] for r in out] == ["n0", "n1", "extra"]
        assert len(router._community_calls) == 1

    def test_fail_soft_counter_exception(self, monkeypatch):
        """计数读取异常 → 不卸载，按原逻辑继续（社区扩展照常执行）。"""
        router = _make_fusion_router(threshold=4, expand=True)

        def _boom():
            raise RuntimeError("counter broken")

        monkeypatch.setattr(qr, "_inflight_now", _boom)
        out = router.retrieve("alpha beta", level=RetrievalLevel.FUSION)
        assert [r["node_id"] for r in out] == ["n0", "n1", "extra"]
        assert len(router._community_calls) == 1


class TestInflightNoLeak:
    """AC5：所有返回路径（含异常/CancelledError）都递减；并发后归零。"""

    def test_decremented_on_exception(self):
        router = _make_router()
        router._hypergraph_retrieve = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            router.retrieve("some query")
        assert qr._inflight_now() == 0

    def test_decremented_on_cancelled_error(self):
        router = _make_router()
        router._hypergraph_retrieve = MagicMock(side_effect=asyncio.CancelledError())
        try:
            router.retrieve("some query")
            raise AssertionError("should have raised CancelledError")
        except asyncio.CancelledError:
            pass
        assert qr._inflight_now() == 0

    def test_concurrent_retrieves_return_to_zero(self):
        """AC5 实测：并发 8 线程检索两轮后 in-flight 计数回到 0。"""
        router = _make_fusion_router(threshold=4, expand=False)
        barrier = threading.Barrier(8)

        def worker(i: int):
            barrier.wait()
            return router.retrieve(
                f"concurrent load shedding query {i}", level=RetrievalLevel.FUSION)

        for _ in range(2):  # 连续跑两轮（AC5「连续跑并发 12 两次」的 8 线程等价）
            with ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(worker, range(8)))
            assert qr._inflight_now() == 0, "并发一轮结束后 in-flight 必须归零"
        # 并发结束后单条查询仍正常（20 条语义的零依赖等价：2 个种子原样返回）
        out = router.retrieve("single after concurrency", level=RetrievalLevel.FUSION)
        assert [r["node_id"] for r in out] == ["n0", "n1"]
        assert qr._inflight_now() == 0
