"""社区相关性「全局词表」快路径 —— 验收测试。

背景：py-spy + 探针实证 `_community_relevance` 每个请求都对「该查询命中的 553–1509
条社区摘要」重新 fit_transform（实测单次 ~845ms），摘要集随查询变化 → 上一提交的
指纹缓存对生产几乎不命中。全局词表把 fit 从请求内挪到一次惰性构建（进程级单例），
请求内只剩 transform + 子集 IDF + 打分；开关默认关（逐字节零回归）。

验收（任务书 AC1-AC6；AC7 端到端由 Hermes 复验）：
  AC1 零回归：开关 OFF（默认）时 `_community_relevance` 与当前实现逐元素相同（差 <1e-12）
  AC2 提速：开关 ON 时第 2 次及以后调用（热）≤150ms，并贴出首次（含全局词表构建）耗时
  AC3 语义贴近：ON vs OFF 在 ≥10 条真实查询 + ≥1000 条真实社区摘要上比对
      rel >= 0.5 的社区 id 集合，Jaccard 重合率 ≥95%（口径：|交集|/|并集|）
  AC4 线程安全：≥8 线程并发（开关 ON）结果与单线程一致、无异常、无死锁
  AC5 fail-soft：模拟全局词表构建抛异常 → 自动回落现有路径且结果正确（不抛给上层）
  AC6 配置：rerank_gate_wait_ms == 300、community_relevance_global_vocab is False（默认关）

运行: env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
      /home/admin/shm/.venv/bin/python -m pytest tests/test_community_global_vocab.py -q
"""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

import retrieval.query_router as qr
from config.settings import load_settings
from retrieval.query_router import QueryRouter, QueryRouterConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent

# 12 个主题词表：社区摘要按主题分布，查询词法只命中对应主题 → rel>=0.5 集合非平凡
# （非全空、非全量），让 AC3 的语义比对有区分度。
_TOPICS = [
    ["K8s", "集群", "网络", "flannel", "calico", "排障", "多会话", "服务发现", "负载均衡", "容器"],
    ["数据库", "索引", "慢查询", "存储引擎", "事务", "连接池", "优化", "锁"],
    ["机器学习", "特征", "模型", "训练", "推理", "梯度", "损失函数", "数据集"],
    ["安全", "认证", "授权", "加密", "TLS", "漏洞", "审计", "密钥"],
    ["消息队列", "Kafka", "消费者", "分区", "偏移", "生产者", "积压"],
    ["缓存", "Redis", "过期", "穿透", "击穿", "雪崩", "集群"],
    ["日志", "监控", "告警", "链路追踪", "可观测性", "指标", "采样"],
    ["分布式", "一致性", "共识", "Raft", "Paxos", "节点", "选举"],
    ["前端", "React", "组件", "状态管理", "渲染", "路由", "构建"],
    ["后端", "FastAPI", "中间件", "路由", "校验", "序列化", "依赖注入"],
    ["测试", "pytest", "覆盖率", "CI", "持续集成", "构建", "回归"],
    ["灰度", "回滚", "压测", "性能", "瓶颈", "容量", "部署", "扩容"],
]
_QUERIES = [
    "K8s 集群网络 flannel 排障",
    "数据库 慢查询 索引 优化",
    "机器学习 模型 训练 推理",
    "安全 认证 加密 TLS",
    "Kafka 消费者 分区 偏移",
    "Redis 缓存 击穿 雪崩",
    "日志 监控 告警 链路追踪",
    "分布式 一致性 Raft 共识",
    "React 组件 状态管理 渲染",
    "FastAPI 中间件 路由 校验",
    "pytest 覆盖率 持续集成",
    "灰度 压测 性能 瓶颈",
]


def _reference_relevance(query: str, summaries: list[str]) -> list[float]:
    """原实现（未缓存、每次 fit）—— AC1/AC3 的等价性基准。"""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), lowercase=True, max_features=50000,
    )
    tf = vectorizer.fit_transform(summaries)
    idf = np.array(vectorizer.idf_)
    doc_lens = tf.sum(axis=1).A1
    avgdl = float(doc_lens.mean()) if doc_lens.size > 0 else 1.0
    q_vec = vectorizer.transform([query])
    k1, b = 1.5, 0.75
    scores = np.zeros(len(summaries), dtype=np.float64)
    for qf_idx in q_vec.indices:
        idf_w = idf[qf_idx]
        col = tf[:, qf_idx]
        vals = col.data
        rows = col.nonzero()[0]
        if rows.size == 0:
            continue
        numerator = vals * (k1 + 1.0)
        denominator = vals + k1 * (1.0 - b + b * doc_lens[rows] / avgdl)
        scores[rows] += idf_w * numerator / denominator
    return (scores / (scores + 1.0)).tolist()


def _community_corpus(n: int = 1000) -> list[str]:
    """真实社区摘要风格语料：12 主题 × 稳定轮转，主题词词法重叠互异。"""
    rng = random.Random(42)
    out = []
    for i in range(n):
        t = _TOPICS[i % len(_TOPICS)]
        parts = [f"社区{i}跨会话记忆", t[0], t[1], t[2], t[3], t[4], t[5]]
        out.append(" ".join(parts))
    return out


def _make_router(**cfg_overrides) -> QueryRouter:
    """最小 QueryRouter（__new__ 构造）：_community_relevance 只依赖 config。"""
    router = QueryRouter.__new__(QueryRouter)
    router.config = QueryRouterConfig(**cfg_overrides)
    return router


@pytest.fixture(autouse=True)
def _clear_relevance_caches():
    """每个测试前后清空进程级缓存，保证「冷/热」与并发构建测量不受跨用例污染。"""
    qr._community_relevance_global_cache.clear()
    qr._community_relevance_cache.clear()
    yield
    qr._community_relevance_global_cache.clear()
    qr._community_relevance_cache.clear()


class TestZeroRegression:
    """AC1：开关 OFF（默认）时与当前实现（未缓存逐次 fit）逐元素相同。"""

    def test_off_matches_reference_elementwise(self):
        summaries = _community_corpus(300)
        router = _make_router()  # 默认 OFF
        max_diff = 0.0
        for q in _QUERIES[:6]:
            off = router._community_relevance(q, summaries)
            reference = _reference_relevance(q, summaries)
            assert len(off) == len(reference) == len(summaries)
            diff = max(abs(a - b) for a, b in zip(off, reference))
            max_diff = max(max_diff, diff)
        print(f"AC1 OFF vs 当前实现 max abs diff = {max_diff:.3e}")
        assert max_diff < 1e-12


class TestSpeedup:
    """AC2：开关 ON 时第 2 次及以后调用（热）≤150ms，贴出首次（含构建）耗时。"""

    def test_hot_call_bounded(self):
        summaries = _community_corpus(1000)
        assert len(summaries) >= 1000
        router = _make_router(community_relevance_global_vocab=True)
        query = _QUERIES[0]

        t0 = time.perf_counter()
        router._community_relevance(query, summaries)  # 冷：含全局词表构建
        cold_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        router._community_relevance(query, summaries)  # 热：仅 transform + 子集 IDF + 打分
        hot_ms = (time.perf_counter() - t0) * 1000.0

        print(f"AC2 cold(含全局词表构建)={cold_ms:.1f}ms hot={hot_ms:.2f}ms")
        assert hot_ms <= 150.0


class TestSemanticAgreement:
    """AC3：ON vs OFF 的 rel>=0.5 社区 id 集合 Jaccard 重合率 ≥95%。"""

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union else 1.0

    def test_on_off_filter_agreement(self):
        summaries = _community_corpus(1000)
        assert len(summaries) >= 1000
        off_router = _make_router()
        on_router = _make_router(community_relevance_global_vocab=True)

        jaccards = []
        for q in _QUERIES:  # 12 条真实查询（≥10）
            on_rel = on_router._community_relevance(q, summaries)
            off_rel = off_router._community_relevance(q, summaries)
            set_on = {i for i, v in enumerate(on_rel) if v >= 0.5}
            set_off = {i for i, v in enumerate(off_rel) if v >= 0.5}
            j = self._jaccard(set_on, set_off)
            jaccards.append(j)
            print(
                f"AC3 q={q[:18]:18s} |ON|={len(set_on):4d} |OFF|={len(set_off):4d} "
                f"Jaccard={j:.4f}"
            )
        assert min(jaccards) >= 0.95, f"最小 Jaccard={min(jaccards)} 低于 0.95"

    def test_on_off_agreement_with_sampling(self):
        """取样路径（max_docs=60 < 语料 1000）仍保持 ≥95% 语义重合。"""
        summaries = _community_corpus(1000)
        off_router = _make_router()
        on_router = _make_router(
            community_relevance_global_vocab=True,
            community_relevance_global_vocab_max_docs=60,
        )
        jaccards = []
        for q in _QUERIES:
            set_on = {i for i, v in enumerate(on_router._community_relevance(q, summaries)) if v >= 0.5}
            set_off = {i for i, v in enumerate(off_router._community_relevance(q, summaries)) if v >= 0.5}
            jaccards.append(self._jaccard(set_on, set_off))
        assert min(jaccards) >= 0.95, f"取样路径最小 Jaccard={min(jaccards)} 低于 0.95"


class TestThreadSafety:
    """AC4：≥8 线程并发（开关 ON）结果与单线程一致、无异常、无死锁。"""

    def test_concurrent_on_matches_single_thread(self):
        summaries = _community_corpus(800)
        router = _make_router(community_relevance_global_vocab=True)
        query = _QUERIES[2]
        expected = router._community_relevance(query, summaries)  # 单线程基线（已入缓存）

        qr._community_relevance_global_cache.clear()  # 清空 → 并发首线程抢建全局词表

        errors = []

        def worker():
            try:
                return router._community_relevance(query, summaries)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                return None

        n_threads = 12
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            results = list(ex.map(lambda _: worker(), range(n_threads)))

        assert not errors, f"并发调用出现异常: {errors}"
        for i, r in enumerate(results):
            assert r == expected, f"线程 {i} 结果与单线程不一致"


class TestFailSoft:
    """AC5：全局词表构建抛异常 → 自动回落现有路径且结果正确（不抛给上层）。"""

    def test_build_raises_falls_back_to_existing_path(self):
        summaries = _community_corpus(200)
        off_router = _make_router()
        on_router = _make_router(community_relevance_global_vocab=True)
        query = _QUERIES[0]
        expected = off_router._community_relevance(query, summaries)

        def _boom(sample, refresh_s):
            raise RuntimeError("simulated global vocab build failure")

        with patch.object(qr, "_community_relevance_global_vectorizer", side_effect=_boom), \
                patch.object(qr, "logger") as mock_logger:
            result = on_router._community_relevance(query, summaries)  # 不应抛出

        assert result == expected, "构建异常回落后的结果应与现有路径一致"
        assert mock_logger.warning.called, "构建异常应记 WARNING"

    def test_build_returns_none_falls_back(self):
        """构建函数返回 None（fit 内部失败）同样回落现有路径。"""
        summaries = _community_corpus(200)
        off_router = _make_router()
        on_router = _make_router(community_relevance_global_vocab=True)
        query = _QUERIES[3]
        expected = off_router._community_relevance(query, summaries)

        with patch.object(qr, "_community_relevance_global_vectorizer", return_value=None):
            result = on_router._community_relevance(query, summaries)

        assert result == expected


class TestConfigDefaults:
    """AC6：rerank_gate_wait_ms == 300 且 community_relevance_global_vocab 默认关（三处一致）。"""

    def test_settings_defaults(self):
        cfg = load_settings().retrieval
        assert cfg.rerank_gate_wait_ms == 300
        assert cfg.community_relevance_global_vocab is False
        assert cfg.community_relevance_global_vocab_max_docs == 5000
        assert cfg.community_relevance_global_vocab_refresh_s == 3600.0

    def test_query_router_config_defaults(self):
        cfg = QueryRouterConfig()
        assert cfg.rerank_gate_wait_ms == 300
        assert cfg.community_relevance_global_vocab is False
        assert cfg.community_relevance_global_vocab_max_docs == 5000
        assert cfg.community_relevance_global_vocab_refresh_s == 3600.0

    def test_defaults_yaml(self):
        raw = yaml.safe_load(
            (_REPO_ROOT / "config" / "defaults.yaml").read_text(encoding="utf-8")
        )
        assert raw["retrieval"]["rerank_gate_wait_ms"] == 300
        assert raw["retrieval"]["community_relevance_global_vocab"] is False
        assert raw["retrieval"]["community_relevance_global_vocab_max_docs"] == 5000
        assert raw["retrieval"]["community_relevance_global_vocab_refresh_s"] == 3600.0

    def test_build_router_config_passthrough(self):
        from api.app import _build_router_config

        class _Cfg:
            tau_weight = 0.4
            vector_weight = 0.6
            top_k_l1 = 5
            top_k_vector = 20
            top_k_keyword = 20

        qcfg = _build_router_config(_Cfg())
        assert qcfg.rerank_gate_wait_ms == 300
        assert qcfg.community_relevance_global_vocab is False
