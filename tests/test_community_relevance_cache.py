"""给 _community_relevance 的 TF-IDF 加进程级缓存 —— 测试。

【并发超时专项·真正修复】社区摘要在社区重建之间不变，但原实现在每个请求里
fit_transform 一次（真实 store 3000 条摘要实测 836.7ms，93% 白付）。本修复缓存
只依赖 summaries 的构建态（vectorizer 含 idf_ / tf / doc_lens / avgdl），query 相关
的 transform + 打分仍每次现算。

验收（任务书 AC1-AC4；AC5/AC6 为重启服务后的端到端并发压测，由 Hermes 复验）：
  AC1 等价性：缓存版 vs 原实现（逐次 fit）逐元素最大绝对差 < 1e-9
  AC2 提速：冷(首次构建) vs 热(命中缓存)，热 ≤ 100ms 且 冷/热 ≥ 5x
  AC3 失效正确：长度相同内容不同的两组 summaries → 各自现算，不串缓存
  AC4 线程安全：≥8 线程并发 → 与单线程一致、无异常
"""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer


def _reference_relevance(query: str, summaries: list[str]) -> list[float]:
    """原实现（未缓存、每次 fit）——AC1/AC3 的等价性基准。"""
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


def _make_router():
    """最小 QueryRouter（__new__ 构造）：_community_relevance 已不依赖任何实例属性。"""
    from retrieval.query_router import QueryRouter
    return QueryRouter.__new__(QueryRouter)


def _realistic_summaries() -> list[str]:
    """真实社区摘要风格（中文工程/运维散文，含 Keywords 风格词表）。"""
    return [
        "K8s 集群网络 flannel calico 多会话排障 记忆",
        "容器编排 部署 服务发现 负载均衡 高可用",
        "数据库 索引 查询优化 慢查询 存储引擎",
        "日志 监控 告警 链路追踪 可观测性",
        "微服务 网关 熔断 限流 降级",
        "Python asyncio 协程 事件循环 并发",
        "机器学习 特征工程 模型训练 推理",
        "分布式 一致性 共识 Raft Paxos",
        "安全 认证 授权 加密 TLS",
        "测试 pytest 覆盖率 持续集成 CI",
        "前端 React 组件 状态管理 渲染",
        "后端 FastAPI 中间件 路由 校验",
        "缓存 Redis 过期 穿透 击穿 雪崩",
        "消息队列 Kafka 消费者 分区 偏移",
        "存储 对象 文件 块 分布式文件系统",
        "网络 TCP UDP HTTP DNS 代理",
        "容器镜像 构建 仓库 漏洞扫描",
        "配置管理 环境变量 密钥 版本控制",
        "灰度发布 蓝绿 金丝雀 回滚",
        "容量规划 压测 性能 瓶颈 优化",
    ]


def _synthetic_corpus(n_docs: int, n_chars: int) -> list[str]:
    """合成语料：每条凑满 n_chars 字符，用于放大 fit 成本、压小打分成本。"""
    rng = random.Random(0)
    words = [
        "K8s", "集群", "网络", "flannel", "calico", "多会话", "排障", "记忆",
        "部署", "容器", "服务", "节点", "配置", "数据", "查询", "存储",
        "监控", "日志", "负载", "扩容", "索引", "缓存", "熔断", "限流",
    ]
    out = []
    for _ in range(n_docs):
        parts = []
        total = 0
        while total < n_chars:
            w = rng.choice(words)
            parts.append(w)
            total += len(w)
        out.append("".join(parts))
    return out


@pytest.fixture(autouse=True)
def _clear_relevance_cache():
    """每个测试前后清空进程级缓存，保证「冷/热」测量不受其他用例污染。"""
    import retrieval.query_router as qr
    qr._community_relevance_cache.clear()
    yield
    qr._community_relevance_cache.clear()


class TestEquivalence:
    """AC1：缓存版与未缓存原实现分数一致。"""

    def test_cached_matches_uncached_reference(self):
        summaries = _realistic_summaries()
        queries = [
            "K8s 集群网络排障",
            "容器编排部署",
            "数据库慢查询优化",
            "日志监控告警",
            "微服务熔断限流",
            "Python 并发",
            "机器学习模型训练",
            "分布式一致性",
            "安全认证加密",
            "测试覆盖率",
            "前端组件",
            "缓存雪崩",
        ]
        router = _make_router()
        max_diff = 0.0
        for q in queries:
            cached = router._community_relevance(q, summaries)
            reference = _reference_relevance(q, summaries)
            assert len(cached) == len(reference) == len(summaries)
            diff = max(abs(a - b) for a, b in zip(cached, reference))
            max_diff = max(max_diff, diff)
        print(f"AC1 max abs diff = {max_diff:.3e}")
        assert max_diff < 1e-9


class TestSpeedup:
    """AC2：热调用 ≤ 100ms 且冷/热 ≥ 5x。"""

    def test_hot_call_bounded_and_speedup(self):
        summaries = _synthetic_corpus(n_docs=600, n_chars=300)
        router = _make_router()
        query = "K8s 网络排障"

        t0 = time.perf_counter()
        router._community_relevance(query, summaries)  # 冷：需构建
        cold_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        router._community_relevance(query, summaries)  # 热：命中缓存
        hot_ms = (time.perf_counter() - t0) * 1000.0

        ratio = cold_ms / hot_ms
        print(f"AC2 cold={cold_ms:.1f}ms hot={hot_ms:.2f}ms ratio={ratio:.1f}x")
        assert hot_ms <= 100.0
        assert ratio >= 5.0


class TestInvalidation:
    """AC3：长度相同内容不同 → 指纹不同、结果各自现算，不串缓存。"""

    def test_same_length_different_content_no_stale_cache(self):
        import retrieval.query_router as qr

        base_a = "K8s 集群网络 flannel calico 排障"
        base_b = "买菜 烹饪 家常菜 西红柿 鸡蛋 食谱"
        width = max(len(base_a), len(base_b)) + 5
        group_a = [base_a + "甲" * (width - len(base_a)) for _ in range(4)]
        group_b = [base_b + "乙" * (width - len(base_b)) for _ in range(4)]
        # 专打弱指纹：条数相同、逐条长度也相同，仅内容不同
        assert len(group_a) == len(group_b)
        assert [len(s) for s in group_a] == [len(s) for s in group_b]
        assert group_a != group_b
        assert qr._community_relevance_fingerprint(group_a) != \
            qr._community_relevance_fingerprint(group_b)

        router = _make_router()
        query = "K8s 网络排障"
        rel_a = router._community_relevance(query, group_a)  # 写入缓存
        rel_b = router._community_relevance(query, group_b)  # 必须失效重建

        ref_a = _reference_relevance(query, group_a)
        ref_b = _reference_relevance(query, group_b)
        assert np.allclose(rel_a, ref_a, atol=1e-12)
        assert np.allclose(rel_b, ref_b, atol=1e-12)
        # 第二组结果不等于第一组缓存结果（内容不同 → 分数不同）
        assert rel_b != rel_a


class TestThreadSafety:
    """AC4：≥8 线程并发同一份 summaries → 与单线程一致、无异常。"""

    def test_concurrent_calls_match_single_thread(self):
        summaries = _synthetic_corpus(n_docs=200, n_chars=200)
        router = _make_router()
        query = "K8s 集群 flannel 网络排障"
        expected = router._community_relevance(query, summaries)  # 单线程基线（已入缓存）

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
