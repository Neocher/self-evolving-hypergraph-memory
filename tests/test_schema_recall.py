"""
Schema 模式蒸馏 + 检索测试（阶段4-1，v6.0.0）
================================================
覆盖：纯规则蒸馏（SSM 回放式频繁模式）生成 SchemaNode（:Conceptual 标签）、
run_once 评测前蒸馏入口（不阻塞在线检索）、_schema_recall 检索通道命中
（公共入口 FUSION retrieve）、graphlite/overgraph 双后端、零回归守卫。
"""
import time

import numpy as np
import pytest

from core.schema_distiller import extract_terms, distill, run_once


# ─── 纯函数：术语提取 + 蒸馏 ────────────────────────────


def test_extract_terms_latin_and_cjk():
    terms = extract_terms("Machine learning research about 机器学习 研究")
    assert "machine" in terms and "learning" in terms
    assert "机器学习" in terms  # CJK 完整段保留（CONTAINS 子串匹配前提）
    assert "the" not in terms   # 停用词
    assert terms == list(dict.fromkeys(terms))  # 去重保序


def test_distill_creates_schema_nodes():
    episodes = [
        {"id": "e1", "content": "机器学习 研究 项目"},
        {"id": "e2", "content": "机器学习 研究 综述"},
        {"id": "e3", "content": "机器学习 研究 实验"},
        {"id": "e4", "content": "天气 今天 下雨"},
    ]
    schemas = distill(episodes, min_support=2)
    assert schemas, "共享频繁模式的组应产出 Schema 节点"
    # 每 Schema 具备属性化字段
    for s in schemas:
        assert s["id"] and s["support"] >= 2
        assert "机器学习" in s["pattern_keywords"]
        assert len(s["source_ids"]) >= 2
    # e4 孤立（无共享模式）不得被蒸馏
    assert all("天气" not in s["pattern_keywords"] for s in schemas)


def test_distill_min_support_boundary():
    episodes = [
        {"id": "e1", "content": "机器学习 研究"},
        {"id": "e2", "content": "机器学习 应用"},
    ]
    assert len(distill(episodes, min_support=3)) == 0  # 支持数不足 → 无 Schema
    assert len(distill(episodes, min_support=2)) >= 1  # 恰好达阈值 → 产出


# ─── 存储原语 + run_once（graphlite 真实库）──────────────


def test_run_once_graphlite_creates_and_recalls(overgraph_store):
    store = overgraph_store
    for i, content in enumerate(["机器学习 研究 项目", "机器学习 研究 综述",
                                 "机器学习 研究 实验"]):
        store.create_episode({"content": content, "created_at": time.time()})

    created = run_once(store, limit=100, min_support=2)
    assert created, "蒸馏应产出 Schema 节点"
    hits = store.query_schema_nodes(["机器学习"])
    assert hits and any(h.get("id") in created for h in hits)
    # 读侧契约：pattern_keywords 为空格连接串（CONTAINS 友好）
    node = hits[0]
    assert "机器学习" in (node.get("pattern_keywords") or "")


def test_run_once_overgraph_creates_and_recalls(overgraph_store):
    store = overgraph_store
    for content in ["机器学习 研究 项目", "机器学习 研究 综述", "机器学习 研究 实验"]:
        store.create_episode({"content": content, "created_at": time.time()})

    created = run_once(store, limit=100, min_support=2)
    assert created
    hits = store.query_schema_nodes(["机器学习"])
    assert hits and any(h.get("id") in created for h in hits)


def test_run_once_overgraph_idempotent_no_accumulation(overgraph_store):
    """P1-2 幂等：重复 run_once（同 episodes）不累积重复 :Conceptual 节点。

    修复前 distill 每轮生成新 uuid4 → 同 pattern 重复 run_once 累积多份节点；
    修复后 uuid5 确定性 id（uuid5 over 完整 pattern）→ 同 id 覆盖（overgraph
    upsert / graphlite INSERT 覆盖），Schema 节点数量不增、产出 id 相同。
    """
    store = overgraph_store
    for content in ["机器学习 研究 项目", "机器学习 研究 综述", "机器学习 研究 实验"]:
        store.create_episode({"content": content, "created_at": time.time()})

    created1 = run_once(store, limit=100, min_support=2)
    assert created1
    count1 = len(store.query_schema_nodes(["机器学习"], limit=100))
    created2 = run_once(store, limit=100, min_support=2)
    assert created2
    count2 = len(store.query_schema_nodes(["机器学习"], limit=100))
    assert count2 == count1, (count1, count2)
    assert created2 == created1  # 确定性 id：两轮产出相同 id（覆盖而非累积）


def test_run_once_graphlite_idempotent_no_accumulation(overgraph_store):
    """P1 幂等（真实 GraphLiteStore）：重复 run_once（同 episodes）不累积
    重复 :Conceptual 节点。

    修复前 graphlite create_schema_node 是裸 INSERT（无覆盖语义），同 id
    重复执行产生重复节点（实测 count 1→2）；修复后两段式守卫（MATCH 命中
    SET 更新 / 未命中 INSERT）→ Schema 节点数量不增、产出 id 相同。
    """
    store = overgraph_store
    for content in ["机器学习 研究 项目", "机器学习 研究 综述", "机器学习 研究 实验"]:
        store.create_episode({"content": content, "created_at": time.time()})

    created1 = run_once(store, limit=100, min_support=2)
    assert created1
    count1 = len(store.query_schema_nodes(["机器学习"], limit=100))
    created2 = run_once(store, limit=100, min_support=2)
    assert created2
    count2 = len(store.query_schema_nodes(["机器学习"], limit=100))
    assert count2 == count1, (count1, count2)
    assert created2 == created1  # 确定性 id：两轮产出相同 id（覆盖而非累积）


def test_create_schema_node_graphlite_upsert_updates_attributes(overgraph_store):
    """P1 幂等（直测 create_schema_node）：同 id 重复写入走 SET 更新路径，
    不新增节点且属性更新（created_at/support 覆盖）。"""
    store = overgraph_store
    schema = {
        "id": "schema-1", "schema_name": "模式A",
        "pattern_keywords": ["机器学习"], "support": 2,
        "source_ids": "[]", "summary": "旧摘要", "created_at": 1.0,
    }
    assert store.create_schema_node(schema) == "schema-1"

    updated = dict(schema)
    updated["support"] = 5
    updated["summary"] = "新摘要"
    assert store.create_schema_node(updated) == "schema-1"

    nodes = store.query_schema_nodes(["机器学习"], limit=100)
    assert len(nodes) == 1, nodes  # 同 id 覆盖，不新增节点
    assert nodes[0]["support"] == 5 and nodes[0]["summary"] == "新摘要", nodes


# ─── 检索通道（公共入口 FUSION）─────────────────────────


@pytest.mark.parametrize("overgraph_store", [{"dimension": 384}], indirect=True)
def test_schema_recall_public_fusion(overgraph_store):
    """FUSION retrieve 公共入口：Schema 节点 append 上下文（尾分缩放 < 种子）。"""
    from retrieval.query_router import QueryRouter, QueryRouterConfig, RetrievalLevel

    store = overgraph_store
    for content in ["机器学习 研究 项目", "机器学习 研究 综述", "机器学习 研究 实验"]:
        store.create_episode({"content": content, "created_at": time.time()})
    run_once(store, limit=100, min_support=2)

    from retrieval.vector_index import VectorIndexAdapter

    class H:
        def __init__(self, dim=384):
            self.dim = dim
            self.dimension = dim

        def embed(self, t):
            rng = np.random.RandomState(hash(t) % (2**31))
            return rng.randn(self.dim).astype(np.float32)

        def embed_batch(self, ts):
            return np.array([self.embed(t) for t in ts])

    enc = H()
    adapter = VectorIndexAdapter(store=store, dimension=384, faiss_id_map={})
    adapter.rebuild([
        {"node_id": store.create_episode(
            {"content": "机器学习 研究 项目", "created_at": time.time()}),
         "embedding": enc.embed("机器学习 研究 项目")},
    ])
    qr = QueryRouter(
        graphlite_store=store, faiss_index=adapter, tfidf_index=None,
        encoder=enc, faiss_id_map={},
        config=QueryRouterConfig(top_k_vector=10, top_k_l1=5, top_k_keyword=10),
    )
    results = qr.retrieve(
        "机器学习 研究",
        query_embedding=enc.embed("机器学习 研究 项目"),
        level=RetrievalLevel.FUSION,
    )
    assert results, results
    schema = [r for r in results if r.get("_source") == "schema"]
    assert schema, results
    max_seed = max(float(r["score"]) for r in results if r.get("_source") != "schema")
    assert all(float(r["score"]) < max_seed for r in schema), (schema, max_seed)
    assert schema[0].get("schema_name"), schema[0]


def test_schema_recall_store_without_methods_noop():
    """store 无 query_schema_nodes → hasattr 守卫假 no-op（零回归）。"""
    from retrieval.query_router import QueryRouter, QueryRouterConfig

    class _Store:
        pass

    qr = QueryRouter(_Store(), None, None, config=QueryRouterConfig())
    out = qr._schema_recall(
        [{"node_id": "x", "content": "c", "score": 0.5}], "机器学习"
    )
    assert out == [{"node_id": "x", "content": "c", "score": 0.5}]
