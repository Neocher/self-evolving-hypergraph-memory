"""
OverGraphStore 集成测试（v6.0.0 OverGraph 引擎，公共入口）
==========================================================
与 GraphLiteStore 行为对拍：create_episode/get/query_cypher/超边/CAS/中文。
全部走 store 公共方法（AGENTS.md 假绿禁令：不直调内部方法绕过生产链路）。

覆盖 design_overgraph_vector.md D 语法契约：零 b64、翻译层（INSERT 节点/边、
逗号 MATCH、RETURN e.*、裸 RETURN、LIKE→CONTAINS、空串哨兵）、CAS、
超边幂等、属性版本 txn 补偿链、熔断器。
"""
import time

import numpy as np
import pytest

pytestmark = pytest.mark.overgraph

# 依赖缺失 → 整模块 skip（不崩收集；统一顶层导入策略，P2#8）
pytest.importorskip("overgraph")


# ─── Episode CRUD / 中文原生 ────────────────────────────


def test_create_get_episode_chinese(overgraph_store):
    """create_episode + get_episode：中文原生（无 {b64}）+ 默认基线字段。"""
    store = overgraph_store
    eid = store.create_episode({"content": "上海天气不错", "created_at": time.time()})
    ep = store.get_episode(eid)
    assert ep is not None
    assert ep["content"] == "上海天气不错"
    assert ep["id"] == eid
    assert ep["version"] == 1
    assert ep["archived"] is False
    assert ep["source_type"] == "direct"
    assert ep["fact_track"] == "active"
    assert "{b64}" not in str(ep)


def test_get_episode_missing(overgraph_store):
    assert overgraph_store.get_episode("no_such_episode") is None


def test_get_episodes_batch(overgraph_store):
    store = overgraph_store
    ids = [store.create_episode({"content": f"内容{i}"}) for i in range(3)]
    rows = store.get_episodes_batch(ids)
    assert len(rows) == 3
    assert {r["id"] for r in rows} == set(ids)
    assert store.get_episodes_batch([]) == []
    assert store.get_episodes_batch(["missing_1", "missing_2"]) == []


def test_query_cypher_chinese_contains(overgraph_store):
    """中文 CONTAINS 子串直查（PoC 实证：OverGraph 中文原生，零 b64）。"""
    store = overgraph_store
    store.create_episode({"content": "上海天气不错", "created_at": time.time()})
    rows = store.query_cypher(
        "MATCH (e:EpisodeNode) WHERE e.content CONTAINS '上海' "
        "RETURN e.id AS id, e.content AS content"
    )
    assert len(rows) == 1
    assert rows[0]["content"] == "上海天气不错"


def test_query_cypher_never_raises(overgraph_store):
    """永不抛异常契约（P0-2）：坏 GQL → []。"""
    store = overgraph_store
    assert store.query_cypher("MATCH (e:EpisodeNode) RETURN e.id AS id") == []
    assert store.query_cypher("THIS IS NOT GQL {{{") == []


def test_active_and_tau(overgraph_store):
    store = overgraph_store
    now = time.time()
    store.create_episode({"id": "ep_tau", "content": "tau节点", "tau_initial": 0.8,
                          "created_at": now})
    store.create_episode({"id": "ep_old", "content": "旧节点",
                          "tau_initial": 0.05, "created_at": now - 100000})
    active = store.get_active_episodes(3600)
    assert any(x["id"] == "ep_tau" for x in active)
    assert not any(x["id"] == "ep_old" for x in active)
    tau = store.get_episodes_by_tau_range(0.5, 1.0)
    assert any(x["id"] == "ep_tau" for x in tau)


# ─── CAS 乐观锁 ────────────────────────────────────────


def test_update_with_version_cas(overgraph_store):
    store = overgraph_store
    eid = store.create_episode({"content": "v0"})
    assert store.update_with_version(eid, {"content": "v1"}, expected_version=1) is True
    ep = store.get_episode(eid)
    assert ep["content"] == "v1" and ep["version"] == 2
    # 旧版本 CAS → False
    assert store.update_with_version(eid, {"content": "v2"}, expected_version=1) is False
    assert store.get_episode(eid)["content"] == "v1"
    # force 写入（不递增版本）
    assert store.update_with_version(eid, {"content": "forced"}, expected_version=None) is True
    assert store.get_episode(eid)["version"] == 2
    # 节点不存在 → False
    assert store.update_with_version("nope", {"content": "x"}, expected_version=1) is False


def test_archive_unarchive_supersedes(overgraph_store):
    store = overgraph_store
    store.create_episode({"id": "ep_old", "content": "旧"})
    new_id = store.create_episode({"content": "新"})
    assert store.archive_node("ep_old", new_id) is True
    assert store.get_episode("ep_old")["archived"] is True
    # SUPERSEDES 血统边
    rows = store.query_cypher(
        "MATCH (a:EpisodeNode {id: 'ep_old'})-[:SUPERSEDES]->(b:EpisodeNode {id: $id}) "
        "RETURN b.id AS id", {"id": new_id})
    assert len(rows) == 1
    # 幂等：重复 archive 不建重复边
    store.archive_node("ep_old", new_id)
    rows = store.query_cypher(
        "MATCH (a:EpisodeNode {id: 'ep_old'})-[:SUPERSEDES]->(b) RETURN count(*) AS cnt")
    assert rows[0]["cnt"] == 1
    # 不存在节点 → False
    assert store.archive_node("missing_node") is False
    # unarchive
    assert store.unarchive("ep_old") is True
    assert store.get_episode("ep_old")["archived"] is False


# ─── 超边 ──────────────────────────────────────────────


def test_hyperedge_flow(overgraph_store):
    store = overgraph_store
    hid = store.create_hyperedge_node({"id": "he_1", "type": "episode", "name": "测试超边"})
    e1 = store.create_episode({"content": "成员一"})
    e2 = store.create_episode({"content": "成员二"})
    store.link_hyperedge_member(hid, e1)
    store.link_hyperedge_member(hid, e2)
    store.link_hyperedge_member(hid, e1)  # 幂等守卫：不产生重复边

    members = store.get_hyperedge_members(hid)
    assert len(members) == 2
    assert {m["id"] for m in members} == {e1, e2}
    # 反查
    hyperedges = store.get_hyperedges_by_node(e1)
    assert len(hyperedges) == 1 and hyperedges[0]["id"] == hid
    # 共享超边邻居扩散
    nb = store.get_hypergraph_neighbors([e1])
    assert nb[e1][0]["id"] == e2
    assert store.get_hypergraph_neighbors([]) == {}
    assert store.get_hypergraph_neighbors(["missing_seed"]) == {}


# ─── 属性版本（txn 补偿链）─────────────────────────────


def test_property_version_supersede_chain(overgraph_store):
    store = overgraph_store
    pid1 = store.create_property_version("Apple", "ceo", "Tim", valid_from=100.0)
    latest = store.get_latest_property_version("Apple", "ceo")
    assert latest["value"] == "Tim" and latest["expired_at"] is None
    pid2 = store.create_property_version("Apple", "ceo", "Cook", valid_from=200.0,
                                         supersedes_id=pid1)
    latest = store.get_latest_property_version("Apple", "ceo")
    assert latest["value"] == "Cook"
    versions = store.get_property_versions("Apple", "ceo")
    assert len(versions) == 2
    old = [v for v in versions if v["id"] == pid1][0]
    assert old["expired_at"] == 200.0
    # 血统边
    rows = store.query_cypher(
        "MATCH (a:PropertyVerNode {id: $a})-[:SUPERSEDES]->(b:PropertyVerNode {id: $b}) "
        "RETURN count(*) AS cnt", {"a": pid1, "b": pid2})
    assert rows[0]["cnt"] == 1


def test_property_version_out_of_order_insert(overgraph_store):
    """乱序中段插入（supersedes_id + superseded_by）：双挂链 + 双向 expired_at。"""
    store = overgraph_store
    p1 = store.create_property_version("Ent", "attr", "v1", valid_from=100.0)
    p2 = store.create_property_version("Ent", "attr", "v2", valid_from=300.0,
                                       supersedes_id=p1)
    p3 = store.create_property_version("Ent", "attr", "v15", valid_from=150.0,
                                       supersedes_id=p1, superseded_by=p2)
    vs = store.get_property_versions("Ent", "attr")
    assert len(vs) == 3
    by_id = {v["id"]: v for v in vs}
    assert by_id[p1]["expired_at"] == 150.0
    assert by_id[p3]["expired_at"] == 300.0
    assert by_id[p2]["expired_at"] is None
    # 血统链 P1→P3→P2（查全部 SUPERSEDES 边，按 id 建集）
    rows = store.query_cypher(
        "MATCH (a:PropertyVerNode)-[:SUPERSEDES]->(b:PropertyVerNode) "
        "RETURN a.id AS a, b.id AS b")
    edges = {(r["a"], r["b"]) for r in rows}
    assert (p1, p3) in edges and (p3, p2) in edges and (p1, p2) not in edges


def test_property_version_prune_and_distinct(overgraph_store):
    store = overgraph_store
    for i in range(12):
        store.create_property_version("BigCo", "field", f"val{i}", valid_from=float(i))
    versions = store.get_property_versions("BigCo", "field")
    assert len(versions) == 12
    removed = store.prune_property_versions("BigCo", "field", max_versions=8)
    assert removed == 4
    versions = store.get_property_versions("BigCo", "field")
    assert len(versions) == 8
    names = store.get_distinct_attr_names()
    assert "field" in names


def test_property_versions_for_entities_normalized(overgraph_store):
    """P1-2 归一化前缀匹配（LIKE 前缀语义 → CONTAINS 等价，OverGraph 无 LIKE）。"""
    store = overgraph_store
    store.create_property_version("Apple Inc", "ceo", "Tim", valid_from=1.0)
    store.create_property_version("Applebee's", "ceo", "X", valid_from=1.0)
    rows = store.get_property_versions_for_entities(["Apple"])
    assert len(rows) == 1 and rows[0]["entity_id"] == "Apple Inc"


# ─── 翻译层（101 处裸 GQL 收敛面）──────────────────────


def test_translation_insert_node_and_query(overgraph_store):
    """INSERT 节点 → typed upsert_node；RETURN s 整节点 → flatten props。"""
    store = overgraph_store
    r = store.execute_cypher(
        "INSERT (:SystemNode {id: $id, payload: $payload})",
        {"id": "sys_1", "payload": "{}"})
    assert r  # mutation 状态行 truthy（GraphLite INSERT 契约）
    rows = store.query_cypher("MATCH (s:SystemNode {id: 'sys_1'}) RETURN s")
    assert len(rows) == 1
    flat = store._flatten_row(rows[0], "s")
    assert flat["id"] == "sys_1" and flat["payload"] == "{}"


def test_translation_comma_match_edge_insert_weight(overgraph_store):
    """逗号 MATCH + INSERT 边（hebbian 模式）→ 重复 MATCH + CREATE + weight SET。"""
    store = overgraph_store
    a = store.create_episode({"content": "A"})
    b = store.create_episode({"content": "B"})
    r = store.execute_cypher(
        "MATCH (a:EpisodeNode {id: $a}), (b:EpisodeNode {id: $b}) "
        "INSERT (a)-[:HEBBIAN_CONNECTION {weight: 0.7}]->(b)",
        {"a": a, "b": b})
    assert r
    conns = store.get_all_hebbian_connections()
    assert len(conns) == 1
    assert abs(float(conns[0]["weight"]) - 0.7) < 1e-6
    all_conns = store.get_all_connections()
    assert abs(all_conns[a][b] - 0.7) < 1e-6


def test_translation_star_return(overgraph_store):
    """RETURN e.* → RETURN e + flatten props（dream_scheduler/user_profile 消费）。"""
    store = overgraph_store
    store.create_episode({"id": "ep_s1", "content": "星号", "created_at": 100.0})
    rows = store.query_cypher(
        "MATCH (e:EpisodeNode {id: 'ep_s1'}) RETURN e.*")
    assert len(rows) == 1
    # P0#3 契约：整节点 props 提升到行顶层（communities/hyperedges/system/
    # gateway 消费方直接 row.get("id")，对齐 GraphLite RETURN e.* 每属性一列）
    assert rows[0]["id"] == "ep_s1"
    assert rows[0]["content"] == "星号"
    assert rows[0]["created_at"] == 100.0
    # OverGraphStore._flatten_row 展开 props
    flat = store._flatten_row(rows[0], "e")
    assert flat["content"] == "星号" and flat["created_at"] == 100.0
    # OverGraphStore._flatten_row 兼容形态（user_profile 消费方零改动）
    from graph.overgraph_store import OverGraphStore
    flat2 = OverGraphStore._flatten_row(rows[0], "e")
    assert flat2["content"] == "星号"


def test_translation_not_exists_purge(overgraph_store):
    """NOT EXISTS → OPTIONAL MATCH + count=0 改写（P0#1：超边孤儿清理真实生效）。

    OverGraph 不支持 EXISTS 子查询/模式谓词 —— 旧实现透传 parse error →
    query_cypher 永不抛契约吞掉 → orphan_count 恒 0 清理 no-op。改写后
    count 与 DETACH DELETE 均按真实无出边语义执行（实测对拍）。
    """
    store = overgraph_store
    store.create_hyperedge_node({"id": "he_has", "type": "episode"})
    store.create_hyperedge_node({"id": "he_orphan", "type": "episode"})
    e1 = store.create_episode({"content": "成员"})
    store.link_hyperedge_member("he_has", e1)
    # 翻译层改写验证（白盒：确认 NOT EXISTS 未原样透传）
    from graph.overgraph_store import _rewrite_not_exists
    q = ("MATCH (h:HyperedgeNode) "
         "WHERE NOT EXISTS { (h)-[:HYPEREDGE_MEMBER]->() } "
         "RETURN count(h) AS cnt")
    rewritten = _rewrite_not_exists(q)
    assert "NOT EXISTS" not in rewritten
    assert "OPTIONAL MATCH" in rewritten and "count(__m)" in rewritten
    # 公共入口：孤儿计数（只 he_orphan 无出边）
    rows = store.query_cypher(q)
    assert rows == [{"cnt": 1}]
    # 公共入口：孤儿清理 DETACH DELETE（he_has 保留，he_orphan 删除）
    store.query_cypher(
        "MATCH (h:HyperedgeNode) "
        "WHERE NOT EXISTS { (h)-[:HYPEREDGE_MEMBER]->() } "
        "DETACH DELETE h")
    assert store.get_node_internal_id("he_has", "HyperedgeNode") is not None
    assert store.get_node_internal_id("he_orphan", "HyperedgeNode") is None
    assert store.query_cypher(q) == [{"cnt": 0}]


def test_hyperedges_list_query_shape(overgraph_store):
    """/hyperedges 路由查询形态（OPTIONAL MATCH + WITH collect + RETURN h.*）。

    P0#2/#3：OverGraph 原生支持 OPTIONAL MATCH/WITH/collect（PoC 实证），
    真实断点是行形态 —— RETURN h.*, member_ids 须把 h props 提升顶层并与
    member_ids 合并（路由消费方 row["id"] 零改动）。
    """
    store = overgraph_store
    store.create_hyperedge_node({"id": "he_a", "type": "episode", "created_at": 1.0})
    store.create_hyperedge_node({"id": "he_b", "type": "episode", "created_at": 2.0})
    e1 = store.create_episode({"content": "成员一"})
    store.link_hyperedge_member("he_a", e1)
    rows = store.query_cypher(
        "MATCH (h:HyperedgeNode) "
        "OPTIONAL MATCH (h)-[:HYPEREDGE_MEMBER]->(e:EpisodeNode) "
        "WITH h, collect(e.id) AS member_ids "
        "RETURN h.*, member_ids ORDER BY h.created_at DESC LIMIT $limit",
        {"limit": 50})
    by_id = {r["id"]: r for r in rows}
    assert set(by_id) == {"he_a", "he_b"}
    assert by_id["he_a"]["member_ids"] == [e1]
    assert by_id["he_b"]["member_ids"] == []
    assert by_id["he_a"]["type"] == "episode"


def test_translation_no_as_bare_columns(overgraph_store):
    """无 AS 裸属性列 key 形态对拍（P1#4）：`RETURN c.id, c.episode_a` → 键为
    `c.id`/`c.episode_a`（communities.py 冲突列表消费方 r.get("c.episode_a")）。"""
    store = overgraph_store
    store.execute_cypher(
        "INSERT (:ConflictNode {id: 'c1', episode_a: 'A', episode_b: 'B', "
        "rule_id: 'r1', detected_at: 1.0, resolved: false})")
    rows = store.execute_cypher(
        "MATCH (c:ConflictNode) WHERE 1=1 "
        "RETURN c.id, c.episode_a, c.episode_b, c.rule_id, "
        "c.detected_at, c.resolved")
    assert len(rows) == 1
    r = rows[0]
    assert r["c.id"] == "c1"
    assert r["c.episode_a"] == "A"
    assert r["c.episode_b"] == "B"
    assert r["c.resolved"] is False
    # 消费方访问形态（与 communities.py L117-146 一致）
    assert r.get("c.episode_a") == "A"
    assert r.get("c.id") == "c1"


def test_translation_var_length_path(overgraph_store):
    """变长路径 RELATES_TO*1..3（P1#5）：ontology_validator 拓扑分真实计算。

    OverGraph 原生支持变长路径（PoC 实证：`*1..3` 有界路径 parse 通过）——
    补公共入口回归，锁死 2 跳命中 / 断连零命中的语义。
    """
    store = overgraph_store
    for name in ("Alpha", "Beta", "Gamma", "Isolated"):
        store.execute_cypher("INSERT (:OntologyEntity {name: $n})", {"n": name})
    store.execute_cypher(
        "MATCH (a:OntologyEntity {name: 'Alpha'}), (b:OntologyEntity {name: 'Beta'}) "
        "INSERT (a)-[:RELATES_TO]->(b)")
    store.execute_cypher(
        "MATCH (a:OntologyEntity {name: 'Beta'}), (b:OntologyEntity {name: 'Gamma'}) "
        "INSERT (a)-[:RELATES_TO]->(b)")
    # 2 跳在 1..3 界内（ontology_validator L992 主循环形态，plain MATCH）
    rows = store.execute_cypher(
        "MATCH (a:OntologyEntity {name: $a_name}) "
        "MATCH (b:OntologyEntity {name: $b_name}) "
        "MATCH (a)-[:RELATES_TO*1..3]-(b) RETURN count(*) AS cnt LIMIT 1",
        {"a_name": "Alpha", "b_name": "Gamma"})
    assert rows == [{"cnt": 1}]
    # 断连实体 → 0 行（plain MATCH 无路径不产出）
    rows = store.execute_cypher(
        "MATCH (a:OntologyEntity {name: $a_name}) "
        "MATCH (b:OntologyEntity {name: $b_name}) "
        "MATCH (a)-[:RELATES_TO*1..3]-(b) RETURN count(*) AS cnt LIMIT 1",
        {"a_name": "Alpha", "b_name": "Isolated"})
    assert rows == [{"cnt": 0}]
    # OPTIONAL MATCH 形态（ontology_validator L1013 共享实体对）不抛错
    rows = store.execute_cypher(
        "MATCH (a:OntologyEntity {name: $a_name}) "
        "MATCH (b:OntologyEntity {name: $b_name}) "
        "OPTIONAL MATCH (a)-[:RELATES_TO*1..3]-(b) RETURN count(*) AS cnt",
        {"a_name": "Alpha", "b_name": "Isolated"})
    assert rows == [{"cnt": 1}]  # OPTIONAL MATCH 无路径 → 1 行 count(*) 计数（同 GraphLite）


def test_translation_remove(overgraph_store):
    """REMOVE 属性（P2#11）：transaction_manager rollback 的 tx_tag 清理语义。

    OverGraph 原生支持 REMOVE（PoC 实证：SET→REMOVE→属性回 None）——
    补公共入口回归，锁死该语义（R1 疑 REMOVE 未翻译，实测已支持）。
    """
    store = overgraph_store
    store.create_episode({"id": "ep_rm", "content": "事务节点"})
    store.query_cypher("MATCH (n {id: 'ep_rm'}) SET n.tx_tag = 'tx_abc'")
    assert store.query_cypher(
        "MATCH (n {id: 'ep_rm'}) RETURN n.tx_tag AS t") == [{"t": "tx_abc"}]
    # transaction_manager.L156-164 同款语句
    store.query_cypher(
        "MATCH (n {id: $id}) WHERE n.tx_tag = $tag REMOVE n.tx_tag",
        {"id": "ep_rm", "tag": "tx_abc"})
    assert store.query_cypher(
        "MATCH (n {id: 'ep_rm'}) RETURN n.tx_tag AS t") == [{"t": None}]


def test_translation_bare_return_and_sentinel(overgraph_store):
    """裸 RETURN 合成（健康检查）+ 空串 $param 哨兵（NOT CONTAINS '' 恒假）。"""
    store = overgraph_store
    assert store.query_cypher("RETURN 1 AS test") == [{"test": 1}]
    store.create_episode({"content": "有内容"})
    rows = store.query_cypher(
        "MATCH (e:EpisodeNode) WHERE NOT e.content CONTAINS $kw RETURN e.id AS id",
        {"kw": ""})
    assert len(rows) == 0


def test_translation_offset_to_skip(overgraph_store):
    """OFFSET → SKIP（OverGraph 无 OFFSET，静默返回空 —— dashboard 分页防坑）。"""
    store = overgraph_store
    for i in range(5):
        store.create_episode({"content": f"c{i}", "created_at": float(i)})
    rows = store.query_cypher(
        "MATCH (e:EpisodeNode) RETURN e.content AS content "
        "ORDER BY e.created_at DESC LIMIT 2 OFFSET 2")
    assert len(rows) == 2
    assert {r["content"] for r in rows} == {"c1", "c2"}  # 第 3/4 新 → OFFSET 2 后取 2 条


def test_translation_standalone_create_node(overgraph_store):
    """独立 CREATE 节点带属性 map（dream_pipeline 超边/社区创建）→ typed upsert。"""
    store = overgraph_store
    store.execute_cypher(
        "CREATE (h:HyperedgeNode {id: $id, type: 'semantic', created_at: $t, "
        "gate_value: 1.0})",
        {"id": "he_create", "t": time.time()})
    he = store.get_node_internal_id("he_create", "HyperedgeNode")
    assert he is not None


def test_translation_multi_edge_insert(overgraph_store):
    """多边 INSERT（ontology_validator RELATES_TO 双向）→ 多 CREATE 单语句。"""
    store = overgraph_store
    store.execute_cypher(
        "INSERT (:OntologyEntity {name: $a})", {"a": "Alpha"})
    store.execute_cypher(
        "INSERT (:OntologyEntity {name: $b})", {"b": "Beta"})
    r = store.execute_cypher(
        "MATCH (a:OntologyEntity {name: $a_name}), (b:OntologyEntity {name: $b_name}) "
        "INSERT (a)-[:RELATES_TO {relation: 'co_occur'}]->(b), "
        "(b)-[:RELATES_TO {relation: 'co_occur'}]->(a)",
        {"a_name": "Alpha", "b_name": "Beta"})
    assert r
    rows = store.query_cypher(
        "MATCH (a:OntologyEntity {name: 'Alpha'})-[r:RELATES_TO]->(b) "
        "RETURN b.name AS name")
    assert {r["name"] for r in rows} == {"Beta"}


# ─── Session / Visual / 命名空间 ───────────────────────


def test_session_flow(overgraph_store):
    store = overgraph_store
    store.ensure_session("sess_1")
    store.ensure_session("sess_1")  # 幂等
    eid = store.create_episode({"content": "会话记忆"})
    store.link_to_session("sess_1", eid)
    store.link_to_session("sess_1", eid)  # 幂等
    memories = store.get_session_memories("sess_1")
    assert len(memories) == 1 and memories[0]["id"] == eid
    assert store.get_or_create_session("sess_2", "meta") == "sess_2"
    assert store.get_or_create_session("sess_2") == "sess_2"


def test_visual_nodes(overgraph_store):
    store = overgraph_store
    store.create_visual_node({"id": "vis_1", "caption": "测试图",
                              "embedding": [0.1, 0.2, 0.3]})
    vn = store.get_visual_node("vis_1")
    assert vn["caption"] == "测试图" and vn["embedding"] == [0.1, 0.2, 0.3]
    nodes = store.get_visual_nodes()
    assert len(nodes) == 1


def test_delete_namespace(overgraph_store):
    store = overgraph_store
    store.ensure_session("ns_1")
    e1 = store.create_episode({"content": "待删1"})
    e2 = store.create_episode({"content": "待删2"})
    store.link_to_session("ns_1", e1)
    store.link_to_session("ns_1", e2)
    deleted = store.delete_namespace("ns_1")
    assert deleted == 2
    assert store.get_episode(e1) is None
    assert store.get_episode(e2) is None


# ─── 熔断器（OverGraphError 计为基础设施失败）────────────


def test_circuit_breaker_counts_overgraph_error(overgraph_store, monkeypatch):
    """OverGraphError → 熔断窗口计数（设计 A7：唯一 SDK 成员）。

    P1#6：不直调 cb.record_failure —— mock 注入抛 OverGraphError 的底层执行
    单元，走 query_cypher 公共入口验证失败计数与 is_open()（熔断真实接线）。
    """
    store = overgraph_store
    from graph.overgraph_store import _INFRA_EXCEPTIONS, OverGraphError
    assert OverGraphError in _INFRA_EXCEPTIONS
    cb = store.circuit_breaker

    def _boom(gql, params=None):
        raise OverGraphError("模拟 infra 失败")

    monkeypatch.setattr(store, "_locked_execute_gql", _boom)
    assert not cb.is_open()
    # 窗口满（window_size 次失败）→ 跳闸（query_cypher 静默返回 [] 保持永不抛）
    for _ in range(cb.window_size):
        assert store.query_cypher("MATCH (e:EpisodeNode) RETURN e.id AS id") == []
    assert cb.is_open()
    # 跳闸后 query_cypher 静默返回 []（永不抛契约）
    assert store.query_cypher("MATCH (e:EpisodeNode) RETURN e.id AS id") == []
    # execute_cypher 抛 CircuitBreakerOpen（写路径显式失败）
    from graph.common import CircuitBreakerOpen
    with pytest.raises(CircuitBreakerOpen):
        store.execute_cypher("MATCH (e:EpisodeNode) RETURN e.id AS id")


# ─── 向量方法（HNSW 主通道）────────────────────────────


def test_vector_search_and_internal_id(overgraph_store):
    store = overgraph_store
    e1 = store.create_episode({"content": "向量一"})
    e2 = store.create_episode({"content": "向量二"})
    rng = np.random.default_rng(3)
    v1 = rng.standard_normal(512).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = rng.standard_normal(512).astype(np.float32)
    v2 /= np.linalg.norm(v2)
    store.batch_upsert_embeddings([
        {"node_id": e1, "embedding": v1},
        {"node_id": e2, "embedding": v2},
    ])
    hits = store.vector_search_dense(5, v1)
    assert hits[0][0] == e1
    assert abs(hits[0][1] - 1.0) < 1e-4
    # internal id 转换
    iid = store.get_node_internal_id(e1)
    assert isinstance(iid, int)
    assert store.get_episode_keys([iid]) == [e1]
    assert e1 in store.get_episode_keys()
    # batch 不破坏 props（读-合并-upsert）
    ep = store.get_episode(e1)
    assert ep["content"] == "向量一" and ep["version"] == 1


# ─── GraphLite/OverGraph 行为对拍（P1#7 真实 diff fixture）────────────


def _build_metric_store(metric: str, tmp_path):
    from graph.overgraph_store import OverGraphStore
    cfg = type("cfg", (), {"database_path": str(tmp_path) + f"_{metric}",
                           "dense_vector_dimension": 512,
                           "dense_vector_metric": metric})()
    s = OverGraphStore(config=cfg)
    s.connect()
    return s


def test_metric_l2_cosine_identical_scores(tmp_path):
    """R1 metric 结论实证补跑（P2#9）：l2/cosine 双开库同向量对拍 score 逐位一致。

    实证：引擎忽略 dense_vector_metric 选项，输出恒为 cosine —— d=1/s-1
    映射对两选项均成立（固化设计文档结论，防误删/误切换）。
    """
    rng = np.random.default_rng(42)
    v1 = rng.standard_normal(512).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = rng.standard_normal(512).astype(np.float32)
    v2 += 0.3 * v1
    v2 /= np.linalg.norm(v2)

    scores: dict[str, list[tuple[str, float]]] = {}
    for metric in ("l2", "cosine"):
        store = _build_metric_store(metric, tmp_path)
        e1 = store.create_episode({"content": "v1"})
        e2 = store.create_episode({"content": "v2"})
        store.batch_upsert_embeddings([
            {"node_id": e1, "embedding": v1},
            {"node_id": e2, "embedding": v2},
        ])
        hits = store.vector_search_dense(5, v1)
        scores[metric] = [(eid, round(float(s), 6)) for eid, s in hits]
        store.close()
    assert [s for _, s in scores["l2"]] == [s for _, s in scores["cosine"]]
    assert abs(scores["l2"][0][1] - 1.0) < 1e-4
    assert len(scores["l2"]) == 2


def test_hnsw_ef_search_wired(overgraph_store, monkeypatch):
    """HNSW ef_search 配置真实透传（P2#10）：config.ef_search → vector_search(ef_search=)。

    m/ef_construction 无 SDK 设置 API（open() 拒绝未知选项，实证）→ 已从
    HNSWConfig/defaults.yaml 移除；仅 ef_search 走 vector_search 生效。
    """
    from graph.overgraph_store import OverGraphStore

    cfg = type("cfg", (), {"database_path": "/tmp/nonexist_ef",
                           "dense_vector_dimension": 512,
                           "dense_vector_metric": "cosine",
                           "ef_search": 96})()
    store = OverGraphStore(config=cfg)
    # 属性须预声明（monkeypatch.setattr 拒绝给实例新增不存在的属性）
    store._db = type("fake_db", (), {"vector_search": None})()
    captured: dict = {}

    def _fake_vector_search(mode, k, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(store._db, "vector_search", _fake_vector_search)
    store.vector_search_dense(5, np.zeros(512, dtype=np.float32))
    assert captured.get("ef_search") == 96
    assert captured.get("label_filter") == {"labels": ["EpisodeNode"]}
    # 未配置 ef_search → 不传参（保持引擎默认）
    store2 = OverGraphStore(config=type("c", (), {
        "database_path": "/tmp/x", "dense_vector_dimension": 512,
        "dense_vector_metric": "cosine"})())
    store2._db = store._db
    captured.clear()
    monkeypatch.setattr(store2._db, "vector_search", _fake_vector_search)
    store2.vector_search_dense(5, np.zeros(512, dtype=np.float32))
    assert "ef_search" not in captured


# ─── 端点级回归（P0#2/#3：/communities /hyperedges /conceptual/analyze）────


def _make_route_app(store):
    """FastAPI 路由装配：dependency_overrides 注入 overgraph store。

    与 tests/test_conflict_revocation.py 同构（真实路由 + 真实 store）。
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes import router, Services, get_services
    from graph.hyperedge import HyperedgeManager

    svc = Services()
    svc.graphlite_store = store
    svc.hyperedge_manager = HyperedgeManager(store)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_services] = lambda: svc
    return TestClient(app)


def test_communities_endpoints_overgraph(overgraph_store):
    """GET /communities + /communities/{id}（OverGraph 后端，P0#3 顶层 props）。"""
    store = overgraph_store
    store.execute_cypher(
        "INSERT (:CommunityNode {id: 'com_1', name: '社区一', summary: '摘要一', "
        "member_count: 3, created_at: 1.0})")
    client = _make_route_app(store)
    r = client.get("/communities")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["communities"][0]["id"] == "com_1"
    assert body["communities"][0]["name"] == "社区一"
    assert body["communities"][0]["summary"] == "摘要一"
    assert body["communities"][0]["member_count"] == 3
    assert body["communities"][0]["keywords"] == []  # 生产不落 keywords → 默认 []
    r = client.get("/communities/com_1")
    assert r.status_code == 200
    assert r.json()["id"] == "com_1"


def test_hyperedges_list_endpoint_overgraph(overgraph_store):
    """GET /hyperedges（OverGraph 后端，OPTIONAL MATCH/WITH/collect 形态）。

    R1 P0#2 报 500：行未扁平化 → row["id"] KeyError。修复后返回
    HyperedgeResponse（id/member_ids/type）。
    """
    store = overgraph_store
    store.create_hyperedge_node({"id": "he_1", "type": "episode", "created_at": 1.0})
    e1 = store.create_episode({"content": "成员"})
    store.link_hyperedge_member("he_1", e1)
    client = _make_route_app(store)
    r = client.get("/hyperedges")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    item = body["hyperedges"][0]
    assert item["id"] == "he_1"
    assert item["member_ids"] == [e1]
    assert item["type"] == "episode"


def test_system_conceptual_analyze_endpoint(overgraph_store):
    """POST /conceptual/analyze（OverGraph 后端，system.py L588-604 消费 c.*）。

    概念聚合为确定性本地逻辑（无 LLM）：2 社区共享 keyword → 发现 1 概念。
    """
    store = overgraph_store
    for i in (1, 2):
        store.execute_cypher(
            f"INSERT (:CommunityNode {{id: 'cc_{i}', name: '社区{i}', "
            f"summary: '摘要{i}', keywords: '[\"共享概念\"]', created_at: {i}.0}})")
    client = _make_route_app(store)
    r = client.post("/conceptual/analyze")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["concepts_found"] >= 1
    assert any(c["concept_name"] == "共享概念" for c in body["concepts"])
