"""Test Schema 自进化 P0-② store 方法（真实 OverGraph 临时库集成）。"""
import sys
sys.path.insert(0, "/home/admin/shm")
import pytest


@pytest.fixture
def og_store(tmp_path):
    """真实 OverGraphStore 临时库。"""
    from graph.overgraph_store import OverGraphStore

    config = type("cfg", (), {"database_path": str(tmp_path / "og_db"),
                              "dense_vector_dimension": 512,
                              "dense_vector_metric": "cosine"})()
    store = OverGraphStore(config=config)
    store.connect()
    yield store


def test_locked_update_entity_props_read_modify_write(og_store):
    eid = og_store.create_entity("张三", entity_type="Person")
    # 第一次写 sidecar
    og_store.locked_update_entity_props(
        eid, lambda p: {**p, "attrs_json": {"title": {"candidates": {}}}})
    # 第二次读-改-写（整包替换语义：不丢 id/name）
    out = og_store.locked_update_entity_props(
        eid, lambda p: {**p, "attrs_json": {"title": {"solidified": {"value": "CEO"}}}})
    assert out["id"] == eid
    assert out["name"] == "张三"
    assert out["attrs_json"]["title"]["solidified"]["value"] == "CEO"


def test_create_rel_edge_and_neighbors(og_store):
    src = og_store.create_entity("华为", entity_type="Organization")
    dst = og_store.create_entity("腾讯", entity_type="Organization")
    og_store.create_rel_edge(src, dst, "PARTNER_WITH", confidence=0.8,
                             evidence_episode_ids=["ep1", "ep2"])
    # 幂等：再建不重复
    og_store.create_rel_edge(src, dst, "PARTNER_WITH", confidence=0.8,
                             evidence_episode_ids=["ep1"])
    neigh = og_store.get_rel_neighbors(src)
    partners = [n for n in neigh if n["predicate"] == "PARTNER_WITH"]
    assert len(partners) == 1, neigh
    assert partners[0]["dst_id"] == dst
    assert partners[0]["confidence"] == 0.8


def test_create_rel_edge_missing_endpoint_raises(og_store):
    src = og_store.create_entity("华为", entity_type="Organization")
    with pytest.raises(Exception):
        og_store.create_rel_edge(src, "ent_not_exist", "PARTNER_WITH")


def test_get_entity_attributes_sidecar(og_store):
    eid = og_store.create_entity("张三", entity_type="Person")
    attrs = og_store.get_entity_attributes(eid)
    assert attrs == {}
    og_store.locked_update_entity_props(
        eid, lambda p: {**p, "attrs_json": {"title": {"solidified": {"value": "CEO"}}}})
    attrs = og_store.get_entity_attributes(eid)
    assert attrs["title"]["solidified"]["value"] == "CEO"


def test_get_entity_relations_sidecar_and_edges(og_store):
    src = og_store.create_entity("华为", entity_type="Organization")
    dst = og_store.create_entity("腾讯", entity_type="Organization")
    og_store.create_rel_edge(src, dst, "PARTNER_WITH", confidence=0.8)
    rels = og_store.get_entity_relations(src)
    assert "PARTNER_WITH" in rels, rels
    assert rels["PARTNER_WITH"][dst]["solidified"] is True


def test_get_entity_episodes_by_episode(og_store):
    ep_id = og_store.create_episode({"id": "ep_x", "content": "华为与腾讯合作。", "source": "test"})
    og_store.link_entity_to_episode("华为", ep_id, entity_type="Organization")
    names = og_store.get_entity_episodes_by_episode(ep_id)
    assert "华为" in names, names
