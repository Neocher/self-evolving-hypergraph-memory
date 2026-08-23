"""Test Schema 自进化检索通道 P1 — 属性匹配扩召回。"""
import sys
sys.path.insert(0, "/home/admin/shm")
import pytest


class _Cfg:
    entity_expansion = type("E", (), {"enabled": True, "boost": 0.9, "attr_boost": 0.85,
                                      "max_results": 10, "max_entities": 3})()


class _Store:
    """模拟 OverGraphStore 属性侧车 + MENTIONS 关联。"""

    def __init__(self):
        self.entities = [
            {"id": "ent_zs", "name": "张三", "attrs_json": {
                "title": {"solidified": {"value": "CEO", "active": True, "conf": 0.8}},
            }},
            {"id": "ent_ls", "name": "李四", "attrs_json": {
                "title": {"solidified": {"value": "CTO", "active": True, "conf": 0.8}},
            }},
            {"id": "ent_noattr", "name": "王五", "attrs_json": {}},
        ]
        self.ep_by_ent = {"张三": ["ep_1", "ep_2"], "李四": ["ep_3"], "王五": ["ep_4"]}

    def get_entities(self, limit=500):
        return self.entities[:limit]

    def get_entity_episodes(self, name, limit=10):
        return self.ep_by_ent.get(name, [])[:limit]

    @staticmethod
    def _decode_sidecar(props, key):
        raw = props.get(key)
        if isinstance(raw, dict):
            return raw
        return {}


class _Router:
    def __init__(self):
        self.config = _Cfg()
        self.graphlite_store = _Store()

    def _attribute_expansion(self, results, query, raw_query, now_ts=None):
        from retrieval.query_router import QueryRouter
        # 复用真实实现：临时绑定 store/config
        r = QueryRouter.__new__(QueryRouter)
        r.config = self.config
        r.graphlite_store = self.graphlite_store
        return r._attribute_expansion(results, query, raw_query, now_ts=now_ts)


def _seed(score=0.8):
    return [{"node_id": "ep_seed", "content": "s", "score": score, "level": "seed"}]


def test_attribute_expansion_hits_solidified_value():
    r = _Router()
    out = r._attribute_expansion(_seed(), "张三 CEO", "张三 CEO")
    levels = {x["level"] for x in out}
    assert "attribute_expansion" in levels, out
    ep_ids = {x["node_id"] for x in out}
    assert "ep_1" in ep_ids and "ep_2" in ep_ids, out
    # 无属性的实体不进入
    assert "ep_4" not in ep_ids, out


def test_attribute_expansion_english_token():
    r = _Router()
    out = r._attribute_expansion(_seed(), "who is CEO", "who is CEO")
    ep_ids = {x["node_id"] for x in out}
    assert "ep_1" in ep_ids, out       # 张三 title=CEO 命中
    assert "ep_3" not in ep_ids, out   # 李四 title=CTO 不含 ceo token


def test_attribute_expansion_no_seed_score_returns_original():
    r = _Router()
    out = r._attribute_expansion([{"node_id": "x", "score": 0.0}], "CEO", "CEO")
    assert out == [{"node_id": "x", "score": 0.0}]


def test_attribute_expansion_skips_candidate_not_solidified():
    r = _Router()
    r.graphlite_store.entities.append(
        {"id": "ent_cand", "name": "赵六", "attrs_json": {
            "title": {"candidates": {"abc": {"value": "CFO"}}}}}  # 候选未固化
    )
    out = r._attribute_expansion(_seed(), "CFO", "CFO")
    ep_ids = {x["node_id"] for x in out}
    assert "ep_5" not in ep_ids  # 赵六无固化属性 → 不进检索


def test_attribute_expansion_score_boost():
    r = _Router()
    out = r._attribute_expansion(_seed(score=0.8), "CEO", "CEO")
    attr_items = [x for x in out if x["level"] == "attribute_expansion"]
    assert attr_items
    assert all(abs(x["score"] - 0.8 * 0.85) < 0.001 for x in attr_items), out
