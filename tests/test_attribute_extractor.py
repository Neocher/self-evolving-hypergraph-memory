"""Test Attribute Extractor — 属性提取（纯规则）。"""
import sys
sys.path.insert(0, "/home/admin/shm")
import pytest
from core.attribute_extractor import extract_attributes, ATTRIBUTE_PATTERNS


class _Ent:
    def __init__(self, name, etype, eid, aliases=None):
        self.name = name
        self.entity_type = etype
        self.entity_id = eid
        self.aliases = aliases or []


def test_extract_attributes_chinese():
    ents = [_Ent("张三", "Person", "ent_zs"), _Ent("某科技", "Organization", "ent_kj")]
    r = extract_attributes("ep1", "张三，现任CEO。某科技成立于2015年，总部位于北京。", ents)
    by = {(x.attr_name, x.attr_value) for x in r}
    assert ("title", "CEO") in by, r
    assert ("founded", "2015") in by, r
    assert ("location", "北京") in by, r
    # partition 区分语言
    assert all(x.partition.endswith("_cn") for x in r)


def test_extract_attributes_english():
    ents = [_Ent("Alice", "Person", "ent_a"), _Ent("Acme Inc", "Organization", "ent_acme")]
    r = extract_attributes("ep2", "Alice is the CEO. Acme Inc founded in 1999, headquartered in New York.", ents)
    by = {(x.attr_name, x.attr_value) for x in r}
    assert ("title", "CEO") in by, r
    assert ("founded", "1999") in by, r
    assert ("location", "New York") in by, r


def test_extract_attributes_no_anchor_drops():
    # 无锚点实体：属性模式命中但实体不存在 → 丢弃（不臆造实体）
    ents = []
    r = extract_attributes("ep3", "张三，现任CEO。某科技成立于2015年。", ents)
    assert r == []


def test_extract_attributes_dedup_same_episode():
    # 同一 (entity, attr, value) 单条 episode 内只取一次
    ents = [_Ent("张三", "Person", "ent_zs")]
    r = extract_attributes("ep4", "张三，现任CEO。张三再次出任CEO。", ents)
    titles = [x for x in r if x.attr_name == "title"]
    assert len(titles) == 1, titles
