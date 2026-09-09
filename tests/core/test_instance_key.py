"""R8 E3 实例身份 key — core/instance_key.py G1 oracle 单测 (TDD 先行).

本体论验收句: 两提及同实例?
    → 由 key 判定: sha1(entity|type|iso_week|obj|obj_desc) 确定性裁决 → 过。

G1 硬性覆盖: ① q0139 形状 (同 obj 异描述 → 异 key) ② 跨年邻接周
③ obj 缺失回退 (行为固定) ④ 同一事件两提及 → 同 key ⑤ normalize_alias 别名归一。
"""
import datetime

import pytest

from core.instance_key import DEFAULT_ALIASES, instance_key, iso_week, normalize_alias


def test_q0139_same_obj_diff_description_diff_key():
    """q0139 形状: 同 obj 'guitar', 修饰语不同 (yellow-with-octopus vs
    shiny-purple) → 必须异 key (obj_desc 进 hash)。"""
    k1 = instance_key("u_amy", "Person", "2023-W42", "guitar", "yellow with octopus")
    k2 = instance_key("u_amy", "Person", "2023-W42", "guitar", "shiny purple")
    assert k1 != k2
    # key 形态: sha1 前 12 位 hex
    assert len(k1) == 12 and all(c in "0123456789abcdef" for c in k1)


def test_cross_year_adjacent_week_diff_key():
    """跨年邻接周: 2023-12-31 (2023-W52) 与 2024-01-01 (2024-W01) iso_week 不同,
    实例 key 亦不同 (iso_week 进 hash, 真日历不 mock)。"""
    w_old = iso_week(datetime.date(2023, 12, 31))
    w_new = iso_week(datetime.date(2024, 1, 1))
    assert w_old == "2023-W52"
    assert w_new == "2024-W01"
    assert w_old != w_new
    k_old = instance_key("u_amy", "Person", w_old, "guitar")
    k_new = instance_key("u_amy", "Person", w_new, "guitar")
    assert k_old != k_new
    # iso_week 入参也接受 date/datetime 本体 (内部归一)
    assert instance_key("u_amy", "Person", datetime.date(2023, 12, 31), "guitar") == k_old


def test_obj_missing_fallback_fixed():
    """obj 缺失 (None/空串) 回退到固定哨兵: 与显式 obj 异 key, 且行为可复现。"""
    k_none = instance_key("u_amy", "Person", "2023-W42", None)
    k_empty = instance_key("u_amy", "Person", "2023-W42", "")
    k_obj = instance_key("u_amy", "Person", "2023-W42", "guitar")
    assert k_none == k_empty
    assert k_none != k_obj
    # 行为固定: 同入参必同 key
    assert k_none == instance_key("u_amy", "Person", "2023-W42", None)


def test_same_event_two_mentions_same_key():
    """同一事件两提及 (同 entity/type/week/obj/obj_desc) → 同 key;
    obj_desc 缺省 None 与显式空串视为等价。"""
    a = instance_key("u_amy", "Person", "2023-W42", "guitar", "yellow with octopus")
    b = instance_key("u_amy", "Person", "2023-W42", "guitar", "yellow with octopus")
    assert a == b
    c = instance_key("u_amy", "Person", "2023-W42", "guitar")
    d = instance_key("u_amy", "Person", "2023-W42", "guitar", "")
    assert c == d


def test_normalize_alias_default_and_injectable():
    """别名归一: 默认词典含 colleague→Rob; 词典可注入覆盖; 未命中原样返回。"""
    assert DEFAULT_ALIASES.get("colleague") == "Rob"
    assert normalize_alias("colleague") == "Rob"
    assert normalize_alias("Rob") == "Rob"
    assert normalize_alias("colleague", {"colleague": "Sam"}) == "Sam"
    assert normalize_alias("colleague", {}) == "colleague"  # 注入空词典 → 关闭默认
    assert normalize_alias("stranger", {"colleague": "Rob"}) == "stranger"


def test_instance_key_normalizes_alias_to_canonical():
    """instance_key 对 entity 应用别名归一: colleague 与 Rob 同 key。"""
    k_alias = instance_key("colleague", "Person", "2023-W42", "guitar")
    k_canon = instance_key("Rob", "Person", "2023-W42", "guitar")
    assert k_alias == k_canon
