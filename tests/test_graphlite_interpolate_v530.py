"""v5.30.0 _interpolate 前缀碰撞修复测试（P0 静默数据丢失）。

背景：旧实现按 dict 序对每个 $key 逐次 str.replace —— '$t1' 是 '$t10' 的
子串前缀，$t1 会先污染 $t10（→ 'one0'）；query_router._entity_match 对
≥10 个候选（t0..t12）生成多段 CONTAINS 时命中此 bug → 检索静默漏检。
修复：单次 re.sub(r"\\$([A-Za-z_]\\w*)", callback, query)，回调按捕获的
完整键名查 params——键序无关、无前缀碰撞；未知键返回原 match（"未命中
不替换"语义保持不变）。

运行: python3 -m pytest tests/test_graphlite_interpolate_v530.py -q
"""
from base64 import b64encode

import numpy as np
import pytest

pytestmark = pytest.mark.graphlite  # 【v6.0.0 legacy】GraphLite 专属语义测试（默认排除，addopts -m 'not graphlite'）

from graph.graphlite_store import GraphLiteStore


def _interp(query: str, params: dict) -> str:
    return GraphLiteStore._interpolate(query, params)


class TestPrefixCollision:

    def test_t1_t10_no_prefix_collision(self):
        """核心：$t1/$t10/$limit 共存，按完整键名替换，无 '$' 残留"""
        params = {"t1": "one", "t10": "ten", "limit": 20}
        q = _interp(
            "MATCH (e) WHERE e.a CONTAINS $t1 OR e.b CONTAINS $t10 "
            "RETURN e.id LIMIT $limit",
            params,
        )
        assert q == (
            "MATCH (e) WHERE e.a CONTAINS 'one' OR e.b CONTAINS 'ten' "
            "RETURN e.id LIMIT 20"
        )
        assert "$" not in q

    def test_t1_t10_reversed_dict_order(self):
        """键序无关：dict 插入序 t10 在前，结果一致"""
        params = {"t10": "ten", "t1": "one", "limit": 20}
        q = _interp("WHERE a=$t1 AND b=$t10 LIMIT $limit", params)
        assert q == "WHERE a='one' AND b='ten' LIMIT 20"
        assert "$" not in q

    def test_no_dollar_residue_for_known_keys(self):
        """完全已知键 → 输出无任何 '$' 残留"""
        params = {"t1": "one", "t2": "two"}
        q = _interp("WHERE x=$t1 AND y=$t2", params)
        assert "$" not in q
        assert q == "WHERE x='one' AND y='two'"


class TestManyParams:

    def test_t0_to_t12_all_segments(self):
        """≥10 参数全量：13 段 CONTAINS 逐段正确替换（实体匹配真实形态）"""
        params = {f"t{i}": i for i in range(13)}
        params["limit"] = 40
        conditions = " OR ".join(f"e.content CONTAINS $t{i}" for i in range(13))
        cypher = (
            f"MATCH (e:EpisodeNode) WHERE {conditions} "
            f"RETURN e.id AS node_id, e.content AS content LIMIT $limit"
        )
        q = _interp(cypher, params)
        for i in range(13):
            assert f"CONTAINS {i}" in q, f"段 t{i} 未替换为字面量 {i}"
            assert f"$t{i}" not in q, f"占位符 $t{i} 残留"
        assert "LIMIT 40" in q
        assert "$" not in q


class TestTypeBranchesPreserved:

    def test_all_types_mixed(self):
        """类型分支保真：str/float/None/空串/中文原生直写/numpy 标量"""
        params = {
            "s": "hello",
            "f": 3.14,
            "n": None,
            "e": "",
            "zh": "中文内容",
            "i": np.float32(1.5),
        }
        q = _interp(
            "SET e.s=$s, e.f=$f, e.n=$n, e.e=$e, e.zh=$zh, e.i=$i",
            params,
        )
        assert "'hello'" in q
        assert "3.14" in q
        assert "NULL" in q
        assert "'__SHM_NO_VALUE__'" in q
        assert "1.5" in q
        assert "$" not in q
        # GraphLite lexer UTF-8 bug 已修复（fork 4452a96）——中文原生直写
        assert "'中文内容'" in q, "中文应原生直写（lexer UTF-8 已修复，不再 b64）"

    def test_empty_string_sentinel(self):
        """空串 → '__SHM_NO_VALUE__' 哨兵（NOT CONTAINS 恒真语义）"""
        q = _interp("WHERE NOT e.new_value CONTAINS $v", {"v": ""})
        assert q == "WHERE NOT e.new_value CONTAINS '__SHM_NO_VALUE__'"


class TestUnknownPlaceholder:

    def test_unknown_key_preserved(self):
        """未知占位符原样保留（未命中不替换语义）"""
        q = _interp("WHERE e.a = $unknown AND e.b = $known", {"known": "x"})
        assert "$unknown" in q
        assert "'x'" in q
        assert "known" in q

    def test_unknown_key_near_known_key(self):
        """未知键与已知键前缀相邻不互相干扰"""
        q = _interp("WHERE a=$t1 OR b=$t12", {"t1": "one", "t12": "twelve"})
        assert q == "WHERE a='one' OR b='twelve'"
        assert "$" not in q


@pytest.mark.parametrize("params", [None, {}])
def test_no_params_returns_query_unchanged(params):
    assert GraphLiteStore._interpolate("MATCH (e) RETURN e", params) == "MATCH (e) RETURN e"
