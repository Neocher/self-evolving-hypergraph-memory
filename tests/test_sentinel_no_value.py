"""空串哨兵收缩回归测试（审计 P0-2）

背景：'__SHM_NO_VALUE__' 哨兵原在通用执行层 _prepare_params 全局替换所有字符串
参数（含 mutation），合法空串属性会被字面存储为哨兵；update_with_version 又不走
此路径 → 同一语义两套行为。

修复：哨兵仅收缩到 read_validate 的 $new_value 比较点（CONTAINS '' 恒真 → 空值
矛盾漏检防护），通用执行层不再替换空串。

判别力验证：
1. 经图库公共入口 execute_cypher 的 MATCH..SET 路径写入空串属性 → query_cypher
   回读必须原样保留（而非哨兵字面量）——旧实现此处 _prepare_params 会把空串
   替换为哨兵落库。
2. read_validate 的 new_value 比较点哨兵仍须生效（收缩不破坏原防护）。
3. 非空串 / 非 new_value 参数不被替换（收缩边界）。
"""
from __future__ import annotations

import time
import uuid

from graph.overgraph_store import _prepare_params

SENTINEL = "__SHM_NO_VALUE__"


def _find_prop(rows, key):
    """在混合形态行中定位属性（兼容 NodeView dict / props / 嵌套包裹）。"""
    for r in rows:
        if not isinstance(r, dict):
            continue
        # 整节点行 `RETURN e`：顶层 props 或 e.props
        for container in (r, r.get("e"), r.get("Node")):
            if isinstance(container, dict):
                props = container.get("props") or {}
                if key in props:
                    return props[key]
        # 直接平铺（RETURN e.id AS id, e.note AS note）
        if key in r:
            return r[key]
    return None


def test_empty_string_prop_roundtrips_via_gql(overgraph_store):
    """经公共 GQL 执行层（MATCH..SET，走 _prepare_params）写入空串 → 回读原样。

    旧实现：_prepare_params 把空串参数替换为哨兵 → SET e.note = $note 落库即
    哨兵字面量（合法空属性被污染）。修复后：通用执行层原样透传 → 回读仍是空串。
    """
    nid = str(uuid.uuid4())
    overgraph_store.create_episode({
        "id": nid,
        "content": "哨兵收缩判别", "created_at": time.time(),
        "tau_initial": 1.0, "source": "test",
    })

    # MATCH..SET 走 _locked_execute_gql → _prepare_params（哨兵替换的发生点）
    rows = overgraph_store.execute_cypher(
        "MATCH (e:EpisodeNode {id: $id}) SET e.note = $note, e.valid = $valid",
        {"id": nid, "note": "", "valid": "ok"},
    )
    assert rows, "MATCH..SET 应返回状态行"

    got = overgraph_store.query_cypher(
        "MATCH (e:EpisodeNode {id: $id}) RETURN e", {"id": nid}
    )
    assert got, "写入后应能回读"
    assert _find_prop(got, "valid") == "ok"
    note = _find_prop(got, "note")
    assert note == "", f"空串被哨兵污染: {note!r}"
    assert note != SENTINEL


def test_read_validate_new_value_sentinel_still_applied():
    """哨兵仍须在 read_validate 的 $new_value 比较点生效。

    判别力：收缩不能破坏原防护语义（CONTAINS '' 恒真 → 空值矛盾漏检防护）。
    """
    out = _prepare_params({"new_value": ""})
    assert out == {"new_value": SENTINEL}, f"new_value 空串应替换哨兵, got {out}"

    out2 = _prepare_params({"new_value": "real"})
    assert out2 == {"new_value": "real"}, "非空 new_value 不得替换"


def test_generic_params_not_replaced():
    """通用参数空串不得替换（收缩点边界）。"""
    out = _prepare_params({"content": ""})
    assert out == {"content": ""}, f"通用空串不得替换, got {out}"

    out2 = _prepare_params({"metadata": {"note": ""}})
    assert out2 == {"metadata": {"note": ""}}, "嵌套 dict 空串也不得替换"

    out3 = _prepare_params({})
    assert out3 is None
    assert _prepare_params(None) is None
