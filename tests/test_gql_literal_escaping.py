"""
GQL 字面量转义 / 注入防护验收测试（v6.27.0 安全修复）
====================================================
针对 Hermes 盘点的 graph/overgraph_store.py 系统性 GQL 字符串拼接注入面：
把用户值直接 f-string 插进 MATCH/WHERE/CONTAINS 查询文本 → 单引号破坏语义、
恶意片段注入。本文件验证修复后的行为（$param 参数化 / 字面量转义）：

- AC1 字面量转义辅助函数 _esc_literal 正确转义 ' \\ \\n \\r
- AC2 含单引号 subject 写入后可正常按 subject 查询（行为断言，非仅不抛异常）
- AC3 注入串不得返回全表（count < N）
- AC4 至少三条函数路径：facts by subject / by id / like-rewrite
"""
import pytest

pytestmark = pytest.mark.overgraph

# 依赖缺失 → 整模块 skip（与 test_overgraph_store.py 同策略）
pytest.importorskip("overgraph")


# ─── AC1：字面量转义辅助函数 ──────────────────────────

def test_esc_literal_escapes_dangerous_chars():
    from graph.overgraph_store import _esc_literal

    assert _esc_literal("O'Brien") == "O\\'Brien"
    assert _esc_literal("a\\b") == "a\\\\b"
    assert _esc_literal("line\nbreak") == "line\\nbreak"
    assert _esc_literal("carriage\rreturn") == "carriage\\rreturn"
    # 无害值原样返回
    assert _esc_literal("plain text") == "plain text"
    assert _esc_literal("") == ""


def test_rewrite_like_escapes_literal():
    from graph.overgraph_store import _rewrite_like

    # 普通前缀：语义等价（尾部 % 剥离）
    assert _rewrite_like("LIKE 'apple%'") == "CONTAINS 'apple'"
    # 反斜杠字面量被转义，避免破坏单引号字面量
    assert _rewrite_like("LIKE 'a\\b%'") == "CONTAINS 'a\\\\b'"
    # 换行字面量被转义
    assert _rewrite_like("LIKE 'a\nb%'") == "CONTAINS 'a\\nb'"


# ─── AC4（路径 3）：like-rewrite 端到端 ────────────────

def test_like_rewrite_path_end_to_end(overgraph_store):
    """LIKE → CONTAINS 重写经 query_cypher 公共入口生效，前缀语义等价。"""
    store = overgraph_store
    store.create_episode({"id": "ep_like", "content": "LIKE 前缀命中"})
    rows = store.query_cypher(
        "MATCH (e:EpisodeNode) WHERE e.content LIKE 'LIKE%' RETURN e.id AS id")
    assert [r["id"] for r in rows] == ["ep_like"]


# ─── AC2：含单引号的值不再破坏查询 ─────────────────────

def test_subject_with_single_quote_roundtrips(overgraph_store):
    store = overgraph_store
    fid = store.create_atomic_fact("O'Brien 机组", "担任", "轮机长")
    rows = store.get_atomic_facts_by_subject("O'Brien 机组")
    assert [r["id"] for r in rows] == [fid]


# ─── AC3：注入串不得返回全表 ──────────────────────────

def test_injection_payload_does_not_return_all(overgraph_store):
    store = overgraph_store
    subjects = ["alpha 一号", "beta 二号", "gamma 三号"]
    for s in subjects:
        store.create_atomic_fact(s, "是", "测试对象")
    payload = "x' OR 1=1 --"
    rows = store.get_atomic_facts_by_subject(payload)
    assert len(rows) < len(subjects)
    assert len(rows) == 0  # 注入串作为字面量不命中任何 subject


# ─── AC4（路径 2）：by id ─────────────────────────────

def test_facts_by_episode_id_with_quote(overgraph_store):
    store = overgraph_store
    store.create_episode({"id": "ep'1", "content": "含引号 ID"})
    fid = store.create_atomic_fact("S", "P", "O", source_episode="ep'1")
    rows = store.get_atomic_facts_by_episode("ep'1")
    assert [r["id"] for r in rows] == [fid]


# ─── 额外路径覆盖：属性版本批量归一化前缀匹配 ─────────

def test_property_versions_for_entities_with_quote(overgraph_store):
    store = overgraph_store
    store.create_property_version("O'Brien Corp", "ceo", "Tim", valid_from=1.0)
    rows = store.get_property_versions_for_entities(["O'Brien"])
    assert len(rows) == 1
    assert rows[0]["entity_id"] == "O'Brien Corp"
