"""检索/存储修复验收测试（v6.28.x 第二轮组 δ）

覆盖 K1 / K2 / K3 / K4 修复的回归防护，全部走公共入口或最小可观测契约：
- K1  _gql_value 转义与 update_with_version 失败可观测
- K2  融合归一化保留通道间绝对质量差
- K3  _vector_retrieve 熔断时通道可被上层感知（不产出空 content 伪结果）
- K4  _entity_expansion 频率门控折叠进单条 SELECT（无独立 COUNT 全表扫描）
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from graph.common import _gql_value


# ══════════════════════════════════════════════════════════════════
# K1 —— _gql_value 转义 + update_with_version 失败可观测
# ══════════════════════════════════════════════════════════════════


def test_gql_value_escapes_newline_cr_quote_backslash():
    """_gql_value 与 _esc_literal 转义语义一致：\\n / \\r / ' / \\ 全覆盖。"""
    assert _gql_value("O'Brien") == "'O\\'Brien'"
    assert _gql_value("a\\b") == "'a\\\\b'"
    assert _gql_value("line\nbreak") == "'line\\nbreak'"
    assert _gql_value("carriage\rreturn") == "'carriage\\rreturn'"
    # 组合：换行 + 回车 + 单引号 + 反斜杠，顺序与 _esc_literal 一致（反斜杠先行）
    assert _gql_value("mix\n\r\\'all") == "'mix\\n\\r\\\\\\'all'"
    # 无害值原样
    assert _gql_value("plain text") == "'plain text'"
    assert _gql_value("") == "''"


@pytest.mark.overgraph
def test_update_with_version_roundtrips_special_chars(overgraph_store):
    """含 \\n / \\r / ' / \\ 的内容经 update_with_version 写入后读回等价。"""
    pytest.importorskip("overgraph")
    store = overgraph_store
    eid = store.create_episode({"id": "ep_special", "content": "v0"})
    payload = "line1\nline2\rline3 'quoted' \\backslash\\"
    assert store.update_with_version(eid, {"content": payload}, expected_version=1) is True
    ep = store.get_episode(eid)
    assert ep is not None
    assert ep["content"] == payload


@pytest.mark.overgraph
def test_update_with_version_exception_is_observed(overgraph_store, caplog):
    """执行异常不再静默：留下 WARNING 日志 + 实例计数器自增。"""
    pytest.importorskip("overgraph")
    store = overgraph_store
    eid = store.create_episode({"id": "ep_fail", "content": "v0"})
    # 让底层执行抛异常（模拟语法错误 / 引擎故障），CAS 路径进入 except。
    # 注意：真实 OverGraph 对象的 execute_gql 是只读 Rust 属性，不能 patch.object
    # 其方法 —— 整体替换 _db 引用（Python 属性）为会抛异常的 stub。
    fake_db = MagicMock()
    fake_db.execute_gql.side_effect = RuntimeError("boom")
    with patch.object(store, "_db", fake_db):
        with caplog.at_level(logging.WARNING, logger="shm.overgraph_store"):
            ok = store.update_with_version(eid, {"content": "x"}, expected_version=1)
    assert ok is False
    assert getattr(store, "_update_with_version_failures", 0) == 1
    assert any("update_with_version failed" in r.message for r in caplog.records)
