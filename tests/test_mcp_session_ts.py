"""R3 P3-2: MCP shm_search session_ts 透传测试（v1 shm/mcp_server.py，无 mcp SDK 依赖）。

MCP 检索工具 schema 加可选 session_ts（float|None），调用透传到 /memories/retrieve。
v2（shm/mcp_server_v2.py，FastMCP）依赖未安装的 mcp SDK，本环境不 import，仅覆盖 v1 入口。
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import shm.mcp_server as ms


def _search_schema() -> dict:
    return next(t for t in ms.TOOLS if t["name"] == "shm_search")["inputSchema"]


class TestMcpSearchSessionTs:

    def test_schema_exposes_session_ts(self):
        schema = _search_schema()
        assert "session_ts" in schema["properties"]
        assert schema["properties"]["session_ts"]["type"] == "number"
        assert "session_ts" not in schema["required"], "session_ts 应可选（向后兼容）"

    def test_passes_session_ts_through(self):
        captured = {}

        def fake_post(path, data):
            captured["path"] = path
            captured["data"] = data
            return {"results": []}

        session_ts = 1_700_000_000.0
        with patch.object(ms, "_shm_post", side_effect=fake_post):
            ms._handle_call("shm_search", {"query": "q", "session_ts": session_ts})
        assert captured["path"] == "/memories/retrieve"
        assert captured["data"]["session_ts"] == session_ts

    def test_omits_session_ts_when_absent(self):
        captured = {}

        def fake_post(path, data):
            captured["data"] = data
            return {"results": []}

        with patch.object(ms, "_shm_post", side_effect=fake_post):
            ms._handle_call("shm_search", {"query": "q"})
        assert "session_ts" not in captured["data"]


_V2_SRC = Path(__file__).resolve().parent.parent / "shm" / "mcp_server_v2.py"


def _v2_search_fn() -> ast.FunctionDef:
    tree = ast.parse(_V2_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "shm_search_memory":
            return node
    raise AssertionError("shm_search_memory 未在 mcp_server_v2.py 中找到")


class TestMcpV2SessionTsStatic:
    """P3-2: v2（FastMCP）session_ts 透传静态断言——不 import 未安装的 mcp SDK。"""

    def test_signature_has_optional_float_session_ts(self):
        fn = _v2_search_fn()
        args = fn.args
        names = [a.arg for a in args.args]
        assert "session_ts" in names, f"session_ts 参数缺失: {names}"
        idx = names.index("session_ts")
        n_defaults = len(args.defaults)
        assert idx >= len(names) - n_defaults, "session_ts 无默认值（应可选）"
        default = args.defaults[idx - (len(names) - n_defaults)]
        assert isinstance(default, ast.Constant) and default.value is None
        ann = args.args[idx].annotation
        assert ann is not None and "float" in ast.unparse(ann), (
            f"标注应为 Optional[float]: {ast.unparse(ann) if ann else None}"
        )

    def test_body_passes_session_ts_through(self):
        fn = _v2_search_fn()
        refs = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Name) and n.id == "session_ts"
        ]
        assert refs, "shm_search_memory 函数体未引用 session_ts（透传缺失）"
