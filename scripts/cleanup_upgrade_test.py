#!/usr/bin/env python3
"""清理升级测试数据 (upgrade-test namespace) — 直连 GraphLite 按内容特征删除。

背景: 升级 v5.21.10 时写入 1 条测试记忆 (source=upgrade-test),
SESSION_MEMBER 边未建立导致 DELETE /memories/namespace 失效 (deleted=0)。
"""
import base64
import os
import sys

sys.path.insert(0, os.path.expanduser("~/GraphLite/bindings/python"))
sys.path.insert(0, os.path.expanduser("~/GraphLite/sdk-python/src"))

from graphlite_sdk import GraphLite  # noqa: E402

DB_PATH = os.path.expanduser("~/self-evolving-hypergraph-memory/data/shm_graphlite_db")


def decode_content(raw):
    """GraphLite 内容可能 {b64} 前缀编码, 匹配前必须解码 (skill 教训)。"""
    if isinstance(raw, str) and raw.startswith("{b64}"):
        try:
            return base64.b64decode(raw[5:]).decode("utf-8")
        except Exception:
            return raw
    return raw


def main():
    db = GraphLite.open(DB_PATH)
    s = db.session("shm")
    s.execute("SESSION SET SCHEMA /shm")
    s.execute("SESSION SET GRAPH /shm")

    r = s.query("MATCH (e:EpisodeNode) RETURN e.id, e.content")
    targets = []
    for row in r.rows:
        # 兼容扁平 (e.id/e.content) 与嵌套 ({'e': {...}}) 两种返回格式
        if "e.id" in row:
            eid, content = row.get("e.id"), row.get("e.content", "")
        else:
            node = row.get("e", {}).get("Node", {})
            props = node.get("properties", {}) if isinstance(node, dict) else {}
            eid = props.get("id", "")
            content = props.get("content", "")
        text = decode_content(content)
        if "升级测试" in text or text.strip() == "upgrade-test":
            targets.append(eid)

    print(f"matched targets: {len(targets)}")
    for eid in targets:
        s.execute(f"MATCH (e:EpisodeNode {{id: '{eid}'}}) DETACH DELETE e")
    print(f"deleted: {len(targets)}")

    # 顺带清理孤儿 SessionNode (upgrade-test)
    try:
        s.execute("MATCH (s:SessionNode {id: 'upgrade-test'}) DETACH DELETE s")
        print("orphan SessionNode cleaned")
    except Exception as exc:
        print(f"session cleanup skipped: {exc}")

    db.close()


if __name__ == "__main__":
    main()
