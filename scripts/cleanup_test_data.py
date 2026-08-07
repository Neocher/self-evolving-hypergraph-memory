#!/usr/bin/env python3
"""
SHM 测试数据清理脚本 — 删除 benchmark/手动测试产生的英文数据。

删除目标(按内容特征):
  1. 英文对话数据(LongMemEval 的英文对话, source=user/assistant)
  2. 手动测试数据(concurrent test write / timing test write / batch test item 等)

保留:
  - 中文真实记忆
  - 系统节点(SessionNode/HyperedgeNode/ConflictNode 等, 非 EpisodeNode)
  - 任何含中文的 EpisodeNode

用法(服务停止后运行, 释放单写者锁):
  python3 scripts/cleanup_test_data.py [--dry-run]
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/GraphLite/bindings/python"))
sys.path.insert(0, os.path.expanduser("~/GraphLite/sdk-python/src"))

# 手动测试数据的特征词(内容前缀/包含)
TEST_MARKERS = [
    "concurrent test write",
    "timing test write",
    "batch test item",
    "speed test write",
    "gpu utilization test write",
    "load test under benchmark",
    "benchmark test write",
    "user discussed topic number",
    "user mentioned car maintenance topic",
    "user bought a brand new car",
    "normal write test without",
    "user mentioned buying a used car",
]

def has_chinese(text: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

def decode_content(raw: str) -> str:
    """解码 GraphLite 存储的 b64 内容。"""
    if raw.startswith("{b64}"):
        import base64
        try:
            return base64.b64decode(raw[5:]).decode("utf-8", errors="ignore")
        except Exception:
            return raw
    return raw

def main() -> None:
    dry_run = "--dry-run" in sys.argv

    from graphlite_sdk import GraphLite
    import glob

    # 定位数据库(与 GraphLiteStore 默认路径一致)
    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "data", "shm_graphlite_db")
    db_dir = os.path.abspath(db_dir)
    print(f"[cleanup] DB: {db_dir}")
    db = GraphLite.open(db_dir)
    s = db.session("shm")
    s.execute("SESSION SET SCHEMA /shm")
    s.execute("SESSION SET GRAPH /shm")

    # 1. 统计所有 EpisodeNode
    result = s.query("MATCH (e:EpisodeNode) RETURN e.id, e.content, e.source")
    all_nodes = []
    for r in result.rows:
        row = {}
        for k, v in r.items():
            # 带别名查询返回扁平 key: "e.source" / "e.id" / "e.content" → 值直接是 str
            if isinstance(v, str):
                row[k.split(".")[-1]] = v
            elif isinstance(v, (int, float)):
                row[k.split(".")[-1]] = v
            elif isinstance(v, dict) and "Node" in v:
                props = v["Node"].get("properties", {})
                for pk, pv in props.items():
                    if isinstance(pv, dict):
                        row[pk] = next(iter(pv.values()))
                    else:
                        row[pk] = pv
            else:
                row[k.split(".")[-1]] = v
        all_nodes.append(row)
    print(f"[cleanup] 总 EpisodeNode: {len(all_nodes)}")

    # 2. 分类: 测试数据 vs 真实数据
    to_delete = []
    kept = []
    for node in all_nodes:
        nid = node.get("id", "")
        content = decode_content(str(node.get("content", "") or ""))
        src = str(node.get("source", "") or "")
        is_test = False

        # 手动测试标记
        low = content.lower()
        for marker in TEST_MARKERS:
            if marker.lower() in low:
                is_test = True
                break

        # 英文对话数据(无中文 + 非系统来源)
        if not is_test and not has_chinese(content) and src in ("user", "assistant"):
            # 检查是否为 LongMemEval 风格对话(有英文句子特征)
            # 保守: 只删明显的英文对话(长度>30 且含多个英文单词)
            words = [w for w in content.split() if w.isalpha() and len(w) > 2]
            if len(words) >= 8 and len(content) > 50:
                is_test = True

        # 【FIX】base64 编码的英文测试数据(解码后判断)
        if not is_test:
            raw = str(node.get("content", "") or "")
            if raw.startswith("{b64}") and not has_chinese(content):
                # 解码后是英文长文本 = 测试数据(LongMemEval 对话)
                words = [w for w in content.split() if w.isalpha() and len(w) > 2]
                if len(words) >= 8 and len(content) > 50:
                    is_test = True

        if is_test:
            to_delete.append(nid)
        else:
            kept.append(nid)

    print(f"[cleanup] 待删除: {len(to_delete)} | 保留: {len(kept)}")

    if dry_run:
        print("[cleanup] DRY-RUN 模式, 不实际删除")
        for nid in to_delete[:5]:
            print(f"  将删除: {nid}")
        db.close()
        return

    # 3. 逐个 DETACH DELETE
    deleted = 0
    for nid in to_delete:
        try:
            s.execute(f"MATCH (e:EpisodeNode {{id: '{nid}'}}) DETACH DELETE e")
            deleted += 1
        except Exception:
            pass
    print(f"[cleanup] 实际删除: {deleted}")
    db.close()

if __name__ == "__main__":
    main()
