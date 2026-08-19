#!/usr/bin/env python3
"""SHM v6.0.0 阶段2 — GraphLite → OverGraph 数据迁移脚本（design_overgraph_vector.md §A9）

迁移接口:
    StoreMigration.dump_graphlite(src)  — 遍历全部已知节点/边 label，typed props
                                          展开 + 遗留 {b64} 一次 decode 落明文
    StoreMigration.load_overgraph(snap, dst) — batch_upsert_nodes/edges + 重放
                                          version/created_at/valid_from/tau_initial
                                          原值（elementKey=node_id 直映，id 落 props；
                                          边 weight 双写 props + 一等字段，GQL 可读）
    StoreMigration.verify(src, dst)    — 节点/边按 label 计数对拍 + 抽样子集
                                          逐字段对拍（含中文 content 逐字节）

用法:
    python scripts/migrate_graphlite_to_overgraph.py --src <gl.db> --dst <og_dir>

退出码: 0 = 迁移 + verify 全通过；1 = 任一步失败/verify 不一致。

铁律: 不碰 core/llm_client.py；GraphLite 后端代码零改动（源库只读打开）。
GraphLite 本版约束（PoC 实证）:
  - `MATCH (n:Label) RETURN n` 返回 {"Node": {labels, properties: {typed}}}；
    未标注 label 的 `MATCH (n)` 可枚举全部节点，但边必须按 label 显式查询
    （`MATCH (a)-[r:L]->(b)` 带关系变量在含边库上偶发 QUERY_ERROR，规避）。
  - INSERT 不能带 version 属性（保留字，QUERY_ERROR）→ 源库 version 由
    create_episode 后置 SET 写入，dump 侧正常读回原值。
  - 属性值为类型标签包装 {"String":..}/{"Number":..}/{"Boolean":..}。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from base64 import b64decode
from typing import Any

# 项目根加入 sys.path（脚本按 scripts/ 目录执行时也能 import shm 包）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger("shm.migrate_overgraph")

# SHM 图模型 label 全集（graphlite_store.py / overgraph_store.py 共用）。
# ConceptualNode 为遗留概念层（core/conceptual_memory.py），源库可能残留；
# 不存在的 label 查询返回空，纳入枚举安全。
NODE_LABELS = [
    "EpisodeNode",
    "HyperedgeNode",
    "SessionNode",
    "VisualNode",
    "PropertyVerNode",
    "CommunityNode",
    "SystemNode",
    "ConflictNode",
    "OntologyType",
    "OntologyEntity",
    "ConceptualNode",
]

# 边 label 全集（代码库 8 类；GraphLite 不支持无 label 边枚举，按 label 显式查询）
EDGE_LABELS = [
    "HYPEREDGE_MEMBER",
    "COMMUNITY_MEMBER",
    "SESSION_MEMBER",
    "SUPERSEDES",
    "HEBBIAN_CONNECTION",
    "RELATES_TO",
    "ALIAS_OF",
    "IS_A",
]

# 重放字段（设计 A9：原值搬运，不重算）
_TIMESTAMP_FIELDS = ("created_at", "valid_from", "expired_at", "tau_initial",
                     "last_seen", "updated_at", "version")


# ─── GraphLite 行解析 helpers ─────────────────────────────────────

def _unwrap_typed(v: Any) -> Any:
    """类型标签 {"String": x}/{"Number": x}/{"Boolean": x} → 原生值。"""
    if isinstance(v, dict) and len(v) == 1:
        (tag, val), = v.items()
        if tag in ("String", "Number", "Boolean"):
            return val
    return v


def _b64_decode_once(v: Any) -> Any:
    """遗留 {b64} 编码一次 decode 落明文（design A9：dump 侧 decode 一次）。"""
    if isinstance(v, str) and v.startswith("{b64}"):
        try:
            return b64decode(v[5:]).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return v
    return v


def _node_props_from_row(row: dict) -> dict:
    """`RETURN n` 行 → props dict（typed 展开 + b64 decode）。

    GraphLite 返回格式 {'n': {'Node': {'id': <elementKey>, 'properties': {...}}}}。
    elementKey id 不在 properties——无 id 属性的节点（OntologyType/
    OntologyEntity）必须并入 props["id"]，否则 dump 后丢失（迁移 verify
    src=23 dst=0 根因，2026-08-20 修复）。
    """
    nd = row.get("n") or row.get("Node") or row
    if isinstance(nd, dict) and "Node" in nd:
        nd = nd["Node"]
    if not isinstance(nd, dict):
        return {}
    props: dict = {}
    for k, v in (nd.get("properties") or {}).items():
        props[k] = _b64_decode_once(_unwrap_typed(v))
    if "id" not in props and nd.get("id") is not None:
        props["id"] = str(nd["id"])
    return props


def _edge_from_row(row: dict) -> dict | None:
    """`RETURN a.id AS src, b.id AS dst, r` 行 → {src, dst, props}。"""
    src, dst = row.get("src"), row.get("dst")
    if src is None or dst is None:
        return None
    ed = row.get("r") or {}
    if isinstance(ed, dict) and "Edge" in ed:
        ed = ed["Edge"]
    props: dict = {}
    if isinstance(ed, dict):
        for k, v in (ed.get("properties") or {}).items():
            props[k] = _b64_decode_once(_unwrap_typed(v))
    return {"src": str(src), "dst": str(dst), "props": props}


def _chunks(items: list, size: int = 1000):
    for i in range(0, len(items), size):
        yield items[i:i + size]


class StoreMigration:
    """GraphLite → OverGraph 数据迁移（dump / load / verify 三段）。"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.last_snapshot: dict | None = None
        self.skipped_no_id = 0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ─── 1. dump ──────────────────────────────────────────────

    def dump_graphlite(self, src: str) -> dict:
        """遍历全部节点/边 label，返回内存快照（b64 已 decode 为明文）。

        SHM 图模型以 id 为 elementKey（A2 直映）；无 id 的节点无法在
        目标库定位/验证，跳过并统计。Snapshot 结构:
            {"nodes": {label: [props, ...]}, "edges": {label: [{src,dst,props}]}}
        """
        from graph.graphlite_store import GraphLiteStore

        cfg = type("cfg", (), {"database_path": src, "max_threads": 4})()
        g = GraphLiteStore(config=cfg)
        g.connect()
        self.skipped_no_id = 0
        try:
            nodes: dict[str, list[dict]] = {}
            for label in NODE_LABELS:
                rows = g.execute_cypher(f"MATCH (n:{label}) RETURN n")
                props_list = []
                for r in rows:
                    p = _node_props_from_row(r)
                    if p.get("id") is not None:
                        props_list.append(p)
                    else:
                        self.skipped_no_id += 1
                nodes[label] = props_list
                self._log(f"  dump {label}: {len(props_list)}")
            edges: dict[str, list[dict]] = {}
            for label in EDGE_LABELS:
                rows = g.execute_cypher(
                    f"MATCH (a)-[r:{label}]->(b) RETURN a.id AS src, b.id AS dst, r"
                )
                edge_list = [e for e in (_edge_from_row(r) for r in rows) if e]
                edges[label] = edge_list
                self._log(f"  dump edge {label}: {len(edge_list)}")
        finally:
            g.close()
        snap = {"nodes": nodes, "edges": edges}
        self.last_snapshot = snap
        return snap

    # ─── 2. load ──────────────────────────────────────────────

    def load_overgraph(self, snap: dict, dst: str, force: bool = False) -> dict:
        """快照批量灌入 OverGraph 库。

        - 节点: batch_upsert_nodes（elementKey=node_id 直映，props 原样含 id）
        - 边: batch_upsert_edges（elementKey→内部 ID 映射；weight 双写
          props + 一等字段 → GQL r.weight 可读，与 OverGraph 自身写路径一致）
        - version/created_at/valid_from/tau_initial 原值重放（零重算）
        """
        from graph.overgraph_store import OverGraphStore

        if os.path.isdir(dst) and os.listdir(dst) and not force:
            raise SystemExit(
                f"目标库 {dst} 已存在且非空；确认覆盖请加 --force（迁移前清空重建）"
            )
        if os.path.isdir(dst):
            import shutil
            shutil.rmtree(dst, ignore_errors=True)

        store_cfg = type("cfg", (), {
            "database_path": dst,
            "dense_vector_dimension": 512,
            "dense_vector_metric": "cosine",
            "ef_search": 64,
        })()
        s = OverGraphStore(config=store_cfg)
        s.connect()
        try:
            db = s.conn
            total_nodes = 0
            for label, props_list in snap["nodes"].items():
                if not props_list:
                    continue
                for chunk in _chunks(props_list):
                    items = []
                    for p in chunk:
                        key = str(p.get("id") or p.get("name") or "")
                        if not key:
                            continue
                        clean = {k: v for k, v in p.items() if v is not None}
                        items.append({"labels": [label], "key": key,
                                      "props": clean})
                    if items:
                        db.batch_upsert_nodes(items)
                        total_nodes += len(items)
                self._log(f"  load {label}: {len(props_list)}")

            # elementKey → 内部 ID（边端点解析；elementKey=props id 直映）
            key_iid: dict[str, int] = {}
            for label, props_list in snap["nodes"].items():
                if not props_list:
                    continue
                for view in db.get_nodes_by_labels(label):
                    key_iid.setdefault(str(view.key), int(view.id))

            total_edges = 0
            skipped_edges = 0
            for label, edge_list in snap["edges"].items():
                if not edge_list:
                    continue
                items = []
                for e in edge_list:
                    fid = key_iid.get(str(e["src"]))
                    tid = key_iid.get(str(e["dst"]))
                    if fid is None or tid is None:
                        skipped_edges += 1
                        continue
                    props = {k: v for k, v in e["props"].items()
                             if v is not None}
                    w = props.get("weight")
                    items.append({
                        "from_id": fid,
                        "to_id": tid,
                        "label": label,
                        "props": props,
                        "weight": float(w) if isinstance(w, (int, float)) else 1.0,
                    })
                for chunk in _chunks(items):
                    if chunk:
                        db.batch_upsert_edges(chunk)
                        total_edges += len(chunk)
                self._log(f"  load edge {label}: {len(items)}"
                          + (f" (skip {skipped_edges})" if skipped_edges else ""))
        finally:
            s.close()
        return {"nodes": total_nodes, "edges": total_edges,
                "skipped_edges": skipped_edges}

    # ─── 3. verify ────────────────────────────────────────────

    def verify(self, src: str, dst: str,
               node_sample: int = 50, edge_sample: int = 50) -> dict:
        """源库 ↔ 目标库逐项对拍。

        - 每 label 节点/边计数相等
        - 每 label 抽样子集（最多 node_sample/edge_sample 个）逐字段相等
          （内容为字符串逐字节比较；含中文）
        - 边抽查走 get_edge_by_triple（props + weight 双字段）
        """
        from graph.graphlite_store import GraphLiteStore
        from graph.overgraph_store import OverGraphStore

        g = GraphLiteStore(config=type("cfg", (), {
            "database_path": src, "max_threads": 4})())
        g.connect()
        s = OverGraphStore(config=type("cfg", (), {
            "database_path": dst, "dense_vector_dimension": 512,
            "dense_vector_metric": "cosine", "ef_search": 64})())
        s.connect()
        report: dict = {"nodes": {}, "edges": {}, "sampled_nodes": 0,
                        "sampled_edges": 0, "mismatches": [],
                        "ok": True}
        try:
            # ── 计数 ──
            for label in NODE_LABELS:
                src_cnt = int(g.execute_cypher(
                    f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"])
                dst_cnt = len(s.conn.get_nodes_by_labels(label))
                ok = src_cnt == dst_cnt
                report["nodes"][label] = {"src": src_cnt, "dst": dst_cnt,
                                          "ok": ok}
                if not ok:
                    report["ok"] = False
                    report["mismatches"].append(
                        f"node count {label}: src={src_cnt} dst={dst_cnt}")
            for label in EDGE_LABELS:
                src_cnt = int(g.execute_cypher(
                    f"MATCH ()-[r:{label}]->() RETURN count(r) AS c")[0]["c"])
                dst_cnt = int(s.conn.execute_gql(
                    f"MATCH ()-[r:{label}]->() RETURN count(r) AS c",
                    mode="auto", allow_full_scan=True)["rows"][0]["c"])
                ok = src_cnt == dst_cnt
                report["edges"][label] = {"src": src_cnt, "dst": dst_cnt,
                                          "ok": ok}
                if not ok:
                    report["ok"] = False
                    report["mismatches"].append(
                        f"edge count {label}: src={src_cnt} dst={dst_cnt}")

            # ── 节点抽样子集逐字段对拍 ──
            for label in NODE_LABELS:
                if report["nodes"][label]["src"] == 0:
                    continue
                rows = g.execute_cypher(
                    f"MATCH (n:{label}) RETURN n.id AS id LIMIT {int(node_sample)}")
                for r in rows:
                    nid = r.get("id")
                    if nid is None:
                        continue
                    src_props = self._gl_node_props(g, label, nid)
                    view = s.conn.get_node_by_key(label, str(nid))
                    dst_props = dict(view.props) if view is not None else None
                    report["sampled_nodes"] += 1
                    if dst_props is None or src_props != dst_props:
                        report["ok"] = False
                        report["mismatches"].append(
                            f"node field {label} {nid}: "
                            f"src={json.dumps(src_props, ensure_ascii=False)[:200]} "
                            f"dst={json.dumps(dst_props, ensure_ascii=False)[:200]}")

            # ── 边抽样子集逐字段对拍（含 weight）──
            key_iid: dict[str, int] = {}
            for label in NODE_LABELS:
                if report["nodes"][label]["src"] == 0:
                    continue
                for view in s.conn.get_nodes_by_labels(label):
                    key_iid.setdefault(str(view.key), int(view.id))
            for label in EDGE_LABELS:
                if report["edges"][label]["src"] == 0:
                    continue
                rows = g.execute_cypher(
                    f"MATCH (a)-[r:{label}]->(b) "
                    f"RETURN a.id AS src, b.id AS dst, r LIMIT {int(edge_sample)}")
                for r in rows:
                    e = _edge_from_row(r)
                    if e is None:
                        continue
                    fid = key_iid.get(e["src"])
                    tid = key_iid.get(e["dst"])
                    if fid is None or tid is None:
                        report["ok"] = False
                        report["mismatches"].append(
                            f"edge endpoint {label} {e['src']}->{e['dst']} "
                            f"missing in dst")
                        continue
                    ev = s.conn.get_edge_by_triple(fid, tid, label)
                    report["sampled_edges"] += 1
                    if ev is None:
                        report["ok"] = False
                        report["mismatches"].append(
                            f"edge missing {label} {e['src']}->{e['dst']}")
                        continue
                    dst_props = dict(ev.props or {})
                    # 源侧 props 的 weight 必须在目标 props 中可读（GQL 契约）
                    if dst_props != e["props"]:
                        report["ok"] = False
                        report["mismatches"].append(
                            f"edge field {label} {e['src']}->{e['dst']}: "
                            f"src={json.dumps(e['props'], ensure_ascii=False)[:200]} "
                            f"dst={json.dumps(dst_props, ensure_ascii=False)[:200]}")
        finally:
            g.close()
            s.close()
        return report

    @staticmethod
    def _gl_node_props(g, label: str, nid: str) -> dict | None:
        """源库单节点 props（与 dump 同一解析路径）。

        GQL `{{id: $id}}` 匹配的是 properties.id——elementKey-id 节点
        （OntologyType/OntologyEntity，id 不在 properties，AGENTS.md 已知坑）
        匹配不到 → label 全扫 + Python 侧按 id 过滤（label 节点数少，可接受）。
        """
        rows = g.query_cypher(f"MATCH (n:{label}) RETURN n")
        for r in rows:
            p = _node_props_from_row(r)
            if p.get("id") == nid:
                return p
        return None


def _build_report(snap: dict, load_result: dict, verify: dict,
                  src: str, dst: str, elapsed: float) -> dict:
    return {
        "src": src,
        "dst": dst,
        "elapsed_sec": round(elapsed, 1),
        "dumped": {k: len(v) for k, v in snap["nodes"].items() if v},
        "dumped_edges": {k: len(v) for k, v in snap["edges"].items() if v},
        "loaded": load_result,
        "verify": verify,
        "passed": bool(verify["ok"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="GraphLite → OverGraph 数据迁移 + verify（SHM v6.0.0 阶段2）")
    ap.add_argument("--src", required=True, help="源 GraphLite 库路径（目录）")
    ap.add_argument("--dst", required=True, help="目标 OverGraph 库路径（目录）")
    ap.add_argument("--force", action="store_true",
                    help="目标库已存在时清空重建")
    ap.add_argument("--node-sample", type=int, default=50,
                    help="verify 每 label 节点抽样子集大小（默认 50）")
    ap.add_argument("--edge-sample", type=int, default=50,
                    help="verify 每 label 边抽样子集大小（默认 50）")
    ap.add_argument("--report", default=None,
                    help="迁移报告 JSON 输出路径（默认 <dst>/migration_report.json）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    m = StoreMigration()
    t0 = time.time()
    try:
        print(f"=== [1/3] dump GraphLite: {args.src} ===", flush=True)
        snap = m.dump_graphlite(args.src)
        print(f"=== [2/3] load OverGraph: {args.dst} ===", flush=True)
        loaded = m.load_overgraph(snap, args.dst, force=args.force)
        print(f"=== [3/3] verify src ↔ dst ===", flush=True)
        v = m.verify(args.src, args.dst,
                     node_sample=args.node_sample, edge_sample=args.edge_sample)
    except Exception:
        print("迁移失败:", flush=True)
        raise

    report = _build_report(snap, loaded, v, args.src, args.dst,
                           time.time() - t0)
    report_path = args.report or os.path.join(args.dst, "migration_report.json")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== 迁移报告 ===", flush=True)
    print(f"节点: dumped={sum(len(v) for v in snap['nodes'].values())} "
          f"loaded={loaded['nodes']}"
          + (f" (跳过无 id 节点 {m.skipped_no_id})" if m.skipped_no_id else ""),
          flush=True)
    print(f"边:   dumped={sum(len(v) for v in snap['edges'].values())} "
          f"loaded={loaded['edges']} skipped={loaded['skipped_edges']}",
          flush=True)
    print(f"verify: {'PASS' if v['ok'] else 'FAIL'} "
          f"(sampled {v['sampled_nodes']} nodes + {v['sampled_edges']} edges)",
          flush=True)
    for lbl, d in v["nodes"].items():
        if d["src"] or d["dst"]:
            print(f"  node {lbl}: src={d['src']} dst={d['dst']} "
                  f"{'OK' if d['ok'] else 'MISMATCH'}", flush=True)
    for lbl, d in v["edges"].items():
        if d["src"] or d["dst"]:
            print(f"  edge {lbl}: src={d['src']} dst={d['dst']} "
                  f"{'OK' if d['ok'] else 'MISMATCH'}", flush=True)
    if v["mismatches"]:
        print("  mismatches:", flush=True)
        for mm in v["mismatches"][:20]:
            print(f"    - {mm}", flush=True)
    print(f"报告已写: {report_path}", flush=True)
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
