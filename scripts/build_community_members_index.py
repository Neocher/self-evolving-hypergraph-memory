# -*- coding: utf-8 -*-
"""构建社区成员 sidecar 索引（可重复执行，v6.31 检索超时修复 Part 4）。

按「目标侧分批」快方向枚举：取全部 EpisodeNode id（约 2,100 个），每批默认 15 个，
`MATCH (c:CommunityNode)-[:COMMUNITY_MEMBER]->(e:EpisodeNode) WHERE e.id IN $batch_ids`
反建整张成员表 {community_id: [episode_id]} → 原子写 data/community_members_index.json。
单批返回行数触达 10,000（引擎静默截断上限）时自动折半重试；任何失败 fail loud
（非零退出）——绝不写出残缺 sidecar。

构建 ~30s 量级（vs 上一轮源侧全表扫描 ~1,449s），可频繁重建（梦境每 10-15 分钟
产生新社区，构建远快于其产出节奏）。

用法:
  # 实测（只统计不写盘）
  env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV /home/admin/shm/.venv/bin/python \
      scripts/build_community_members_index.py --dry-run
  # 正式构建（原子替换）
  ... scripts/build_community_members_index.py
  # 指定输出路径 / 数据库路径 / 批大小
  ... scripts/build_community_members_index.py --output /tmp/idx.json \
      --db /path/to/shm_overgraph_db --page-size 15

仅 stdlib + 现有 store；不依赖服务进程。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.overgraph_store import OverGraphStore, _atomic_write_json


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_db_path() -> str:
    return os.environ.get(
        "SHM_OVERGRAPH_DB",
        os.path.join(_project_root(), "data", "shm_overgraph_db"),
    )


def _default_output_path() -> str:
    return os.path.join(_project_root(), "data", "community_members_index.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计不写盘")
    parser.add_argument("--output", default=None,
                        help="输出 JSON 路径（默认 data/community_members_index.json）")
    parser.add_argument("--db", default=None,
                        help="OverGraph 数据库路径（默认 data/shm_overgraph_db）")
    parser.add_argument("--page-size", type=int, default=15,
                        help="目标侧分批大小，即每批 episode id 数（默认 15）")
    args = parser.parse_args(argv)

    db_path = args.db or _default_db_path()
    output_path = args.output or _default_output_path()
    cfg = type("cfg", (), {
        "database_path": db_path,
        "dense_vector_dimension": 512,
        "dense_vector_metric": "cosine",
    })()
    store = OverGraphStore(config=cfg)
    store.connect()
    try:
        t0 = time.time()
        edge_total, index = store.build_community_members_index(
            page_size=args.page_size
        )
        elapsed = time.time() - t0
        print(
            f"build: edge_total={edge_total} community_total={len(index)} "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        if args.dry_run:
            return 0
        payload = {
            "_meta": {
                "built_at": time.time(),
                "edge_total": int(edge_total),
                "community_total": len(index),
            },
        }
        payload.update(index)
        _atomic_write_json(output_path, payload)
        print(f"wrote: {output_path}", flush=True)
        return 0
    except Exception as e:
        print(f"build failed: {e}", file=sys.stderr, flush=True)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
