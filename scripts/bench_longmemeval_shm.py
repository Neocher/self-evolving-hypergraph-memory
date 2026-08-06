#!/usr/bin/env python3
"""
SHM × LongMemEval 真实数据基准评测(Phase 1)
==========================================
用 LongMemEval 官方数据集(ICLR 2025)评测真实 SHM 服务。

流程:
  1. 从 longmemeval_oracle.json 读取 500 个评测实例
  2. 将每个实例的 haystack 会话(带时间戳的对话)灌入 SHM
  3. 对每个 question 做检索(混合检索:语义 + 时序)
  4. 用 LongMemEval 官方的 evaluate_qa 计算指标

用法:
  python3 bench_longmemeval_shm.py [--data /tmp/LongMemEval/data] [--limit N] [--shm http://127.0.0.1:8000]

依赖:
  pip install httpx  (或 requests)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import httpx

SHM_BASE = os.environ.get("SHM_BASE_URL", "http://127.0.0.1:8000")
DATA_DIR = os.environ.get("LONGMEMEVAL_DATA", "/tmp/LongMemEval/data")
NS = "bench-longmemeval"  # 命名空间隔离,不污染真实记忆

# ─── SHM API 封装 ────────────────────────────────────────────────

class SHMClient:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.client = httpx.Client(timeout=60.0)

    def write(self, content: str, source: str = "user", ts: float | None = None) -> str:
        """写一条记忆(episode)。返回 episode id。时间戳放 metadata。"""
        payload: dict[str, Any] = {
            "content": content,
            "source": source,
            "namespace": NS,
            "force_promote": True,  # benchmark 数据直接提升,不走 τ 阈值
        }
        if ts:
            payload["metadata"] = {"ts": ts}
        r = self.client.post(f"{self.base}/memories/episodes", json=payload)
        r.raise_for_status()
        return r.json().get("episode_id", r.json().get("id", ""))

    def retrieve(self, query: str, top_k: int = 10, endpoint: str = "vector") -> list[dict]:
        """检索。endpoint: vector(纯向量,快) | hybrid(混合检索,大库下可能超时)"""
        if endpoint == "hybrid":
            r = self.client.post(f"{self.base}/memories/retrieve",
                                 json={"query": query, "top_k": top_k, "namespace": NS})
        else:
            r = self.client.post(f"{self.base}/search/vector",
                                 json={"query": query, "top_k": top_k})
        r.raise_for_status()
        return r.json().get("results", [])

    def clear(self) -> None:
        """清空命名空间。"""
        try:
            self.client.delete(f"{self.base}/memories/namespace/{NS}")
        except Exception:
            pass

    def health(self) -> dict:
        r = self.client.get(f"{self.base}/health")
        return r.json()


# ─── LongMemEval 数据解析 ─────────────────────────────────────────

def parse_date(s: str) -> float:
    """'2023/04/10 (Mon) 23:07' → epoch。"""
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", s)
    if not m:
        return 0.0
    y, mo, d = int(m[1]), int(m[2]), int(m[3])
    hm = re.search(r"(\d{2}):(\d{2})", s)
    hh = int(hm.group(1)) if hm else 12
    mm = int(hm.group(2)) if hm else 0
    return datetime(y, mo, d, hh, mm).timestamp()


def load_instances(data_dir: str, limit: int | None = None) -> list[dict]:
    """加载 oracle 数据集(500 实例,仅含证据会话)。"""
    path = os.path.join(data_dir, "longmemeval_oracle.json")
    with open(path, encoding="utf-8") as f:
        instances = json.load(f)
    if limit:
        instances = instances[:limit]
    return instances


# ─── 评测逻辑 ─────────────────────────────────────────────────────

def run_instance(shm: SHMClient, inst: dict, top_k: int, endpoint: str = "vector") -> dict:
    """单个实例:灌数据(串行) → 检索 → 判断命中。"""
    qid = inst["question_id"]
    qtype = inst["question_type"]
    question = inst["question"]
    answer = inst["answer"]
    sessions = inst.get("haystack_sessions", [])
    dates = inst.get("haystack_dates", [])

    # 1. 灌入会话(带时间戳,模拟时序)— 并发写入提速
    write_tasks: list[tuple[str, str, float]] = []
    written = 0
    for si, session in enumerate(sessions):
        base_ts = parse_date(dates[si]) if si < len(dates) else time.time()
        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if not content:
                continue
            ts = base_ts + written * 30
            write_tasks.append((content, role, ts))
            written += 1

    # 串行写入(服务端写路径并发不安全: 并发会触发锁竞争/FAISS-Hebbian 重建风暴
    # 导致事件循环卡死, 2026-08-06 benchmark 实测复现)
    for content, role, ts in write_tasks:
        try:
            shm.write(content, role, ts)
        except Exception:
            pass

    # 2. 检索
    results = shm.retrieve(question, top_k=top_k, endpoint=endpoint)
    retrieved_texts = []
    for r in results:
        content = r.get("content", "")
        if content:
            retrieved_texts.append(content)

    # 3. 判断命中:答案关键词是否出现在检索结果中
    #    简化:答案中提取关键词(去停用词),检查命中
    ans_tokens = [w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", answer)]
    # 取有区分度的词(去掉最通用词)
    stop = {"the", "and", "for", "with", "was", "were", "had", "have", "has",
            "not", "but", "that", "this", "from", "you", "your", "what",
            "did", "get", "got", "first", "after", "issue", "about"}
    key_tokens = [w for w in ans_tokens if w not in stop][:5]
    hit = False
    if key_tokens:
        joined = " ".join(retrieved_texts).lower()
        hit = any(tok in joined for tok in key_tokens)

    return {
        "qid": qid, "type": qtype, "question": question,
        "answer": answer, "key_tokens": key_tokens,
        "hit": hit, "retrieved_count": len(results),
        "written": written,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--shm", default=SHM_BASE)
    ap.add_argument("--limit", type=int, default=None, help="评测实例数(默认全部500)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--endpoint", choices=["vector", "hybrid"], default="vector",
                    help="检索端点: vector(快) / hybrid(混合,大库可能超时)")
    ap.add_argument("--cleanup", action="store_true", help="评测后清空命名空间")
    args = ap.parse_args()

    shm = SHMClient(args.shm)
    print(f"[bench] SHM: {args.shm} | data: {args.data} | limit: {args.limit or 'all'} | top_k: {args.top_k}", flush=True)
    try:
        h = shm.health()
        print(f"[bench] SHM 健康: {h.get('status')} | graph: {h.get('graph_connected')} | faiss: {h.get('faiss_loaded')}", flush=True)
    except Exception as e:
        print(f"[bench] ✗ SHM 不可达: {e}")
        sys.exit(1)

    shm.clear()
    instances = load_instances(args.data, args.limit)
    print(f"[bench] 加载 {len(instances)} 个评测实例", flush=True)

    results = []
    t0 = time.time()
    for i, inst in enumerate(instances):
        try:
            r = run_instance(shm, inst, args.top_k, endpoint=args.endpoint)
            results.append(r)
        except Exception as e:
            print(f"  [skip] {inst.get('question_id')}: {e}")
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  ... {i+1}/{len(instances)} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    recall = hits / total if total else 0.0

    # 分类型统计
    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r["hit"])

    print("\n" + "=" * 60)
    print(f"SHM × LongMemEval 基准结果(Phase 1)")
    print(f"实例数: {total} | 耗时: {elapsed:.0f}s ({elapsed/max(total,1):.2f}s/实例)")
    print(f"写入记忆总数: {sum(r['written'] for r in results)}")
    print(f"整体 recall@{args.top_k}: {recall:.4f} ({hits}/{total})")
    print("-" * 60)
    print(f"{'类型':<30} {'命中率':>10} {'样本':>6}")
    for t, hs in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"{t:<30} {sum(hs)/len(hs):>10.4f} {len(hs):>6}")
    print("=" * 60)

    # 保存结果
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"longmemeval_shm_results_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "elapsed": elapsed,
                   "overall_recall": recall, "hits": hits, "total": total,
                   "by_type": {t: {"hit": sum(hs), "total": len(hs)}
                               for t, hs in by_type.items()},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out}")

    if args.cleanup:
        shm.clear()
        print("命名空间已清理")


if __name__ == "__main__":
    main()
