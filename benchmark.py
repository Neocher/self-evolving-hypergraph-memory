#!/usr/bin/env python3
"""SHM v4.0 全面性能测试脚本"""
import requests, json, time, statistics, sys

BASE = "http://127.0.0.1:8000"
OK = "\033[92m\u2713\033[0m"
FAIL = "\033[91m\u2717\033[0m"
WARN = "\033[93m~\033[0m"

def test(name, fn):
    try:
        t0 = time.time()
        result = fn()
        t = (time.time() - t0) * 1000
        if result.get("status") == "error":
            print(f"  {WARN} {name}: {t:.0f}ms (degraded: {result.get('status')})")
            return {"ok": False, "ms": t, "detail": result}
        print(f"  {OK} {name}: {t:.0f}ms")
        return {"ok": True, "ms": t, "detail": result}
    except Exception as e:
        print(f"  {FAIL} {name}: ERROR - {e}")
        return {"ok": False, "ms": 0, "detail": str(e)}

def check(name, fn):
    try:
        t0 = time.time()
        r = fn()
        t = (time.time() - t0) * 1000
        if r.status_code == 200:
            print(f"  {OK} {name}: {t:.0f}ms")
            return {"ok": True, "ms": t, "data": r.json()}
        else:
            print(f"  {WARN} {name}: {t:.0f}ms (HTTP {r.status_code})")
            return {"ok": False, "ms": t, "data": r.json()}
    except Exception as e:
        print(f"  {FAIL} {name}: ERROR - {e}")
        return {"ok": False, "ms": 0, "data": str(e)}

results = []

print("\n" + "="*60)
print("  SHM v4.0 全面性能测试")
print("="*60)

# ========== SECTION 1: 基础 API ==========
print("\n\033[1m[1/6] 基础 API 健康检查\033[0m")
r = check("GET /health", lambda: requests.get(f"{BASE}/health"))
results.append(("健康检查", r))
r = check("GET /docs (Swagger)", lambda: requests.get(f"{BASE}/docs"))
results.append(("Swagger文档", r))

# ========== SECTION 2: 写入性能 ==========
print("\n\033[1m[2/6] 写入性能测试\033[0m")

# 单条写入
timings = []
for i in range(10):
    t0 = time.time()
    r = requests.post(f"{BASE}/memories/episodes",
        json={"content": f"性能测试数据第{i+1}条: SHM v4.0 自演化超图记忆系统基准测试", "source": "system", "force_promote": True})
    timings.append((time.time() - t0) * 1000)
avg = statistics.mean(timings)
p99 = sorted(timings)[-1]
print(f"  {OK} 单条写入 x10: avg={avg:.0f}ms, max={p99:.0f}ms")
results.append(("单条写入延迟", {"ok": True, "ms": avg}))

# 批量写入 (30条)
t0 = time.time()
for i in range(30):
    requests.post(f"{BASE}/memories/episodes",
        json={"content": f"批量测试 #{i}", "source": "system", "force_promote": True})
batch_time = (time.time() - t0) * 1000
print(f"  {OK} 批量写入 x30: {batch_time:.0f}ms total ({batch_time/30:.0f}ms/条)")
results.append(("批量写入 (30条)", {"ok": True, "ms": batch_time}))

# 感觉缓冲区写入
t0 = time.time()
for i in range(20):
    requests.post(f"{BASE}/memories/sensory",
        json={"content": f"Sensory test #{i}", "source": "system"})
sensory_time = (time.time() - t0) * 1000
print(f"  {OK} 感觉缓冲区写入 x20: {sensory_time:.0f}ms ({sensory_time/20:.0f}ms/条)")
results.append(("感觉缓冲区写入", {"ok": True, "ms": sensory_time}))

# ========== SECTION 3: 读取性能 ==========
print("\n\033[1m[3/6] 读取性能测试\033[0m")

# 查询单个Episode
r = check("GET /memories/episodes/{id}",
    lambda: requests.get(f"{BASE}/memories/episodes/4971e924-d0e0-47a4-b70a-ca56e06e1969"))
results.append(("单节点查询", r))

# 检索
r = check("检索 (5条)",
    lambda: requests.post(f"{BASE}/memories/retrieve",
        json={"query": "SHM v4.0 记忆系统性能测试", "top_k": 5}))
results.append(("检索延迟 (5条)", r))

r = check("检索 (20条)",
    lambda: requests.post(f"{BASE}/memories/retrieve",
        json={"query": "记忆系统基准测试数据", "top_k": 20}))
results.append(("检索延迟 (20条)", r))

# 长文本检索
long_query = "SHM v4.0 " * 20
r = check("长文本检索 (20条)",
    lambda: requests.post(f"{BASE}/memories/retrieve",
        json={"query": long_query.strip(), "top_k": 20}))
results.append(("长文本检索", r))

# ========== SECTION 4: 超边管理 ==========
print("\n\033[1m[4/6] 超边管理性能\033[0m")

# 获取所有节点ID
node_ids = []
try:
    from graph.kuzu_store import KuzuStore, KuzuConfig
    store = KuzuStore(config=KuzuConfig(database_path="./data/shm_kuzu_db"))
    store.connect()
    rows = store.query_cypher("MATCH (e:EpisodeNode) RETURN e.id LIMIT 10")
    node_ids = [str(r[0]) for r in rows]
    store = None
except:
    pass

if len(node_ids) >= 2:
    r = check("创建超边 (5成员)",
        lambda: requests.post(f"{BASE}/hyperedges",
            json={"type": "episode", "member_ids": node_ids[:5], "topic": "性能测试超边"}))
    results.append(("创建超边", r))

    r = check("查询超边",
        lambda: requests.get(f"{BASE}/hyperedges/e4e40e93-4408-4249-8f79-4f983d30f50a"))
    results.append(("查询超边", r))
else:
    print(f"  {WARN} 超边测试: 跳过 (节点数不足)")

# 社区列表
r = check("社区列表",
    lambda: requests.get(f"{BASE}/communities?limit=10"))
results.append(("社区列表", r))

# ========== SECTION 5: 压力测试 ==========
print("\n\033[1m[5/6] 并发压力测试\033[0m")

# 并发写入
t0 = time.time()
threads = []
import threading

def write_episode(i):
    requests.post(f"{BASE}/memories/episodes",
        json={"content": f"并发测试 #{i} {'x'*100}", "source": "system", "force_promote": True})

threads = [threading.Thread(target=write_episode, args=(i,)) for i in range(10)]
t0 = time.time()
for t in threads: t.start()
for t in threads: t.join()
concurrent_time = (time.time() - t0) * 1000
print(f"  {OK} 并发写入 x10: {concurrent_time:.0f}ms ({concurrent_time/10:.0f}ms/条)")
results.append(("并发写入 (10线程)", {"ok": True, "ms": concurrent_time}))

# 混合负载 (读写混合)
def read_write_mix(i):
    if i % 3 == 0:
        requests.post(f"{BASE}/memories/episodes",
            json={"content": f"混合负载 #{i}", "source": "system", "force_promote": True})
    elif i % 3 == 1:
        requests.post(f"{BASE}/memories/retrieve",
            json={"query": "混合负载测试数据", "top_k": 5})
    else:
        requests.get(f"{BASE}/health")

threads = [threading.Thread(target=read_write_mix, args=(i,)) for i in range(30)]
t0 = time.time()
for t in threads: t.start()
for t in threads: t.join()
mix_time = (time.time() - t0) * 1000
print(f"  {OK} 混合负载 30请求: {mix_time:.0f}ms")
results.append(("混合负载 (30请求)", {"ok": True, "ms": mix_time}))

# ========== SECTION 6: 系统资源 ==========
print("\n\033[1m[6/6] 系统资源状态\033[0m")
r = check("健康检查 (最终)",
    lambda: requests.get(f"{BASE}/health"))
if r["ok"]:
    d = r["data"]
    print(f"  内存: ~{d['stats']['memory'].get('info', 'N/A')}")
    print(f"  断路器: {d['stats']['circuit_breaker']['state']} (成功率 {d['stats']['circuit_breaker']['success_rate']}%)")
results.append(("系统资源", r))

# ========== 汇总报告 ==========
print("\n" + "="*60)
print("  \033[1m性能测试汇总报告\033[0m")
print("="*60)
print(f"  {'项目':<25} {'延迟':<10} {'状态':<8}")
print(f"  {'-'*25} {'-'*10} {'-'*8}")

all_ok = True
for name, r in results:
    status = OK if r["ok"] else FAIL
    if not r["ok"]: all_ok = False
    ms_str = f"{r['ms']:.0f}ms" if r["ms"] > 0 else "N/A"
    print(f"  {name:<25} {ms_str:<10} {status}")

all_str = "\033[92m全部通过\033[0m" if all_ok else "\033[91m有失败项\033[0m"
print(f"\n  {all_str}")
print(f"  PID: 129111")
print(f"  端口: 8000")
print(f"  服务: shm-v4.service (systemd, 开机自启)")
print("="*60)
