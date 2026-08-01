#!/usr/bin/env python3
"""SHM健康监控脚本 — 每30分钟检查一次，异常时报警"""
import subprocess, json, os, time
from datetime import datetime

LOG = os.path.expanduser("~/.hermes/shm_monitor.log")
ALERT_SENT = os.path.expanduser("~/.hermes/.shm_alert_sent")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")

def check():
    try:
        r = subprocess.run(
            "curl -s --max-time 5 http://localhost:8000/health",
            shell=True, capture_output=True, text=True, timeout=10
        )
        if not r.stdout:
            log("CRITICAL: SHM health endpoint unreachable")
            return False
        d = json.loads(r.stdout)
        s = d.get("stats", {})
        cb = s.get("circuit_breaker", {})
        
        log(f"OK | dreams={s.get('dream_run_count', 0)} "
            f"| nodes={s.get('node_count', 0)} "
            f"| faiss={s.get('faiss_index_size', 0)} "
            f"| graphlite={d.get('graphlite_connected')} "
            f"| circuit={cb.get('state', '?')}")

        # 异常检测
        issues = []
        if not d.get("graphlite_connected"):
            issues.append("GraphLite disconnected")
        if cb.get("state") == "open":
            issues.append(f"Circuit breaker OPEN ({cb.get('success_rate', 0)}% success)")
        if not d.get("faiss_loaded"):
            issues.append("FAISS not loaded")
        
        if issues:
            log(f"WARNING: {'; '.join(issues)}")
            # 写告警标记（供外部读取）
            with open(ALERT_SENT, "w") as f:
                f.write(f"{'|'.join(issues)}\n{time.time()}")
            return False
        
        # 清除告警标记
        if os.path.exists(ALERT_SENT):
            os.remove(ALERT_SENT)
        return True
        
    except Exception as e:
        log(f"CHECK_FAILED: {e}")
        return False

if __name__ == "__main__":
    check()
