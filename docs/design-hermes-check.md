# 检测脚本设计 v2：scripts/check_hermes_integration.py（CC 审查修订版）

## 要求
- 纯 Python 标准库（urllib/json/os/sys/re），无第三方依赖
- 6 项检查，输出 PASS/WARN/FAIL + FIX: 修复建议
- 退出码：全 PASS=0，任一 FAIL=1
- 每项 try/except 包裹，单项失败不中断

## 检查项（v2 修订）

1. **SHM 存活**：GET /health（5s 超时）
   - PASS: HTTP 200 且 status in ("ok", "degraded")  ← M2
   - degraded → WARN（不判 FAIL，降级仍可应答）
   - error/非200 → FAIL

2. **插件目录位置**：~/.hermes/plugins/shm_v5/ 存在含 __init__.py
   - 若 ~/.hermes/plugins/memory/shm_v5/ 存在（错误位置）→ 明确提示移动

3. **插件可加载（真实发现机制）** ← M1
   - 先 os.environ.setdefault("HERMES_HOME", 解析值)（必须！否则 loader 扫错目录假绿）
   - sys.path 插入 $HERMES_HOME/hermes-agent
   - from plugins.memory import discover_memory_providers
   - providers = {n: a for n, d, a in discover_memory_providers()}
   - PASS: "shm_v5" in providers 且 available=True

4. **config 匹配** ← M4
   - 缩进感知解析 ~/.hermes/config.yaml：找顶层 memory: 段 ← S3
   - 断言 memory.provider == "shm_v5" **且 memory_enabled == true**（缺一即 FAIL：
     插件加载但永不自动检索 = 静默失败另半边）
   - 值可能带引号，strip 处理

5. **prefetch 端到端** ← M3
   - p = providers["shm_v5"] 实例（或 load_memory_provider）
   - ctx = p.prefetch("SHM 记忆系统")（显式 10s 超时）← S2
   - 返回类型：若 str → 断言含 "SHM v5 记忆检索" 且非空
   - 若 list/dict → 断言非空（字段兼容）
   - 注意：top_k 正确性（坑2）不在此覆盖——文档明确"检查5 覆盖连通性，top_k 参数正确性由 SHM 日志侧验证"
   - prefetch 异常/超时 → FAIL

6. **状态注入**：p.system_prompt_block() 含 "记忆系统" 或 "节点"
   - 异常 → FAIL

## 输出格式

```
=== Hermes ↔ SHM 对接检测 ===
[1/6] SHM 服务存活       ... PASS (v5.26.0, 5735 节点)
[2/6] 插件目录位置        ... PASS (~/.hermes/plugins/shm_v5/)
[3/6] 插件可加载          ... PASS (SHMv5MemoryProvider)
[4/6] config provider     ... PASS (shm_v5, memory_enabled=true)
[5/6] prefetch 端到端     ... PASS (3 条记忆)
[6/6] 状态注入            ... PASS
结论: 6/6 PASS — Hermes ↔ SHM 对接成功
```

- FAIL 行带 FIX: 建议
- --debug 参数打印异常 traceback ← S4
- 检查1 FAIL 时检查5/6 标注 "上游依赖未满足（跳过判级）" ← S5

## 边界
- HERMES_HOME 默认 ~/.hermes，环境变量覆盖，import 前 setdefault
- hermes-agent 源码目录探测（$HERMES_HOME/hermes-agent）
- 无 yaml 依赖，缩进感知行扫描
