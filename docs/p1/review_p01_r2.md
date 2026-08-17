# P1 R2 复核任务书（Codex — R1 修复终审）

## 背景

Codex R1 终审判定需修改（1 P1 + 1 P1 + 2 P2 + 1 P3）→ OpenCode R1 修复已完成（全量 1003 passed）。本任务：R2 复核闭环。

## R1 修复内容（待复核）

1. **P1-1** `_agentic_round` 补 `_mesa_synthesis`（query_router.py:1685，与 _finish:1301 顺序一致）
2. **P1-2** api/app.py:150-153 改 `getattr(mesa, ...)` 显式缺省（零值不吞）
3. **P2-1** `_MESA_BOOST_MAX=0.59`（self_evolving.py:49）+ validate 边界（:103）+ 规则 6 用上界
4. **P2-2** MesaConfig.max_nodes >=1 校验（settings.py:122）+ `_mesa_synthesis` 前置判断
5. **P3-1** 规则 6 `mesa_enabled and` 条件（self_evolving.py:313/317）+ report_probe 带 mesa_enabled

## 复核要求（read_file 静态分析）

1. P1-1：`_agentic_round` 中 `_mesa_synthesis` 调用位置正确（_community_expansion 后）？与 _finish 顺序一致？
2. P1-2：api/app.py 四字段透传用 getattr 显式缺省？boost=0.0 不再被 or 吞？
3. P2-1：mesa_boost 演化上界 0.59 < community boost 0.6（分数契约保持）？validate 边界一致？
4. P2-2：max_nodes=0 不再合成？MesaConfig 校验正确？
5. P3-1：mesa_enabled=False 时规则 6 不调 mesa_boost？report_probe 构造快照带 mesa_enabled？
6. 全量回归证据：1003 passed 是否与实现一致？

## 输出格式

- 判定：通过 / 需修改
- 缺陷清单（🔴 P0 / 🟠 P1 / 🟡 P2 / ⚪ P3，含文件:行号 + 证据）
- 修复建议
