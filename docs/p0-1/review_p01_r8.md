# P0-1 终审任务书（Codex R8 — 最终复核 v6）

Codex R7 复核判定需再修改（1 P1 + 1 P2）。Hermes 按 R7 修法补齐：

## Hermes 修复内容（需复核）
1. **R7-P1 后继读取空行/vf=None 静默退化**（`graph/graphlite_store.py` ~654）：`if rows and rows[0].get("vf") is not None` 后加 `else: raise QueryError(...)`——读不到后继行/vf=None 时抛一致性异常，走现有回滚重抛路径（不再静默保留 succ_ts=now）
2. **R7-P2 回归测试**（`tests/test_property_temporal.py`）：新增 2 用例
   - `test_mid_insert_rollback_succ_read_exception`：execute_cypher 抛错 → 回滚新节点 + 异常传播
   - `test_mid_insert_rollback_succ_read_empty`：空结果（后继不存在）→ 一致性错误 + 回滚 + 异常传播

## 验证（Hermes 实测）
- pytest tests/test_property_temporal.py → 54 passed（+2）
- 全量 pytest（排除已知 flaky）→ 949 passed

## 审核重点
1. read_file 验证 else 分支：QueryError 抛在 try 内（会走现有 except 回滚重抛）、QueryError 已 import
2. 2 个新测试的注入方式（execute_cypher monkeypatch）与断言（新节点回滚、异常传播）
3. 是否引入新缺陷
4. 版本 5.47.0 一致

## 输出格式
- 每修复点：✅ 已闭环 / ❌ 未闭环（+文件行号+修法）
- 新问题（如有）：严重度 + 位置 + 修法
- 终审判定：最终通过 / 需再修改
