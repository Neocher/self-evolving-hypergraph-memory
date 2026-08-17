# P1 R4 复核任务书（Codex — R3 修复终审）

## 背景

Codex R3 复核发现 1 P1 + 3 P3 → OpenCode R3 修复完成（全量 1009 passed）。本任务：R4 复核闭环。

## R3 修复内容（待复核）

1. **P1-3** 规则 6 只基于 source=="retrieve" 快照（self_evolving.py:235-238 过滤 probe 快照）
2. **P3-5** mesa_relevance 改名 mesa_avg_score（5 处同步）
3. **P3-6** _version.py:22 描述"命中多且强→升；零命中→降；中间/弱命中→维持"
4. **P3-7** 全量 1009 passed 实际输出作证据

## 复核要求（read_file 静态分析）

1. P1-3：规则 6 计算 avg_mesa_hit/avg_mesa_avg_score 前过滤 source=="probe"？探针快照不触发 mesa_boost 调整？新增回归测试覆盖？
2. P3-5：mesa_avg_score 全仓无残留 mesa_relevance？
3. P3-6：_version.py 描述与实际三分支一致？
4. 综合判定：是否可闭环发布 v5.49.0？

## 输出格式

- 判定：通过 / 需修改
- 缺陷清单（🔴 P0 / 🟠 P1 / 🟡 P2 / ⚪ P3，含文件:行号 + 证据）
- 修复建议
