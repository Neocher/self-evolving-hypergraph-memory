# P1 R3 修复任务书（OpenCode — Codex R3 复核缺陷修复）

## 背景

Codex R3 复核：R2 修复基本通过，但发现 1 P1 + 3 P3。本任务修复后须再派 Codex 复核闭环。

## 缺陷清单与修法

### P1-3 🟠：健康探针零命中误降 mesa_boost
- 位置：`retrieval/self_evolving.py:664-676`（探针快照固定 source="probe"、mesa_hit_count=0）+ :725-727（仍传 mesa_enabled）+ :321-323（avg_mesa_hit==0 无条件降）
- 问题：MESA 开启时，任意 recall<0.5 的梦境健康探针都会持续压低 mesa_boost（与 MESA 实际检索质量无关），最终可能压到 0
- 修法：**规则 6 只基于 source=="retrieve" 的快照计算/调整**——在 DiagnosisEngine 计算 avg_mesa_hit/avg_mesa_rel 前过滤 source=="probe" 快照（或探针触发时跳过 MESA 调整）

### P3-5 ⚪：mesa_relevance 语义与命名不一致
- 位置：`retrieval/self_evolving.py:135`（注释"合成节点平均分"）+ :607-610（sum(score)/hit_count）+ :315-316（注释/阈值按 avg_relevance）
- 问题：score = rel_score × min_seed_score × boost（query_router.py:2075），不是原始相关度；阈值 0.3 受种子分和 boost 缩放，低种子分场景可能长期不升
- 修法（二选一，Karpathy 最少代码）：
  - **方案 A（推荐）**：改名 `mesa_avg_score`（语义对齐：合成节点平均分）+ 注释同步；阈值口径按 score（0.3 可保持或调小）
  - 方案 B：存原始 rel_score（合成节点额外字段）后统一阈值口径（改动大）
- 采用方案 A（改名 + 注释同步，最小改动）

### P3-6 ⚪：_version.py:22 规则描述滞后
- 位置：`shm/_version.py:22` 写"命中多→升、零命中→降"，实际三分支（:317-323：avg_hit>=1 且 avg_rel>=0.3 升，零命中降，中间/弱命中维持）
- 修法：同步为"命中多且强→升；零命中→降；中间/弱命中→维持"

### P3-7 ⚪：1007 passed 缺静态证据
- 位置：`.pytest_cache/v/cache/nodeids`（1156 个收集 ID）+ 无 pytest 输出日志
- 问题：空 lastfailed 不能独立证明通过数量；nodeids 含 deselected 测试
- 修法：保留/附上干净运行输出（`/tmp/pytest_p01.log` 已有 1007 passed 输出可作证据）；说明 nodeids 是收集数非通过数

## 验收标准（AC）

1. P1-3：新增测试——mesa_enabled=True + source="probe" 快照不触发 mesa_boost 调整（探针不误降）；source="retrieve" 快照正常调整
2. P3-5：mesa_relevance 改名 mesa_avg_score（字段/注释/消费方同步）
3. P3-6：_version.py:22 描述同步三分支
4. P3-7：保留 /tmp/pytest_p01.log 全量输出作证据（已有）
5. 全量测试通过（`-p no:randomly --deselect tests/test_core_engine.py::TestTauDecay::test_decay_threshold_candidate`）

## 关键约束

- 只改任务相关文件，先 read_file 确认实际结构再改
- 版本不 bump（v5.49.0 未发布，同天修复）
