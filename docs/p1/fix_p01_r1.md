# P1 R1 修复任务书（OpenCode — Codex R1 终审缺陷修复）

## 背景

Codex R1 终审判定需修改（1 P1 + 1 P1 + 2 P2 + 1 P3，全部实现级直接修）。修复后须再派 Codex 复核闭环。

## 缺陷清单与修法

### P1-1 🟠：`_agentic_round` 未接入 MESA（Agentic 路径静默跳过合成节点）
- 位置：`retrieval/query_router.py:1682-1684`（`_agentic_round` 的 `_community_expansion` 后）
- 问题：`retrieve()` 在 :1321 直接返回 `_agentic_retrieve`，绕过含 `_mesa_synthesis` 的 `_finish`。`_agentic_round` 只调 `_community_expansion`/`_visual_recall`/归档过滤，无 `_mesa_synthesis` → agentic 路径 MESA 不生效，且自演化统计 `mesa_hit_count=0` 误判
- 修法：`_agentic_round` 的 `_community_expansion` 后补 `results = self._mesa_synthesis(results, query, raw_query)`（保持与 `_finish` 顺序一致）

### P1-2 🟠：api/app.py 透传 `and/or` 吞掉合法零值
- 位置：`api/app.py:499-502`
- 问题：`rcfg.mesa.boost or 0.4` → `boost=0.0` 被改成 0.4；`threshold=0.0`→0.5；`max_nodes=0`→5。但 `EvolvableParams.validate` 允许 `mesa_boost=0.0`（合法值），启动透传静默覆盖操作者配置
- 修法：改为显式缺省判断（`rcfg.mesa.boost if getattr(rcfg, "mesa", None) else 0.4` 或 `getattr(rcfg.mesa, "boost", 0.4)`），避免 0.0/0 被 or 吞掉

### P2-1 🟡：mesa_boost 演化上界可破坏"低于 community 成员"分数契约
- 位置：`retrieval/self_evolving.py:305`（规则 6 上界 `min(1.0,...)`）+ :98（validate 钳 [0,1]）
- 问题：设计注释要求 mesa_boost 严格 < community_expansion.boost=0.6（query_router.py:199/:2022），但演化上界 1.0 允许 >0.6 → 同社区合成分可超社区成员
- 修法：mesa_boost 演化上界改 `< community_expansion.boost`（如 0.59），validate 边界同步改 `[0, 0.59]`（或 0.6 开区间）。注：注释与实现需一致

### P2-2 🟡：mesa_max_nodes 下界无校验，零/负数仍合成 1 条
- 位置：`config/settings.py:112-117`（MesaConfig）+ `retrieval/query_router.py:2073-2076`（append 后 break）
- 问题：`max_nodes=0/-1` 时循环先 append 再 break → 首个候选仍合成
- 修法：`MesaConfig.max_nodes` 加 `>=1` 校验；`_mesa_synthesis` 循环前先判 `if max_nodes <= 0: return results`（或先判断再 append）

### P3-1 ⚪：mesa_relevance 无消费方 + mesa_enabled=False 时参数漂移
- 位置：`retrieval/self_evolving.py:594-610`（快照写入）+ :304-309（规则 6 只用 avg_mesa_hit）+ :650-665（report_probe 未带 mesa_hit_count）
- 问题：mesa_relevance 只写快照无诊断消费方；report_probe 低 recall 一律按 avg_mesa_hit==0 降 mesa_boost，recall 量级未参与；mesa_enabled=False 时仍漂移 mesa_boost
- 修法：规则 6 加 `mesa_enabled=False` 跳过（不调 mesa_boost）；report_probe 构造快照带 mesa_hit_count（从探针结果统计或默认 0）

## 验收标准（AC）

1. 新增/更新测试覆盖：P1-1（agentic 路径下 MESA 生效）、P1-2（boost=0.0 不被 or 吞掉）、P2-2（max_nodes=0 不合成）
2. 全量测试通过（`-p no:randomly --deselect tests/test_core_engine.py::TestTauDecay::test_decay_threshold_candidate`）
3. 默认关零回归不变

## 关键约束

- 只改任务相关文件，先 read_file 确认实际结构再改
- 版本号不 bump（v5.49.0 未发布，同天修复）
