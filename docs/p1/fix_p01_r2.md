# P1 R2 修复任务书（OpenCode — Codex R2 复核缺陷修复）

## 背景

Codex R2 复核：R1 五项修复全部通过，但发现新缺陷（1 P2 + 3 P3）。本任务修复后须再派 Codex 复核闭环。

## 缺陷清单与修法

### P2-3 🟡：配置层可绕过 mesa_boost 上界（破坏分数契约）
- 位置：`config/settings.py:115`（MesaConfig.boost 无校验）+ `retrieval/query_router.py:199,2049`（透传并消费）+ `retrieval/self_evolving.py:49,103,314`（仅演化层有界）
- 问题：`mesa.boost=0.7` 配置时合成节点可超社区成员（0.6）；`community_expansion.boost` 调低（如 0.5）时 0.59 仍超。硬编码 0.59 与实际配置漂移
- 修法（运行时 clamp 最稳）：
  1. `_mesa_synthesis` 读取 boost 后做运行时 clamp：`boost = min(boost, community_boost * 0.95)`（community_boost 从 `self.config` 或 get_settings().retrieval.community_expansion.boost 取，确保合成节点严格低于社区成员）
  2. `MesaConfig` 加 `__post_init__` 校验：`boost < community_expansion.boost`（或至少 <=0.59），不满足 raise ValueError
  3. 保持 `_MESA_BOOST_MAX=0.59` 演化上界不变（自演化层已正确）

### P3-2 ⚪：mesa_relevance 仍无消费方
- 位置：`retrieval/self_evolving.py:135,603,619`（定义/写入）+ 规则 6（:308-318 只用 avg_mesa_hit）
- 修法（二选一，Karpathy 最少代码）：
  - **方案 A（推荐）**：规则 6 加命中强度：`avg_hit >= 1 and avg_relevance >= 0.3` 才升 mesa_boost（把 mesa_relevance 作为命中质量信号消费）
  - 方案 B：删除 mesa_relevance 字段（若 A 太复杂）
- 采用方案 A（mesa_relevance 已有写入，消费它闭环）

### P3-3 ⚪：_version.py:20 文档不一致
- 位置：`shm/_version.py:20` 写 `validate[0,1]`
- 修法：改为 `validate[0,0.59]`（对齐 _MESA_BOOST_MAX）

### P3-4 ⚪：pytest cache lastfailed 残留 40 failed（验证证据混淆）
- 位置：`.pytest_cache/v/cache/lastfailed`
- 问题：历史失败的测试残留（含 test_decay_threshold_candidate deselected），非当前通过态；Codex 无法静态证实 1003 passed
- 修法：清空 `.pytest_cache/v/cache/lastfailed`（写空 {} 或删除文件）；重跑全量测试后重新生成干净 cache

## 验收标准（AC）

1. P2-3：新增测试验证——mesa.boost=0.7 配置时运行时 clamp 到 < community_boost；community_expansion.boost=0.5 时 mesa 合成分仍低于成员
2. P3-2：规则 6 用 mesa_relevance（命中强度信号）
3. P3-3：_version.py:20 改 validate[0,0.59]
4. P3-4：.pytest_cache 清理 + 全量测试重新验证
5. 全量测试通过（`-p no:randomly --deselect tests/test_core_engine.py::TestTauDecay::test_decay_threshold_candidate`）

## 关键约束

- 只改任务相关文件，先 read_file 确认实际结构再改
- 版本不 bump（v5.49.0 未发布，同天修复）
