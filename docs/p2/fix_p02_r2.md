# P2 R2 修复任务书（OpenCode — Codex R2 复核缺陷修复）

## 背景

Codex R2 复核：R1 六点 4 通过 2 未完全修复，另发现 1 P0 + 2 P1。本任务修复后须再派 Codex 复核闭环。

## 缺陷清单与修法

### P0-2 🔴：ontology_validator KeyError → 写入 500
- 位置：`core/ontology_validator.py:270-272`（假设 merged 中每个 info 有 conflict_keys）+ :264（extended 全量合并）+ `api/app.py:352`（含 attr_aliases 的 extended_types 原样传入）+ `api/routes/write.py:592`（write_validate）+ :1128（_classify_ontology_type 不在 try 内）
- 问题：`attr_aliases` 是 `{canonical: [alias...]}`（list 值无 conflict_keys 键），`_classify_ontology_type` 访问 info["conflict_keys"] → KeyError → 写入 500（已有 alias 表时重启后正常写入即炸）
- 修法：`ontology_validator.py` 合并时过滤非类型键——`_merged_ontology_types` 只保留 `isinstance(v, dict) and "conflict_keys" in v` 的项（或显式排除 _RESERVED_KEYS）；补回归测试：`extended_types={"attr_aliases": {...}}` 下 write_validate 不抛异常

### P1-4 🟠：alias 识别只覆盖单 token 英文（中文/多词 alias 漏）
- 位置：`retrieval/query_router.py:2381-2389`（re.finditer(r'[a-z]{2,}') 只提取单 token 英文）
- 问题：`_apply_attr_ops` 允许中文 alias 写入（ontology_evolution.py:257/:263），测试也存了 "营收"，但这些 alias 到 `_extract_property_terms` 后收不到 → `_expand_attr_aliases` 无从扩展
- 修法：`_extract_property_terms` 中 alias 识别改为**直接子串匹配**：对 alias_words 做 `if a in query.lower(): terms.add(a)`（覆盖中文 + 多词英文 alias）；补 `{"revenue": ["营业额"]}` + "Apple 营业额" 的公共入口测试

### P1-5 🟠：alias 表只启动注入一次（梦境写盘后不刷新）
- 位置：`api/app.py:514`（启动注入）+ `retrieval/query_router.py:285`（存 self._attr_aliases）+ `core/ontology_evolution.py:399-400`（梦境写盘）+ `core/dream_pipeline.py:796`（无回写）
- 问题：梦境 `_ontology_evolution_step` 成功产生 attr_op 写盘后，已运行的 QueryRouter._attr_aliases 不刷新 → 新学 alias 重启才生效
- 修法：`_ontology_evolution_step` 成功产生 attr_op 后，更新活动路由器的 alias map——新增 `set_attr_aliases(aliases)` 方法（经 retrieval_guard 更新内层 _qr），或 `OntologyEvolution.evolve` 返回 attr_op 时回调更新；若实现复杂，明确记录为重启生效（降级 P2 文档化）

## 验收标准（AC）

1. P0-2：新增回归测试——extended_types 含 attr_aliases 时 write_validate 不抛异常（写入 200）
2. P1-4：新增公共入口测试——{"revenue": ["营业额"]} + "Apple 营业额" 命中 canonical revenue
3. P1-5：梦境写盘后 alias map 刷新（或明确文档化重启生效）
4. 全量测试通过（`-p no:randomly --deselect tests/test_core_engine.py::TestTauDecay::test_decay_threshold_candidate`）+ 独立原始日志

## 关键约束

- 只改任务相关文件，先 read_file 确认实际结构再改
- 版本不 bump（v5.50.0 未发布，同天修复）
