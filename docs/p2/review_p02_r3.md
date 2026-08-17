# P2 R3 复核任务书（Codex — R2 修复终审）

## 背景

Codex R2 复核发现 1 P0 + 2 P1 → OpenCode R2 修复完成（全量 1043 passed，日志 /tmp/pytest_p02_r2.log）。本任务：R3 复核闭环。

## R2 修复内容（待复核）

1. **P0-2** validator 过滤非类型键（ontology_validator.py `_merged_ontology_types` 只保留 isinstance(v,dict) and "conflict_keys" in v 的项）——attr_aliases 不再 KeyError → 写入 500
2. **P1-4** alias 识别直接子串匹配（_extract_property_terms 对 alias_words 做 if a in query.lower()）——中文/多词 alias 也能识别
3. **P1-5** set_attr_aliases 运行时刷新（query_router.py:2334）——梦境写盘后更新活动路由器 alias map

## 复核要求（read_file 静态分析）

1. P0-2：_merged_ontology_types 过滤逻辑正确（保留 conflict_keys dict、滤除 attr_aliases）？write_validate 在 extended_types 含 attr_aliases 时不抛异常？回归测试覆盖？
2. P1-4：_extract_property_terms 子串匹配正确（中文/多词 alias 识别）？不误伤英文单 token？公共入口测试（{"revenue":["营业额"]} + "Apple 营业额"）？
3. P1-5：set_attr_aliases 实现正确（更新 self._attr_aliases）？梦境写盘后调用链完整（_ontology_evolution_step → set_attr_aliases）？经 retrieval_guard 更新内层 _qr？
4. 全量证据：/tmp/pytest_p02_r2.log 尾部 1043 passed？
5. 综合判定：是否可闭环发布 v5.50.0？

## 输出格式

- 判定：通过 / 需修改
- 缺陷清单（🔴 P0 / 🟠 P1 / 🟡 P2 / ⚪ P3，含文件:行号 + 证据）
- 修复建议
