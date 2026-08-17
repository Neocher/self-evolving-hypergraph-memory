# P2 R5 复核任务书（Codex — R4 修复终审）

## 背景

Codex R4 复核发现 1 P1（启动注入路径遗漏）→ OpenCode R4 修复完成（全量 1050 passed，日志 /tmp/pytest_p02_r4.log）。本任务：R5 复核闭环。

## R4 修复内容（待复核）

1. **P1-7** QueryRouter.__init__:288 改为 `self._attr_aliases = attr_aliases if isinstance(attr_aliases, dict) else {}`（根因修复，覆盖构造注入路径）+ 启动路径测试

## 复核要求（read_file 静态分析）

1. P1-7：:288 isinstance 校验正确（非 dict → {}）？构造注入路径全覆盖？启动路径测试（attr_aliases=["x"] → 不抛异常）？
2. 全量证据：/tmp/pytest_p02_r4.log 尾部 1050 passed？
3. 综合判定：是否可闭环发布 v5.50.0？

## 输出格式

- 判定：通过 / 需修改
- 一句话总结（可发布 / 需再修）
