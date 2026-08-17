# P2 R4 复核任务书（Codex — R3 修复终审）

## 背景

Codex R3 复核发现 1 P1 + 1 P3 → OpenCode R3 修复完成（全量 1048 passed，日志 /tmp/pytest_p02_r3.log）。本任务：R4 复核闭环。

## R3 修复内容（待复核）

1. **P1-6** alias 子串匹配收窄（query_router.py:2410 `(" " in a or any(ord(ch) > 127 for ch in a)) and a in ql`）——仅 CJK/空格 alias 子串匹配，纯 ASCII 单 token 交给精确 token 通道
2. **P3-2** set_attr_aliases isinstance 校验（query_router.py:2341 `if not isinstance(aliases, dict)`）——非 dict 降级 {}

## 复核要求（read_file 静态分析）

1. P1-6：子串匹配收窄正确（CJK/空格 alias 仍命中，ASCII 单 token 不再误伤）？"Apple incoming" 不命中 income？词边界保证恢复？回归测试覆盖？
2. P3-2：set_attr_aliases 非 dict 降级 {} 正确？属性通道不静默失效？
3. 全量证据：/tmp/pytest_p02_r3.log 尾部 1048 passed？
4. 综合判定：是否可闭环发布 v5.50.0？

## 输出格式

- 判定：通过 / 需修改
- 缺陷清单（🔴 P0 / 🟠 P1 / 🟡 P2 / ⚪ P3，含文件:行号 + 证据）
- 修复建议
