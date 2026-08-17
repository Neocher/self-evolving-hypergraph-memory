# P2 R3 修复任务书（OpenCode — Codex R3 复核缺陷修复）

## 背景

Codex R3 复核：P0-2/P1-5 通过，P1-4 子串匹配过宽误伤英文单 token + 1 P3。本任务修复后须再派 Codex 复核闭环。

## 缺陷清单与修法

### P1-6 🟠：alias 子串匹配过宽，误伤英文单 token alias
- 位置：`retrieval/query_router.py:2400-2405`（对 alias_words 全量 `if a in ql`）
- 问题：直接子串匹配未排除纯 ASCII 单 token alias，破坏词边界保证——alias `income` 会让 "Apple incoming" 命中 income → _expand_attr_aliases 扩出 revenue 并过滤属性版本；同理 age/sales 误伤 manager/salesforce。下方 :2406-2413 已有精确 token + _attr_name_matches 词边界通道
- 修法：仅对**含 CJK 或空格**的 alias 使用子串匹配，纯 ASCII 单 token 交给下方精确 token 通道：
  ```python
  if (" " in a or any(ord(ch) > 127 for ch in a)) and a in ql:
  ```
- 补回归测试：`_extract_property_terms("Apple incoming", {"revenue"}, {"revenue": ["income"]})` 不得包含 income

### P3-2 ⚪：attr_aliases 非 dict 校验
- 位置：`core/dream_pipeline.py:820` / `retrieval/query_router.py:2340`（attr_aliases 仅按 truthy 接收）
- 问题：若文件顶层 attr_aliases 为 list/string（手工损坏或旧文件），`_extract_property_terms` 的 `.items()` 抛异常，被外层 try 吞掉 → 属性时间通道静默降级
- 修法：`QueryRouter.set_attr_aliases` / 相关刷新方法统一 `if not isinstance(aliases, dict): aliases = {}` 后再赋值

## 验收标准（AC）

1. P1-6：新增测试——"Apple incoming" 不命中 income（词边界保持）；中文/多词 alias 仍命中
2. P3-2：新增测试——set_attr_aliases(非 dict) 不抛异常（降级空 dict）
3. 全量测试通过（`-p no:randomly --deselect tests/test_core_engine.py::TestTauDecay::test_decay_threshold_candidate`）+ 独立原始日志

## 关键约束

- 只改任务相关文件，先 read_file 确认实际结构再改
- 版本不 bump（v5.50.0 未发布，同天修复）
