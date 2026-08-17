# P2 R4 修复任务书（OpenCode — Codex R4 复核缺陷修复）

## 背景

Codex R4 复核：P1-6 通过，P3-2 只修了运行时刷新路径，启动注入路径未覆盖（QueryRouter.__init__:285 未做 isinstance 校验）→ 文件损坏时属性通道仍静默降级。本任务修复后须再派 Codex 复核闭环。

## 缺陷清单与修法

### P1-7 🟠：QueryRouter.__init__ 启动注入未做 dict 校验
- 位置：`retrieval/query_router.py:285`（`self._attr_aliases = attr_aliases or {}`）
- 问题：api/app.py:514 直接把 `ontology_extended_types.get("attr_aliases")` 注入 QueryRouter，该值可能是 truthy 的 list/string；:285 未做 dict 校验 → :2396 `.items()` 抛 AttributeError → 被外层 try 吞掉 → 属性通道静默降级（文件损坏时）
- 修法（根因修复）：`:285` 改为 `self._attr_aliases = attr_aliases if isinstance(attr_aliases, dict) else {}`（覆盖所有构造注入路径）
- 补测试：构造注入非 dict attr_aliases 的启动路径测试（QueryRouter 构造 attr_aliases=["x"] → 属性通道不抛异常、静默空表）

## 验收标准（AC）

1. P1-7：新增测试——QueryRouter 构造注入非 dict attr_aliases → self._attr_aliases == {} 不抛异常
2. 全量测试通过（`-p no:randomly --deselect tests/test_core_engine.py::TestTauDecay::test_decay_threshold_candidate`）+ 独立原始日志

## 关键约束

- 只改任务相关文件，先 read_file 确认实际结构再改
- 版本不 bump（v5.50.0 未发布，同天修复）
