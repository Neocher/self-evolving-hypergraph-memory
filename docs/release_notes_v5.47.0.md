# v5.47.0 — Entity-Property-Time（P0-1）

## 核心能力：实体属性时间三维建模
- **存储**：独立 `PropertyVerNode` + `SUPERSEDES` 血统边（P→new→S 双挂链 + 双向 expired_at），N=8 版本惰性裁剪
- **写入**：`entity_resolver` 版本编排（幂等/单调时间戳/同值时序判定/非原子写补偿），挂 `_run_entity_resolver` 写线程
- **时间注入**：`relation_extractor` 中文年份正则 `(?<!\d)(?:19|20)\d{2}(?!\d)` → attr_year → valid_from
- **检索**：`_property_temporal_retrieve` 独立通道（latest / at_time / current 三模式），年末语义 + expired_at 校验 + 相对时间词换算 + 属性词过滤
- **Schema 零迁移**：`AttributeDef` 末尾 `temporal: bool = False`

## 工程加固（Codex 八轮终审 R1-R8 闭环）
- 乱序写入血统链完整性（中段插入双挂链 + 删除旧 P→S 边）
- 同值历史写入建版本（at_time 历史时点可查）
- 补偿路径完整回滚（4 个失败点 + 后继读取失败抛一致性异常）
- GraphLite 双 MATCH DELETE 静默无效坑（实测规避，单 MATCH 边模式）
- 注入测试 6 类（SET/P→new/SET new/new→S/读取异常/空结果）

## 验证
- `tests/test_property_temporal.py`：54 passed
- 全量 pytest：949 passed（1 flaky deselected + 1 skipped）
- Codex 终审 R8：**最终通过**

## 依赖
- Neocher/GraphLite fork ≥ 4452a96（UTF-8 修复 + 多语句 + ID()）
- bge-small-zh-v1.5（512d, CPU）
