# P2 实施任务书 — Schema 演化深化 P0（OpenCode）

## 背景

CC 设计审查已通过（设计全文在 /home/admin/shm/design_p02.md，决策 1-6 + 关键假设 1-6）。本任务按 **CC 设计 P0 范围**实施：属性别名合并 + 中文映射学习的零回归最小闭环。

## 实施清单（严格按 CC 设计，勿偏离）

### 1. `get_distinct_attr_names()` 只读查询（D2）
- `graph/graphlite_store.py` 新增：
  ```python
  def get_distinct_attr_names(self) -> list[str]:
      """【v5.50.0 P2】全部属性名清单（PropertyVerNode.attr_name distinct）。
      只读，复用 query_cypher 永不抛异常契约；GraphLite 失败 → []。"""
  ```
  - GQL：`MATCH (p:PropertyVerNode) RETURN DISTINCT p.attr_name`
  - 结果提取 attr_name 值，失败/空 → []

### 2. `_apply_attr_ops` + attr_aliases 写入（D1/D3）
- `core/ontology_evolution.py` 新增 `_apply_attr_ops(parsed, current, distinct_attrs)` 纯函数：
  - 输入：LLM 响应 dict 的 `attr_ops` 数组字段 `[{"canonical": str, "aliases": [str], "op": "merge_alias"}]`
  - 守卫：max 1 attr_op/轮（顺序尝试第一个过守卫的）；canonical/alias 均过 `_is_generic_key`（非泛词）；**canonical 必须 ∈ distinct_attrs**（否则孤儿 alias 无消费方 → skip）
  - 写入：extended JSON per-type `"attr_aliases": {"<canonical>": ["<alias1>", ...]}`（canonical 对齐 `{relation}_{key}` 格式）
- `evolve_once`（:272）：
  - `_build_prompt` 注入 distinct attr_name 清单（prompt 里列出当前属性名，供 LLM 决策别名合并）
  - LLM 响应解析加 attr_ops 处理：`action` 分支后新增 `_apply_attr_ops` 调用（与类型决策正交，可同轮发生）
  - `evolve_once` 签名加 `distinct_attrs: Optional[list] = None`（None → skip attr_ops）
- 落盘：复用 `_extended_only` + `_atomic_write`（失败 → skip 不声称成功）

### 3. `_expand_attr_aliases` 纯函数 + 通道内消费（D3）
- `retrieval/query_router.py` 新增纯函数：
  ```python
  def _expand_attr_aliases(terms: list[str], aliases: dict) -> list[str]:
      """【v5.50.0 P2】属性词归一：term 命中 alias 表 → 扩展出 canonical。
      纯增量（只可能多命中）；空表/无命中 → 返回原 terms。"""
  ```
  - 对每个 term：若在 aliases 的某个 alias 列表里（或 term == canonical）→ 追加 canonical；否则保留原 term
  - 去重保序
- `_property_temporal_retrieve`（:2366 附近）：`_extract_property_terms` 之后、`_attr_name_matches` 过滤之前插入：
  ```python
  terms = self._expand_attr_aliases(terms, self._attr_aliases or {})
  ```
  - `self._attr_aliases` 为空/None → 恒等短路（行为与现状逐字节等价）

### 4. QueryRouter 注入 attr_aliases 接线（D5）
- `retrieval/query_router.py` `__init__` 加参数 `attr_aliases: Optional[dict] = None`（存 `self._attr_aliases`）
- `api/app.py` 构造 QueryRouter 时注入：`attr_aliases=ontology_evolution.merged().get("attr_aliases")` 或 load_extended 读取（参考 app.py:348-352 已有 extended_types 注入模式）
- 空表 → 短路（零回归）

### 5. 测试（新增 tests/test_schema_attr_ops.py）
- get_distinct_attr_names：mock GraphLite 返回 attr_name 列表；失败 → []
- _apply_attr_ops：canonical ∈ distinct_attrs 才写入；canonical 不在 → skip；泛词 → skip；max 1/轮
- _expand_attr_aliases：term 命中 alias → 扩展 canonical；空表 → 原样；去重保序
- 通道内消费：mock _property_temporal_retrieve 走公共入口（防假绿）——构造 QueryRouter 注入 attr_aliases，验证属性查询命中 canonical

## 验收标准（AC）

1. 全量测试通过（含新增测试）无回归
2. attr_aliases 为空（默认）时检索逐字节等价（零回归）
3. 版本四处同步 5.50.0（_version.py/pyproject.toml/VERSION/README.md）+ VERSION_SUMMARY v5.50.0 段
4. 不做：属性分裂/废弃/值冲突检测（P2 砍掉）、miss 信号回喂（P1）、classify_with_extended 激活（暂缓）——留给后续迭代

## 关键约束

- 只改任务相关文件，遵循原风格
- 先 read_file 确认实际结构再改（防按行号盲改）
- GraphLite 失败静默降级（不抛异常）
- 版本号：v5.50.0，版本名 "Schema-AttrOps"（`__version_name__`）
