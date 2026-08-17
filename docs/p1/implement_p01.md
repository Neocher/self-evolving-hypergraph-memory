# P1 实施任务书 — MESA 记忆增强检索 + 学习式 Dreaming（OpenCode）

## 背景

CC 设计审查已通过（设计全文在 /home/admin/shm/design_p01.md，决策表 D1-D8 + 关键假设 K1-K7）。本任务按 **CC 设计 P0 范围**实施：MESA 最小闭环 + 零回归。

## 实施清单（严格按 CC 设计，勿偏离）

### 1. MesaConfig + defaults.yaml mesa 段（D8）
- `config/settings.py` 新增 `MesaConfig` 类（紧随 `VisualRecallConfig` 之后）：
  ```python
  @dataclass
  class MesaConfig:
      enabled: bool = False        # MESA 合成节点通道开关（默认关，与 community 默认开不同）
      boost: float = 0.4           # score = rel × min_seed × boost（严格 < community_expansion.boost=0.6）
      threshold: float = 0.5       # BM25-on-summary relevance 阈值（对齐 community_expansion）
      max_nodes: int = 5           # 每查询最多合成节点数
  ```
- `config/defaults.yaml` 新增 `mesa:` 段（对齐 community_expansion 结构，默认关）：
  ```yaml
  mesa:                            # v5.49.0 MESA 记忆增强检索（默认关）
    enabled: false
    boost: 0.4
    threshold: 0.5
    max_nodes: 5
  ```
- `RetrievalConfig`（settings.py:122 附近）加 `mesa: MesaConfig = field(default_factory=MesaConfig)`

### 2. QueryRouterConfig 加 mesa_* 字段（D7/K5）
- `retrieval/query_router.py` QueryRouterConfig（:192 附近）加：
  ```python
  mesa_enabled: bool = False
  mesa_boost: float = 0.4
  mesa_threshold: float = 0.5
  mesa_max_nodes: int = 5
  ```

### 3. QueryRouter._mesa_synthesis（D1-D4）
- 在 `_finish` 内、`_community_expansion` 之后、`_visual_recall` 之前（:1293 后）新增调用：
  ```python
  results = self._mesa_synthesis(results, query, raw_query)
  ```
- 新方法（对齐 `_community_expansion` 模式，:1875）：
  - 开关：`if not self.config.mesa_enabled or not results: return results`（默认关零回归）
  - 语义：seeds = top-5 node_id → `store.get_communities_by_seeds(seeds)` → `_community_relevance(query, summaries)` 算 rel → rel < mesa_threshold 丢弃 → 合成节点 append：
    ```python
    {
        "node_id": comm_id,           # community_id（跨查询自去重 + 可回溯）
        "content": summary,           # 社区摘要（梦境产物）
        "score": round(rel * min_seed_score * self.config.mesa_boost, 6),
        "level": "mesa_synthesis",
        "_source": "mesa",
        "fact_track": "active",       # 不给 core 标记（避免误吃 ×1.1 boost）
        # 无 archived 字段（_filter_archived 恒保留，与 visual/property 一致）
    }
    ```
  - min_seed_score = min(top-5 种子分)；max_nodes 限制 append 数
  - try/except 静默降级（GraphLite 失败/异常 → 返回原 results）
  - **数学保证**：mesa_boost=0.4 < community_expansion.boost=0.6 → 合成节点低于本社区原始成员；0.4<1 → 低于种子

### 4. EvolvableParams + _sync_params（D5/D7）
- `retrieval/self_evolving.py` EvolvableParams 加 `mesa_boost: float = 0.4` + validate 边界（[0, 1]）
- `_sync_params` 同步 `mesa_boost` 到 `QueryRouterConfig.mesa_boost`（写 `self._qr.config`，对齐 agentic 先例）

### 5. RetrievalSnapshot + 统计（D5）
- `RetrievalSnapshot` 加 `mesa_hit_count: int = 0` + `mesa_relevance: float = 0.0`
- `SelfEvolvingRetrieval.retrieve` 构造快照时统计：最终结果中 `level=="mesa_synthesis"` 的条数 + 平均 rel（命中率信号）

### 6. DiagnosisEngine 规则（D5）
- 加一条 mesa 规则：mesa_hit_count 高 → 建议提升 mesa_boost；低 → 维持/降低

### 7. api/app.py 透传（K7 桥接）
- `api/app.py:492-499` QRCfg 构造加：
  ```python
  mesa_enabled=getattr(rcfg, "mesa", None) and rcfg.mesa.enabled or False,
  mesa_boost=getattr(rcfg, "mesa", None) and rcfg.mesa.boost or 0.4,
  mesa_threshold=getattr(rcfg, "mesa", None) and rcfg.mesa.threshold or 0.5,
  mesa_max_nodes=getattr(rcfg, "mesa", None) and rcfg.mesa.max_nodes or 5,
  ```

### 8. 测试（新增 tests/test_mesa_synthesis.py）
- 默认关：mesa_enabled=False 时 `_mesa_synthesis` 返回原 results（主通道 bit 级等价）
- 开启后：mock GraphLite 返回社区 → 合成节点 append（level=mesa_synthesis, score=rel×min_seed×0.4）
- score 数学保证：合成节点 < 种子分 < 社区成员分（0.4 < 0.6）
- 阈值：rel < threshold 时丢弃
- max_nodes 限制
- GraphLite 异常 → 静默降级返回原 results

## 验收标准（AC）

1. 全量测试通过（含新增测试）无回归
2. mesa.enabled=False（默认）时主检索字节级等价（零回归）
3. 版本四处同步 5.49.0（_version.py/pyproject.toml/VERSION/README.md）
4. 不做：MESA 合成节点持久化（P2）、keywords/topics 落库（P1）、LLM 打分（P2）——留给后续迭代

## 关键约束

- 只改任务相关文件，遵循原风格
- 先 read_file 确认实际结构再改（防按行号盲改）
- GraphLite 失败静默降级（不抛异常）
- 版本号：v5.49.0，版本名 "Mesa-Synthesis"（`__version_name__`）
- VERSION_SUMMARY 顶部插入 v5.49.0 段
