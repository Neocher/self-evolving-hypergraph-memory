# P1 R1 复核任务书（Codex — MESA 实施终审）

## 背景

SHM v5.49.0 MESA（Memory-Enhanced Search Architecture，记忆增强检索架构）实施完成。CC 设计审查（design_p01.md）→ OpenCode 实施（implement_p01.md）→ 本任务：Codex 终审。

**设计核心**：`_finish` 内新增默认关闭的 `_mesa_synthesis` 补充通道，把梦境社区 summary 按 `rel×min_seed×0.4` 的相对尾分缩放 append 为 `level="mesa_synthesis"` 的虚拟合成节点，并把 `mesa_boost` 挂进现有 ConfigEvolver（复用回滚保护），以「合成节点命中率 + 健康探针 recall」为奖励驱动权重演化，全程零回归。

## 实施内容（待复核）

1. `config/settings.py`：MesaConfig（enabled=False/boost=0.4/threshold=0.5/max_nodes=5）+ RetrievalConfig.mesa 字段
2. `config/defaults.yaml`：mesa: 段（默认关）
3. `retrieval/query_router.py`：QueryRouterConfig mesa_* 字段（:198-199）+ `_mesa_synthesis`（:2008）
4. `retrieval/self_evolving.py`：EvolvableParams.mesa_boost（:80）+ validate 边界 + _sync_params + RetrievalSnapshot.mesa_hit_count/mesa_relevance（:129-130）+ DiagnosisEngine 规则 6（:302-305）
5. `api/app.py`：QRCfg 构造透传 mesa_*（:499-500）
6. `tests/test_mesa_synthesis.py`：8 passed（默认关零回归/开启合成节点/score 数学保证/阈值/max_nodes/异常降级）

## 复核要求（read_file 静态分析，不实测）

按 AGENTS.md 审计要求：

1. **追踪调用链**：`_mesa_synthesis` 是否在 `_finish` 内被调用（query_router.py:1293 附近）？调用顺序是否正确（_community_expansion 之后）？
2. **配置源一致性（CC 设计 K5/K7 关键）**：`_mesa_synthesis` 读 `self.config`（QueryRouterConfig）而非 `get_settings()`——确认 EvolvableParams._sync_params 能写入 QueryRouterConfig.mesa_boost（自演化生效点）？api/app.py 透传是否完整（enabled/boost/threshold/max_nodes 四字段）？
3. **score 数学保证**：合成节点 score = rel×min_seed×mesa_boost(0.4) < community_expansion 成员 rel×min_seed×0.6 < 种子分——验证代码实现与设计一致？
4. **零回归契约**：mesa_enabled=False 默认时 `_mesa_synthesis` 首行短路返回原 results？主检索字节级等价？
5. **合成节点标记**：level="mesa_synthesis"/_source="mesa"/fact_track="active"（不给 core 标记避免误吃 ×1.1）/无 archived 字段（恒保留）？node_id=community_id 跨查询自去重？
6. **生命周期**：无新增持久化（查询时虚拟生成）——确认没有引入孤儿节点/持久化残留？
7. **静默降级**：GraphLite 异常/无 communities → 返回原 results 不抛异常？
8. **DiagnosisEngine 规则**：mesa 规则是否合理（命中多→升 boost；命中少→降/维持）？边界钳制 [0,1]？
9. **版本四处**：_version.py（5.49.0/Mesa-Synthesis/2026-08-17/VERSION_SUMMARY v5.49.0 段）/pyproject.toml/VERSION/README.md 一致？
10. **遗留标记**：确认 shm/_version.py 无 merge 冲突标记残留（已手工清理 `>>>>>>> f8eeb75`）？

## 输出格式

- 判定：通过 / 需修改
- 缺陷清单（严重度分级 🔴 P0 / 🟠 P1 / 🟡 P2 / ⚪ P3，含文件:行号 + 证据）
- 修复建议（具体修法）
