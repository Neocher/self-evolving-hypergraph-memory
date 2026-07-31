# SHM 剩余问题修复设计 v2 — cdlib 社区检测 (v5.19.1)

## 背景

core/dream_pipeline.py `_detect_communities()` (line 529-580) 优先使用 cdlib Leiden，
但在 cdlib 0.4.1 中 `cdlib.algorithms` 不是模块属性（algorithms 是子模块），
运行时报 `module 'cdlib' has no attribute 'algorithms'`。

## 根因（已确认 + CC 审查补充）

1. `import cdlib` 后直接访问 `cdlib.algorithms.leiden` 失败 —— 0.4.1 中需显式 `from cdlib import algorithms`
2. cdlib Leiden 依赖额外包 `leidenalg`（缺失时抛 ImportError，现有 except 已能覆盖）
3. **CC 补充 #1**：同一方法内 Louvain 分支 `except ImportError: pass` 与 Leiden 不一致，且
   `next_comm = len(partition) // max(1, len(set(partition.values())))` 计算的是"平均社区大小"
   而非"下一个可用社区 ID"，极端情况下孤立节点并入错误社区（潜伏 bug）
4. **CC 补充 #2**：cdlib 未在 requirements.txt 声明（干净环境 ImportError → 永远走 Louvain）

## 修复方案（CC 审查修订版）

修改 `core/dream_pipeline.py` `_detect_communities()`：

```python
# Leiden 分支（cdlib）
try:
    from cdlib import algorithms as cdlib_algorithms
    from networkx import convert_node_labels_to_integers

    H = convert_node_labels_to_integers(G, label_attribute="_orig_id")  # 防属性冲突
    communities_list = cdlib_algorithms.leiden(H)
    partition: dict[str, int] = {}
    for comm_idx, comm in enumerate(communities_list.communities):
        for int_id in comm:
            orig_id = H.nodes[int_id].get("_orig_id", str(int_id))
            partition[orig_id] = comm_idx
    next_comm = len(communities_list.communities)
    for node in G.nodes:
        if node not in partition:
            partition[node] = next_comm
            next_comm += 1
    return partition
except ImportError:
    # cdlib 未装 / leidenalg 缺失 → 依赖不可用，回退 Louvain
    logger.warning("cdlib/leidenalg unavailable, falling back to Louvain", exc_info=True)
except Exception:
    # 算法运行失败或胶水代码 bug → 回退，但按 error 记录避免掩盖真 bug
    logger.error("cdlib Leiden failed unexpectedly, falling back to Louvain", exc_info=True)

# Louvain 分支（networkx）
try:
    from networkx.algorithms.community import louvain_communities

    partition = {}
    for comm_idx, comm in enumerate(louvain_communities(G)):
        for node_id in comm:
            partition[node_id] = comm_idx
    # 修复 next_comm bug：下一个可用社区 ID = max+1
    next_comm = (max(partition.values()) + 1) if partition else 0
    for node in G.nodes:
        if node not in partition:
            partition[node] = next_comm
            next_comm += 1
    return partition
except Exception:
    logger.warning("networkx Louvain unavailable, falling back to connected components", exc_info=True)

# 连通分量兜底（不变）
```

**关键变化：**
1. `import cdlib` → `from cdlib import algorithms as cdlib_algorithms`
2. `except ImportError` 拆两层：ImportError → warning（依赖缺失），Exception → error（运行失败）
3. Louvain 分支同步修复异常处理 + `next_comm = max+1` bug
4. `label_attribute="_orig_id"` 防节点属性冲突
5. requirements.txt 补 `cdlib>=0.4.1`

## 验收标准

1. `python3 -c "from cdlib import algorithms; print(hasattr(algorithms, 'leiden'))"` → True
2. 直接调用 `_detect_communities` 不抛异常（Leiden 或 Louvain 或连通分量成功）
3. 极端场景测试：孤立节点 + 3 社区时 `next_comm` 不碰撞
4. 现有测试无回归
5. Health API 版本号 v5.19.1

## 不做的事

- 不安装 leidenalg（增加依赖，Louvain 已够用）
- 不重构社区检测逻辑
- 不修改其他文件
