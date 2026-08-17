# P1 R3 复核任务书（Codex — R2 修复终审）

## 背景

Codex R2 复核发现 4 新缺陷（P2-3 配置层上界 / P3-2 mesa_relevance / P3-3 _version 文档 / P3-4 pytest cache）→ OpenCode R2 修复完成（全量 1007 passed）。本任务：R3 复核闭环。

## R2 修复内容（待复核）

1. **P2-3** 运行时 clamp：`_mesa_synthesis` :2050-2056 `boost = min(boost, community_boost * 0.95)`（community_boost 从配置取）+ MesaConfig 校验
2. **P3-2** 规则 6 消费 mesa_relevance：:236 avg_mesa_rel + :310-315 命中质量信号（avg_hit>=1 and avg_relevance>=0.3 才升）
3. **P3-3** _version.py:20 validate[0,0.59]（对齐 _MESA_BOOST_MAX）
4. **P3-4** .pytest_cache/v/cache/lastfailed = {}（已清空）+ 全量 1007 passed 实际输出

## 复核要求（read_file 静态分析）

1. P2-3：clamp 逻辑正确？community_boost 来源正确（配置读取）？clamp 后合成节点数学保证恢复（< 社区成员 < 种子）？MesaConfig 校验？
2. P3-2：规则 6 消费 mesa_relevance 逻辑正确（命中质量信号）？不会误降/误升？
3. P3-3：_version.py 文档与实际一致？
4. P3-4：cache 清空 + 全量证据（1007 passed 是否合理）
5. 综合判定：是否可闭环发布？

## 输出格式

- 判定：通过 / 需修改
- 缺陷清单（🔴 P0 / 🟠 P1 / 🟡 P2 / ⚪ P3，含文件:行号 + 证据）
- 修复建议
