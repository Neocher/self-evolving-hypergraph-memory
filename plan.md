# SHM v5.14.1 → v6.0 优化实施计划

**Generated**: 2026-07-28 | **Source**: CC analysis of 12 optimization directions

---

## 12 方向复杂度评估

| # | 方向 | 梯队 | 复杂度 | 代码量 | 时间 | 核心依赖 |
|:-:|:-----|:----|:--------|:-------|:-----|:---------|
| 1 | 记忆投毒防御 | T1 | 🔴高 | 800-1500行 | 2-3周 | BLAKE3链 |
| 2 | 写入消解 (Write Reconciliation) | T1 | 🔴高 | 1000-2000行 | 2-4周 | 本体系统 |
| 3 | 时序相位旋转 (RoMem) | T1 | 🟡中 | 500-800行 | 1-2周 | τ衰减 |
| 4 | 多模态记忆 | T1 | 🔴高 | 2000-4000行 | 3-6周 | 向量存储 |
| 5 | 认证 + 鉴权 | T2 | 🟡中 | 500-1000行 | 1-2周 | 网关层 |
| 6 | 速率限制 + 配额 | T2 | 🟢低 | 300-600行 | 0.5-1周 | #5 |
| 7 | 可观测面板 (Dashboard) | T2 | 🟡中 | 2000-4000行 | 2-4周 | 无 |
| 8 | 向量DB可插拔 | T2 | 🟡中 | 600-1200行 | 1-2周 | FAISS |
| 9 | Benchmark 套件 | T3 | 🟡中 | 800-1500行 | 1-2周 | 无 |
| 10 | MCP Registry 提交 | T3 | 🟢低 | 100-300行 | 3-5天 | MCP Server |
| 11 | A2A 全协议合规 | T3 | 🟡中 | 800-1500行 | 1-2周 | A2A适配器 |
| 12 | 记忆可视化 (Web UI) | T3 | 🔴高 | 2000-4000行 | 2-4周 | #7 |

---

## 推荐实施顺序

```
Phase 1: 安全基座 (Week 1-3)     🔴 不可跳过
  └── #5 认证+鉴权 → #6 速率限制

Phase 2: 社区引爆 (Week 3-6)     🟢 核心增长
  └── #9 Benchmark → #10 MCP Registry → #7 Dashboard → #12 可视化

Phase 3: 生产硬化 (Week 6-10)    🟡 深度能力
  └── #1 投毒防御 ∥ #2 写入消解 ∥ #11 A2A合规

Phase 4: 扩展生态 (Week 10-14)   🟢 锦上添花
  └── #8 向量DB可插拔 ∥ #3 时序相位 ∥ #4 多模态记忆
```

---

## 最短路社区引爆路径 🚀

```
Week 1-2:  #5 认证 (安全是公开前提)
Week 2-3:  #6 速率限制
Week 3-4:  #9 Benchmark (定量对比数据)
Week 4:    #10 MCP Registry PR (89K★ 流量入口)
Week 5-6:  #7 Dashboard MVP
Week 6:    📢 发布: Blog + Show HN + V2EX + 即刻
```

**引爆 Checklist:**
- MCP Server 出现在官方 Registry
- Benchmark 对比图 (SHM vs Mem0 vs Letta vs Engram)
- Dashboard 截图/GIF
- 博客: "SHM: A Self-Evolving Memory System"
- Show HN: "SHM — Self-evolving hypergraph memory with MCP/A2A/ACP support (7 unique capabilities)"

---

## 各方向详细验收标准

### #5 认证 + 鉴权 (🟡中, 500-1000行)
| AC | 描述 |
|:---|:-----|
| AC1 | 无 token 请求 → 401 Unauthorized |
| AC2 | 错误 token → 403 Forbidden |
| AC3 | API key 可通过 CLI 创建/撤销/轮换 |
| AC4 | 认证覆盖 HTTP :8000 / A2A :8001 / ACP :8770 三个入口 |
| AC5 | MCP stdio :8002 使用文件系统权限控制 |
| AC6 | 密钥存文件 (chmod 600)，不存数据库 |
| AC7 | 默认关闭 auth (DEV_MODE=true) 保持开发体验 |

### #6 速率限制 + 配额 (🟢低, 300-600行)
| AC | 描述 |
|:---|:-----|
| AC1 | Per-agent/Per-IP 速率限制 (requests/min) |
| AC2 | 配额系统: 写入配额/检索配额/梦境配额 |
| AC3 | 超限 → 429 Too Many Requests + Retry-After header |
| AC4 | 配额可配置 (环境变量) |
| AC5 | 默认值合理: 1000 req/min, 10k writes/day |

### #9 Benchmark 套件 (🟡中, 800-1500行)
| AC | 描述 |
|:---|:-----|
| AC1 | 至少 3 测试集: LongMemEval / MemEval / 自建 QA |
| AC2 | 对比方: Mem0 + Letta + Engram + 官方MCP Memory |
| AC3 | 指标: Recall / Precision / F1 / Latency P50/P95 |
| AC4 | 结果输出: Markdown 表格 + JSON + HTML |
| AC5 | CI 可触发: `python -m benchmark` |
| AC6 | 每次 SHM 版本变更自动触发 |

### #10 MCP Registry 提交 (🟢低, 100-300行)
| AC | 描述 |
|:---|:-----|
| AC1 | MCP Server 通过 `@modelcontextprotocol/inspector` 验证 |
| AC2 | 提交 PR 到 github.com/modelcontextprotocol/servers |
| AC3 | PR 含 README 文档和配置示例 |
| AC4 | 实现 tools/search 作为 Required 工具 |

### #7 可观测面板 (🟡中, 2000-4000行)
| AC | 描述 |
|:---|:-----|
| AC1 | Web 仪表盘 (FastAPI + Jinja2/React) |
| AC2 | 实时显示: 记忆总数、检索延迟、梦境状态、FAISS 索引大小 |
| AC3 | 图表: 检索成功率趋势、节点增长率 |
| AC4 | 日志查看: 最近 N 条操作日志 |

### #1 记忆投毒防御 (🔴高, 800-1500行)
| AC | 描述 |
|:---|:-----|
| AC1 | 语义矛盾检测 → 告警 (不自动阻断) |
| AC2 | 可疑写入隔离到 quarantine 区 |
| AC3 | 误报率 <5% |
| AC4 | 检测延迟 <50ms |
| AC5 | 审计链记录投毒检测事件 |

### #2 写入消解 (🔴高, 1000-2000行)
| AC | 描述 |
|:---|:-----|
| AC1 | 三策略: LWW (Last-Write-Wins) / Merge / Additive |
| AC2 | 并发 10 agent × 100 writes 无数据丢失 |
| AC3 | 冲突日志可查询 |
| AC4 | Supersession_of 链完整 |

---

## 版本里程碑

| 版本 | 内容 | 预计 | 优先级 |
|:-----|:------|:-----|:-------|
| **v5.15.0** | 认证 + 速率限制 | 2026-08-11 | 🟢 **下一阶段** |
| v5.16.0 | Dashboard MVP + Benchmark | 2026-08-25 | 🟡 |
| v5.17.0 | 记忆可视化 + MCP PR | 2026-09-08 | 🟡 |
| **v6.0.0** | 投毒防御 + 写入消解 + A2A合规 | 2026-10-06 | 🔴 |
| v6.1.0 | 向量DB可插拔 + 时序相位 + 多模态 | 2026-11-01 | 🟢 |

---

## 依赖关系图

```
#5 认证  ←──── #6 速率限制 (依赖认证身份)
  │
  ├── #9 Benchmark (认证后才有安全公开的 API)
  ├── #10 MCP Registry (认证后才有安全 MCP Server)
  │
  └── #7 Dashboard (依赖认证)
        │
        └── #12 可视化 (UI 层面扩展 #7)

#1 投毒防御  ← 独立 (基于 BLAKE3 链)
#2 写入消解  ← 独立 (基于事务管理器)
#3 时序相位  ← 独立 (基于 τ 衰减)
#4 多模态    ← 独立 (基于向量存储)
#8 向量DB    ← 独立 (FAISS 抽象)
#11 A2A合规  ← 独立 (基于现有 A2A 适配器)
```

---

## Good First Issue 池

| 难度 | 方向 | 描述 |
|:-----|:-----|:------|
| 🟢入门 | #6 | 添加 Redis/内存后备的速率限制器 |
| 🟢入门 | #9 | 在 benchmark 中添加 Mem0 基线 |
| 🟢入门 | #10 | 提交 MCP Registry PR (仅文档) |
| 🟡中级 | #8 | 提取 VectorDB 抽象接口 |
| 🟡中级 | #3 | 实现 RoMem 相位旋转公式 |
| 🔴高级 | #1 | 语义矛盾检测模型 |
| 🔴高级 | #2 | MVCC 写入消解引擎 |
