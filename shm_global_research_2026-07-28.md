# SHM Global Research — Final Consolidated Report
**Date**: 2026-07-28 | **Direction**: 自进化 (Self-Evolving)

## Pipeline Status

| Pipeline | Status | Duration | Output |
|:---------|:-------|:---------|:-------|
| Track 1: Graph/Hypergraph | ✅ | 398s | 10 papers + 6 trends |
| Track 2: Cognitive Architecture | ✅ | 174s | 10 papers + convergence map |
| Track 3: Production Systems | ✅ | 232s | 10 papers + 6 projects |
| Direct Arxiv Search | ✅ | parallel | 25+ papers with arXiv IDs |
| D+F Container | ⏳ building | — | DaoCloud mirror pull |

## Key Findings — 自进化 (Self-Evolving) Focus

### 🔥 Top 3 Papers Most Relevant to SHM

| Rank | Paper | arXiv | Core Insight | SHM Gap |
|:----:|:------|:------|:------------|:--------|
| **1** | **EvolveMem** | 2605.13941 | 检索配置自演化：LLM诊断→配置调整→自动回滚→探索。LoCoMo +25.7% | **P0**: 固定评分+融合策略，缺少自演化 |
| **2** | **Language Models Need Sleep** | 2605.26099 | SSM块睡眠N轮离线巩固→清空KV cache。验证SSM睡眠巩固路线 | **同构验证**: Dream pipeline领先行业 |
| **3** | **Retain or Consolidate?** | 2607.17545 | 预算感知的保留vs整合门控选择 | **P1**: DualAdaptiveGate缺预算感知 |

### Track 1: Graph/Hypergraph Memory (10 papers)

Key papers for SHM:
- **AriGraph** (2407.04363): KG world model + episodic memory outperforms summarization
- **Oracle Agent Memory** (2607.13157): DB-native memory, 93.8% LongMemEval, 10.7x fewer tokens
- **RoMem** (2604.11544): Continuous phase rotation for temporal KG — τ-Hebbian对标
- **TMA-NM** (2606.24322): Memory poisoning defense for agent memory
- **LightRAG** (2410.05779): Graph-structured RAG, lightweight retrieval

**Trend**: Flat RAG → Graph-Structured Agent Memory, temporal awareness, hypergraph under-explored

### Track 2: Cognitive Architecture + Consolidation (10 papers)

- **MemGPT** (2310.08560, 47 cites): OS-inspired virtual context management
- **Reflexion** (2303.11366, 272 cites): Verbal RL with episodic memory
- **Zep**: Temporal KG memory — closest commercial competitor
- **Sleep consolidation** (Born 2005, 1862 cites): NREM/REM cycle role
- **GEM** (2017, 497 cites): Gradient episodic memory — SSM遗忘机制数学基础

**Trend**: 3 waves — Simple RAG → MemGPT(OS隐喻) → Self-evolving(2026)

### Track 3: Production Systems (10 papers + 6 projects)

- **MemTX**: Transactional belief commit — two-phase commit for agent memory
- **MemClaw**: Governed shared memory for multi-agent with scope/supersession/provenance/policy
- **WorldDB**: Vector-graph-of-worlds with ontology-aware write reconciliation
- **MRMS**: Multi-resolution memory substrate — validates SHM's 5-layer design
- **AdaMem**: Learning what to remember — learnable write gate replacing hard thresholds

**Open source landscape**:
- Letta: 23,991★ (stateful agents, no SSM/hypergraph/dream)
- Chroma: 28,892★ (vector search only)
- Mem0: popular memory layer (no graph structure)
- MS GraphRAG: industrial graph RAG (offline index)
- Graphiti (Zep): real-time KG (commercial closed)

## SHM Advantage Summary

**Unique advantages** (no complete open-source/publication match):
- SSM-driven memory evolution ✅ (only Language Models Need Sleep 2026)
- Dream pipeline (sleep consolidation) ✅ (only sleep paper 2026)
- Hypergraph structure ✅ (HyperMem 2026, just started)
- τ-Hebbian temporal decay ✅ (only RoMem 2026 for temporal KG)
- 5-layer memory architecture ✅ (MRMS 2026 validates multi-resolution)

**Critical gaps** (ordered by priority):
| Priority | Gap | Reference | Action Needed |
|:--------:|:----|:----------|:--------------|
| **P0** | 检索策略自演化 | EvolveMem | LLM诊断→配置调整→回滚保护 |
| **P1** | 事务性记忆 | MemTX | 原子写操作语义 |
| **P1** | 预算感知门控 | Retain or Consolidate? | α动态预算学习 |
| **P1** | 过巩固防护 | Manufactured Confidence | 置信度校准 |
| **P2** | 可学习遗忘 | AdaMem | τ硬衰减→自适应遗忘 |
| **P2** | 多Agent共享 | MemClaw | 共享记忆治理 |

## Files Generated

| File | Size | Content |
|:-----|:-----|:--------|
| `shm_global_research_2026-07-28.md` | 6.8K | Consolidated report (this) |
| `graph_hypergraph_memory_agents_report.md` | ~175 lines | Track 1 full report |
| `track2_cognitive_architecture_memory.md` | full | Track 2 full report |
| `track3_production_memory_systems_report.md` | ~31K, 600 lines | Track 3 full report |
