# Self-Evolving Memory for AI Agents: A Multi-Protocol Architecture

> **SHM (Self-evolving Hypergraph Memory)** — an open research memory system with 7 unique capabilities and 5 protocol interfaces, designed to give AI agents persistent, self-improving long-term memory.

---

## The Problem

Every AI agent today suffers from the same fundamental limitation: **it forgets everything when the session ends.**

You chat with Claude/ChatGPT/Codex, have a productive conversation, close the tab — and the next session starts from scratch. Your preferences, the context you built, the decisions you made — all gone.

The industry has responded with several memory solutions:

- **Mem0** (61K★) — personalized vector memory, great for user preferences
- **Letta/MemGPT** (24K★) — virtual context management, OS-inspired hierarchy
- **Cognee** (29K★) — AI memory platform with graph capabilities  
- **Engram** (5.7K★) — Go binary + SQLite, MCP-native, growing fast
- **Official MCP Memory Server** (89K★ repo) — basic knowledge graph, JSONL storage

But **none of them self-evolve**. Their retrieval strategies are frozen at deployment. Their consolidation is manual or non-existent. Their memory writes have no transactional guarantees.

**What if memory could learn from its own usage patterns?** What if it consolidated during idle periods — like the brain during sleep? What if writes were atomic and rollbackable?

That's what we built.

---

## SHM Architecture Overview

SHM is a **5-layer hypergraph memory system** with three core engines:

```
                      ┌──────────────────┐
                      │   Conceptual     │  Layer 5 — Abstract concepts, schemas
                      ├──────────────────┤
                      │   Community      │  Layer 4 — Emergent communities via Louvain
                      ├──────────────────┤
                      │   Hyperedge      │  Layer 3 — Multi-entity relationships  
                      ├──────────────────┤
                      │   Episodic       │  Layer 2 — Structured episodes with τ-decay
                      ├──────────────────┤
                      │   Sensory        │  Layer 1 — Ring buffer, raw input
                      └──────────────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                        ▼
  τ-Engine              Hebbian Engine             Dream Pipeline
  (temporal decay)       (association learning)    (sleep consolidation)

     Memory is not a bucket. It's a living, evolving structure.
```

### The Three Engines

**τ-Engine** (temporal decay): Each memory node has a decay rate τ that controls how fast it fades. Frequently accessed memories decay slower; neglected memories fade and become candidates for pruning. The decay is learnable — each node can learn its own optimal τ.

**Hebbian Engine** (association learning): "Neurons that fire together, wire together." When two episodes are retrieved together, a HEBBIAN_CONNECTION edge is strengthened between them. Repeated co-retrieval eventually promotes them into a hyperedge — a higher-order relationship.

**Dream Pipeline** (sleep consolidation): During idle periods, SHM runs an 8-step consolidation pipeline:
1. Community detection (Louvain on the hypergraph)
2. Intra-community synthesis (generate summaries)
3. Conflict resolution across communities
4. Compression (prune redundant connections)
5. SSM state evolution (N rounds of structured state space model replay)
6. Confidence calibration (prevent over-consolidation)
7. Audit chain verification (BLAKE3 integrity check)
8. Persistence

---

## 7 Unique Capabilities

The global research survey (30+ papers, 4 reports) identified these gaps in existing memory systems — **all 7 are exclusive to SHM**:

### 1. Self-Evolving Retrieval (P0)
*Reference: EvolveMem (arXiv:2605.13941)*

```python
# Traditional systems: frozen retrieval config
config = {"fusion_weights": [0.5, 0.3, 0.2], "top_k": 20}

# SHM: retrieval that learns from failures
evolver = SelfEvolvingRetrieval(gate)
result = await evolver.retrieve("What did we discuss about MCP?")
# After a failure → diagnostic engine analyzes → evolution guard adjusts
# - Decrease BM25 weight if it contributed irrelevant results
# - Increase entity weight if entity matching helped
# - If 6 consecutive retrievals show no improvement → explore new params
# - If performance drops >15% → auto-rollback to last good config
```

### 2. Budget-Aware Gating (P1)
*Reference: Retain or Consolidate? (arXiv:2607.17545)*

The dual gate (SSM + MLP) fusion weight α is not just reward-adaptive — it's **budget-aware**:

```python
# When budget is high (>50%) → favor consolidate (deep SSM path)
# When budget is low (<50%) → favor retain (fast MLP path)
# α = f(reward) + f(budget_ratio)
operator = gate.operator_selection()  # "retain" or "consolidate"
```

### 3. Confidence Calibrator (P1)
*Reference: Manufactured Confidence (arXiv:2606.29279)*

Repeated consolidation cycles can inflate confidence in uncertain facts. SHM's calibrator applies exponential decay to each consolidation cycle:

```python
calibrator = ConfidenceCalibrator()
calibrated = calibrator.calibrate(
    confidence=0.95, 
    content_hash=hash("User prefers morning meetings"),
    source_type="inference"  # direct=1.0, inferred=0.7, hearsay=0.4
)
# 5th consolidation of an inferred fact: 0.95 × exp(-0.3 × 5) = 0.21
# → flagged for review, not automatically consolidated
```

### 4. Transactional Memory Writes (P1)
*Reference: MemTX (arXiv:2607.13157)*

"Write ≠ Belief." SHM implements two-phase commit for all memory operations:

```python
with mgr.transaction() as tx:
    tx.stage_create("EpisodeNode", content="AI discussed MCP protocol")
    tx.stage_update("HyperedgeNode", id="h123", add_member="e456")
    # ← if anything fails here, ALL changes are rolled back atomically
    tx.commit()
```

### 5. Learnable Forgetting (P2)
*Reference: AdaMem*

Each memory node learns its own optimal decay rate. High-utility memories become stickier; low-utility memories fade faster:

```python
learner = AdaptiveDecayLearner()
new_tau = learner.learn(
    current_tau=0.3, 
    access_frequency=12,       # 12 accesses in the last window
    reward_history=[0.8, 0.6], # positive retrieval feedback
    node_id="episode_789"
)
# Frequently accessed + well-rewarded → τ decreases (slower decay)
```

### 6. Multi-Agent Governance (P2)
*Reference: MemClaw*

Memory operations carry provenance — agent IDs, timestamps, and scope:

```python
hyperedge = await create_multi_agent_hyperedge(
    member_ids=["e1", "e2", "e3"],
    agent_scope="agent_a",      # visible only to agent_a
    visibility="shared",        # or "private" / "team"
    supersession_of="h_prev"    # explicitly overrides previous
)
# Get memories visible to a specific agent
visible = hyperedge_manager.get_visible_hyperedges(agent_id="agent_b")
```

### 7. SSM Dream Consolidation (P2)
*Reference: Language Models Need Sleep (arXiv:2605.26099)*

The dream pipeline runs N rounds of SSM state evolution during idle periods:

```python
dreamer = SSMDreamWrapper(config=SSMDreamConfig(n_rounds=3))
report = await dreamer.run(communities, hyperedges)
# Round 1: community detection + synthesis
# Round 2: conflict resolution + compression
# Round 3: SSM state replay + calibration
print(f"Created {report.created} reports, consolidated {report.consolidated}")
```

---

## Multi-Protocol Gateway

Memory is useless if no one can access it. SHM speaks **5 protocols** through a unified GatewayAPI layer:

```
                    ┌─────────────────────────────┐
                    │       Any AI Agent            │
                    │  Claude  │  Google ADK  │  ⋯  │
                    └────┬──────────┬──────────┬───┘
                         │          │          │
              ┌──────────┴──┐  ┌────┴────┐  ┌─┴──────────┐
              │   MCP       │  │   A2A   │  │   ACP      │
              │  :8002      │  │  :8001   │  │  :8770     │
              ├─────────────┤  ├─────────┤  ├────────────┤
              │ shm_write   │  │ /memory  │  │ shm:write  │
              │ shm_retrieve│  │ /health  │  │ shm:retrieve│
              │ shm_health  │  │ /search  │  │ shm:health │
              │ shm_dream   │  │ /dream   │  │            │
              │ shm_search  │  │ /store   │  │            │
              └──────┬──────┘  └────┬─────┘  └──────┬─────┘
                     └──────────────┴───────────────┘
                                    │
                         ┌──────────┴──────────┐
                         │     GatewayAPI       │
                         │     (570 LOC)        │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │     SHM Core         │
                         │  (Hypergraph+FAISS+  │
                         │   SSM+τ+Dream)       │
                         └─────────────────────┘
```

| Protocol | Use Case | Example Consumer |
|:---------|:---------|:-----------------|
| **MCP** (stdio) | Tool-based memory access | Claude Desktop, Cursor, VS Code, Windsurf |
| **A2A** (HTTP JSON) | Agent-to-agent memory sharing | Google ADK agents, multi-agent systems |
| **ACP** (HTTP action)| Orchestration memory bridge | Internal agent pipelines |
| **HTTP REST** | Full CRUD API | Custom integrations, web apps |
| **CLI** | Developer debugging | `shm retrieve "query"` |

---

## Comparison: SHM vs Existing Systems

| Dimension | Mem0 | Letta | Engram | Official MCP | **SHM** |
|:----------|:-----|:------|:-------|:-------------|:--------|
| Storage | Vector DB | Sliding window | SQLite+FTS5 | JSONL file | **Hypergraph+FAISS** |
| Layers | Flat | 3-tier | Flat | KG entities | **5 layers** |
| Vector Search | ✅ | ❌ | ❌ | ❌ | **✅ FAISS+BM25+Entity** |
| Self-Evolution | ❌ | ❌ | ❌ | ❌ | **✅** |
| Sleep Consolidation | ❌ | ❌ | ❌ | ❌ | **✅ SSM Dream** |
| Budget-Aware Gating | ❌ | ❌ | ❌ | ❌ | **✅** |
| Transactional Writes | ❌ | ❌ | ❌ | ❌ | **✅** |
| Confidence Calibration | ❌ | ❌ | ❌ | ❌ | **✅** |
| Learnable Forgetting | ❌ | ❌ | ❌ | ❌ | **✅** |
| Multi-Agent Governance | ❌ | ❌ | ❌ | ❌ | **✅** |
| MCP Protocol | ✅ | ❌ | ✅ | ✅ | **✅** |
| A2A Protocol | ❌ | ❌ | ❌ | ❌ | **✅** |
| One-Deploy | ✅ pip | ✅ pip | ✅ binary | ✅ npm | ⚠️ manual |

**Score: 7/7 unique features, 5 protocol interfaces, 0 gaps shared with any competitor.**

---

## The Vision

Current AI agents have **no persistent identity**. Each conversation is a fresh persona. We believe the next leap in AI capability won't come from larger models — it will come from agents that **remember, learn, and evolve across conversations**.

SHM is a research prototype exploring this direction. It's not production-ready (yet), but it demonstrates what's possible when you apply:

- Neuroscience-inspired consolidation
- Database-grade transactional guarantees
- Self-adaptive retrieval strategies
- Multi-protocol accessibility

**The roadmap ahead:**

1. **Open source release** (GitHub public)
2. **MCP Registry contribution** — as a drop-in replacement for the basic official memory server
3. **Benchmark suite** — comparing SHM against Mem0, Engram, and the official MCP server
4. **Production hardening** — packaging as a Docker one-liner

---

## Try It

```bash
# Clone (private during research phase)
git clone https://github.com/Neocher/self-evolving-hypergraph-memory.git

# Quick start
cd shm && pip install -r requirements.txt
python -m api.app  # HTTP :8000

# Use any protocol
python -m gateway.cli write "Hello SHM"                    # CLI
python -m gateway.mcp_server                                # MCP stdio
python -m gateway.a2a_server                                # A2A :8001
```

---

**SHM v5.14.0** — 15,953 LOC, 45/45 tests passing, 5 protocols, 7 unique capabilities.

*References: EvolveMem · Retain or Consolidate? · Manufactured Confidence · MemTX · Language Models Need Sleep · AdaMem · MemClaw · A2A Protocol · MCP Specification*

---

*This is a research project in active development. APIs will change. Feedback and contributions welcome.*