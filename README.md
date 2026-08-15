# SHM — Self-evolving Hypergraph Memory

**v5.40.0** | *8 unique capabilities · 5 protocol interfaces · 3 cognitive engines*

> Memory that learns, consolidates, and evolves — like the brain.
>
> Not just storage. **Evolution.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-ready-orange)](https://modelcontextprotocol.io)
[![A2A](https://img.shields.io/badge/A2A-ready-blueviolet)](https://github.com/google/A2A)
[![Tests](https://img.shields.io/badge/tests-730%2F731-green)]()

---

## 🔥 What is SHM?

SHM is an **open research memory system** for AI agents. It gives agents **persistent, self-improving long-term memory** that:

- **Self-evolves** its retrieval strategy from usage patterns
- **Consolidates** during idle periods (like sleep)
- **Guarantees** atomic, rollbackable writes
- **Calibrates** confidence to prevent over-consolidation
- **Governs** multi-agent memory with provenance + scoping

**5 protocols, 1 unified backend:**

```ascii
┌─────────────────────────────────────────────────────┐
│  MCP (:8002)  A2A (:8001)  ACP (:8770)  CLI  HTTP  │
└────────────────────┬────────────────────────────────┘
                     │
              GatewayAPI (570 LOC)
                     │
           ┌─────────┴─────────┐
           │    SHM Core        │
           │  Hypergraph+FAISS  │
           │  SSM+τ+Dream      │
           └───────────────────┘
```

---

## 🏗 Architecture

### 5 Memory Layers

```ascii
Layer 5: Conceptual    — Abstract concepts, schemas
Layer 4: Community     — Emergent clusters (Louvain)
Layer 3: Hyperedge     — Multi-entity relationships
Layer 2: Episodic      — Structured episodes (τ-decay)
Layer 1: Sensory       — Ring buffer (raw input)
```

### 3 Cognitive Engines

| Engine | Function | Analogy |
|:-------|:---------|:--------|
| **τ-Engine** | Learnable temporal decay for each memory node | Forgetting curve (Ebbinghaus) |
| **Hebbian Engine** | Strengthens associations on co-retrieval | "Fire together, wire together" |
| **Dream Pipeline** | 8-step sleep consolidation: community detection → SSM replay → calibration | Hippocampal replay |

---

## ✨ 8 Unique Capabilities

| # | Capability | Paper | SHM Only? |
|:-:|:-----------|:------|:----------|
| 1 | **Self-Evolving Retrieval** | EvolveMem (2605.13941) | ✅ |
| 2 | **Budget-Aware Gating** | Retain or Consolidate? (2607.17545) | ✅ |
| 3 | **Confidence Calibrator** | Manufactured Confidence (2606.29279) | ✅ |
| 4 | **Transactional Memory** | MemTX (2607.13157) | ✅ |
| 5 | **Learnable Forgetting** | AdaMem | ✅ |
| 6 | **Multi-Agent Governance** | MemClaw | ✅ |
| 7 | **SSM Dream Consolidation** | Language Models Need Sleep (2605.26099) | ✅ |
| 8 | **User-Profile** | Profile-Graph Memory (2606.06036) | ✅ |

**Benchmark: 0/8 shared with Mem0, Letta, Engram, or Zep.**

---

## 🚀 Quick Start

### Clone & Run

```bash
git clone https://github.com/Neocher/self-evolving-hypergraph-memory.git
cd self-evolving-hypergraph-memory

pip install -r requirements.txt
python -m api.app                # HTTP REST :8000
python -m gateway.mcp_server     # MCP stdio :8002
python -m gateway.a2a_server     # A2A HTTP :8001
python -m gateway.cli health     # CLI
```

### Docker

```bash
docker compose up -d  # Full stack
```

### Use via CLI

```bash
# Write
python -m gateway.cli write "Learned about MCP protocol today"

# Retrieve
python -m gateway.cli retrieve "MCP protocol" --top-k 5

# Health
python -m gateway.cli health

# Trigger dream consolidation
python -m gateway.cli dream
```

### Use via MCP (Claude Desktop)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "shm": {
      "command": "python3",
      "args": ["-m", "gateway.mcp_server"],
      "env": {"SHM_BASE_URL": "http://127.0.0.1:8000"}
    }
  }
}
```

### Use via Python

```python
from gateway.gateway_api import GatewayAPI

api = GatewayAPI(...)  # init from existing Services
await api.write_sensory("Hello SHM", source="user")
results = await api.retrieve("What did I learn today?")
```

---

## 📡 Protocol Comparison

| Protocol | Port | Use Case | Example Consumer |
|:---------|:-----|:---------|:-----------------|
| **HTTP REST** | 8000 | Full CRUD | Any HTTP client |
| **MCP stdio** | 8002 | Tool-based memory | Claude Desktop, Cursor, VS Code |
| **A2A JSON** | 8001 | Agent↔Agent sharing | Google ADK, multi-agent |
| **ACP** | 8770 | Orchestration bridge | Internal agent pipelines |
| **CLI** | — | Developer debugging | Terminal |

---

## 🚀 Installation

### 1. Build the GraphLite engine (required)

SHM's storage engine is [GraphLite](https://github.com/GraphLite-AI/GraphLite)
(Apache-2.0), an embedded graph database written in Rust. Its Python SDK is **not
published on PyPI** — clone and build it first:

> ⚠️ **v5.31.4+ 必须使用修复版 fork**（`Neocher/GraphLite`，含 UTF-8 lexer 修复
> `4452a96`）。上游 main 自 2026-01 停滞，其 lexer 在多字节 UTF-8 字符处 panic
> （`end byte index not a char boundary`）。SHM v5.31.4 起已去除 b64 透明编码
> （中文原生直写），若搭配旧引擎，中文 INSERT/CONTAINS/LIKE 将全部 PANIC。

```bash
git clone https://github.com/Neocher/GraphLite ~/GraphLite   # 修复版 fork（4452a96+）
cd ~/GraphLite && source ~/.cargo/env && cargo build --release -p graphlite-ffi
```

Then make the SDK discoverable (add to your shell profile):

```bash
export GRAPHLITE_BINDINGS=~/GraphLite/bindings/python
export GRAPHLITE_SDK=~/GraphLite/sdk-python/src
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt   # or: pip install -e .
```

### 3. Run

```bash
shm-server                     # starts API on :8000
# or
python3 run_server.py
```

### 4. Run as a systemd service (optional, recommended)

Two deployment modes — pick one:

```bash
# A. User-level (no root, desktop). Crash auto-restart + start at boot:
bash install.sh --user
systemctl --user status shm-server      # verify
journalctl --user -u shm-server -f      # logs

# B. System-level (root, multi-user server):
sudo bash install.sh
systemctl status shm
```

User mode defaults: `SHM_EMBEDDING__DEVICE=cuda` (GPU), GraphLite SDK at
`$HOME/GraphLite`, linger enabled. Override with env vars, e.g.:

```bash
SHM_EMBEDDING__DEVICE=cpu bash install.sh --user   # no-GPU machine
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -q
# 526/526 passed (1 skipped) — v5.28 含写队列卡死修复 8 用例；无 DEEPSEEK_API_KEY 环境为 525/526 (1 skipped)
```

---

## 📊 Comparison to Existing Systems

| Feature | Mem0 | Letta | Engram | Official MCP | **SHM** |
|:--------|:-----|:------|:-------|:-------------|:---|
| Storage | Vector DB | Window | SQLite | JSONL | **Hypergraph+FAISS** |
| Layers | Flat | 3-tier | Flat | KG | **5 layers** |
| Self-Evolution | ❌ | ❌ | ❌ | ❌ | **✅** |
| Sleep Consolidation | ❌ | ❌ | ❌ | ❌ | **✅ SSM Dream** |
| Budget Gating | ❌ | ❌ | ❌ | ❌ | **✅** |
| Transactional Writes | ❌ | ❌ | ❌ | ❌ | **✅** |
| Confidence Calibration | ❌ | ❌ | ❌ | ❌ | **✅** |
| Learnable Forgetting | ❌ | ❌ | ❌ | ❌ | **✅** |
| Multi-Agent Governance | ❌ | ❌ | ❌ | ❌ | **✅** |
| MCP Protocol | ✅ | ❌ | ✅ | ✅ | **✅** |
| A2A Protocol | ❌ | ❌ | ❌ | ❌ | **✅** |

---

## 🗺 Roadmap

- [x] **P0**: Self-Evolving Retrieval
- [x] **P1**: Budget Gating + Confidence Calibrator + Transactional Memory
- [x] **P2**: Learnable Forgetting + Multi-Agent Governance + SSM Dream
- [x] **GW1**: MCP + CLI + GatewayAPI
- [x] **GW2**: A2A + ACP
- [ ] **OSS**: Public release + MCP Registry + blog post
- [ ] **Prod**: Docker one-liner, pip install

---

## 📚 References

- EvolveMem: Self-Evolving Retrieval (arXiv:2605.13941)
- Retain or Consolidate? (arXiv:2607.17545)
- Manufactured Confidence (arXiv:2606.29279)
- MemTX: Transactional Belief Commit (arXiv:2607.13157)
- Language Models Need Sleep (arXiv:2605.26099)
- AdaMem: Learnable Forgetting
- MemClaw: Governed Shared Memory
- A2A Protocol (Google, 25K★)
- MCP Specification (Anthropic, 89K★)

---

## 📄 License

MIT

---

*Built with τ-Hebbian-Dream + 30+ research papers surveyed.*
