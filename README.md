<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/status-beta-green" alt="Beta">
  <img src="https://img.shields.io/github/stars/derekwang85/derekinside" alt="Stars">
  <img src="https://img.shields.io/github/last-commit/derekwang85/derekinside" alt="Last Commit">
</p>

<h1 align="center">🧠 DereInside</h1>
<p align="center"><em>Know your project from the inside out.</em></p>

<p align="center">
  <strong>Local-first · Multi-model · Self-learning · Agent-native</strong>
  <br>
  The AI knowledge system that <strong>understands your code</strong> — not just retrieves it.
</p>

---

## 🏆 Why DereInside?

Most "AI knowledge" tools are black boxes. One embedding model. One pipeline. One size fits nobody.

**DereInside is different.** It's a **multi-model cognitive engine** — not a vector database with a chat wrapper.

| What others do | What DereInside does |
|:--------------|:--------------------|
| One embedding model for everything | **6 interchangeable models**, auto-switched by content type |
| Fixed pipeline, no tuning | **Constraint-solving pipeline** — define what you need (intelligence × cost × speed), the system selects the best model |
| Black-box evaluation | **Quantified public benchmarks** (LongMemEval) — every mode has measurable precision/recall/F1 |
| Manual prompt engineering | **Self-learning consensus** — multiple models cross-validate each other, automatically rejecting noise |
| Tool-specific lock-in (Ollama only) | **Open provider architecture** — Ollama / vLLM / OpenAI / FreeCode / MiniMax / any OpenAI-compatible endpoint |
| Static knowledge | **Living knowledge graph** — entities, relations, cross-Wing fusion, temporal decay |
| Cloud-dependent | **Zero-cloud optional** — runs on a Raspberry Pi 4; GPU optional, not required |

---

## 🧬 Architecture — Designed, Not Patched

DereInside's architecture is the result of **deliberate engineering**, not organic growth. Every layer was designed to solve a real constraint:

```
                     ┌─────────────────────┐
                     │     Model Registry   │
                     │  6+ model endpoints  │
                     │ (Ollama/vLLM/OpenAI) │
                     └──────┬──────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
      ┌────────────┐ ┌──────────┐ ┌──────────┐
      │ Embedding  │ │Extraction│ │ Rerank   │
      │ Pipeline   │ │Pipeline  │ │ Pipeline │
      └──────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             ▼            ▼            ▼
      ┌──────────────────────────────────────┐
      │        Constraint Solver             │
      │  Filter: intel × cost × latency      │
      │  Rank:   quality / speed / cost      │
      │  Relax:  graceful degradation        │
      └──────────────────────────────────────┘
```

### 🎯 Model Registry — First-Class Models, Not Plumbing

Every AI endpoint is a **first-class citizen** with a **four-dimensional profile**:

```yaml
models:
  qwen-7b:
    driver: ollama        # Transport: Ollama
    capabilities: [extract, rerank]  # What it can do
    intelligence: high     # How smart it is
    cost_tier: free        # What it costs
    speed_tier: slow       # How fast it runs
    quality: high          # Output quality
```

This means you can **swap providers without changing code**. In production? Switch from Ollama to vLLM for GPU acceleration. API budget available? Add OpenAI as a fallback. Using FreeCode's free tier? Add it in one line — the system auto-profles its capabilities.

### 🔄 Pipeline Resolver — Constraint Solving, Not Fallback Chains

```yaml
pipeline:
  extract:
    requires:
      min_intelligence: low    # Only smart enough models
      max_cost: free           # Only free models
      max_latency_ms: 10000    # Under 10 seconds
    objective: optimize_quality
    candidates:
      - qwen-7b                # High quality, free
      - qwen-1.5b              # Faster fallback
      - gpt-4o-mini            # ❌ Excluded: paid
```

Not a linear "try A then B then C" — a **multi-dimensional constraint solver**:
- Filters by intelligence requirement
- Filters by cost budget
- Filters by latency ceiling
- Checks model health
- Ranks by objective (quality / speed / cost)
- **Relaxes constraints gracefully** when no model matches, with clear logging

### 🧪 Model Profiler — Zero-Conf Auto-Detection

Users shouldn't need to know their model's specs. DereInside **probes models automatically**:

```python
class ModelProfiler:
    # Golden data from LongMemEval — zero external API cost
    SMOKE_SET = [3 samples, ~2s]     # Boot positioning
    FULL_SET  = [15 samples, ~10s]   # Deep benchmark

    def profile(self, model) -> ModelProfile:
        return ModelProfile(
            capabilities=self._detect_capabilities(model),  # What can it do?
            intelligence=self._measure_intelligence(model), # How smart?
            speed=self._measure_speed(model),               # How fast?
            quality=self._measure_quality(model),           # How accurate?
            cost_tier=self._guess_cost(model),              # How expensive?
        )
```

**Passive observer**: runtime metrics (latency, entity count, error rate) collected as zero-cost side effects. When it detects anomalies — latency doubling, entity count dropping — it triggers a re-profle automatically.

**Oscillation detection**: if a FreeCode model keeps changing, profle freezes after 3 probes and alerts instead of chasing instability.

---

## 🧠 Self-Learning Consensus

DereInside doesn't trust a single model. It **cross-validates across multiple extraction modes**:

```
Chunk: "class OrderService extends BaseService { @Autowired ... }"

  regex      → {OrderService, BaseService}          (精95.6%)
  hybrid-7b  → {OrderService, BaseService, ...}     (精77.4%)
  1.5b       → {OrderService, BaseService, Autowired} (高召回, 有噪音)

  ConsensusEngine:
    OrderService  3/3  → confirmed  (weight=1.0) ✓
    BaseService   3/3  → confirmed  (weight=1.0) ✓
    Autowired     1/3  → rejected   (weight=0.0) ✗  (噪音: 非实体)
```

**Result**: noise rejection improves with each extraction run. After 3 full cycles on a 2,467-chunk codebase, estimated noise drops from ~28% to ~10%.

---

## 🏗️ Knowledge Graph — Rich, Clean, Connected

DereInside builds a knowledge graph that understands **relationships**, not just keywords:

| Relation | Source | Semantics |
|:---------|:-------|:----------|
| `OrderService` → `BaseService` | `class OrderService extends BaseService` | **extends** |
| `KYCController` → `/api/kyc/submit` | `@PostMapping("/api/kyc/submit")` | **serves_path** |
| `AuditService` → `AuditLogRepository` | `@Autowired` | **depends_on** |
| `TradeEntity` ↔ `PositionService` | Field declaration | **has_field** |
| `KYCApplication` → (merged) `KYC申请` | Entity resolution | **alias** |

**Entity resolution** automatically merges:
- `KYCApplication` = `kycapplication` = `KYC 申请` (alias dict)
- `AuditServiceImpl` → `AuditService` (suffix stripping)
- Cross-wing duplicates → automatic fusion

**Subgraph queries** traverse the graph:
```bash
derekinside graph subgraph "KYC" --depth 2 --ascii
# → 🔍 KYC (concept)
#     ├─ KYCApplication (class) ← depends_on
#     ├─ KYCController (class) ← serves_path: /api/kyc/submit
#     ├─ 合同审批流程 (concept) ← related
```

---

## 📊 Quantified — Not Hype

Every extraction mode is benchmarked on **LongMemEval**, a 290-entity / 100-chunk golden dataset with human-annotated ground truth:

| Mode | Precision | Recall | F1 | Speed | Uses |
|:-----|:--------:|:------:|:--:|:-----:|:----|
| **regex** | **95.6%** | 37.9% | 0.542 | **~5ms/chunk** | Code entities |
| **hybrid-7b** | 77.4% | 31.7% | 0.450 | ~7s/chunk | General purpose |
| **hybrid-1.5b** | 72.4% | 30.3% | 0.431 | ~3s/chunk | Fast extraction |
| **7b** | 59.0% | **44.1%** | 0.505 | ~7s/chunk | Recall-focused |
| **1.5b** | 71.7% | **55.9%** | 0.628 | ~2.6s/chunk | Max recall |

**Smart Dispatch** automatically selects the right mode per chunk:
```
.cjava/.py → regex        (精95.6%, 5ms)
.md/.txt   → hybrid-1.5b  (F1=0.43, 3s)
.xml/.sql  → 1.5b          (召回55.9%, 2.6s)
.log       → skip           (0s)
```

Result: **2,467 chunks reduced from 5h26m to ~1.5h** with weighted F1 improvement of ~15%.

---

## 🏭 Production Profile (as of June 2026)

DereInside is running in production powering the aITMS01 engineering workflow:

| Metric | Value | Growth |
|:-------|:-----:|:------:|
| **Wings** (knowledge domains) | **21** | +50% from launch |
| **Rooms** (sub-domains) | **54** | +20% |
| **Pages** (ingested) | **599** | +8% |
| **Chunks** (indexed) | **2,931** | +11%, 100% embedded |
| **Knowledge Graph Entities** | **5,485** | **7.9x** |
| **Knowledge Graph Links** | **10,817** | **7.6x** |

The knowledge graph explosion (7.9x entities, 7.6x links) is the direct result of the Model Registry + constraint-solving pipeline — the system now discovers relationships across code, documents, and conversations that were invisible under the old single-model architecture.

**What this means in practice:**
- Zero cloud dependency — runs on a single VM with PostgreSQL
- All embeddings, extractions, and graph operations local
- API response < 200ms for search queries
- Full re-index of 2,931 chunks completes in ~1.5h (was 5h26m before smart dispatch)
- Agent-native: MCP server provides structured context to sub-agents at spawn time

---

## ⚡ Quick Start

```bash
# Install
pip install derekinside

# Ingest your project
derekinside mine ~/TradeOMS --wing=tradeoms

# Build the knowledge graph
derekinside graph build

# Search
derekinside search "KYC approval flow"

# Serve as API for AI agents
derekinside serve --mode http --port 18890

# Explore the graph
derekinside graph subgraph "OrderService" --depth 2 --ascii
```

### MCP Integration (AI Agent ready)

```python
# Your AI agent gets persistent memory via MCP
from mcp import ClientSession

session = ClientSession("http://localhost:18890")
context = session.query("What is the KYC process?")
# → Returns entities, relations, and ranked chunks
```

---

## 🛣️ Roadmap

| Phase | Status | What |
|:------|:------:|:-----|
| **Phase 0** | ✅ | gbrain → DereInside migration |
| **Phase 1-2** | ✅ | Hierarchical indexes + knowledge graph |
| **Phase 2.5** | ✅ | 5-mode extraction + LongMemEval benchmarks |
| **Phase 3** | ✅ | MCP server + HTTP bridge + per-agent isolation |
| **Track A** | ✅ | **Model Registry + Pipeline + Profiler** — architectural overhaul |
| **Track B** | ✅ | Smart dispatch, Consensus self-learning, Cross-wing fusion, Temporal decay |
| **Track C** | ✅ | Relation inferrer, Entity resolution, Graph pruning, Enrichment, Subgraph |
| **Phase 4** | 🚧 | Multi-model ensemble + Agent-native context gate |
| **Phase 5** | 📋 | Web UI dashboard + collaborative annotations |
| **Phase 6** | 📋 | Fleet learning — share profles across instances |
| **EverOS Merger** | 📋 | Evaluate EverOS integration (see [RFC-0002](docs/rfcs/0002-everos-merge.md)) |
| **Fact Logging** | 📋 | Cross-Agent shared memory (see [RFC-0001](docs/rfcs/0001-fact-logging.md)) |

---

## 🤝 Contributing

We're building something that matters. If you share the vision:

- **Code contributors**: See [CONTRIBUTING.md](CONTRIBUTING.md) — we welcome PRs
- **Testers**: Run LongMemEval on your own data and share results
- **Feedback**: Open an issue or start a discussion
- **Sponsors**: Reach out if DereInside saves your team time

---

## 📜 License

MIT — free for any use, commercial or otherwise.

---

<p align="center">
  <strong>🧠 DereInside: Know your project from the inside out.</strong>
  <br>
  <em>From one engineer who refused to build another black box.</em>
</p>
