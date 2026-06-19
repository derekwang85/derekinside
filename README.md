# DereInside 🧠🔍

> **从内部给予知识 — 名字源自拉丁语 `dare`（给予）→ `dere`，合 `inside` 为 DereInside**

DereInside is a local-first AI knowledge system. It ingests code, docs, conversations, and designs — then surfaces the right context when you need it.

Born from lessons learned running [gbrain](https://github.com/derekwang85/dolphin-knowledge-base) in production for 6 months across 265+ pages and 15,000+ chunks, DereInside is a complete rewrite that takes the best ideas from:

| Project | Stars | What we took |
|---------|-------|-------------|
| [mempalace](https://github.com/mempalace/mempalace) | 55k★ | Hierarchical wings/rooms/drawers, verbatim storage, LLM rerank |
| [mem0](https://github.com/mem0ai/mem0) | 58k★ | Adaptive memory updates, dedup+merge |
| [graphiti](https://github.com/getzep/graphiti) | 27k★ | Real-time knowledge graphs, entity extraction |
| [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) | 3.7k★ | PageRank propagation over entity graph |
| [iwe](https://github.com/Aiixu/iwe) | 1k★ | Markdown-native memory that humans can read |

## Quick start

```bash
pip install derekinside

# Ingest a project
derekinside mine ~/TradeOMS --wing=tradeoms

# Search
derekinside search "KYC approval flow"

# Wake up (load context for a new session)
derekinside wake
```

## Hardware Requirements

### Benchmarks (实测, Intel Xeon 8352S 32核 / 62GB RAM / CPU-only ollama)

| Operation | Latency | Bottleneck |
|:----------|:-------:|:-----------|
| Embedding (bge-m3, 300字) | **2.3s** | 🔴 CPU (单条) |
| Embedding (batch 16条) | **~10s** (0.6s/条) | 🟡 CPU (批量有 4x 加速) |
| Hybrid search (不含embed) | **~8ms** | 🟢 DB |
| Graph propagation | **~2ms** | 🟢 内存 |
| LLM rerank (qwen2.5-coder:1.5b) | **~3-5s** | 🟡 CPU |
| LLM rerank (qwen2.5-coder:7b) | **~30-60s** | 🔴 CPU (不推荐实时) |
| Regex entity extraction | **~5ms/chunk** | 🟢 CPU |
| PostgreSQL stats | **~2ms** | 🟢 DB |
| HTTP bridge (cold) | **<0.5s** | 🟢 |

### Recommended Configurations

| Tier | Use Case | CPU | RAM | GPU | Disk | Estimated QPS |
|:-----|:---------|:---:|:---:|:---:|:----:|:-------------:|
| **A — Personal** | CLI only, single user | 4 cores | 8 GB | None | 50 GB SSD | ~0.4 |
| **B — Team** | HTTP + MCP, 3-5 agents | 8 cores | 32 GB | T4 / RTX 3060 12GB | 200 GB NVMe | ~50 |
| **C — Production** | Multi-agent, enterprise | 16 cores | 64 GB | A10 / 2xT4 | 1 TB NVMe | ~200 |

**Tier A** runs entirely on CPU. Embedding takes ~2.3s/query — suitable for CLI usage.
**Tier B** adds a GPU for sub-100ms embedding, enabling real-time HTTP/MCP access.
**Tier C** adds connection pooling and horizontal scaling for concurrent multi-agent workloads.

### Storage Scaling

| Knowledge | Chunks | DB Size | Disk Needed |
|:----------|:------:|:-------:|:-----------:|
| 1,000 files | ~5,000 | ~100 MB | Any |
| 100,000 files | ~500,000 | ~10 GB | 50 GB SSD |
| 1,000,000 files | ~5,000,000 | ~100 GB | 200 GB NVMe |
| 10,000,000 files | ~50,000,000 | ~1 TB | SSD array |

**Formula:** 1 file ≈ 5 chunks ≈ 2 MB storage (with pgvector index).

### Known Constraints

- **Embedding is the bottleneck:** All search operations pass through ollama embedding. CPU-only gives ~0.4 QPS. A GPU (T4 or better) pushes this to 20-50 QPS.
- **Embedding cache helps:** Repeated queries skip ollama entirely (4.6ms vs 3322ms — **720x speedup**). Built-in LRU cache (maxsize=256) in the HTTP bridge.
- **LLM rerank is slow on CPU:** The 7B model takes 30-60s per prompt. Use the 1.5B model for near-real-time or disable rerank for sub-100ms search.
- **Connection pool recommended:** For concurrent access, use `pool_min/pool_max` (default 1/5) in production.

## Architecture

```
derekinside/
├── cli.py              # CLI entry point
├── bridge/
│   ├── http.py         # REST API
│   └── mcp.py          # MCP server for agent-native access
├── indexer/
│   ├── chunker.py      # Smart chunking
│   ├── embedder.py     # Embedding (bge-m3, pluggable)
│   └── entity.py       # Entity extraction → knowledge graph
├── storage/
│   ├── pgvector.py     # Vector storage (PostgreSQL + pgvector)
│   ├── graph.py        # Knowledge graph (entity/relation tables)
│   └── timeline.py     # Temporal metadata
├── search/
│   ├── hybrid.py       # Vector + keyword + temporal
│   ├── reranker.py     # LLM rerank on top-k results
│   └── propagation.py  # Graph propagation
└── sync/
    ├── git.py          # Git-based knowledge sync
    └── filesystem.py   # Local directory watching
```

## Roadmap

- **Phase 1** — Hierarchical indexes + LLM rerank + temporal weighting
- **Phase 2** — Knowledge graph + entity extraction + PageRank propagation
- **Phase 3** — MCP server + per-agent isolation

## License

MIT
