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
