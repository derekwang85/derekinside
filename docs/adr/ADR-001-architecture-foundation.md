# ADR-001: DereInside Architecture Foundation

**Status**: Accepted (2026-06-19)
**Context**: From gbrain (local postgresql+pgvector bridge) to derekinside
**Decisions**:

## 1. Project Scope

DereInside is a local-first AI knowledge system. It ingests code, docs, conversations, and designs — then surfaces the right context for AI agents and humans.

**In scope**: Semantic search, knowledge graph, temporal context, MCP integration
**Out of scope**: Real-time collaboration, cloud sync, multi-tenant hosting

## 2. Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.10+ | gbrain was Python; agent ecosystem is Python |
| Storage | PostgreSQL 17 + pgvector | Proven in gbrain, ACID, pgvector native |
| Embedding | bge-m3 via Ollama | Already running, 1024d, CPU-friendly |
| CLI | click | Standard Python CLI framework |
| API | HTTP (FastAPI) + MCP | Dual: FastAPI for REST, MCP for agent-native |
| Config | YAML | Single config file, source-controllable |

## 3. Architecture Principles

1. **Pluggable storage** — The `storage` module defines a base interface. pgvector is default; others can be dropped in
2. **Verbatim storage** — Original text stored as-is, not summarized
3. **Hierarchical** — Wing → Room → Drawer, inspired by mempalace
4. **No lock-in** — No API keys required (local Ollama)
5. **Phase-gated** — Each phase stable before next begins

## 4. Deviations from gbrain

| Aspect | gbrain | derekinside |
|--------|--------|-------------|
| Storage | Implicit schema | Explicit migrations |
| Scope | Flat chunks | Wing/Room/Drawer |
| Search | Vector + keyword RRF | + temporal + rerank + graph |
| Access | HTTP only | HTTP + MCP |
| Config | Hardcoded paths | YAML config file |
| Knowledge | Implicit | Entity-relation graph |

**Consequences**: New project, not a gbrain fork. gbrain continues running until Phase 1 stable.
