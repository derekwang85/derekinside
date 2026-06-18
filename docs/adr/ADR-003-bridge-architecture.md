# ADR-003: DereInside Bridge Architecture

**Status**: Accepted (2026-06-19)
**Context**: Phase 3 — adding external API access (REST + MCP) and per-agent isolation to derekinside.
**Decisions**:

## 1. Protocol Choice

Two protocols supported:

| Protocol | Transport | Client | Use Case |
|----------|-----------|--------|----------|
| HTTP (REST) | TCP | Any HTTP client (curl, browser, apps) | Human and programmatic access |
| MCP (JSON-RPC 2.0) | stdio | Claude Desktop, MCP-compatible agents | Agent-native access |

**Rationale**: HTTP is universal; MCP is the emerging standard for AI agent tool access. Supporting both covers all access patterns without lock-in.

## 2. Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| HTTP Framework | FastAPI | Modern, async, auto-docs, high adoption |
| HTTP Server | uvicorn | Standard ASGI server, paired with FastAPI |
| MCP Protocol | Custom stdio | JSON-RPC 2.0 over stdin/stdout, no extra deps |
| Auth | Shared token (HMAC compare) | Simple, constant-time, env var fallback |

## 3. Agent Isolation Design

Each agent gets its own wing namespace:

```
Agents Table (agents)
  agent_id  →  wing = "agent-{id}"
  wings     →  "agent-{id}/memory" (rooms/pages/chunks)

HTTP Headers
  X-Agent-ID → scopes search to that agent's wing
  X-DEREINSIDE-TOKEN → auth token
```

**Why not table-level isolation**: The existing wing/room/pages/chunks hierarchy already provides natural namespacing. No schema changes needed.

## 4. MCP Tool Mapping

| Tool | REST Endpoint | Description |
|------|---------------|-------------|
| derekinside_search | POST /api/v1/search | Semantic search |
| derekinside_status | GET /api/v1/status | System health |
| derekinside_wings | GET /api/v1/wings | List wings |
| derekinside_graph_stats | GET /api/v1/graph/stats | Graph statistics |
| derekinside_graph_entity | GET /api/v1/graph/entity/{name} | Entity lookup |
| derekinside_wake | POST /api/v1/wake | Session context |

## 5. Deviations from gbrain bridge

| Aspect | gbrain | derekinside |
|--------|--------|-------------|
| Protocol | HTTP only | HTTP + MCP |
| Auth | None | Optional token-based |
| Endpoints | Single query endpoint | 6+ endpoints + 6 MCP tools |
| Agent isolation | None | X-Agent-ID namespacing |
| Documentation | None | Auto-generated FastAPI docs at /docs |

**Consequences**: MCP enables direct integration with Claude Desktop and other MCP clients. REST enables scripting and human access. Both share the same backend logic through the search/storage layers.
