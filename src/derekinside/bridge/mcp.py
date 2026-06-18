"""
derekinside — MCP (Model Context Protocol) server.

Exposes derekinside as MCP tools over stdio transport.
Compatible with Claude Desktop and other MCP clients.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from typing import Any

from derekinside.config import load_config
from derekinside.storage.pgvector import VectorStore
from derekinside.indexer.embedder import Embedder
from derekinside.search.hybrid import HybridSearch, SearchRequest
from derekinside.search.propagation import GraphPropagator

logger = logging.getLogger(__name__)

# ── JSON-RPC helpers ──────────────────────────────────────────


def rpc_error(id: Any, code: int, message: str, data: Any = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message},
        **({"data": {"details": str(data)}} if data else {}),
    }


def rpc_result(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


# ── Tool definitions ──────────────────────────────────────────

_TOOLS = [
    {
        "name": "derekinside_search",
        "description": "Semantic search across indexed knowledge. Returns relevant chunks with scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Number of results",
                    "default": 10,
                },
                "wing": {"type": "string", "description": "Filter by wing (optional)"},
                "room": {"type": "string", "description": "Filter by room (optional)"},
                "use_kg": {
                    "type": "boolean",
                    "description": "Enable graph propagation",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "derekinside_status",
        "description": "Show system health and index stats (wings, rooms, pages, chunks, graph)",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "derekinside_wings",
        "description": "List all wings with room/page counts",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "derekinside_graph_stats",
        "description": "Knowledge graph statistics (entities, relations, links, type breakdown)",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "derekinside_graph_entity",
        "description": "Look up an entity in the knowledge graph by name",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Entity name to look up"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "derekinside_wake",
        "description": "Get recent context for session initialization",
        "inputSchema": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": "How many hours back",
                    "default": 24,
                },
                "wing": {"type": "string", "description": "Filter by wing"},
            },
        },
    },
]


class MCPServer:
    """MCP protocol server over stdio transport."""

    def __init__(self, config_path: str = ""):
        self._cfg = load_config()
        self._store = VectorStore(
            dsn=self._cfg.database.dsn,
            schema=self._cfg.database.schema,
        )
        self._store.connect()
        self._store.ensure_schema()

        self._embedder = Embedder(
            url=self._cfg.embedding.url,
            model=self._cfg.embedding.model,
            dimensions=self._cfg.embedding.dimensions,
        )

        self._searcher = HybridSearch(store=self._store, config=self._cfg.storage)

        self._kg = None
        self._propagator = None
        if self._cfg.knowledge_graph.enabled:
            from derekinside.storage.graph import KnowledgeGraph as KG

            self._kg = KG(self._store)
            self._kg.ensure_schema()
            self._propagator = GraphPropagator(self._kg, enabled=True)

    def run(self) -> None:
        """Read JSON-RPC messages from stdin, write responses to stdout."""
        # Send server info
        self._send(
            rpc_result(
                None,
                {
                    "protocol": "2025-03-26",
                    "name": "derekinside",
                    "version": "0.1.0",
                    "tools": _TOOLS,
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                    },
                },
            )
        )

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                self._handle(msg)
            except json.JSONDecodeError as e:
                self._send(rpc_error(None, -32700, "Parse error", str(e)))

    # ── Message handling ──────────────────────────────────────

    def _send(self, msg: dict) -> None:
        """Write a JSON-RPC message to stdout."""
        payload = json.dumps(msg, ensure_ascii=False)
        sys.stdout.write(f"{payload}\n")
        sys.stdout.flush()

    def _handle(self, msg: dict) -> None:
        msg_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {}) or {}

        if method == "ping":
            self._send(rpc_result(msg_id, {"status": "ok"}))

        elif method == "tools/list":
            self._send(rpc_result(msg_id, {"tools": _TOOLS}))

        elif method == "tools/call":
            self._handle_tool_call(
                msg_id, params.get("name", ""), params.get("arguments", {})
            )

        elif method == "resources/list":
            self._send(rpc_result(msg_id, {"resources": []}))

        elif method == "initialize":
            self._send(
                rpc_result(
                    msg_id,
                    {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"tools": {}, "resources": {}},
                        "serverInfo": {"name": "derekinside", "version": "0.1.0"},
                    },
                )
            )

        else:
            self._send(rpc_error(msg_id, -32601, f"Method not found: {method}"))

    def _handle_tool_call(self, msg_id: Any, name: str, args: dict) -> None:
        try:
            if name == "derekinside_search":
                self._handle_search(msg_id, args)
            elif name == "derekinside_status":
                self._handle_status(msg_id, args)
            elif name == "derekinside_wings":
                self._handle_wings(msg_id, args)
            elif name == "derekinside_graph_stats":
                self._handle_graph_stats(msg_id, args)
            elif name == "derekinside_graph_entity":
                self._handle_graph_entity(msg_id, args)
            elif name == "derekinside_wake":
                self._handle_wake(msg_id, args)
            else:
                self._send(rpc_error(msg_id, -32602, f"Unknown tool: {name}"))
        except Exception:
            self._send(
                rpc_error(msg_id, -32603, "Internal error", traceback.format_exc())
            )

    # ── Tool implementations ──────────────────────────────────

    def _handle_search(self, msg_id: Any, args: dict) -> None:
        query = args.get("query", "")
        top_k = args.get("top_k", 10)
        wing = args.get("wing")
        room = args.get("room")
        use_kg = args.get("use_kg", False)

        t0 = time.time()
        query_emb = self._embedder.embed(query)

        req = SearchRequest(
            query=query,
            embedding=query_emb,
            top_k=top_k,
            wing=wing,
            room=room,
        )
        resp = self._searcher.search(req)

        if use_kg and self._propagator:
            resp.results = self._propagator.propagate(query_emb, resp.results, top_k)

        elapsed = time.time() - t0

        self._send(
            rpc_result(
                msg_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": f"🔍 {resp.total} results ({elapsed:.2f}s)",
                        },
                        {
                            "type": "text",
                            "text": json.dumps(
                                [r.to_dict() for r in resp.results[:5]],
                                indent=2,
                                ensure_ascii=False,
                            ),
                        },
                    ],
                },
            )
        )

    def _handle_status(self, msg_id: Any, _args: dict) -> None:
        stats = self._store.stats()
        lines = [
            f"DereInside v0.1.0 — {stats['wings']} wings, {stats['rooms']} rooms",
            f"Pages: {stats['pages']}  |  Chunks: {stats['chunks']} ({stats['embedding_percent']}% embedded)",
        ]
        if self._kg:
            try:
                gs = self._kg.stats()
                lines.append(
                    f"Graph: {gs['entities']} entities, {gs['relations']} relations, {gs['entity_chunk_links']} links"
                )
            except Exception:
                pass
        self._send(
            rpc_result(
                msg_id,
                {
                    "content": [{"type": "text", "text": "\n".join(lines)}],
                },
            )
        )

    def _handle_wings(self, msg_id: Any, _args: dict) -> None:
        wings = self._store.list_wings()
        lines = [
            f"  🏛️  {w.name}/ — {w.room_count} rooms, {w.page_count} pages"
            for w in wings
        ]
        self._send(
            rpc_result(
                msg_id,
                {
                    "content": [{"type": "text", "text": "\n".join(lines)}],
                },
            )
        )

    def _handle_graph_stats(self, msg_id: Any, _args: dict) -> None:
        if not self._kg:
            self._send(rpc_error(msg_id, -32603, "Knowledge graph not enabled"))
            return
        gs = self._kg.stats()
        lines = [
            f"Entities: {gs['entities']}",
            f"Relations: {gs['relations']}",
            f"Links: {gs['entity_chunk_links']}",
        ]
        if gs["type_breakdown"]:
            lines.append("Types:")
            for etype, count in gs["type_breakdown"].items():
                lines.append(f"  {etype}: {count}")
        self._send(
            rpc_result(
                msg_id,
                {
                    "content": [{"type": "text", "text": "\n".join(lines)}],
                },
            )
        )

    def _handle_graph_entity(self, msg_id: Any, args: dict) -> None:
        if not self._kg:
            self._send(rpc_error(msg_id, -32603, "Knowledge graph not enabled"))
            return
        name = args.get("name", "")
        entity = self._kg.get_entity_by_name(name)
        if not entity:
            # Try search
            matches = self._kg.search_entities(name, limit=5)
            if matches:
                self._send(
                    rpc_result(
                        msg_id,
                        {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Entity '{name}' not found. Did you mean: "
                                    + ", ".join(m.name for m in matches),
                                }
                            ],
                        },
                    )
                )
            else:
                self._send(
                    rpc_result(
                        msg_id,
                        {
                            "content": [
                                {"type": "text", "text": f"Entity '{name}' not found."}
                            ],
                        },
                    )
                )
            return

        text = f"Entity: {entity.name} ({entity.entity_type})"
        relations = self._kg.get_relations_for_entity(entity.id)
        if relations:
            text += f"\nRelations ({len(relations)}):"
            for r in relations[:10]:
                target = self._kg.get_entity(r.target_entity_id)
                source = self._kg.get_entity(r.source_entity_id)
                tname = target.name if target else f"#{r.target_entity_id}"
                sname = source.name if source else f"#{r.source_entity_id}"
                text += f"\n  {sname} → {tname} ({r.relation_type})"

        self._send(
            rpc_result(
                msg_id,
                {
                    "content": [{"type": "text", "text": text}],
                },
            )
        )

    def _handle_wake(self, msg_id: Any, args: dict) -> None:
        hours = args.get("hours", 24)
        wing = args.get("wing")
        query_emb = self._embedder.embed("recent changes updates modifications")
        req = SearchRequest(
            query="wake context",
            embedding=query_emb,
            top_k=10,
            wing=wing,
            temporal_boost=True,
            recent_days=hours // 24 + 1,
        )
        resp = self._searcher.search(req)
        lines = [
            f"  🏛️  {r.wing_name}/{r.room_name} → {r.chunk_text[:150].replace(chr(10), ' ')}..."
            for r in resp.results
        ]
        self._send(
            rpc_result(
                msg_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Recent context ({resp.total} items):\n"
                            + "\n".join(lines),
                        }
                    ],
                },
            )
        )
