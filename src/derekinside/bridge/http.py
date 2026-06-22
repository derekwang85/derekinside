"""
derekinside — HTTP (FastAPI) bridge.

Provides REST API for all derekinside operations.
Supports per-agent isolation via X-Agent-ID header.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from derekinside.bridge.auth import Auth
from derekinside.storage.pgvector import VectorStore
from derekinside.storage.graph import KnowledgeGraph
from derekinside.storage.subgraph import build_subgraph
from derekinside.indexer.embedder import Embedder
from derekinside.indexer.entity import EntityExtractor
from derekinside.search.hybrid import HybridSearch, SearchRequest

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU cache for query embeddings keyed by query text."""

    def __init__(self, maxsize: int = 256):
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, query: str) -> Optional[list[float]]:
        if query in self._cache:
            self._cache.move_to_end(query)
            self._hits += 1
            return self._cache[query]
        self._misses += 1
        return None

    def put(self, query: str, embedding: list[float]) -> None:
        self._cache[query] = embedding
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0,
        }


def create_app(
    store: VectorStore,
    embedder: Embedder,
    auth: Optional[Auth] = None,
    kg: Optional[KnowledgeGraph] = None,
    extractor: Optional[EntityExtractor] = None,
) -> Any:
    """Create and configure the FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI required: pip install 'derekinside[http]'")

    app = FastAPI(title="DereInside", version="0.1.0")

    # Embedding cache shared across requests
    embed_cache = EmbeddingCache(maxsize=256)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    searcher = HybridSearch(store=store)

    if auth is None:
        auth = Auth()

    def _check_auth(req: Request) -> None:
        if auth and auth.enabled:
            token = req.headers.get(auth._config.header, "")
            if not auth.check(token):
                raise HTTPException(status_code=401, detail="Unauthorized")

    def _get_agent_wing(req: Request) -> Optional[str]:
        agent_id = req.headers.get("X-Agent-ID", "")
        if agent_id and kg:
            from derekinside.bridge.agent_store import AgentStore

            as_ = AgentStore(store)
            info = as_.get_agent(agent_id)
            if info:
                return info.wing
        return None

    # ── Endpoints ─────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/v1/status")
    async def api_status(request: Request):
        _check_auth(request)
        stats = store.stats()
        result = {
            "wings": stats["wings"],
            "rooms": stats["rooms"],
            "pages": stats["pages"],
            "chunks": stats["chunks"],
            "embedded_chunks": stats["embedded_chunks"],
            "version": "0.1.0",
        }
        if kg:
            try:
                gs = kg.stats()
                result["graph"] = {
                    "entities": gs["entities"],
                    "relations": gs["relations"],
                    "links": gs["entity_chunk_links"],
                }
            except Exception:
                pass
        return result

    @app.post("/api/v1/search")
    async def api_search(request: Request):
        _check_auth(request)
        body = await request.json()
        query = body.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="query required")

        top_k = body.get("top_k", 20)
        wing = body.get("wing") or _get_agent_wing(request)
        room = body.get("room")
        use_recent = body.get("use_recent", False)
        before = body.get("before")
        after = body.get("after")

        t0 = time.time()

        # Embedding cache lookup
        q = query.strip().lower()
        cached = embed_cache.get(q)
        if cached:
            query_embedding = cached
            cache_hit = True
        else:
            query_embedding = embedder.embed(query)
            embed_cache.put(q, query_embedding)
            cache_hit = False

        req = SearchRequest(
            query=query,
            embedding=query_embedding,
            top_k=top_k,
            wing=wing,
            room=room,
            temporal_boost=use_recent,
            before=before,
            after=after,
        )
        resp = searcher.search(req)

        elapsed = time.time() - t0

        return {
            "query": query,
            "total": resp.total,
            "timing_ms": round(elapsed * 1000, 1),
            "cache_hit": cache_hit,
            "cache_stats": embed_cache.stats(),
            "results": [r.to_dict() for r in resp.results],
        }

    @app.get("/api/v1/graph/stats")
    async def api_graph_stats(request: Request):
        _check_auth(request)
        if not kg:
            raise HTTPException(status_code=404, detail="Knowledge graph not enabled")
        return kg.stats()

    @app.get("/api/v1/graph/entity/{name}")
    async def api_graph_entity(name: str, request: Request):
        _check_auth(request)
        if not kg:
            raise HTTPException(status_code=404, detail="Knowledge graph not enabled")
        entity = kg.get_entity_by_name(name)
        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
        relations = kg.get_relations_for_entity(entity.id)
        chunk_ids = kg.get_chunks_for_entity(entity.id, limit=20)
        return {
            "entity": entity.to_dict(),
            "relations": [r.to_dict() for r in relations],
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunk_ids),
        }

    @app.get("/api/v1/wings")
    async def api_wings(request: Request):
        _check_auth(request)
        return [
            {"name": w.name, "rooms": w.room_count, "pages": w.page_count}
            for w in store.list_wings()
        ]

    @app.post("/api/v1/wake")
    async def api_wake(request: Request):
        _check_auth(request)
        try:
            body = await request.json()
        except Exception:
            body = {}
        hours = body.get("hours", 24)
        query_embedding = embedder.embed("recent changes updates modifications")
        req = SearchRequest(
            query="wake context",
            embedding=query_embedding,
            top_k=10,
            wing=body.get("wing") or _get_agent_wing(request),
            temporal_boost=True,
            recent_days=hours // 24 + 1,
        )
        resp = searcher.search(req)
        return {
            "context": [
                {
                    "wing": r.wing_name,
                    "room": r.room_name,
                    "source": r.source_path or r.title or r.slug,
                    "preview": r.chunk_text[:200],
                }
                for r in resp.results
            ],
            "total": resp.total,
        }

    @app.get("/")
    async def frontend():
        from fastapi.responses import HTMLResponse
        from pathlib import Path
        html = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
        if html.exists():
            return HTMLResponse(html.read_text())
        return HTMLResponse("<h1>DereInside</h1><p>Frontend not found</p>")

    @app.get("/api/v1/entities")
    async def list_entities(request: Request, q: str = "", limit: int = 100):
        _check_auth(request)
        if not kg:
            raise HTTPException(status_code=404, detail="Knowledge graph not enabled")
        if q:
            entities = kg.search_entities(q, limit=limit)
        else:
            entities = kg.list_entities(limit=limit)
        result = []
        for e in entities:
            result.append({
                "id": e.id,
                "name": e.name,
                "entity_type": e.entity_type,
                "chunks": len(kg.get_chunks_for_entity(e.id, limit=1000)),
            })
        return {"entities": result, "total": len(result)}

    @app.get("/api/v1/graph/subgraph")
    async def api_subgraph(request: Request):
        _check_auth(request)
        if not kg:
            return JSONResponse({"error": "Knowledge graph disabled"}, status_code=400)
        entity = request.query_params.get("entity", "")
        depth = int(request.query_params.get("depth", 2))
        if not entity:
            return JSONResponse({"error": "Missing 'entity' param"}, status_code=400)
        sg = build_subgraph(kg, entity, max_depth=depth)
        if sg is None:
            matches = kg.search_entities(entity, limit=5)
            return JSONResponse({
                "error": f"Entity '{entity}' not found",
                "suggestions": [{"name": m.name, "type": m.entity_type} for m in matches],
            }, status_code=404)
        return JSONResponse(sg.to_dict())


    @app.post("/api/v1/mine")
    async def api_mine(request: Request):
        """Ingest a file's content into derekinside."""
        _check_auth(request)
        body = await request.json()

        file_path = body.get("path", "")
        file_content = body.get("content", "")
        wing = body.get("wing", "obsidian")
        room = body.get("room", "notes")

        if not file_content:
            raise HTTPException(status_code=400, detail="content required")

        from derekinside.indexer.chunker import chunk_text, detect_strategy
        from pathlib import Path as PPath

        strategy = detect_strategy(file_path) if file_path else "markdown"
        chunks = chunk_text(file_content, strategy)

        if not chunks:
            return {"status": "skipped", "reason": "no chunks"}

        wing_id = store.get_or_create_wing(wing, "Obsidian vault")
        room_id = store.get_or_create_room(wing_id, room, "")

        page_id = store.insert_page(
            room_id=room_id,
            slug=file_path,
            title=PPath(file_path).stem if file_path else "note",
            source_path=file_path,
            source_kind="file",
            page_kind=strategy,
        )

        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts)

        batch_data = []
        for ci, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            batch_data.append({
                "page_id": page_id,
                "chunk_index": ci,
                "chunk_text": chunk.text,
                "token_count": chunk.token_count,
                "embedding": emb,
            })

        store.insert_chunks_batch(batch_data)

        return {
            "status": "ok",
            "file": file_path,
            "wing": wing,
            "room": room,
            "chunks": len(chunks),
        }

    @app.get("/api/v1/cache/stats")
    async def cache_stats(request: Request):
        _check_auth(request)
        return embed_cache.stats()

    return app


def serve_http(
    store: VectorStore,
    embedder: Embedder,
    auth: Optional[Auth] = None,
    kg: Optional[KnowledgeGraph] = None,
    extractor: Optional[EntityExtractor] = None,
    host: str = "0.0.0.0",
    port: int = 18890,
) -> None:
    """Start the HTTP server."""
    if not HAS_FASTAPI:
        print("❌ FastAPI not installed. Run: pip install 'derekinside[http]'")
        return

    app = create_app(store, embedder, auth, kg, extractor)
    print(f"🌐 HTTP bridge starting on http://{host}:{port}")
    print(f"   Documentation: http://{host}:{port}/docs")
    print(f"   Health: http://{host}:{port}/health")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=300,
    )
    # Note: uvicorn timeout_keep_alive=300 gives ollama 5 min for slow embedding
