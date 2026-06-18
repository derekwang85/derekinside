"""
derekinside — Know your project from the inside out.

CLI entry point — Phase 1 + Phase 2.
"""

from __future__ import annotations

import json as _json
import logging
import time
from pathlib import Path

import click

from derekinside.config import load_config
from derekinside.storage.pgvector import VectorStore
from derekinside.storage.graph import KnowledgeGraph
from derekinside.indexer.embedder import Embedder
from derekinside.indexer.chunker import chunk_file, detect_strategy
from derekinside.indexer.entity import EntityExtractor
from derekinside.search.hybrid import HybridSearch, SearchRequest
from derekinside.search.reranker import Reranker
from derekinside.search.propagation import GraphPropagator

# ── Logging ────────────────────────────────────────────────────


def setup_logging(level: str = "info", log_file: str = "") -> None:
    fmt = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
    handlers: list = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(path)))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
    )


# ── Shared Context ─────────────────────────────────────────────


class DereContext:
    """Shared state passed to CLI commands."""

    def __init__(self, config_path: str = ""):
        cfg_path = Path(config_path) if config_path else None
        self.cfg = load_config(cfg_path)
        self.store = VectorStore(
            dsn=self.cfg.database.dsn,
            schema=self.cfg.database.schema,
        )
        self.graph = KnowledgeGraph(self.store)
        self.embedder = Embedder(
            url=self.cfg.embedding.url,
            model=self.cfg.embedding.model,
            dimensions=self.cfg.embedding.dimensions,
        )
        self.extractor = EntityExtractor(
            url="http://localhost:11434/api/generate",
            model=self.cfg.knowledge_graph.extraction_model,
            enabled=self.cfg.knowledge_graph.enabled,
        )
        self.reranker = Reranker(
            url="http://localhost:11434/api/generate",
            model=self.cfg.search.rerank.model,
            enabled=self.cfg.search.rerank.enabled,
        )
        self.propagator = GraphPropagator(
            graph=self.graph,
            enabled=self.cfg.knowledge_graph.enabled,
        )
        self.searcher = HybridSearch(
            store=self.store,
            config=self.cfg.storage,
        )

    def connect(self):
        self.store.connect()
        self.store.ensure_schema()
        if self.cfg.knowledge_graph.enabled:
            self.graph.ensure_schema()


# ── CLI Group ──────────────────────────────────────────────────


@click.group()
@click.version_option(version="0.1.0")
@click.option("--config", "-c", default="", help="Path to config.yaml")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx, config, verbose):
    """DereInside — local-first AI knowledge system."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose


def _get_ctx(click_ctx) -> DereContext:
    """Get or create DereContext."""
    if "dere" not in click_ctx.obj:
        dc = DereContext(config_path=click_ctx.obj.get("config_path", ""))
        setup_logging(
            "debug" if click_ctx.obj.get("verbose") else "info", dc.cfg.logging.file
        )
        dc.connect()
        click_ctx.obj["dere"] = dc
    return click_ctx.obj["dere"]


# ── status ─────────────────────────────────────────────────────


@main.command()
@click.pass_context
def status(click_ctx):
    """Show system health and index stats."""
    dere = _get_ctx(click_ctx)
    stats = dere.store.stats()
    click.echo(f"DereInside v0.1.0 — {stats['wings']} wings, {stats['rooms']} rooms")
    click.echo(
        f"Pages: {stats['pages']}  |  Chunks: {stats['chunks']} ({stats['embedding_percent']}% embedded)"
    )
    click.echo()

    # Graph stats if enabled
    if dere.cfg.knowledge_graph.enabled:
        try:
            gs = dere.graph.stats()
            click.echo(
                f"Knowledge Graph: {gs['entities']} entities, {gs['relations']} relations"
            )
            click.echo(f"  Chunk links: {gs['entity_chunk_links']}")
            if gs["type_breakdown"]:
                click.echo("  Entity types:")
                for etype, count in gs["type_breakdown"].items():
                    click.echo(f"    {etype}: {count}")
        except Exception as e:
            click.echo(f"  ⚠️  Graph: {e}")

    click.echo()
    click.echo("Wing breakdown:")
    for wname, info in stats["wing_breakdown"].items():
        click.echo(f"  🏛️  {wname}/ — {info['rooms']} rooms, {info['pages']} pages")


# ── wings ──────────────────────────────────────────────────────


@main.command()
@click.pass_context
def wings(click_ctx):
    """List all wings with stats."""
    dere = _get_ctx(click_ctx)
    for w in dere.store.list_wings():
        click.echo(f"  🏛️  {w.name}/ — {w.room_count} rooms, {w.page_count} pages")


# ── rooms ──────────────────────────────────────────────────────


@main.command()
@click.argument("wing", default=None, required=False)
@click.pass_context
def rooms(click_ctx, wing):
    """List rooms, optionally filtered by wing."""
    dere = _get_ctx(click_ctx)
    wing_id = None
    if wing:
        for w in dere.store.list_wings():
            if w.name == wing:
                wing_id = w.id
                break
        if wing_id is None:
            click.echo(f"Wing '{wing}' not found")
            return

    for r in dere.store.list_rooms(wing_id):
        click.echo(f"  📂 {r.name} — {r.page_count} pages (wing_id={r.wing_id})")


# ── mine ───────────────────────────────────────────────────────


@main.command()
@click.argument("path")
@click.option("--wing", default=None, help="Assign to a wing")
@click.option("--room", default=None, help="Assign to a room")
@click.option("--mode", default="files", type=click.Choice(["files", "convos"]))
@click.option("--pattern", "-p", multiple=True, help="File patterns to include")
@click.option("--dry-run", is_flag=True, help="Preview without writing")
@click.pass_context
def mine(click_ctx, path, wing, room, mode, pattern, dry_run):
    """Ingest files or conversations into the palace."""
    dere = _get_ctx(click_ctx)
    target = Path(path).expanduser()

    if not target.exists():
        click.echo(f"❌ Path not found: {target}")
        return

    click.echo(f"🔍 Mining {target} (wing={wing or 'auto'}, mode={mode})...")

    files: list[Path] = []
    if target.is_file():
        files = [target]
    elif target.is_dir():
        if pattern:
            for p in pattern:
                files.extend(target.rglob(p))
        else:
            for ext in (
                ".md",
                ".py",
                ".java",
                ".vue",
                ".ts",
                ".js",
                ".yaml",
                ".yml",
                ".json",
                ".xml",
                ".sql",
                ".txt",
                ".rst",
                ".go",
                ".rs",
                ".rb",
                ".php",
                ".html",
                ".css",
                ".scss",
                ".kt",
                ".swift",
            ):
                files.extend(target.rglob(f"*{ext}"))
        ignore_dirs = {
            "node_modules",
            "target",
            "dist",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
        }
        files = [f for f in files if not any(d in f.parts for d in ignore_dirs)]
    else:
        click.echo(f"❌ Not a file or directory: {target}")
        return

    files.sort()
    click.echo(f"📄 Found {len(files)} files")

    if not files:
        return

    if not wing:
        wing = target.name if target.is_dir() else target.parent.name
    if not room:
        room = "files"

    wing_id = dere.store.get_or_create_wing(wing, f"Mined from {path}")
    room_id = dere.store.get_or_create_room(wing_id, room, "")

    if dry_run:
        click.echo(f"\n📋 [DRY-RUN] Would ingest {len(files)} files into {wing}/{room}")
        for f in files[:20]:
            rel = f.relative_to(target) if target.is_dir() else f.name
            click.echo(f"   📄 {rel}")
        if len(files) > 20:
            click.echo(f"   ... +{len(files)-20} more")
        return

    imported = 0
    total_chunks = 0
    skipped = 0

    for fpath in files:
        try:
            rel = fpath.relative_to(target) if target.is_dir() else fpath.name
            strategy = detect_strategy(fpath)
            chunks = chunk_file(fpath, strategy)
            if not chunks:
                skipped += 1
                continue

            page_id = dere.store.insert_page(
                room_id=room_id,
                slug=str(rel),
                title=fpath.stem,
                source_path=str(fpath),
                source_kind="file",
                page_kind=strategy,
            )

            texts = [c.text for c in chunks]
            embeddings = dere.embedder.embed_batch(texts)

            batch_data = []
            for ci, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                batch_data.append(
                    {
                        "page_id": page_id,
                        "chunk_index": ci,
                        "chunk_text": chunk.text,
                        "token_count": chunk.token_count,
                        "embedding": emb,
                    }
                )

            dere.store.insert_chunks_batch(batch_data)

            imported += 1
            total_chunks += len(chunks)
            click.echo(f"  ✅ {rel} → {len(chunks)} chunks", err=False)

        except Exception as e:
            click.echo(f"  ⚠️  {getattr(fpath, 'name', fpath)}: {e}", err=True)

    click.echo(
        f"\n📊 Results: {imported} files imported, {total_chunks} chunks, {skipped} skipped"
    )


# ── search ─────────────────────────────────────────────────────


@main.command()
@click.argument("query")
@click.option("--wing", default=None, help="Search within a wing")
@click.option("--room", default=None, help="Search within a room")
@click.option("--top-k", default=20, type=int, help="Number of results")
@click.option("--rerank", is_flag=True, help="Enable LLM reranking")
@click.option("--kg", "use_kg", is_flag=True, help="Enable graph propagation")
@click.option("--recent", is_flag=True, help="Enable temporal boost")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
@click.pass_context
def search(click_ctx, query, wing, room, top_k, rerank, use_kg, recent, json_out):
    """Semantic search across indexed knowledge."""
    dere = _get_ctx(click_ctx)
    t0 = time.time()

    query_embedding = dere.embedder.embed(query)

    req = SearchRequest(
        query=query,
        embedding=query_embedding,
        top_k=top_k,
        wing=wing,
        room=room,
        temporal_boost=recent or dere.cfg.search.temporal_boost.enabled,
        recent_days=dere.cfg.search.temporal_boost.recent_days,
        recent_weight=dere.cfg.search.temporal_boost.recent_weight,
        rerank=rerank or dere.cfg.search.rerank.enabled,
    )

    resp = dere.searcher.search(req)

    if req.rerank:
        resp.results = dere.reranker.rerank(query, resp.results, top_k)

    # Graph propagation
    use_kg = use_kg or dere.cfg.knowledge_graph.enabled
    if use_kg:
        resp.results = dere.propagator.propagate(query_embedding, resp.results, top_k)

    elapsed = time.time() - t0

    if json_out:
        output = {
            "query": query,
            "total": resp.total,
            "kg_enabled": use_kg,
            "timing_ms": round(elapsed * 1000, 1),
            "results": [r.to_dict() for r in resp.results],
        }
        click.echo(_json.dumps(output, indent=2, ensure_ascii=False))
        return

    click.echo(
        f"\n🔍 '{query}' — {resp.total} results ({elapsed:.2f}s, "
        f"rerank={'on' if req.rerank else 'off'}, "
        f"kg={'on' if use_kg else 'off'})"
    )
    click.echo("─" * 60)
    for i, r in enumerate(resp.results):
        score_str = f"{r.score:.3f}" if r.score else "---"
        wing_room = f"{r.wing_name}/{r.room_name}" if r.wing_name else ""
        click.echo(f"\n[{i+1}] (score={score_str}) {wing_room}")
        click.echo(f"    📄 {r.source_path or r.title or r.slug or '(unknown)'}")
        preview = r.chunk_text[:200].replace("\n", " ")
        click.echo(f"    {preview}...")


# ── wake ───────────────────────────────────────────────────────


@main.command()
@click.option("--wing", default=None, help="Load context for a wing")
@click.option("--hours", default=24, type=int, help="How many hours back")
@click.pass_context
def wake(click_ctx, wing, hours):
    """Load session context from recent knowledge."""
    dere = _get_ctx(click_ctx)
    click.echo(f"🌅 Waking up (wing={wing or 'all'}, last {hours}h)...")

    req = SearchRequest(
        query="wake context",
        embedding=dere.embedder.embed("recent changes updates modifications"),
        top_k=10,
        wing=wing,
        temporal_boost=True,
        recent_days=hours // 24 + 1,
    )
    resp = dere.searcher.search(req)

    click.echo(f"\n📋 Recent context ({resp.total} items):")
    for r in resp.results:
        preview = r.chunk_text[:150].replace("\n", " ")
        click.echo(f"  🏛️  {r.wing_name}/{r.room_name} → {preview}...")


# ── graph ──────────────────────────────────────────────────────


@main.group()
@click.pass_context
def graph(click_ctx):
    """Knowledge graph commands."""
    _get_ctx(click_ctx)  # ensure connected


@graph.command(name="stats")
@click.pass_context
def graph_stats(click_ctx):
    """Show knowledge graph statistics."""
    dere = _get_ctx(click_ctx)
    gs = dere.graph.stats()
    click.echo(f"Entities: {gs['entities']}")
    click.echo(f"Relations: {gs['relations']}")
    click.echo(f"Entity-Chunk links: {gs['entity_chunk_links']}")
    if gs["type_breakdown"]:
        click.echo("\nEntity types:")
        for etype, count in gs["type_breakdown"].items():
            click.echo(f"  {etype}: {count}")
    if gs["most_connected"]:
        click.echo("\nMost connected entities:")
        for e in gs["most_connected"]:
            click.echo(f"  {e['name']} ({e['type']}) — {e['relations']} relations")


@graph.command(name="entity")
@click.argument("name")
@click.pass_context
def graph_entity(click_ctx, name):
    """Look up an entity and its connected chunks."""
    dere = _get_ctx(click_ctx)
    entity = dere.graph.get_entity_by_name(name)
    if not entity:
        # Try search
        matches = dere.graph.search_entities(name, limit=10)
        if matches:
            click.echo(f"Entity '{name}' not found. Did you mean:")
            for m in matches:
                click.echo(f"  {m.name} ({m.entity_type})")
        else:
            click.echo(f"Entity '{name}' not found.")
        return

    click.echo(f"Entity: {entity.name} ({entity.entity_type})")
    if entity.metadata:
        click.echo(f"Metadata: {entity.metadata}")

    # Relations
    relations = dere.graph.get_relations_for_entity(entity.id)
    if relations:
        click.echo(f"\nRelations ({len(relations)}):")
        for r in relations:
            if r.source_entity_id == entity.id:
                target = dere.graph.get_entity(r.target_entity_id)
                tname = target.name if target else f"#{r.target_entity_id}"
                click.echo(f"  → {tname} ({r.relation_type}, w={r.weight})")
            else:
                source = dere.graph.get_entity(r.source_entity_id)
                sname = source.name if source else f"#{r.source_entity_id}"
                click.echo(f"  ← {sname} ({r.relation_type}, w={r.weight})")

    # Connected chunks
    chunk_ids = dere.graph.get_chunks_for_entity(entity.id, limit=10)
    if chunk_ids:
        click.echo(f"\nLinked chunks ({len(chunk_ids)}):")
        for cid in chunk_ids:
            # Get chunk preview via direct query
            with dere.store.conn.cursor() as cur:
                cur.execute("SELECT chunk_text[:200] FROM chunks WHERE id = %s", (cid,))
                row = cur.fetchone()
                if row:
                    preview = row[0].replace("\n", " ")
                    click.echo(f"  #{cid}: {preview}...")


@graph.command(name="search")
@click.argument("query")
@click.option("--type", "entity_type", default=None, help="Filter by entity type")
@click.option("--limit", default=20, type=int, help="Max results")
@click.pass_context
def graph_search(click_ctx, query, entity_type, limit):
    """Search entities by name."""
    dere = _get_ctx(click_ctx)
    results = dere.graph.search_entities(query, entity_type, limit)

    if not results:
        click.echo(f"No entities found matching '{query}'")
        return

    click.echo(f"Entities matching '{query}':")
    for e in results:
        # Count linked chunks
        chunk_ids = dere.graph.get_chunks_for_entity(e.id, limit=1)
        link_count = "?" if not chunk_ids else "..."
        try:
            with dere.store.conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM entity_chunks WHERE entity_id = %s", (e.id,)
                )
                link_count = cur.fetchone()[0]
        except Exception:
            pass
        click.echo(f"  {e.name} ({e.entity_type}) — {link_count} chunks")


@graph.command(name="build")
@click.option("--batch", default=20, type=int, help="Chunks per batch")
@click.option("--max-chunks", default=0, type=int, help="Max chunks to process (0=all)")
@click.pass_context
def graph_build(click_ctx, batch, max_chunks):
    """Build knowledge graph: extract entities from all chunks."""
    dere = _get_ctx(click_ctx)
    click.echo("🏗️  Building knowledge graph — extracting entities from chunks...")

    # Count available chunks
    with dere.store.conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE chunk_text IS NOT NULL AND LENGTH(chunk_text) > 80"
        )
        total = cur.fetchone()[0]
    if max_chunks and max_chunks < total:
        total = max_chunks
    click.echo(f"📊 {total} chunks to process (min 80 chars)")

    # Process in batches
    offset = 0
    processed = 0
    entity_count = 0
    relation_count = 0
    link_count = 0
    errors = 0

    batch = max(1, min(batch, 50))

    while True:
        if max_chunks and offset >= max_chunks:
            break

        with dere.store.conn.cursor() as cur:
            cur.execute(
                "SELECT id, chunk_text FROM chunks "
                "WHERE chunk_text IS NOT NULL AND LENGTH(chunk_text) > 80 "
                "ORDER BY id LIMIT %s OFFSET %s",
                (batch, offset),
            )
            rows = cur.fetchall()
            if not rows:
                break

        for chunk_id, chunk_text in rows:
            try:
                result = dere.extractor.extract(chunk_text)
                if result.is_empty():
                    continue

                # Bulk-import entities and relations
                entities = []
                for ent in result.entities:
                    entities.append(
                        {
                            "name": ent.name,
                            "entity_type": ent.entity_type,
                        }
                    )

                links = []
                entity_name_map = {}
                for e in entities:
                    eid = dere.graph.get_or_create_entity(e["name"], e["entity_type"])
                    entity_name_map[e["name"]] = eid
                    links.append(
                        {
                            "entity_name": e["name"],
                            "chunk_id": chunk_id,
                            "relevance": 1.0,
                        }
                    )
                    entity_count += 1

                # Relations
                rel_data = []
                for rel in result.relations:
                    if rel.source in entity_name_map and rel.target in entity_name_map:
                        rel_data.append(
                            {
                                "source_name": rel.source,
                                "target_name": rel.target,
                                "relation_type": rel.relation_type,
                                "weight": 1.0,
                            }
                        )

                imp = dere.graph.bulk_import(entities, rel_data, links)
                entity_count += imp["entities"]
                relation_count += imp["relations"]
                link_count += imp["links"]

            except Exception as e:
                errors += 1
                if errors <= 5:
                    click.echo(f"  ⚠️  Chunk #{chunk_id}: {e}", err=True)

            processed += 1
            if processed % 10 == 0:
                click.echo(
                    f"  ⏳ {processed}/{total} chunks, {entity_count} entities, {relation_count} relations...",
                    err=False,
                )

        offset += batch
        time.sleep(0.2)

    click.echo("\n✅ Graph build complete:")
    click.echo(f"   Processed: {processed} chunks ({errors} errors)")
    click.echo(f"   Entities:  {entity_count}")
    click.echo(f"   Relations: {relation_count}")
    click.echo(f"   Links:     {link_count}")

    # Verify
    gs = dere.graph.stats()
    click.echo(
        f"\n📊 Graph stats: {gs['entities']} entities, {gs['relations']} relations, {gs['entity_chunk_links']} links"
    )


# ── serve ──────────────────────────────────────────────────────


@main.command()
@click.option("--port", default=18890, type=int, help="HTTP port")
@click.pass_context
def serve(click_ctx, port):
    """Start HTTP bridge (FastAPI)."""
    click.echo("🌐 Starting HTTP bridge...")
    click.echo("   FastAPI bridge coming in Phase 3")
    click.echo("   For now, use the CLI: derekinside search 'query'")


if __name__ == "__main__":
    main()
