"""
derekinside — Know your project from the inside out.

CLI entry point — Phase 1 + Phase 2.
"""

from __future__ import annotations

import json as _json
import logging

logger = logging.getLogger(__name__)
import os
import time
from pathlib import Path

import click
from pathlib import Path

from derekinside.config import load_config
from derekinside.storage.pgvector import VectorStore
from derekinside.storage.graph import KnowledgeGraph
from derekinside.indexer.chunker import chunk_file, detect_strategy
from derekinside.search.hybrid import HybridSearch, SearchRequest
from derekinside.search.propagation import GraphPropagator
from derekinside.engine.engine import Engine
from derekinside.indexer.relation_inferrer import RelationInferrer
from derekinside.indexer.merge import merge_entity_sets, merge_relation_sets
from derekinside.indexer.entity_resolver import EntityResolver
from derekinside.indexer.graph_pruner import GraphPruner
from derekinside.storage.subgraph import build_subgraph
from derekinside.indexer.fusion import EntityFusion
from derekinside.indexer.consensus import ConsensusEngine
from derekinside.indexer.enricher import EntityEnricher

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
        self.propagator = GraphPropagator(
            graph=self.graph,
            enabled=self.cfg.knowledge_graph.enabled,
        )
        self.searcher = HybridSearch(
            store=self.store,
            config=self.cfg.storage,
        )
        # Engine — unified model registry + pipeline resolver + profiler
        # Build model config from the flat config structure
        self.engine = Engine(self.cfg.to_dict())

    def connect(self):
        self.store.connect()
        self.store.ensure_schema()
        if self.cfg.knowledge_graph.enabled:
            self.graph.ensure_schema()
        self.engine.start()


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
            embeddings = dere.engine.embed_batch(texts)

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
@click.option("--before", default=None, help="Only chunks created before this ISO datetime")
@click.option("--after", default=None, help="Only chunks created after this ISO datetime")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
@click.pass_context
def search(click_ctx, query, wing, room, top_k, rerank, use_kg, recent, before, after, json_out):
    """Semantic search across indexed knowledge.

    Examples:
        derekinside search "KYC" --before "2026-06-18T00:00:00"
        derekinside search "order" --after "2026-06-01" --before "2026-06-15"
    """
    dere = _get_ctx(click_ctx)
    t0 = time.time()

    query_embedding = dere.engine.embed(query)

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
        before=before,
        after=after,
    )

    resp = dere.searcher.search(req)

    if req.rerank:
        resp.results = dere.engine.rerank(query, [r.text for r in resp.results])

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
        embedding=dere.engine.embed("recent changes updates modifications"),
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
            with dere.store.cursor() as cur:
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
            with dere.store.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM entity_chunks WHERE entity_id = %s", (e.id,)
                )
                link_count = cur.fetchone()[0]
        except Exception:
            pass
        click.echo(f"  {e.name} ({e.entity_type}) — {link_count} chunks")


@graph.command(name="subgraph")
@click.argument("entity")
@click.option("--depth", default=2, type=int, help="Max traversal depth")
@click.option("--ascii", is_flag=True, help="Output as ASCII tree")
@click.pass_context
def graph_subgraph(click_ctx, entity, depth, ascii):
    """Show subgraph centered on an entity."""
    dere = _get_ctx(click_ctx)
    sg = build_subgraph(dere.graph, entity, max_depth=depth)

    if sg is None:
        click.echo(f"Entity '{entity}' not found.")
        # Try search
        matches = dere.graph.search_entities(entity, limit=5)
        if matches:
            click.echo("Did you mean:")
            for m in matches:
                click.echo(f"  {m.name} ({m.entity_type})")
        return

    if ascii:
        click.echo(sg.to_ascii())
    else:
        import json
        click.echo(json.dumps(sg.to_dict(), ensure_ascii=False, indent=2))


@graph.command(name="consensus")
@click.option("--sample", default=50, type=int, help="Edge entities to re-examine")
@click.option("--dry-run", is_flag=True, default=True, help="Preview only (default)")
@click.pass_context
def graph_consensus(click_ctx, sample, dry_run):
    """Cross-validate edge entities across extraction modes."""
    dere = _get_ctx(click_ctx)
    ce = ConsensusEngine(dere.engine, dere.graph)
    ce.ensure_schema()
    result = ce.learn_from_graph(sample_count=sample, dry_run=dry_run)
    confirmed = sum(1 for v in result.verdicts if v.status == "confirmed")
    uncertain = sum(1 for v in result.verdicts if v.status == "uncertain")
    rejected = sum(1 for v in result.verdicts if v.status == "rejected")
    click.echo(f"Entities evaluated: {len(result.verdicts)}")
    click.echo(f"  Confirmed:  {confirmed}")
    click.echo(f"  Uncertain:  {uncertain}")
    click.echo(f"  Rejected:   {rejected}")
    click.echo(f"  Time:       {result.runtime_ms:.0f}ms")
    if not dry_run:
        click.echo("  (Learning table updated)")


@graph.command(name="fuse")
@click.option("--dry-run", is_flag=True, default=True, help="Preview only (default)")
@click.pass_context
def graph_fuse(click_ctx, dry_run):
    """Merge duplicate entities across wings."""
    dere = _get_ctx(click_ctx)
    fusion = EntityFusion(dere.graph)
    report = fusion.fuse(dry_run=dry_run)
    click.echo(f"Duplicates: {report.total_duplicates}")
    click.echo(f"Merged:     {report.merged}")
    click.echo(f"Deleted:    {report.deleted}")
    click.echo(f"Rels:       {report.relations_transferred}")
    click.echo(f"Chunks:     {report.chunks_relinked}")
    if report.errors:
        click.echo(f"Errors:     {report.errors}")
    if report.details and len(report.details) <= 20:
        for d in report.details:
            click.echo(f"  {d}")
    elif report.details:
        click.echo(f"  ... {len(report.details)} total details")


@graph.command(name="enrich")
@click.option("--limit", default=50, type=int, help="Max entities to enrich")
@click.option("--model", default="qwen2.5-coder:7b", help="LLM model for description generation")
@click.pass_context
def graph_enrich(click_ctx, limit, model):
    """Generate descriptions for entities using offline LLM."""
    dere = _get_ctx(click_ctx)
    enricher = EntityEnricher(dere.graph, model_name=model)
    click.echo(f"Enriching up to {limit} entities without descriptions...")
    result = enricher.enrich_batch(limit=limit)
    click.echo(f"Done: {result['enriched']} enriched, {result['failed']} failed, out of {result['total']}")


@graph.command(name="build")
@click.option("--batch", default=20, type=int, help="Chunks per batch")
@click.option("--max-chunks", default=0, type=int, help="Max chunks to process (0=all)")
@click.option("--llm", is_flag=True, help="[Deprecated] Enable LLM extraction (use --mode instead)")
@click.option("--mode", default="", type=click.Choice(["", "regex", "1.5b", "7b", "hybrid-1.5b", "hybrid-7b"]),
              help="Entity extraction mode (overrides config)")
@click.option("--list-modes", is_flag=True, help="List available extraction modes")
@click.pass_context
def graph_build(click_ctx, batch, max_chunks, llm, mode, list_modes):
    """Build knowledge graph: extract entities from all chunks."""
    from derekinside.indexer.entity import _EXTRACTION_MODES

    if list_modes:
        click.echo("📋 Available entity extraction modes:")
        for mname, mdesc in _EXTRACTION_MODES.items():
            click.echo(f"  {mname:15s} — {mdesc}")
        return

    dere = _get_ctx(click_ctx)

    # Resolve mode: CLI --mode overrides config
    effective_mode = mode or dere.cfg.knowledge_graph.entity_extraction.mode
    if llm and not effective_mode:
        effective_mode = "hybrid-1.5b"  # --llm backward compat

    # Mode selection is now handled by PipelineResolver
    # CLI --mode flag overrides the pipeline strategy
    if effective_mode and effective_mode != dere.cfg.pipeline.get('extract', {}).get('mode', ''):
        logger.info("CLI mode override: %s", effective_mode)

    click.echo(f"🧠 Entity extraction mode: {effective_mode}")
    click.echo("🏗️  Building knowledge graph — extracting entities from chunks...")

    # Count available chunks
    with dere.store.cursor() as cur:
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

        with dere.store.cursor() as cur:
            cur.execute(
                "SELECT id, chunk_text FROM chunks "
                "WHERE chunk_text IS NOT NULL AND LENGTH(chunk_text) > 80 "
                "ORDER BY id LIMIT %s OFFSET %s",
                (batch, offset),
            )
            rows = cur.fetchall()
            if not rows:
                break

        inferrer = RelationInferrer()
        resolver = EntityResolver()

        for chunk_id, chunk_text in rows:
            processed += 1
            try:
                # Step 1: LLM/regex entity extraction
                result = dere.engine.extract(chunk_text)

                # Step 2: Get file extension for structured inference
                ext = ""
                with dere.store.cursor() as cur:
                    cur.execute(
                        "SELECT p.source_path FROM pages p "
                        "JOIN chunks c ON c.page_id = p.id "
                        "WHERE c.id = %s", (chunk_id,)
                    )
                    page_row = cur.fetchone()
                    source_path = page_row[0] if page_row else ""
                    ext = Path(source_path).suffix if source_path else ""

                # Step 3: Entity extraction result
                entities = []
                if hasattr(result, 'entities'):
                    entities = result.entities
                elif isinstance(result, dict):
                    entities = result.get('entities', [])

                # Step 4: Resolve entity names (dedup)
                resolved_map = {}
                for ent in entities:
                    name = ent.name if hasattr(ent, 'name') else ent.get('name', '')
                    if name:
                        resolved_map[name] = resolver.resolve(name)

                # Step 5: Create entities and build name->id map
                entity_name_map = {}
                for ent in entities:
                    name = ent.name if hasattr(ent, 'name') else ent.get('name', '')
                    if not name:
                        continue
                    etype = ent.entity_type if hasattr(ent, 'entity_type') else ent.get('type', 'concept')
                    canonical, _ = resolved_map.get(name, (name, False))
                    eid = dere.graph.get_or_create_entity(canonical, etype)
                    entity_name_map[name] = eid
                    entity_count += 1

                # Step 6: Link entities to chunk
                for eid in set(entity_name_map.values()):
                    dere.graph.link_entity_to_chunk(eid, chunk_id, relevance=1.0)
                    link_count += 1

                # Step 7: Relations from LLM extraction
                relations = []
                if hasattr(result, 'relations'):
                    relations = result.relations
                elif isinstance(result, dict):
                    relations = result.get('relations', [])

                for rel in relations:
                    source = rel.source if hasattr(rel, 'source') else rel.get('source', '')
                    target = rel.target if hasattr(rel, 'target') else rel.get('target', '')
                    rtype = rel.relation_type if hasattr(rel, 'relation_type') else rel.get('type', 'related')
                    src_eid = entity_name_map.get(source)
                    tgt_eid = entity_name_map.get(target)
                    if src_eid and tgt_eid:
                        dere.graph.add_relation(src_eid, tgt_eid, rtype, 1.0)
                        relation_count += 1

                # Step 8: Structured relation inference
                if ext:
                    inferred = inferrer.infer(chunk_text, chunk_id, ext)
                    for rel in inferred:
                        src_eid = entity_name_map.get(rel.source)
                        tgt_eid = entity_name_map.get(rel.target)
                        if rel.source and not src_eid:
                            src_eid = dere.graph.get_or_create_entity(rel.source, 'class')
                            entity_name_map[rel.source] = src_eid
                            entity_count += 1
                        if rel.target and not tgt_eid:
                            tgt_eid = dere.graph.get_or_create_entity(rel.target, 'module')
                            entity_name_map[rel.target] = tgt_eid
                            entity_count += 1
                        if src_eid and tgt_eid:
                            dere.graph.add_relation(src_eid, tgt_eid, rel.relation_type, rel.weight)
                            relation_count += 1
                if result.is_empty():
                    continue

                # Create entities and build name→id map
                entity_name_map: dict[str, int] = {}
                for ent in result.entities:
                    eid = dere.graph.get_or_create_entity(ent.name, ent.entity_type)
                    entity_name_map[ent.name] = eid
                    entity_count += 1

                # Link entities to this chunk
                for name, eid in entity_name_map.items():
                    dere.graph.link_entity_to_chunk(eid, chunk_id, relevance=1.0)
                    link_count += 1

                # Create relations
                for rel in result.relations:
                    if rel.source in entity_name_map and rel.target in entity_name_map:
                        dere.graph.add_relation(
                            entity_name_map[rel.source],
                            entity_name_map[rel.target],
                            rel.relation_type,
                            1.0,
                        )
                        relation_count += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    click.echo(f"  ⚠️  Chunk #{chunk_id}: {e}", err=True)

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
@click.option(
    "--mode",
    default="http",
    type=click.Choice(["http", "mcp", "both"]),
    help="Serve mode: http (REST), mcp (stdio), both",
)
@click.option("--port", default=18890, type=int, help="HTTP port")
@click.option("--host", default="0.0.0.0", help="HTTP bind address")
@click.pass_context
def serve(click_ctx, mode, port, host):
    """Start bridge server (HTTP, MCP, or both)."""
    dere = _get_ctx(click_ctx)

    if mode == "http" or mode == "both":
        from derekinside.bridge.http import serve_http
        from derekinside.bridge.auth import Auth, AuthConfig

        ac = AuthConfig(
            enabled=dere.cfg.mcp_server.enabled,
            token=os.environ.get("DEREINSIDE_TOKEN", ""),
        )
        auth = Auth(ac)

        import threading

        http_thread = threading.Thread(
            target=serve_http,
            args=(dere.store, dere.embedder),
            kwargs={
                "auth": auth,
                "kg": dere.graph if dere.cfg.knowledge_graph.enabled else None,
                "extractor": dere.extractor
                if dere.cfg.knowledge_graph.enabled
                else None,
                "host": host,
                "port": port,
            },
            daemon=True,
        )
        http_thread.start()
        click.echo(f"🌐 HTTP bridge starting on http://{host}:{port}")

    if mode == "mcp" or mode == "both":
        from derekinside.bridge.mcp import MCPServer

        mcp = MCPServer()
        if mode == "both":
            click.echo("📡 MCP server starting on stdio...")
            import time

            time.sleep(1)
            mcp.run()
        else:
            click.echo("📡 MCP server starting on stdio...")
            mcp.run()

    if mode == "http":
        import time

        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            click.echo("\n👋 Shutting down...")

    if mode == "mcp":
        pass  # mcp.run() blocks


if __name__ == "__main__":
    main()
