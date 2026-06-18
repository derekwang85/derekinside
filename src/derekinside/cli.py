"""
derekinside — Know your project from the inside out.

CLI entry point, CLI dispatcher.
"""

import click


@click.group()
@click.version_option(version="0.1.0")
def main():
    """DereInside — local-first AI knowledge system."""
    pass


@main.command()
@click.argument("path")
@click.option("--wing", default=None, help="Assign to a wing")
@click.option("--mode", default="files", type=click.Choice(["files", "convos"]))
def mine(path, wing, mode):
    """Ingest files or conversations into the palace."""
    click.echo(f"Mining {path} (wing={wing}, mode={mode})...")
    # TODO: Phase 1 implementation


@main.command()
@click.argument("query")
@click.option("--wing", default=None, help="Search within a wing")
@click.option("--room", default=None, help="Search within a room")
@click.option("--top-k", default=20, help="Number of results")
def search(query, wing, room, top_k):
    """Semantic search across indexed knowledge."""
    click.echo(f"Searching '{query}' (wing={wing}, room={room}, top_k={top_k})...")
    # TODO: Phase 1 implementation


@main.command()
@click.option("--wing", default=None, help="Load context for a wing")
def wake(wing):
    """Load session context from recent knowledge."""
    click.echo(f"Waking up (wing={wing})...")
    # TODO: Phase 3 implementation


@main.command()
def status():
    """Show system health and index stats."""
    click.echo("DereInside status...")
    # TODO


@main.command()
def serve():
    """Start HTTP bridge."""
    click.echo("Starting HTTP bridge on port 18890...")
    # TODO


if __name__ == "__main__":
    main()
