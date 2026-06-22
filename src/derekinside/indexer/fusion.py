"""
derekinside — Cross-Wing Entity Fusion.

Merges entities with the same canonical name across different wings.
Runs as a background task (weekly cron or manual derekinside graph fuse).

Strategy:
  1. Identifies entities with the same canonicalized name in different wings
  2. Merges them into a single canonical entity
  3. Links all chunks from all wings to the merged entity
  4. Transfers all relations to the merged entity
  5. Removes duplicate entities
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from derekinside.storage.graph import KnowledgeGraph
from derekinside.indexer.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)


@dataclass
class FusionReport:
    total_duplicates: int = 0
    merged: int = 0
    relations_transferred: int = 0
    chunks_relinked: int = 0
    deleted: int = 0
    errors: int = 0
    details: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Duplicates: {self.total_duplicates}, Merged: {self.merged}, "
            f"Rels transferred: {self.relations_transferred}, "
            f"Chunks relinked: {self.chunks_relinked}, Deleted: {self.deleted}"
        )


class EntityFusion:
    """
    Cross-Wing Entity Fusion.

    Finds entities with the same canonical name in different wings and merges them.
    Uses EntityResolver for name canonicalization.
    """

    def __init__(self, graph: KnowledgeGraph):
        self._graph = graph
        self._resolver = EntityResolver()

    def fuse(self, dry_run: bool = True) -> FusionReport:
        """
        Run fusion across all wings.

        Args:
            dry_run: If True, only report what would be merged

        Returns:
            FusionReport with statistics
        """
        report = FusionReport()

        # Step 1: Find duplicate entity names across wings
        with self._graph.cursor() as cur:
            # Get all entities grouped by canonicalized name
            cur.execute("""
                SELECT e.id, e.name, e.entity_type, w.name as wing_name
                FROM entities e
                JOIN entity_chunks ec ON ec.entity_id = e.id
                JOIN chunks c ON c.id = ec.chunk_id
                JOIN pages p ON p.id = c.page_id
                JOIN rooms r ON r.id = p.room_id
                JOIN wings w ON w.id = r.wing_id
                ORDER BY e.name
            """)
            rows = cur.fetchall()

        # Group by canonical name
        canonical_groups: dict[str, list[dict]] = {}
        for eid, name, etype, wing in rows:
            canonical, _ = self._resolver.resolve(name)
            canonical_groups.setdefault(canonical, []).append({
                "id": eid, "name": name, "type": etype, "wing": wing,
            })

        # Find duplicate groups (same canonical name in >1 wing or >1 entity)
        for canonical, group in canonical_groups.items():
            if len(group) < 2:
                continue
            # Check if they're actually in different wings
            wings = set(e["wing"] for e in group)
            if len(wings) < 2 and len(group) < 2:
                continue

            report.total_duplicates += 1

            # Pick the "best" entity to keep (prefer higher precision type)
            # Priority: class > function > module > api > concept
            type_priority = {"class": 0, "function": 1, "module": 2, "api": 3, "concept": 4}
            group.sort(key=lambda e: type_priority.get(e["type"], 5))

            keep = group[0]
            merge_targets = group[1:]

            detail = (
                f"[{'MERGE' if not dry_run else 'DRY'}] "
                f"Canonical '{canonical}': "
                f"keep #{keep['id']} ({keep['name']}, {keep['type']}, {keep['wing']}) "
                f"← {len(merge_targets)} duplicates"
            )
            report.details.append(detail)

            if dry_run:
                continue

            # Merge all duplicates into the keeper
            for dup in merge_targets:
                dup_id = dup["id"]
                try:
                    # Transfer chunk links
                    with self._graph.cursor() as cur:
                        cur.execute("""
                            UPDATE entity_chunks
                            SET entity_id = %s
                            WHERE entity_id = %s
                            AND NOT EXISTS (
                                SELECT 1 FROM entity_chunks
                                WHERE entity_id = %s AND chunk_id = entity_chunks.chunk_id
                            )
                        """, (keep["id"], dup_id, keep["id"]))
                        report.chunks_relinked += cur.rowcount

                    # Transfer relations from this duplicate
                    with self._graph.cursor() as cur:
                        # Incoming relations: point to keeper
                        cur.execute("""
                            UPDATE relations
                            SET target_entity_id = %s
                            WHERE target_entity_id = %s
                            AND NOT EXISTS (
                                SELECT 1 FROM relations
                                WHERE source_entity_id = relations.source_entity_id
                                AND target_entity_id = %s
                                AND relation_type = relations.relation_type
                            )
                        """, (keep["id"], dup_id, keep["id"]))
                        report.relations_transferred += cur.rowcount

                        # Outgoing relations: point from keeper
                        cur.execute("""
                            UPDATE relations
                            SET source_entity_id = %s
                            WHERE source_entity_id = %s
                            AND NOT EXISTS (
                                SELECT 1 FROM relations
                                WHERE source_entity_id = %s
                                AND target_entity_id = relations.target_entity_id
                                AND relation_type = relations.relation_type
                            )
                        """, (keep["id"], dup_id, keep["id"]))
                        report.relations_transferred += cur.rowcount

                    # Delete relations pointing to the duplicate
                    with self._graph.cursor() as cur:
                        cur.execute(
                            "DELETE FROM relations WHERE source_entity_id = %s OR target_entity_id = %s",
                            (dup_id, dup_id),
                        )

                    # Delete entity_chunks for the duplicate
                    with self._graph.cursor() as cur:
                        cur.execute(
                            "DELETE FROM entity_chunks WHERE entity_id = %s AND entity_id != %s",
                            (dup_id, keep["id"]),
                        )

                    # Delete the duplicate entity
                    with self._graph.cursor() as cur:
                        cur.execute("DELETE FROM entities WHERE id = %s", (dup_id,))

                    report.merged += 1
                    report.deleted += 1

                except Exception as e:
                    logger.error("Failed to merge entity #%d into #%d: %s", dup_id, keep["id"], e)
                    report.errors += 1

        return report

    def run_fusion_cron(self, dry_run: bool = False) -> FusionReport:
        """Wrapper for cron job with timing."""
        logger.info("Starting cross-wing entity fusion (dry_run=%s)", dry_run)
        start = time.time()
        report = self.fuse(dry_run=dry_run)
        elapsed = time.time() - start
        logger.info(
            "Fusion complete: %s (%.1fs)",
            report.summary(), elapsed,
        )
        return report
