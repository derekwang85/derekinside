"""
derekinside — Graph Pruning.

Cleans up noise entities from the knowledge graph:
  1. Entities that link to only 1 chunk → low-confidence, mark for pruning
  2. Stop-word entities (common English words extracted as concepts)
  3. 1.5b-only entities (not confirmed by 7b/regex)
  4. Orphan entities (no chunk links, no relations, created >30 days ago)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from derekinside.storage.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# ── Stop-word list for entity names that are almost always noise ──

_STOP_ENTITIES: set[str] = {
    # Common English words
    "data",
    "info",
    "information",
    "content",
    "text",
    "file",
    "code",
    "name",
    "type",
    "value",
    "key",
    "id",
    "list",
    "map",
    "set",
    "array",
    "object",
    "class",
    "method",
    "function",
    "field",
    "property",
    "result",
    "response",
    "request",
    "input",
    "output",
    "status",
    "config",
    "configuration",
    "setting",
    "option",
    "param",
    "parameter",
    "item",
    "tag",
    "label",
    "title",
    "desc",
    "description",
    "summary",
    "index",
    "node",
    "path",
    "url",
    "uri",
    "link",
    "ref",
    "reference",
    "system",
    "service",
    "module",
    "component",
    "util",
    "utils",
    "helper",
    "api",
    "rest",
    "soap",
    "json",
    "xml",
    "yaml",
    "yml",
    "sql",
    # Java/Kotlin default
    "string",
    "integer",
    "boolean",
    "long",
    "double",
    "float",
    "byte",
    "char",
    "short",
    "void",
    "object",
    "exception",
    "throwable",
    "serialversionuid",
    "logger",
    "log",
    # Framework noise
    "autowired",
    "inject",
    "resource",
    "override",
    "deprecated",
    "suppresswarnings",
    "test",
    "before",
    "after",
    "setup",
    "teardown",
    # Development
    "todo",
    "fixme",
    "hack",
    "xxx",
    "note",
}


@dataclass
class PruneReport:
    removed: int = 0
    archived: int = 0
    lowered_weight: int = 0
    skipped: int = 0
    details: list[str] = field(default_factory=list)


class GraphPruner:
    """
    Knowledge graph noise cleaner.
    Run with --dry-run to preview, without --dry-run to execute.
    """

    def __init__(self, graph: KnowledgeGraph):
        self._graph = graph

    def prune(
        self,
        min_chunk_links: int = 2,
        dry_run: bool = True,
        days_old: int = 30,
    ) -> PruneReport:
        """
        Prune noise entities.

        Args:
            min_chunk_links: Minimum chunk links for an entity to survive
            dry_run: If True, only report what would be removed
            days_old: Remove orphan entities older than this many days
        """
        report = PruneReport()
        cutoff_ts = time.time() - days_old * 86400

        with self._graph.cursor() as cur:
            # 1. Entity with too few chunk links
            cur.execute(
                """
                SELECT e.id, e.name, e.entity_type, e.metadata,
                       COUNT(ec.id) as link_count
                FROM entities e
                LEFT JOIN entity_chunks ec ON ec.entity_id = e.id
                GROUP BY e.id, e.name, e.entity_type, e.metadata
                HAVING COUNT(ec.id) < %s
                ORDER BY link_count, e.id
            """,
                (min_chunk_links,),
            )
            sparse = cur.fetchall()
            report.details.append(
                f"Sparse entities (< {min_chunk_links} links): {len(sparse)}"
            )

            for row in sparse:
                eid, name, etype, metadata, links = row
                lower_name = name.lower()
                should_remove = False
                reason = ""

                # Check stop words
                if lower_name in _STOP_ENTITIES:
                    should_remove = True
                    reason = "stop-word"
                # Check noise patterns
                elif re.match(r"^[a-z]{2,4}$", name):  # short lowercase words
                    should_remove = True
                    reason = "too-short"
                elif re.match(r"^[0-9.]+$", name):  # pure numbers
                    should_remove = True
                    reason = "numeric"
                # 1.5b-only with no metadata
                elif links == 1 and name.islower() and len(name) < 5:
                    should_remove = True
                    reason = "1.5b-noise"

                if should_remove:
                    if dry_run:
                        report.details.append(
                            f"  [DRY] #{eid} '{name}' ({etype}): {reason}"
                        )
                    else:
                        cur.execute("DELETE FROM entities WHERE id = %s", (eid,))
                        report.details.append(
                            f"  [RMD] #{eid} '{name}' ({etype}): {reason}"
                        )
                    report.removed += 1

            # 2. Orphan entities (no chunk links, no relations, old)
            cur.execute(
                """
                SELECT e.id, e.name, e.entity_type
                FROM entities e
                WHERE NOT EXISTS (SELECT 1 FROM entity_chunks ec WHERE ec.entity_id = e.id)
                  AND NOT EXISTS (SELECT 1 FROM relations r
                                  WHERE r.source_entity_id = e.id OR r.target_entity_id = e.id)
                  AND EXTRACT(EPOCH FROM e.created_at) < %s
                ORDER BY e.id
            """,
                (cutoff_ts,),
            )
            orphans = cur.fetchall()
            report.details.append(
                f"Orphan entities (> {days_old}d, no links): {len(orphans)}"
            )

            for row in orphans:
                eid, name, etype = row
                if dry_run:
                    report.details.append(f"  [ARCHIVE DRY] #{eid} '{name}' ({etype})")
                else:
                    cur.execute("DELETE FROM entities WHERE id = %s", (eid,))
                report.archived += 1

            # 3. Lower weight of entities with only 1b-only confidence
            # (marked by _mode field in metadata)
            cur.execute("""
                SELECT id, name, metadata
                FROM entities
                WHERE metadata->>'_mode' IN ('1.5b', '7b')
                  AND NOT EXISTS (SELECT 1 FROM entity_chunks ec
                                  WHERE ec.entity_id = entities.id
                                  AND ec.relevance < 0.5)
                  AND metadata->>'weight' IS DISTINCT FROM '0.3'
            """)
            low_confidence = cur.fetchall()
            report.details.append(
                f"Low-confidence entities (1.5b/7b only): {len(low_confidence)}"
            )

            if not dry_run:
                for row in low_confidence:
                    eid, name, meta = row
                    meta_dict = meta if isinstance(meta, dict) else {}
                    meta_dict["weight"] = 0.3
                    meta_dict["pruned"] = True
                    cur.execute(
                        "UPDATE entities SET metadata = %s WHERE id = %s",
                        (meta_dict, eid),
                    )
                    report.lowered_weight += 1

        return report
