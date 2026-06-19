"""
derekinside — Knowledge Graph storage layer.

Entity-relation graph built on PostgreSQL for:
- Entity extraction (class, function, module, concept, API)
- Relational links between entities (imports, calls, contains, related)
- Entity↔Chunk linking for graph-augmented search
- PageRank propagation through the graph
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from derekinside.storage.pgvector import VectorStore


def _ensure_json(value: Any) -> Any:
    if isinstance(value, dict):
        try:
            from psycopg.types.json import Json

            return Json(value)
        except ImportError:
            return value
    return value


logger = logging.getLogger(__name__)


@dataclass
class Entity:
    id: int = 0
    name: str = ""
    entity_type: str = "concept"  # class, function, module, api, concept, person
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "type": self.entity_type}


@dataclass
class Relation:
    id: int = 0
    source_entity_id: int = 0
    target_entity_id: int = 0
    relation_type: str = "related"  # imports, calls, contains, associated, related
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source_entity_id,
            "target": self.target_entity_id,
            "type": self.relation_type,
            "weight": self.weight,
        }


class KnowledgeGraph:
    """Entity-relation knowledge graph backed by PostgreSQL."""

    def __init__(self, store: VectorStore):
        self._store = store

    @property
    def conn(self):
        return self._store.conn

    def cursor(self):
        """Shortcut: get a cursor from the store's connection pool."""
        return self._store.cursor()

    # ── Schema ─────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        """Create entity/relation tables if they don't exist."""
        with self.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT 'concept',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
                CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id SERIAL PRIMARY KEY,
                    source_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    target_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL DEFAULT 'related',
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(source_entity_id, target_entity_id, relation_type)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id);
                CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id);
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entity_chunks (
                    id SERIAL PRIMARY KEY,
                    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    relevance REAL DEFAULT 1.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(entity_id, chunk_id)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_chunks_eid ON entity_chunks(entity_id);
                CREATE INDEX IF NOT EXISTS idx_entity_chunks_cid ON entity_chunks(chunk_id);
            """)

    # ── Entities ───────────────────────────────────────────────

    def get_or_create_entity(
        self, name: str, entity_type: str = "concept", metadata: Optional[dict] = None
    ) -> int:
        """Get entity ID by name, create if not exists. Returns (id, created)."""
        with self.cursor() as cur:
            cur.execute("SELECT id FROM entities WHERE name = %s", (name,))
            row = cur.fetchone()
            if row:
                return row[0]

            cur.execute(
                "INSERT INTO entities (name, entity_type, metadata) VALUES (%s, %s, %s) RETURNING id",
                (name, entity_type, _ensure_json(metadata or {})),
            )
            return cur.fetchone()[0]

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, name, entity_type, metadata FROM entities WHERE name = %s",
                (name,),
            )
            row = cur.fetchone()
            if row:
                return Entity(
                    id=row[0], name=row[1], entity_type=row[2], metadata=row[3] or {}
                )
            return None

    def get_entity(self, entity_id: int) -> Optional[Entity]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, name, entity_type, metadata FROM entities WHERE id = %s",
                (entity_id,),
            )
            row = cur.fetchone()
            if row:
                return Entity(
                    id=row[0], name=row[1], entity_type=row[2], metadata=row[3] or {}
                )
            return None

    def search_entities(
        self, query: str, entity_type: Optional[str] = None, limit: int = 20
    ) -> list[Entity]:
        with self.cursor() as cur:
            if entity_type:
                cur.execute(
                    "SELECT id, name, entity_type, metadata FROM entities "
                    "WHERE name ILIKE %s AND entity_type = %s "
                    "ORDER BY LENGTH(name) LIMIT %s",
                    (f"%{query}%", entity_type, limit),
                )
            else:
                cur.execute(
                    "SELECT id, name, entity_type, metadata FROM entities "
                    "WHERE name ILIKE %s "
                    "ORDER BY LENGTH(name) LIMIT %s",
                    (f"%{query}%", limit),
                )
            return [
                Entity(id=r[0], name=r[1], entity_type=r[2], metadata=r[3] or {})
                for r in cur.fetchall()
            ]

    def list_entities(
        self, entity_type: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> list[Entity]:
        with self.cursor() as cur:
            if entity_type:
                cur.execute(
                    "SELECT id, name, entity_type, metadata FROM entities "
                    "WHERE entity_type = %s ORDER BY id LIMIT %s OFFSET %s",
                    (entity_type, limit, offset),
                )
            else:
                cur.execute(
                    "SELECT id, name, entity_type, metadata FROM entities "
                    "ORDER BY id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
            return [
                Entity(id=r[0], name=r[1], entity_type=r[2], metadata=r[3] or {})
                for r in cur.fetchall()
            ]

    # ── Relations ──────────────────────────────────────────────

    def add_relation(
        self,
        source_id: int,
        target_id: int,
        relation_type: str = "related",
        weight: float = 1.0,
    ) -> int:
        """Add a directed relation between two entities."""
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO relations (source_entity_id, target_entity_id, relation_type, weight) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (source_entity_id, target_entity_id, relation_type) "
                "DO UPDATE SET weight = EXCLUDED.weight "
                "RETURNING id",
                (source_id, target_id, relation_type, weight),
            )
            return cur.fetchone()[0]

    def add_relation_batch(self, relations: list[dict]) -> int:
        """Batch insert relations. Each dict: source_id, target_id, relation_type, weight."""
        if not relations:
            return 0
        with self.cursor() as cur:
            for rel in relations:
                cur.execute(
                    "INSERT INTO relations (source_entity_id, target_entity_id, relation_type, weight) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (source_entity_id, target_entity_id, relation_type) "
                    "DO UPDATE SET weight = EXCLUDED.weight",
                    (
                        rel.get("source_id") or rel.get("source_entity_id"),
                        rel.get("target_id") or rel.get("target_entity_id"),
                        rel.get("relation_type", "related"),
                        rel.get("weight", 1.0),
                    ),
                )
            return len(relations)

    def get_relations_for_entity(
        self, entity_id: int, direction: str = "both"
    ) -> list[Relation]:
        """Get relations connected to an entity."""
        with self.cursor() as cur:
            if direction == "outgoing":
                cur.execute(
                    "SELECT id, source_entity_id, target_entity_id, relation_type, weight "
                    "FROM relations WHERE source_entity_id = %s",
                    (entity_id,),
                )
            elif direction == "incoming":
                cur.execute(
                    "SELECT id, source_entity_id, target_entity_id, relation_type, weight "
                    "FROM relations WHERE target_entity_id = %s",
                    (entity_id,),
                )
            else:
                cur.execute(
                    "SELECT id, source_entity_id, target_entity_id, relation_type, weight "
                    "FROM relations WHERE source_entity_id = %s OR target_entity_id = %s",
                    (entity_id, entity_id),
                )
            return [
                Relation(
                    id=r[0],
                    source_entity_id=r[1],
                    target_entity_id=r[2],
                    relation_type=r[3],
                    weight=r[4],
                )
                for r in cur.fetchall()
            ]

    # ── Entity ↔ Chunk Links ───────────────────────────────────

    def link_entity_to_chunk(
        self, entity_id: int, chunk_id: int, relevance: float = 1.0
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO entity_chunks (entity_id, chunk_id, relevance) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (entity_id, chunk_id) DO UPDATE SET relevance = EXCLUDED.relevance "
                "RETURNING id",
                (entity_id, chunk_id, relevance),
            )
            return cur.fetchone()[0]

    def link_entity_to_chunks_batch(self, links: list[dict]) -> int:
        """Batch link entities to chunks. Each dict: entity_id, chunk_id, relevance."""
        if not links:
            return 0
        with self.cursor() as cur:
            for link in links:
                cur.execute(
                    "INSERT INTO entity_chunks (entity_id, chunk_id, relevance) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (entity_id, chunk_id) DO UPDATE SET relevance = EXCLUDED.relevance",
                    (link["entity_id"], link["chunk_id"], link.get("relevance", 1.0)),
                )
            return len(links)

    def get_chunks_for_entity(self, entity_id: int, limit: int = 50) -> list[int]:
        """Get chunk IDs linked to an entity, ordered by relevance."""
        with self.cursor() as cur:
            cur.execute(
                "SELECT chunk_id FROM entity_chunks "
                "WHERE entity_id = %s ORDER BY relevance DESC LIMIT %s",
                (entity_id, limit),
            )
            return [r[0] for r in cur.fetchall()]

    def get_entities_for_chunk(self, chunk_id: int) -> list[Entity]:
        """Get entities linked to a specific chunk."""
        with self.cursor() as cur:
            cur.execute(
                "SELECT e.id, e.name, e.entity_type, e.metadata "
                "FROM entities e "
                "JOIN entity_chunks ec ON ec.entity_id = e.id "
                "WHERE ec.chunk_id = %s "
                "ORDER BY ec.relevance DESC",
                (chunk_id,),
            )
            return [
                Entity(id=r[0], name=r[1], entity_type=r[2], metadata=r[3] or {})
                for r in cur.fetchall()
            ]

    def get_related_chunks(
        self, chunk_id: int, depth: int = 1, limit: int = 20
    ) -> list[dict]:
        """
        Get chunks related to a given chunk through the entity graph.
        Returns list of {chunk_id, path_score}.
        """
        # Step 1: entities in this chunk
        entities = self.get_entities_for_chunk(chunk_id)
        if not entities:
            return []

        entity_ids = [e.id for e in entities]
        placeholders = ",".join(str(e) for e in entity_ids)

        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT ec.chunk_id, COUNT(DISTINCT ec.entity_id) AS shared_entities,
                       AVG(ec.relevance) AS avg_relevance
                FROM entity_chunks ec
                WHERE ec.entity_id IN ({placeholders})
                  AND ec.chunk_id != %s
                GROUP BY ec.chunk_id
                ORDER BY shared_entities DESC, avg_relevance DESC
                LIMIT %s
                """,
                (chunk_id, limit),
            )
            return [
                {
                    "chunk_id": r[0],
                    "shared_count": r[1],
                    "avg_relevance": float(r[2] or 0),
                }
                for r in cur.fetchall()
            ]

    # ── Graph Statistics ───────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entities")
            entities = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM relations")
            relations = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM entity_chunks")
            links = cur.fetchone()[0]
            cur.execute(
                "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type ORDER BY COUNT(*) DESC"
            )
            type_breakdown = {r[0]: r[1] for r in cur.fetchall()} if entities else {}
            # Most connected entities
            cur.execute("""
                SELECT e.name, e.entity_type, COUNT(r.id) AS rel_count
                FROM entities e
                JOIN relations r ON r.source_entity_id = e.id OR r.target_entity_id = e.id
                GROUP BY e.id, e.name, e.entity_type
                ORDER BY rel_count DESC
                LIMIT 10
            """)
            most_connected = (
                [
                    {"name": r[0], "type": r[1], "relations": r[2]}
                    for r in cur.fetchall()
                ]
                if entities
                else []
            )

        return {
            "entities": entities,
            "relations": relations,
            "entity_chunk_links": links,
            "type_breakdown": type_breakdown,
            "most_connected": most_connected,
        }

    # ── Bulk Import ────────────────────────────────────────────

    def bulk_import(
        self,
        entities: list[dict],
        relations: list[dict],
        entity_chunk_links: list[dict],
    ) -> dict[str, int]:
        """
        Bulk import entities, relations, and links.
        entities: [{name, entity_type, metadata}]
        relations: [{source_name, target_name, relation_type, weight}]
        entity_chunk_links: [{entity_name, chunk_id, relevance}]
        """
        entity_id_map: dict[str, int] = {}
        imported = {"entities": 0, "relations": 0, "links": 0}

        for ent in entities:
            eid = self.get_or_create_entity(
                name=ent["name"],
                entity_type=ent.get("entity_type", "concept"),
                metadata=ent.get("metadata"),
            )
            entity_id_map[ent["name"]] = eid
            imported["entities"] += 1

        for rel in relations:
            src = entity_id_map.get(rel["source_name"])
            tgt = entity_id_map.get(rel["target_name"])
            if src and tgt:
                self.add_relation(
                    src,
                    tgt,
                    relation_type=rel.get("relation_type", "related"),
                    weight=rel.get("weight", 1.0),
                )
                imported["relations"] += 1

        for link in entity_chunk_links:
            eid = entity_id_map.get(link["entity_name"])
            if eid:
                self.link_entity_to_chunk(
                    eid,
                    link["chunk_id"],
                    relevance=link.get("relevance", 1.0),
                )
                imported["links"] += 1

        return imported
