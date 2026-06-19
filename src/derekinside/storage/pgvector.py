"""
derekinside — pgvector storage layer.

Manages wings/rooms/pages/chunks schema and provides:
- Full-text search (PostgreSQL tsvector)
- Vector search (cosine similarity via pgvector)
- Hybrid search (RRF fusion)
- Temporal boost for recent documents
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError:
    psycopg = None
    ConnectionPool = None

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: int
    page_id: int
    slug: str | None
    title: str | None
    source_path: str | None
    chunk_index: int
    chunk_text: str
    token_count: int | None
    score: float = 0.0
    wing_name: str = ""
    room_name: str = ""
    created_at: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "page_id": self.page_id,
            "slug": self.slug,
            "title": self.title,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "chunk_text": self.chunk_text[:500],
            "token_count": self.token_count,
            "score": round(self.score, 4),
            "wing": self.wing_name,
            "room": self.room_name,
        }


@dataclass
class WingInfo:
    id: int
    name: str
    description: str | None
    page_count: int = 0
    room_count: int = 0


@dataclass
class RoomInfo:
    id: int
    wing_id: int
    name: str
    description: str | None
    page_count: int = 0
    created_at: Any = None


@dataclass
class PageInfo:
    id: int
    room_id: int
    slug: str | None
    title: str | None
    source_path: str | None
    source_kind: str | None
    page_kind: str | None
    created_at: Any = None
    updated_at: Any = None
    chunk_count: int = 0


class VectorStore:
    """PostgreSQL + pgvector storage manager.

    Uses connection pool for concurrent access.
    Default pool: min_size=1, max_size=5.
    """

    def __init__(
        self, dsn: str, schema: str = "public",
        pool_min: int = 1, pool_max: int = 5, **kwargs
    ):
        if psycopg is None:
            raise ImportError(
                "psycopg is required. Install: pip install derekinside[pgvector]"
            )

        self._dsn = dsn
        self._schema = schema
        self._pool: Optional[ConnectionPool] = None
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._extra_conn_kwargs = kwargs

    # ── Connection Pool ─────────────────────────────────────────

    def connect(self) -> None:
        """Initialize connection pool (creates schema if needed)."""
        if self._pool is not None:
            return

        self._pool = ConnectionPool(
            conninfo=self._dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
            open=True,
            configure=lambda conn: setattr(conn, "autocommit", True),
            **self._extra_conn_kwargs,
        )

        # Ensure schema exists
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET search_path TO {self._schema}")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    @property
    def conn(self) -> psycopg.Connection:
        """Get a connection from pool. Returns context-managed connection.

        Usage:
            with store.conn as conn:
                with conn.cursor() as cur:
                    ...
        """
        if self._pool is None:
            self.connect()
        return self._pool.connection()

    def cursor(self):
        """Shortcut: borrow connection + get cursor in one context.

        Usage:
            with store.cursor() as cur:
                cur.execute(...)
        """
        from contextlib import contextmanager

        @contextmanager
        def _cursor():
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    yield cur

        return _cursor()

    # ── Schema ─────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        """Create tables/indexes if they don't exist."""
        with self.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS wings (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rooms (
                    id SERIAL PRIMARY KEY,
                    wing_id INTEGER REFERENCES wings(id),
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(wing_id, name)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pages (
                    id SERIAL PRIMARY KEY,
                    room_id INTEGER REFERENCES rooms(id),
                    slug TEXT,
                    title TEXT,
                    source_path TEXT,
                    source_kind TEXT,
                    page_kind TEXT,
                    created_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ,
                    metadata JSONB DEFAULT '{}'
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    page_id INTEGER REFERENCES pages(id),
                    chunk_index INTEGER,
                    chunk_text TEXT,
                    token_count INTEGER,
                    embedding vector(1024),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Vector index (IVFFlat — good enough for ~2k chunks)
            cur.execute("""
                DO $$ BEGIN
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                EXCEPTION WHEN OTHERS THEN
                    NULL;
                END $$;
            """)
            # tsvector index for hybrid search
            cur.execute("""
                DO $$ BEGIN
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', coalesce(chunk_text, ''))) STORED;
                EXCEPTION WHEN OTHERS THEN NULL;
                END $$;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON chunks(page_id);
            """)

    # ── Wings ──────────────────────────────────────────────────

    def get_or_create_wing(self, name: str, description: str = "") -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO wings (name, description) VALUES (%s, %s) "
                "ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name RETURNING id",
                (name, description),
            )
            return cur.fetchone()[0]

    def list_wings(self) -> list[WingInfo]:
        with self.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT w.id, w.name, w.description,
                       COUNT(DISTINCT r.id) AS room_count,
                       COUNT(DISTINCT p.id) AS page_count
                FROM wings w
                LEFT JOIN rooms r ON r.wing_id = w.id
                LEFT JOIN pages p ON p.room_id = r.id
                GROUP BY w.id ORDER BY w.name
            """)
            return [WingInfo(**r) for r in cur.fetchall()]

    # ── Rooms ──────────────────────────────────────────────────

    def get_or_create_room(self, wing_id: int, name: str, description: str = "") -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO rooms (wing_id, name, description) VALUES (%s, %s, %s) "
                "ON CONFLICT (wing_id, name) DO UPDATE SET name=EXCLUDED.name RETURNING id",
                (wing_id, name, description),
            )
            return cur.fetchone()[0]

    def list_rooms(self, wing_id: Optional[int] = None) -> list[RoomInfo]:
        with self.cursor(row_factory=dict_row) as cur:
            if wing_id:
                cur.execute(
                    "SELECT r.*, COUNT(p.id) AS page_count FROM rooms r "
                    "LEFT JOIN pages p ON p.room_id = r.id "
                    "WHERE r.wing_id = %s GROUP BY r.id ORDER BY r.name",
                    (wing_id,),
                )
            else:
                cur.execute(
                    "SELECT r.*, COUNT(p.id) AS page_count FROM rooms r "
                    "LEFT JOIN pages p ON p.room_id = r.id "
                    "GROUP BY r.id ORDER BY r.name"
                )
            return [RoomInfo(**r) for r in cur.fetchall()]

    # ── Pages ──────────────────────────────────────────────────

    def insert_page(
        self,
        room_id: int,
        slug: str = "",
        title: str = "",
        source_path: str = "",
        source_kind: str = "file",
        page_kind: str = "doc",
        metadata: Optional[dict] = None,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO pages (room_id, slug, title, source_path, source_kind, page_kind, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (
                    room_id,
                    slug,
                    title,
                    source_path,
                    source_kind,
                    page_kind,
                    psycopg.types.json.Json(metadata or {}) if psycopg else "{}",
                ),
            )
            return cur.fetchone()[0]

    def list_pages(
        self, room_id: Optional[int] = None, limit: int = 100, offset: int = 0
    ) -> list[PageInfo]:
        with self.cursor(row_factory=dict_row) as cur:
            if room_id:
                cur.execute(
                    "SELECT p.*, (SELECT COUNT(*) FROM chunks WHERE page_id = p.id) AS chunk_count "
                    "FROM pages p WHERE p.room_id = %s ORDER BY p.id LIMIT %s OFFSET %s",
                    (room_id, limit, offset),
                )
            else:
                cur.execute(
                    "SELECT p.*, (SELECT COUNT(*) FROM chunks WHERE page_id = p.id) AS chunk_count "
                    "FROM pages p ORDER BY p.id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
            return [PageInfo(**r) for r in cur.fetchall()]

    # ── Chunks ─────────────────────────────────────────────────

    def insert_chunk(
        self,
        page_id: int,
        chunk_index: int,
        chunk_text: str,
        token_count: int = 0,
        embedding: Optional[list[float]] = None,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO chunks (page_id, chunk_index, chunk_text, token_count, embedding)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (page_id, chunk_index, chunk_text, token_count, embedding),
            )
            return cur.fetchone()[0]

    def insert_chunks_batch(self, chunks: list[dict]) -> int:
        """Batch insert chunks. Each dict: page_id, chunk_index, chunk_text, token_count, embedding."""
        if not chunks:
            return 0
        with self.cursor() as cur:
            records = [
                (
                    c["page_id"],
                    c["chunk_index"],
                    c["chunk_text"],
                    c.get("token_count", 0),
                    c.get("embedding"),
                )
                for c in chunks
            ]
            cur.executemany(
                """INSERT INTO chunks (page_id, chunk_index, chunk_text, token_count, embedding)
                   VALUES (%s, %s, %s, %s, %s)""",
                records,
            )
        return len(chunks)

    # ── Search ─────────────────────────────────────────────────

    def search_vector(
        self,
        embedding: list[float],
        top_k: int = 20,
        wing: Optional[str] = None,
        room: Optional[str] = None,
    ) -> list[SearchResult]:
        """Pure vector (cosine) search."""
        embed_str = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
        where = ""
        params: list = []
        if wing:
            where = " AND w.name = %s"
            params.append(wing)
        if room:
            where += " AND r.name = %s"
            params.append(room)

        sql = f"""
            SELECT c.id AS chunk_id, p.id AS page_id, p.slug, p.title, p.source_path,
                   c.chunk_index, c.chunk_text, c.token_count,
                   1 - (c.embedding <=> %s::vector) AS score,
                   w.name AS wing_name, r.name AS room_name, p.created_at
            FROM chunks c
            JOIN pages p ON p.id = c.page_id
            JOIN rooms r ON r.id = p.room_id
            JOIN wings w ON w.id = r.wing_id
            WHERE c.embedding IS NOT NULL {where}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        full_params = [embed_str] + params + [embed_str, top_k]

        with self.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, full_params)
            return [SearchResult(**r) for r in cur.fetchall()]

    def search_keyword(
        self,
        query: str,
        top_k: int = 20,
        wing: Optional[str] = None,
        room: Optional[str] = None,
    ) -> list[SearchResult]:
        """Full-text search via tsvector."""
        where = ""
        params: list = []
        if wing:
            where += " AND w.name = %s"
            params.append(wing)
        if room:
            where += " AND r.name = %s"
            params.append(room)

        sql = f"""
            SELECT c.id AS chunk_id, p.id AS page_id, p.slug, p.title, p.source_path,
                   c.chunk_index, c.chunk_text, c.token_count,
                   ts_rank(c.tsv, plainto_tsquery(%s)) AS score,
                   w.name AS wing_name, r.name AS room_name, p.created_at
            FROM chunks c
            JOIN pages p ON p.id = c.page_id
            JOIN rooms r ON r.id = p.room_id
            JOIN wings w ON w.id = r.wing_id
            WHERE c.tsv @@ plainto_tsquery(%s) {where}
            ORDER BY score DESC
            LIMIT %s
        """
        full_params = [query, query] + params + [top_k]

        with self.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, full_params)
            return [SearchResult(**r) for r in cur.fetchall()]

    def search_hybrid(
        self,
        query: str,
        embedding: list[float],
        top_k: int = 20,
        wing: Optional[str] = None,
        room: Optional[str] = None,
        temporal_boost: bool = False,
        recent_days: int = 7,
        recent_weight: float = 1.5,
    ) -> list[SearchResult]:
        """
        Hybrid search: vector + keyword (RRF fusion), optional temporal boost.

        RRF: score = 1/(k + rank_vector) + 1/(k + rank_keyword)
        Temporal boost: if created_at within recent_days, multiply score by recent_weight
        """
        k = 60  # RRF constant

        vec_results = self.search_vector(embedding, top_k * 2, wing, room)
        kw_results = self.search_keyword(query, top_k * 2, wing, room)

        # Build rank maps {chunk_id: rank}
        vec_ranks = {r.chunk_id: i + 1 for i, r in enumerate(vec_results)}
        kw_ranks = {r.chunk_id: i + 1 for i, r in enumerate(kw_results)}

        results_map: dict[int, SearchResult] = {}

        # Best result objects
        for r in vec_results + kw_results:
            if r.chunk_id not in results_map:
                results_map[r.chunk_id] = r

        # Compute RRF scores
        import datetime

        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff = now - datetime.timedelta(days=recent_days)

        for cid, result in results_map.items():
            v_rank = vec_ranks.get(cid, len(vec_results) + 1)
            k_rank = kw_ranks.get(cid, len(kw_results) + 1)
            score = 1.0 / (k + v_rank) + 1.0 / (k + k_rank)

            # Temporal boost
            if temporal_boost and result.created_at:
                try:
                    if (
                        hasattr(result.created_at, "tzinfo")
                        and result.created_at.tzinfo is not None
                    ):
                        result_ts = result.created_at
                    else:
                        result_ts = result.created_at.replace(
                            tzinfo=datetime.timezone.utc
                        )
                    if result_ts >= cutoff:
                        score *= recent_weight
                except (ValueError, AttributeError):
                    pass

            result.score = score

        # Sort by score desc
        ranked = sorted(results_map.values(), key=lambda r: -r.score)
        return ranked[:top_k]

    # ── Stats ──────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM wings")
            wings = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM rooms")
            rooms = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM pages")
            pages = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks")
            chunks = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
            embedded = cur.fetchone()[0]
            cur.execute(
                "SELECT w.name, COUNT(DISTINCT r.id) rooms, COUNT(DISTINCT p.id) pages "
                "FROM wings w JOIN rooms r ON r.wing_id=w.id "
                "JOIN pages p ON p.room_id=r.id "
                "GROUP BY w.id, w.name ORDER BY w.name"
            )
            wing_breakdown = {
                r[0]: {"rooms": r[1], "pages": r[2]} for r in cur.fetchall()
            }

        return {
            "wings": wings,
            "rooms": rooms,
            "pages": pages,
            "chunks": chunks,
            "embedded_chunks": embedded,
            "embedding_percent": (embedded * 100 // chunks) if chunks else 0,
            "wing_breakdown": wing_breakdown,
        }
