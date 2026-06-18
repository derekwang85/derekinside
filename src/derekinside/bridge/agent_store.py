"""
derekinside — Per-agent namespace isolation.

Each agent gets its own wing namespace for isolation.
Mapping stored in agents table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from derekinside.storage.pgvector import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    agent_id: str
    name: str
    wing: str
    room: str = "memory"
    created_at: Optional[str] = None


class AgentStore:
    """Per-agent namespace isolation backed by the wing/room hierarchy."""

    def __init__(self, store: VectorStore):
        self._store = store

    @property
    def conn(self):
        return self._store.conn

    def ensure_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT '',
                    wing TEXT NOT NULL,
                    room TEXT NOT NULL DEFAULT 'memory',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

    def register_agent(self, agent_id: str, name: str = "") -> AgentInfo:
        """Register a new agent or return existing one."""
        wing = f"agent-{agent_id}"
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agents (agent_id, name, wing) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (agent_id) DO UPDATE SET name = EXCLUDED.name "
                "RETURNING agent_id, name, wing, room, created_at",
                (agent_id, name or agent_id, wing),
            )
            row = cur.fetchone()
            # Ensure wing exists
            self._store.get_or_create_wing(wing, f"Agent {name or agent_id} namespace")
            return AgentInfo(
                agent_id=row[0],
                name=row[1],
                wing=row[2],
                room=row[3],
                created_at=str(row[4]) if row[4] else None,
            )

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT agent_id, name, wing, room, created_at FROM agents WHERE agent_id = %s",
                (agent_id,),
            )
            row = cur.fetchone()
            if row:
                return AgentInfo(
                    agent_id=row[0],
                    name=row[1],
                    wing=row[2],
                    room=row[3],
                    created_at=str(row[4]) if row[4] else None,
                )
            return None

    def list_agents(self) -> list[AgentInfo]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT agent_id, name, wing, room, created_at FROM agents ORDER BY agent_id"
            )
            return [
                AgentInfo(
                    agent_id=r[0],
                    name=r[1],
                    wing=r[2],
                    room=r[3],
                    created_at=str(r[4]) if r[4] else None,
                )
                for r in cur.fetchall()
            ]
