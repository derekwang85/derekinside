"""
derekinside — Hybrid search orchestrator.

Coordinates vector, keyword, temporal, and reranker stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from derekinside.storage.pgvector import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SearchRequest:
    query: str
    embedding: list[float]
    top_k: int = 20
    wing: Optional[str] = None
    room: Optional[str] = None
    temporal_boost: bool = False
    recent_days: int = 7
    recent_weight: float = 1.5
    rerank: bool = False
    before: Optional[str] = None
    after: Optional[str] = None


@dataclass
class SearchResponse:
    results: list[SearchResult] = field(default_factory=list)
    total: int = 0
    timing: dict = field(default_factory=dict)


class HybridSearch:
    """Orchestrates multi-stage search."""

    def __init__(self, store: VectorStore, config: Optional[dict] = None):
        self._store = store
        self._config = config or {}

    def search(self, req: SearchRequest) -> SearchResponse:
        import time

        t0 = time.time()

        hybrid = self._config.get("hybrid_search", True)

        if hybrid:
            results = self._store.search_hybrid(
                query=req.query,
                embedding=req.embedding,
                top_k=req.top_k,
                wing=req.wing,
                room=req.room,
                temporal_boost=req.temporal_boost,
                recent_days=req.recent_days,
                recent_weight=req.recent_weight,
                before=req.before,
                after=req.after,
            )
        else:
            results = self._store.search_vector(
                embedding=req.embedding,
                top_k=req.top_k,
                wing=req.wing,
                room=req.room,
                before=req.before,
                after=req.after,
            )

        t1 = time.time()

        return SearchResponse(
            results=results,
            total=len(results),
            timing={
                "search_ms": round((t1 - t0) * 1000, 1),
            },
        )
