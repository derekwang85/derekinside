"""
derekinside — Graph propagation for re-ranking search results.

After initial hybrid/vector search, propagate scores through the
entity-relation graph to boost chunks connected to highly-relevant entities.
Inspired by HippoRAG's PageRank propagation approach.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from derekinside.storage.pgvector import SearchResult
from derekinside.storage.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def _softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax."""
    import math

    max_v = max(values) if values else 0
    exps = [math.exp((v - max_v) / temperature) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class GraphPropagator:
    """
    Graph-based score propagation for search results.

    After initial search:
    1. Find entities in top-k chunks
    2. Boost chunk scores for shared entities
    3. Propagate scores through entity relation graph
    4. Return re-ranked results with combined scores
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        enabled: bool = False,
        entity_weight: float = 0.3,  # weight of entity sharing boost
        relation_weight: float = 0.2,  # weight of relation propagation
        max_iterations: int = 3,  # PageRank iterations
        damping: float = 0.85,
    ):  # PageRank damping factor
        self._graph = graph
        self._enabled = enabled
        self._entity_weight = entity_weight
        self._relation_weight = relation_weight
        self._max_iterations = max_iterations
        self._damping = damping

    @property
    def enabled(self) -> bool:
        return self._enabled

    def propagate(
        self, query_embedding: list[float], results: list[SearchResult], top_k: int = 20
    ) -> list[SearchResult]:
        """
        Propagate search scores through the entity graph.

        1. Entity sharing boost: chunks sharing entities get score boost
        2. Graph propagation: entity scores spread through relations
        3. Re-rank by combined score
        """
        if not self._enabled or not results:
            return results

        try:
            return self._do_propagate(results, top_k)
        except Exception as e:
            logger.warning("Graph propagation failed: %s — using original results", e)
            return results[:top_k]

    def _do_propagate(
        self, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Core propagation logic."""

        # ── Phase 1: Get entities for each chunk ──
        chunk_entities: dict[int, list[dict]] = {}
        for r in results:
            try:
                ents = self._graph.get_entities_for_chunk(r.chunk_id)
                chunk_entities[r.chunk_id] = [
                    {"id": e.id, "name": e.name, "type": e.entity_type} for e in ents
                ]
            except Exception:
                chunk_entities[r.chunk_id] = []

        # ── Phase 2: Entity sharing boost ──
        # For each pair of chunks sharing entities, boost both scores
        entity_chunks: dict[int, list[int]] = defaultdict(list)
        for chunk_id, ents in chunk_entities.items():
            for e in ents:
                entity_chunks[e["id"]].append(chunk_id)

        # Build shared-entity boost map {chunk_id: boost_factor}
        shared_boost: dict[int, float] = defaultdict(float)
        for eid, linked_chunks in entity_chunks.items():
            if len(linked_chunks) > 1:
                boost = 1.0 + (len(linked_chunks) - 1) * 0.1
                for cid in linked_chunks:
                    shared_boost[cid] = max(shared_boost[cid], boost)

        # ── Phase 3: Graph propagation ──
        # Collect all unique entity IDs involved
        all_entity_ids = set()
        for ents in chunk_entities.values():
            for e in ents:
                all_entity_ids.add(e["id"])

        # Build entity graph within our subgraph
        entity_graph: dict[int, set[int]] = defaultdict(set)
        for eid in all_entity_ids:
            try:
                relations = self._graph.get_relations_for_entity(eid)
                for rel in relations:
                    neighbor = (
                        rel.target_entity_id
                        if rel.source_entity_id == eid
                        else rel.source_entity_id
                    )
                    if neighbor in all_entity_ids:
                        entity_graph[eid].add(neighbor)
            except Exception:
                pass

        # PageRank on entity subgraph
        entity_scores: dict[int, float] = {
            eid: 1.0 / len(all_entity_ids) if all_entity_ids else 0
            for eid in all_entity_ids
        }

        for _ in range(self._max_iterations):
            new_scores: dict[int, float] = {}
            for eid in all_entity_ids:
                rank_sum = 0.0
                for neighbor in entity_graph.get(eid, set()):
                    out_degree = len(entity_graph.get(neighbor, set()))
                    if out_degree > 0:
                        rank_sum += entity_scores[neighbor] / out_degree
                new_scores[eid] = (1 - self._damping) + self._damping * rank_sum
            entity_scores = new_scores

        # Normalize entity scores
        if entity_scores:
            max_score = max(entity_scores.values())
            if max_score > 0:
                entity_scores = {k: v / max_score for k, v in entity_scores.items()}

        # ── Phase 4: Combine scores ──
        for r in results:
            ents = chunk_entities.get(r.chunk_id, [])
            entity_score = 0.0
            if ents:
                entity_score = sum(entity_scores.get(e["id"], 0) for e in ents) / len(
                    ents
                )

            boost = shared_boost.get(r.chunk_id, 1.0)
            combined = (
                r.score * (1 - self._entity_weight - self._relation_weight)
                + entity_score * self._relation_weight * boost
                + (boost - 1.0) * self._entity_weight
            )
            r.score = combined

        # Sort by combined score
        results.sort(key=lambda r: -r.score)
        return results[:top_k]
