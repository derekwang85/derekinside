"""
derekinside — Consensus Engine (自主学习).

Multi-model cross-validation for entity extraction.
Instead of relying on user feedback, multiple extraction modes cross-validate each other.

Key insight from LongMemEval data:
  1.5b and 7b have only ~40% entity overlap
  → They are complementary, not competing
  → Consensus across modes is a strong signal for entity quality

Flow:
  1. Run multiple extraction modes on the same chunk
  2. Evaluate each detected entity's consensus score
  3. auto_confirm (>=0.65) → store with full weight
  4. uncertain (0.35-0.65) → mark for review
  5. suspicious (0.15-0.35) → store with reduced weight
  6. auto_reject (<0.15) → discard
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from derekinside.storage.graph import KnowledgeGraph
from derekinside.engine.engine import Engine

logger = logging.getLogger(__name__)

# ── Mode weights (from LongMemEval precision data) ──

MODE_WEIGHTS: dict[str, float] = {
    "regex": 1.0,          # 精确率 95.6%
    "hybrid-7b": 0.9,      # 精确率 77.4%
    "hybrid-1.5b": 0.8,    # 精确率 72.4%
    "7b": 0.7,             # 精确率 59.0%
    "1.5b": 0.5,           # 71.7% 召回高但 28.3% 噪音
    "gpt-4o-mini": 0.95,   # 云端信任
}

CONSENSUS_THRESHOLDS = {
    "auto_confirm": 0.65,
    "uncertain_low": 0.35,
    "suspicious_low": 0.15,
}


@dataclass
class ConsensusVerdict:
    entity_name: str
    entity_type: str
    consensus_score: float
    total_weight: float
    confirmed_by: list[str]
    rejected_by: list[str]
    status: str  # confirmed | uncertain | suspicious | rejected


@dataclass
class ConsensusResult:
    verdicts: list[ConsensusVerdict] = field(default_factory=list)
    runtime_ms: float = 0.0
    learned_entities: int = 0
    rejected_entities: int = 0


class ConsensusEngine:
    """
    Multi-model cross-validation consensus engine.
    Re-extracts edge-case entities with multiple modes to determine quality.
    """

    def __init__(self, engine: Engine, graph: KnowledgeGraph):
        self._engine = engine
        self._graph = graph

    def ensure_schema(self) -> None:
        with self._graph.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entity_learning (
                    entity_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT 'concept',
                    consensus_score REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'unknown',
                    confirmed_by TEXT[] DEFAULT '{}',
                    rejected_by TEXT[] DEFAULT '{}',
                    last_evaluated_at TIMESTAMPTZ DEFAULT NOW(),
                    eval_count INTEGER DEFAULT 1,
                    PRIMARY KEY (entity_name, entity_type)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_learning_status ON entity_learning(status)")

    def cross_validate(self, detections: list[dict],
                       available_modes: list[str] | None = None) -> list[ConsensusVerdict]:
        """Cross-validate entity detections across available modes."""
        if not detections:
            return []

        grouped: dict[str, dict] = {}
        all_modes = list(MODE_WEIGHTS.keys())
        if available_modes:
            all_modes = [m for m in all_modes if m in available_modes]
        total_weight = sum(MODE_WEIGHTS.get(m, 0.5) for m in all_modes)

        for d in detections:
            name = d["name"]
            etype = d.get("type", "concept")
            mode = d.get("mode", "unknown")
            if name not in grouped:
                grouped[name] = {"type": etype, "found_by": set(), "not_found_by": set()}
            grouped[name]["found_by"].add(mode)

        for name, g in grouped.items():
            for m in all_modes:
                if m not in g["found_by"]:
                    g["not_found_by"].add(m)

        verdicts = []
        for name, g in grouped.items():
            score = sum(MODE_WEIGHTS.get(m, 0.5) for m in g["found_by"]) / max(total_weight, 0.01)
            if score >= CONSENSUS_THRESHOLDS["auto_confirm"]:
                status = "confirmed"
            elif score >= CONSENSUS_THRESHOLDS["uncertain_low"]:
                status = "uncertain"
            elif score >= CONSENSUS_THRESHOLDS["suspicious_low"]:
                status = "suspicious"
            else:
                status = "rejected"
            verdicts.append(ConsensusVerdict(
                entity_name=name, entity_type=g["type"],
                consensus_score=score, total_weight=total_weight,
                confirmed_by=sorted(g["found_by"]),
                rejected_by=sorted(g["not_found_by"]),
                status=status,
            ))

        verdicts.sort(key=lambda v: -v.consensus_score)
        return verdicts

    def learn_from_graph(self, sample_count: int = 200,
                         dry_run: bool = True) -> ConsensusResult:
        """Re-extract edge entities to cross-validate."""
        result = ConsensusResult()
        start = time.time()

        with self._graph.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.name, e.entity_type, COUNT(ec.id) as links
                FROM entities e
                LEFT JOIN entity_chunks ec ON ec.entity_id = e.id
                GROUP BY e.id, e.name, e.entity_type
                HAVING COUNT(ec.id) <= 3
                ORDER BY COUNT(ec.id) ASC LIMIT %s
            """, (sample_count,))
            edge_entities = cur.fetchall()

        for eid, ename, etype, _ in edge_entities:
            with self._graph.cursor() as cur:
                cur.execute("""
                    SELECT c.chunk_text FROM chunks c
                    JOIN entity_chunks ec ON ec.chunk_id = c.id
                    WHERE ec.entity_id = %s LIMIT 2
                """, (eid,))
                chunks = [r[0] for r in cur.fetchall() if r[0]]

            if not chunks:
                continue

            detections = []
            for chunk_text in chunks:
                try:
                    extraction = self._engine.extract(chunk_text)
                    entities = (extraction.get("entities", [])
                                if isinstance(extraction, dict) else [])
                    found = any(
                        (isinstance(e, dict) and e.get("name") == ename)
                        or (hasattr(e, "name") and e.name == ename)
                        for e in entities
                    )
                    detections.append({"name": ename, "type": etype,
                                       "mode": "current_pipeline", "found": found})
                except Exception:
                    detections.append({"name": ename, "type": etype,
                                       "mode": "current_pipeline", "found": False})

            verdicts = self.cross_validate(detections)
            for v in verdicts:
                if v.entity_name == ename:
                    result.verdicts.append(v)
                    if v.status == "rejected":
                        result.rejected_entities += 1
                    else:
                        result.learned_entities += 1
                    if not dry_run:
                        self._store_verdict(v)

        result.runtime_ms = (time.time() - start) * 1000
        return result

    def _store_verdict(self, v: ConsensusVerdict) -> None:
        with self._graph.cursor() as cur:
            cur.execute("""
                INSERT INTO entity_learning
                    (entity_name, entity_type, consensus_score, status,
                     confirmed_by, rejected_by, last_evaluated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (entity_name, entity_type) DO UPDATE SET
                    consensus_score = EXCLUDED.consensus_score,
                    status = EXCLUDED.status,
                    confirmed_by = EXCLUDED.confirmed_by,
                    rejected_by = EXCLUDED.rejected_by,
                    last_evaluated_at = NOW(),
                    eval_count = entity_learning.eval_count + 1
            """, (v.entity_name, v.entity_type, v.consensus_score,
                  v.status, v.confirmed_by, v.rejected_by))

    def filter_rejected(self, entities: list[dict]) -> list[dict]:
        """Filter out rejected entities (used during graph_build)."""
        if not entities:
            return entities
        rejected = set()
        with self._graph.cursor() as cur:
            cur.execute("SELECT entity_name FROM entity_learning WHERE status = 'rejected'")
            rejected = {r[0].lower() for r in cur.fetchall()}
        if not rejected:
            return entities
        return [
            e for e in entities
            if (e.get("name", "") if isinstance(e, dict)
                else getattr(e, "name", "")).lower() not in rejected
        ]
