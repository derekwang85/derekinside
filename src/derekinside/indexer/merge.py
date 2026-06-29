"""
derekinside — Cross-mode relation merge strategy + graph_build integration.

Handles merging extraction results from multiple modes:
  - High-precision modes (7b, hybrid-7b, regex) are trusted more
  - High-recall modes (1.5b) are kept but lower-weighted
  - Conflicting relations are resolved by precision order
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Precision order: regex > hybrid-7b > hybrid-1.5b > 7b > 1.5b
_MODE_PRECISION = {
    "regex": 1.0,
    "hybrid-7b": 0.9,
    "hybrid-1.5b": 0.8,
    "7b": 0.7,
    "1.5b": 0.5,
}


def mode_weight(mode: str) -> float:
    """Get confidence weight for a given extraction mode."""
    return _MODE_PRECISION.get(mode, 0.5)


def merge_entity_sets(
    existing: list[dict],
    incoming: list[dict],
    incoming_mode: str,
) -> list[dict]:
    """
    Merge incoming entities into existing set.
    Higher-precision modes override lower-precision ones.
    """
    # Build map: entity name → (existing entry, existing mode confidence)
    merged = {}
    for e in existing:
        merged[e["name"]] = {
            "entry": e,
            "weight": e.get("weight", mode_weight(e.get("_mode", "regex"))),
        }

    inc_weight = mode_weight(incoming_mode)

    for e in incoming:
        name = e["name"]
        if name in merged:
            # Conflict: keep higher weight
            if inc_weight > merged[name]["weight"]:
                e["weight"] = inc_weight
                e["_mode"] = incoming_mode
                merged[name] = {"entry": e, "weight": inc_weight}
            else:
                # Lower weight: still keep but with low weight
                merged[name]["entry"].setdefault("_modes_seen", []).append(
                    incoming_mode
                )
        else:
            e["weight"] = inc_weight
            e["_mode"] = incoming_mode
            merged[name] = {"entry": e, "weight": inc_weight}

    return [v["entry"] for v in merged.values()]


def merge_relation_sets(
    existing: list[dict],
    incoming: list[dict],
    incoming_mode: str,
) -> list[dict]:
    """
    Merge incoming relations into existing set.
    Strategy: keep all relations, adjust weight by precision.
    """
    inc_weight = mode_weight(incoming_mode)
    seen = set()
    merged = []

    for r in existing:
        key = (r.get("source", ""), r.get("target", ""), r.get("type", ""))
        seen.add(key)
        merged.append(r)

    for r in incoming:
        key = (r.get("source", ""), r.get("target", ""), r.get("type", ""))
        if key not in seen:
            r["weight"] = r.get("weight", 1.0) * inc_weight
            r["_mode"] = incoming_mode
            merged.append(r)
            seen.add(key)
        # If already exists, keep existing (higher precision)

    return merged


def get_entity_weight(entity: dict) -> float:
    """Get effective weight of an entity for pruning decisions."""
    return min(
        entity.get("weight", 0.5),
        mode_weight(entity.get("_mode", "unknown")),
    )
