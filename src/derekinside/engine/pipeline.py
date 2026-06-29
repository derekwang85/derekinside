"""
derekinside — Pipeline Resolver.

Multi-dimensional constraint solver for model selection.
Not a linear fallback chain — selects models based on:
  - Intelligence required (min_intelligence)
  - Cost budget (max_cost)
  - Latency ceiling (max_latency_ms)
  - Health status
Then ranks by objective (optimize_quality|speed|cost).
"""

from __future__ import annotations

import logging

from derekinside.engine.model import (
    ModelEndpoint,
    NoModelSatisfies,
    intel_rank,
    cost_rank,
)

logger = logging.getLogger(__name__)
from derekinside.engine.registry import ModelRegistry


class PipelineResolver:
    """
    Constraint-based model selector for a given capability.

    Usage:
        resolver = PipelineResolver(registry, config["pipeline"])
        model = resolver.select("embed")
        result = model.invoke("embed", text="hello")
    """

    def __init__(self, registry: ModelRegistry, pipeline_config: dict):
        self._registry = registry
        self._pipelines = pipeline_config or {}

    # ── Public API ──

    def select(self, capability: str, context: dict | None = None) -> ModelEndpoint:
        """
        Select the best model for a given capability.

        1. Read pipeline config for this capability.
        2. Filter candidates by constraints (intelligence, cost, latency, health).
        3. Relax constraints if no model matches (graceful degradation).
        4. Rank by objective and return the best.
        """
        pl = self._pipelines.get(capability)
        if not pl:
            raise NoModelSatisfies(f"No pipeline configured for '{capability}'")

        candidates = pl.get("candidates", [])
        requires = pl.get("requires", {})
        objective = pl.get("objective", "optimize_quality")

        # Step 1: Strict constraint filtering
        eligible = self._filter(candidates, requires, strict=True)
        if eligible:
            return self._rank(eligible, objective)[0]

        # Step 2: Relaxed constraint filtering (degradation)
        logger.info(
            "No model satisfies strict constraints for '%s', relaxing", capability
        )
        relaxed = self._relax(requires)
        eligible = self._filter(candidates, relaxed, strict=False)
        if eligible:
            logger.warning(
                "Using relaxed constraints for '%s' — quality may degrade", capability
            )
            return self._rank(eligible, objective)[0]

        raise NoModelSatisfies(
            f"No available model for '{capability}' "
            f"(candidates={candidates}, requires={requires})"
        )

    def select_batch(self, capability: str, count: int = 1) -> list[ModelEndpoint]:
        """
        Select multiple models (for parallel execution).
        Returns top-N models sorted by objective.
        """
        pl = self._pipelines.get(capability)
        if not pl:
            raise NoModelSatisfies(f"No pipeline configured for '{capability}'")

        candidates = pl.get("candidates", [])
        requires = pl.get("requires", {})
        objective = pl.get("objective", "optimize_quality")

        eligible = self._filter(candidates, requires, strict=True)
        if not eligible:
            relaxed = self._relax(requires)
            eligible = self._filter(candidates, relaxed, strict=False)

        if not eligible:
            raise NoModelSatisfies(f"No available models for '{capability}'")

        ranked = self._rank(eligible, objective)
        return ranked[:count]

    # ── Filtering ──

    def _filter(
        self, candidate_names: list[str], requires: dict, strict: bool
    ) -> list[ModelEndpoint]:
        """Filter candidates by constraints."""
        eligible = []
        for name in candidate_names:
            try:
                model = self._registry.get(name)
            except KeyError:
                continue

            if self._satisfies(model, requires):
                eligible.append(model)
        return eligible

    def _satisfies(self, model: ModelEndpoint, requires: dict) -> bool:
        """Check if model satisfies all constraints."""
        # Always check health first
        try:
            h = model.health()
            if h.get("status") != "ok":
                return False
            actual_latency = h.get("latency_ms", 99999)
        except Exception:
            return False

        if model.profile:
            # Override profile latency with actual health check latency
            self._update_latency_from_health(model, actual_latency)
            try:
                return model.profile.satisfies(requires)
            except Exception:
                pass

        return True

    def _update_latency_from_health(self, model, actual_ms):
        """Update profile latency if health check returned a real value."""
        if model.profile and actual_ms and actual_ms < 99999:
            model.profile.avg_latency_ms = min(
                model.profile.avg_latency_ms,
                actual_ms,
            )

    def _relax(self, requires: dict) -> dict:
        """Relax constraints progressively. Only relax fields that were in original."""
        relaxed = dict(requires)
        # Relax intelligence: only if originally specified
        if "min_intelligence" in relaxed:
            current = relaxed["min_intelligence"]
            order = ["very_high", "high", "medium", "low", "n/a"]
            idx = order.index(current) if current in order else len(order) - 1
            if idx < len(order) - 1:
                relaxed["min_intelligence"] = order[idx + 1]
        # Relax latency: double
        if "max_latency_ms" in relaxed:
            relaxed["max_latency_ms"] = relaxed["max_latency_ms"] * 3
        # Relax cost: one tier up
        if "max_cost" in relaxed:
            cost_order = ["free", "paid_low", "paid_high"]
            current_cost = relaxed["max_cost"]
            idx = cost_order.index(current_cost) if current_cost in cost_order else 0
            if idx < 2:
                relaxed["max_cost"] = cost_order[idx + 1]
        return relaxed

    # ── Ranking ──

    def _rank(self, models: list[ModelEndpoint], objective: str) -> list[ModelEndpoint]:
        """Sort by objective: optimize_speed, optimize_quality, optimize_cost."""
        if not models:
            return models

        if objective == "optimize_speed":
            models.sort(key=lambda m: m.profile.avg_latency_ms if m.profile else 9999)
        elif objective == "optimize_quality":
            models.sort(key=lambda m: -intel_rank(m.intelligence) if m.profile else 0)
        elif objective == "optimize_cost":
            models.sort(key=lambda m: cost_rank(m.cost_tier) if m.profile else 0)
        return models
