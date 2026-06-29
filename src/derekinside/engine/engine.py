"""
derekinside — Engine facade.

Single entry point for all AI capabilities.
Wraps ModelRegistry + PipelineResolver + PassiveObserver + ModelProfiler.
Replaces the old embedder/extractor/reranker scattered initialization.
"""

from __future__ import annotations

import logging
from typing import Any

from derekinside.engine.registry import ModelRegistry
from derekinside.engine.pipeline import PipelineResolver
from derekinside.engine.profiler import ModelProfiler, PassiveObserver

logger = logging.getLogger(__name__)


class Engine:
    """
    Unified engine facade.

    Usage:
        engine = Engine(config)
        engine.start()
        emb = engine.embed("hello")
        entities = engine.extract("class Foo extends Bar")
        engine.record_usage("model-x", "embed", latency_ms=200, ...)
        engine.close()
    """

    def __init__(self, config: dict):
        self._config = config
        self._registry = ModelRegistry(config)
        self._profiler = ModelProfiler(self._registry)
        self._observer = PassiveObserver(self._profiler, self._registry)
        self._resolver = PipelineResolver(self._registry, config.get("pipeline", {}))
        self._started = False

        # Wire profiler into registry
        self._registry.set_profiler(self._profiler)

    # ── Lifecycle ──

    def start(self) -> None:
        """Initialize all models and run health checks."""
        if self._started:
            return
        self._started = True
        results = self._registry.health_all()
        healthy = sum(1 for v in results.values() if v.get("status") == "ok")
        total = len(results)
        logger.info("Engine started: %d/%d models healthy", healthy, total)

    def close(self) -> None:
        """Release all resources."""
        self._registry.close()
        self._started = False

    @property
    def healthy(self) -> int:
        return sum(
            1 for v in self._registry.health_all().values() if v.get("status") == "ok"
        )

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def profiler(self) -> ModelProfiler:
        return self._profiler

    @property
    def observer(self) -> PassiveObserver:
        return self._observer

    # ── Unified API ──

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        model = self._resolver.select("embed")
        result = model.invoke("embed", text=text)
        self._record(model.name, "embed", result)
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Uses driver's batch method for efficiency."""
        model = self._resolver.select("embed")
        # Use the driver's batch method (if available via config hint)
        if hasattr(model, "embed_batch"):
            result = model.embed_batch(texts)
        else:
            result = [model.invoke("embed", text=t) for t in texts]
        self._record(model.name, "embed", result)
        return result

    def extract(self, text: str, mode: str = "") -> dict:
        """Extract entities from text."""
        model = self._resolver.select("extract")
        result = model.invoke("extract", text=text, mode=mode)
        entity_count = len(
            result.get("entities", []) if isinstance(result, dict) else []
        )
        self._record(model.name, "extract", result, entity_count=entity_count)
        return result

    def rerank(self, query: str, chunks: list[str]) -> list[float]:
        """Re-rank chunks by relevance to query."""
        model = self._resolver.select("rerank")
        result = model.invoke("rerank", query=query, chunks=chunks)
        self._record(model.name, "rerank", result)
        return result

    def generate(self, prompt: str) -> str:
        """Generate text (generic completion)."""
        model = self._resolver.select("generate")
        result = model.invoke("generate", prompt=prompt)
        self._record(model.name, "generate", result)
        return result

    # ── Passive Observation ──

    def _record(
        self, model_name: str, capability: str, result: Any, entity_count: int = 0
    ) -> None:
        """Record side-effect data for passive observation."""
        success = result is not None and (
            not isinstance(result, dict) or "error" not in result
        )
        json_valid = isinstance(result, (dict, list))
        self._observer.record(
            model_name=model_name,
            capability=capability,
            latency_ms=0,  # caller should override with actual timing
            success=success,
            entity_count=entity_count,
            output_json_valid=json_valid,
        )

    # ── Config helpers ──

    def list_models(self, capability: str | None = None) -> list[dict]:
        """List registered models with their profiles."""
        return [
            {
                "name": m.name,
                "intelligence": m.intelligence,
                "cost_tier": m.cost_tier,
                "speed_tier": m.speed_tier,
                "capabilities": list(m.profile.capabilities.keys())
                if m.profile
                else [],
                "health": m.health().get("status", "unknown"),
            }
            for m in self._registry.list(capability)
        ]
