"""
derekinside — ModelEndpoint.

Unified interface for all AI model endpoints.
A model = a driver (transport) + an api_model (model name) + capabilities.

Attributes include the four-dimensional profile:
  - intelligence: low | medium | high | very_high
  - cost_tier:    free | paid_low | paid_high
  - speed_tier:   fast | medium | slow | batch
  - quality:      low | medium | high | very_high
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Enums for the four-dimensional model profile ──

INTELLIGENCE_ORDER = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
COST_ORDER = {"free": 0, "paid_low": 1, "paid_high": 2}
SPEED_ORDER = {"batch": 0, "slow": 1, "medium": 2, "fast": 3}
QUALITY_ORDER = {"low": 0, "medium": 1, "high": 2, "very_high": 3}


def intel_rank(val: str) -> int:
    return INTELLIGENCE_ORDER.get(val, 0)


def cost_rank(val: str) -> int:
    return COST_ORDER.get(val, 0)


def speed_rank(val: str) -> int:
    return SPEED_ORDER.get(val, 0)


def quality_rank(val: str) -> int:
    return QUALITY_ORDER.get(val, 0)


# ── Capability Not Supported Error ──


class CapabilityNotSupported(NotImplementedError):
    """Raised when a model does not support the requested capability."""


class NoModelSatisfies(RuntimeError):
    """Raised when no model can satisfy the pipeline constraints."""


class AllModelsDown(RuntimeError):
    """Raised when all models in a fallback chain are unhealthy."""


# ── Model Profile (four-dimensional + metadata) ──


@dataclass
class ModelProfile:
    """Auto-detected or user-specified model profile."""

    capabilities: dict[str, bool] = field(default_factory=dict)
    intelligence: str = "medium"
    intelligence_confidence: float = 0.5
    speed_tier: str = "medium"
    avg_latency_ms: float = 1000.0
    quality: str = "medium"
    cost_tier: str = "free"
    probe_count: int = 0
    last_probed_at: float = 0.0
    oscillating: bool = False
    history: list[dict] = field(default_factory=list)

    def satisfies(self, requires: dict) -> bool:
        """Check if this profile satisfies the pipeline constraints."""
        min_intel = requires.get("min_intelligence", "low")
        max_cost = requires.get("max_cost", "free")
        max_latency = requires.get("max_latency_ms", 99999)

        if intel_rank(self.intelligence) < intel_rank(min_intel):
            return False
        if cost_rank(self.cost_tier) > cost_rank(max_cost):
            return False
        if self.avg_latency_ms > max_latency:
            return False
        if self.oscillating:
            return False
        return True

    @classmethod
    def from_config(cls, cfg: dict) -> ModelProfile:
        """Build a profile from user config (or empty if auto-detect).
        Sets avg_latency_ms based on speed_tier when not provided."""
        speed = cfg.get("speed_tier", "medium")
        latency_hints = {"fast": 50, "medium": 1000, "slow": 5000, "batch": 15000}
        # Parse capabilities from config (supports both list and dict forms)
        raw_caps = cfg.get("capabilities", [])
        if isinstance(raw_caps, list):
            capabilities = {c: True for c in raw_caps}
        elif isinstance(raw_caps, dict):
            capabilities = raw_caps
        else:
            capabilities = {}
        return cls(
            capabilities=capabilities,
            intelligence=cfg.get("intelligence", "medium"),
            intelligence_confidence=1.0,  # user-specified = trusted
            speed_tier=speed,
            avg_latency_ms=cfg.get("avg_latency_ms", latency_hints.get(speed, 1000)),
            quality=cfg.get("quality", "medium"),
            cost_tier=cfg.get("cost_tier", "free"),
            probe_count=999,  # user-specified → never auto-probe
        )


# ── Model Endpoint (unified abstraction) ──


class ModelEndpoint(ABC):
    """
    Unified interface for one named AI model endpoint.

    - One driver implementation per API protocol (Ollama, vLLM, OpenAI).
    - One ModelEndpoint instance = one (url, api_model) pair.
    - Capabilities are discovered at registration or via profiling.

    Subclasses must override invoke() and may override health().
    """

    def __init__(self, name: str, config: dict):
        self._name = name
        self._config = config
        # If user provided explicit four-dimensional attributes, create profile immediately
        has_user_profile = any(
            k in config for k in ["intelligence", "cost_tier", "speed_tier", "quality"]
        )
        if has_user_profile:
            self._profile = ModelProfile.from_config(config)
        else:
            self._profile = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> ModelProfile | None:
        return self._profile

    @profile.setter
    def profile(self, p: ModelProfile) -> None:
        self._profile = p

    @property
    def intelligence(self) -> str:
        return self._profile.intelligence if self._profile else "medium"

    @property
    def cost_tier(self) -> str:
        return self._profile.cost_tier if self._profile else "free"

    @property
    def speed_tier(self) -> str:
        return self._profile.speed_tier if self._profile else "medium"

    @abstractmethod
    def invoke(self, capability: str, **kwargs) -> Any:
        """
        Call the model with the given capability.

        capability: "embed" | "extract" | "rerank" | "generate"
        kwargs depend on capability.

        Raises CapabilityNotSupported if the model doesn't support this capability.
        """
        ...

    def health(self) -> dict:
        """
        Quick health check. Uses a minimal invoke on a known capability.
        Tries embed first (most models support it), then extract.
        """
        import time

        start = time.time()
        try:
            # Try the model's primary capability
            caps = self._profile.capabilities if self._profile else {}
            if caps.get("embed"):
                self.invoke("embed", text="test")
            elif caps.get("extract"):
                self.invoke("extract", text="hello world")
            else:
                # Unknown: try embed, fallback to extract
                try:
                    self.invoke("embed", text="test")
                except CapabilityNotSupported:
                    self.invoke("extract", text="hello world")
            elapsed = int((time.time() - start) * 1000)
            return {"status": "ok", "latency_ms": elapsed, "name": self._name}
        except CapabilityNotSupported:
            return {
                "status": "ok",
                "latency_ms": 0,
                "name": self._name,
                "note": "no supported capability checked",
            }
        except Exception as e:
            return {"status": "down", "error": str(e)[:200], "name": self._name}

    def close(self) -> None:
        """Release resources. Subclasses may override."""
