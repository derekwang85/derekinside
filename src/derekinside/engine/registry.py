"""
derekinside — ModelRegistry.

Central registry for all model endpoints.
Handles: registration, discovery, health, auto-profile on first use.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from derekinside.engine.model import (
    ModelEndpoint,
    ModelProfile,
)
from derekinside.drivers.ollama import OllamaModel
from derekinside.drivers.vllm import VLLMModel
from derekinside.drivers.openai import OpenAIModel

logger = logging.getLogger(__name__)

# ── Driver map ──

_DRIVERS = {
    "ollama": OllamaModel,
    "vllm": VLLMModel,
    "openai": OpenAIModel,
}

# ── Cache path for model profiles ──

_PROFILE_CACHE_DIR = Path.home() / ".derekinside" / "model-profiles"


class ModelRegistry:
    """
    Central registry for all model endpoints.

    - Loads models from config.yaml models section.
    - Each model gets a driver instance (OllamaModel, VLLMModel, etc.).
    - Supports auto-profiling on first use (delegated to ModelProfiler).
    - Thread-safe.
    """

    def __init__(self, config: dict):
        self._lock = threading.Lock()
        self._models: dict[str, ModelEndpoint] = {}
        self._profiler = None  # set externally by DereContext
        self._init_from_config(config.get("models", {}))
        self._load_profiles()

    def set_profiler(self, profiler) -> None:
        """Attach the ModelProfiler (external to avoid circular import)."""
        self._profiler = profiler

    # ── Initialization ──

    def _init_from_config(self, models_config: dict) -> None:
        """Create ModelEndpoint instances from config."""
        for name, cfg in models_config.items():
            driver_name = cfg.get("driver", "")
            driver_cls = _DRIVERS.get(driver_name)
            if not driver_cls:
                logger.warning(
                    "Unknown driver '%s' for model '%s', skipping", driver_name, name
                )
                continue
            with self._lock:
                self._models[name] = driver_cls(name, cfg)

    def _load_profiles(self) -> None:
        """Load cached profiles from disk."""
        if not _PROFILE_CACHE_DIR.exists():
            return
        for name, model in self._models.items():
            cache_file = _PROFILE_CACHE_DIR / f"{name}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text())
                    model.profile = ModelProfile(**data)
                except Exception as e:
                    logger.debug("Failed to load profile for %s: %s", name, e)

    def _save_profile(self, name: str) -> None:
        """Save a single model profile to disk cache."""
        model = self._models.get(name)
        if not model or not model.profile:
            return
        _PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _PROFILE_CACHE_DIR / f"{name}.json"
        try:
            cache_file.write_text(json.dumps(model.profile.__dict__, default=str))
        except Exception as e:
            logger.warning("Failed to save profile for %s: %s", name, e)

    # ── Access ──

    def get(self, name: str) -> ModelEndpoint:
        """Get a model by name. Auto-probes if needed when profiler is attached."""
        with self._lock:
            model = self._models.get(name)
            if not model:
                raise KeyError(f"Model '{name}' not registered")
        # Auto-profile: if no profile and profiler attached, probe
        if model.profile is None and self._profiler is not None:
            logger.info("Auto-profiling model '%s' on first use", name)
            profile = self._profiler.get_or_probe(name, mode="boot")
            model.profile = profile
            self._save_profile(name)
        return model

    def list(self, capability: str | None = None) -> list[ModelEndpoint]:
        """List all models, optionally filtered by capability."""
        with self._lock:
            models = list(self._models.values())
        if capability:
            models = [m for m in models if self._has_capability(m, capability)]
        return models

    def _has_capability(self, model: ModelEndpoint, capability: str) -> bool:
        """Check if model supports capability (via profile or driver)."""
        if model.profile and model.profile.capabilities:
            return model.profile.capabilities.get(capability, False)
        return True  # unknown = assume capable (will fail at invoke if not)

    def health(self, name: str) -> dict:
        """Health check for one model."""
        try:
            model = self._models[name]
            return model.health()
        except KeyError:
            return {"status": "unknown", "error": f"Model '{name}' not found"}

    def health_all(self) -> dict[str, dict]:
        """Health check for all models."""
        results = {}
        for name in list(self._models.keys()):
            results[name] = self.health(name)
        return results

    def refresh_profile(self, name: str, profile: ModelProfile) -> None:
        """Update a model's profile and persist."""
        with self._lock:
            model = self._models.get(name)
            if model:
                model.profile = profile
                self._save_profile(name)

    def close(self) -> None:
        """Close all model connections."""
        for model in self._models.values():
            try:
                model.close()
            except Exception:
                pass
