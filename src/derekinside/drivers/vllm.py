"""
derekinside — vLLM driver.

Routes embeds to a vLLM server via OpenAI-compatible Embedding API.
Primarily for GPU-accelerated embedding (bge-m3, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from derekinside.engine.model import ModelEndpoint, CapabilityNotSupported

logger = logging.getLogger(__name__)


class VLLMModel(ModelEndpoint):
    """One (url, api_model) pair on a vLLM server.
    OpenAI-compatible Embedding API for GPU-accelerated embedding.
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self._url = config.get("url", "http://localhost:8080/v1/embeddings")
        self._api_model = config.get("api_model", "BAAI/bge-m3")
        self._api_key = config.get("api_key", "")
        self._dimensions = config.get("dimensions", 1024)
        self._timeout = config.get("timeout", 60.0)
        self._batch_size = config.get("batch_size", 32)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = httpx.Client(timeout=self._timeout, headers=headers)

    def invoke(self, capability: str, **kwargs) -> Any:
        if capability == "embed":
            return self._embed(kwargs.get("text", ""))
        raise CapabilityNotSupported(f"vLLM supports 'embed' only, not '{capability}'")

    def _embed(self, text: str) -> list[float]:
        if not text.strip():
            # Return a zero vector of correct dimensions
            return [0.0] * self._dimensions
        resp = self._client.post(
            self._url,
            json={
                "model": self._api_model,
                "input": text,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("data", [{}])[0].get("embedding", [])
        if not emb:
            raise ValueError(f"Empty embedding response: {text[:80]}...")
        return emb

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding (GPU-accelerated)."""
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                resp = self._client.post(
                    self._url,
                    json={"model": self._api_model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = [d["embedding"] for d in data.get("data", [])]
                results.extend(embeddings[: len(batch)])
            except Exception as e:
                logger.warning("VLLM batch embed failed at offset %d: %s", i, e)
                # Fallback: one by one
                for t in batch:
                    try:
                        results.append(self._embed(t))
                    except Exception:
                        results.append([0.0] * self._dimensions)
        return results

    def close(self) -> None:
        self._client.close()
