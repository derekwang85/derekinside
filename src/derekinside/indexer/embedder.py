# DEPRECATED — Use Engine instead. Will be removed after migration.
"""
derekinside — Embedding via Ollama (bge-m3). [DEPRECATED]

Generates vector embeddings for text chunks.
Supports batching for efficiency.
"""

from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)


class Embedder:
    """Ollama-based embedding client."""

    def __init__(
        self,
        url: str = "http://localhost:11434/api/embed",
        model: str = "bge-m3",
        dimensions: int = 1024,
    ):
        self._url = url
        self._model = model
        self._dimensions = dimensions
        self._client = httpx.Client(timeout=120.0)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        resp = self._client.post(
            self._url,
            json={"model": self._model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError(f"Empty embedding response for: {text[:80]}...")
        return embeddings[0]

    def embed_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Embed multiple texts in batches."""
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = self._client.post(
                    self._url,
                    json={"model": self._model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings", [])
                results.extend(embeddings[: len(batch)])
            except Exception as e:
                logger.warning("Batch embed failed at offset %d: %s", i, e)
                # Fallback: embed one by one
                for t in batch:
                    try:
                        results.append(self.embed(t))
                    except Exception as e2:
                        logger.error("Single embed failed: %s", e2)
                        results.append([0.0] * self._dimensions)
            if i + batch_size < len(texts):
                time.sleep(0.1)  # brief cooldown
        return results

    def count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars ≈ 1 token for English)."""
        return len(text) // 4 + 1

    def close(self) -> None:
        self._client.close()
