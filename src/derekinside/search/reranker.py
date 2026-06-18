"""
derekinside — LLM reranker.

Takes top-k search results and re-ranks using an LLM via Ollama.
Improves relevance for complex or domain-specific queries.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from derekinside.storage.pgvector import SearchResult

logger = logging.getLogger(__name__)

_RERANK_PROMPT = """You are a relevance judge. Given a query and a list of text passages, score each passage 0-10 for relevance to the query.

Query: {query}

Passages:
{passages}

Return a JSON array of scores, one per passage, in order. Example: [8, 3, 9, 1]
Only return the JSON array, no explanation."""


class Reranker:
    """LLM-based reranker using Ollama."""

    def __init__(
        self,
        url: str = "http://localhost:11434/api/generate",
        model: str = "qwen2.5-coder:7b",
        enabled: bool = False,
    ):
        self._url = url
        self._model = model
        self._enabled = enabled
        self._client = httpx.Client(timeout=60.0)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: Optional[int] = None
    ) -> list[SearchResult]:
        """
        Rerank results using LLM scoring.
        Returns top_k results sorted by LLM score.
        """
        if not self._enabled or not results:
            return results

        if top_k is None:
            top_k = len(results)

        # Build passage list (truncate each to 500 chars)
        passages = []
        for i, r in enumerate(results):
            text = r.chunk_text[:500].replace("\n", " ")
            passages.append(f"[{i}] {text}")

        prompt = _RERANK_PROMPT.format(
            query=query,
            passages="\n\n".join(passages),
        )

        try:
            t0 = time.time()
            resp = self._client.post(
                self._url,
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 256, "temperature": 0.1},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "").strip()
            elapsed = time.time() - t0
            logger.info("Reranker took %.1fs for %d results", elapsed, len(results))

            # Parse JSON scores
            import json as _json

            # Extract JSON array from response (may be wrapped in markdown)
            json_str = raw
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
            if raw.startswith("[") and not raw.startswith("```"):
                json_str = raw

            scores = _json.loads(json_str)
            if not isinstance(scores, list) or len(scores) != len(results):
                logger.warning("Reranker returned unexpected format: %s", raw[:200])
                return results[:top_k]

            # Apply scores
            for r, s in zip(results, scores):
                r.score = float(s) / 10.0

            # Sort by new score
            results.sort(key=lambda r: -r.score)

        except Exception as e:
            logger.warning("Reranker failed: %s — falling back to original order", e)

        return results[:top_k]

    def close(self) -> None:
        self._client.close()
