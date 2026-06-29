"""
derekinside — Ollama driver.

Routes invocations to an Ollama server via HTTP API.
Supports: embed, extract, rerank, generate.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from derekinside.engine.model import ModelEndpoint, CapabilityNotSupported

logger = logging.getLogger(__name__)


class OllamaModel(ModelEndpoint):
    """One (url, api_model) pair on an Ollama server."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self._url = config.get("url", "http://localhost:11434/api/generate")
        self._embed_url = config.get("embed_url", "http://localhost:11434/api/embed")
        self._api_model = config.get("api_model", "qwen2.5-coder:7b")
        self._timeout = config.get("timeout", 120.0)
        self._client = httpx.Client(timeout=self._timeout)

    def invoke(self, capability: str, **kwargs) -> Any:
        if capability == "embed":
            return self._embed(kwargs.get("text", ""))
        elif capability == "extract":
            return self._extract(kwargs.get("text", ""))
        elif capability == "rerank":
            return self._rerank(
                kwargs.get("query", ""),
                kwargs.get("chunks", []),
            )
        elif capability == "generate":
            return self._generate(kwargs.get("prompt", ""))
        raise CapabilityNotSupported(f"Ollama does not support '{capability}'")

    # ── Embed ──

    def _embed(self, text: str) -> list[float]:
        if not text.strip():
            return []
        resp = self._client.post(
            self._embed_url,
            json={"model": self._api_model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError(f"Empty embedding response for: {text[:80]}...")
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding helper (not part of invoke)."""
        results: list[list[float]] = []
        for i in range(0, len(texts), 16):
            batch = texts[i : i + 16]
            try:
                resp = self._client.post(
                    self._embed_url,
                    json={"model": self._api_model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings", [])
                results.extend(embeddings[: len(batch)])
            except Exception as e:
                logger.warning("Batch embed failed at offset %d: %s", i, e)
                for t in batch:
                    try:
                        results.append(self._embed(t))
                    except Exception:
                        results.append([0.0] * 1024)
        return results

    # ── Extract (via generate) ──

    _EXTRACT_PROMPT = (
        "Extract named entities from this text. "
        "Types: class, function, module, api, concept. "
        'Return only JSON: {{"entities":[{{"name":"X","type":"class"}}]}} '
        'No explanation. Empty: {{"entities":[]}}\n\nTEXT:\n{text}'
    )

    def _extract(self, text: str) -> dict:
        if len(text) < 20:
            return {"entities": []}

        prompt = self._EXTRACT_PROMPT.format(text=text[:1500])
        try:
            resp = self._client.post(
                self._url,
                json={
                    "model": self._api_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 200, "temperature": 0.1},
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return self._parse_entities(raw)
        except Exception as e:
            logger.debug("Extraction failed: %s", e)
            return {"entities": []}

    def _parse_entities(self, raw: str) -> dict:
        json_str = raw.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        brace_s = json_str.find("{")
        brace_e = json_str.rfind("}")
        if 0 <= brace_s < brace_e:
            json_str = json_str[brace_s : brace_e + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"entities": []}

    # ── Rerank (via generate, lightweight) ──

    _RERANK_PROMPT = (
        "Rate relevance (0-10) of each chunk to the query. "
        'Return JSON: {{"scores": [0, 5, 8]}}\n\n'
        "Query: {query}\nChunks: {chunks}"
    )

    def _rerank(self, query: str, chunks: list[str]) -> list[float]:
        if not chunks:
            return []
        # For large top-k, only rerank top 10
        target = chunks[:10]
        prompt = self._RERANK_PROMPT.format(
            query=query,
            chunks="\n---\n".join(c[:200] for c in target),
        )
        try:
            resp = self._client.post(
                self._url,
                json={
                    "model": self._api_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 50, "temperature": 0.1},
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return self._parse_scores(raw, len(chunks))
        except Exception:
            return [1.0] * len(chunks)

    def _parse_scores(self, raw: str, expected_len: int) -> list[float]:
        try:
            data = json.loads(raw)
            scores = data.get("scores", [])
            if len(scores) < expected_len:
                scores += [1.0] * (expected_len - len(scores))
            return [float(s) / 10.0 for s in scores[:expected_len]]
        except (json.JSONDecodeError, TypeError, ValueError):
            return [1.0] * expected_len

    # ── Generate ──

    def _generate(self, prompt: str) -> str:
        try:
            resp = self._client.post(
                self._url,
                json={
                    "model": self._api_model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            logger.error("Generation failed: %s", e)
            raise

    # ── Lifecycle ──

    def close(self) -> None:
        self._client.close()
