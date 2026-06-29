"""
derekinside — OpenAI driver.

Routes invocations to OpenAI / MiniMax / FreeCode / any OpenAI-compatible API.
Supports: embed, extract, rerank, generate.

FreeCode / any OpenAI-compatible endpoint:
    Set base_url + api_key in config.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from derekinside.engine.model import ModelEndpoint, CapabilityNotSupported

logger = logging.getLogger(__name__)


class OpenAIModel(ModelEndpoint):
    """
    OpenAI-compatible API model.

    Can connect to:
      - OpenAI (default: https://api.openai.com/v1)
      - FreeCode (set base_url: https://api.freecode.com/v1)
      - MiniMax (set base_url: https://api.minimax.chat/v1)
      - Any OpenAI-compatible endpoint
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self._api_model = config.get("api_model", "gpt-4o-mini")
        self._api_key = config.get("api_key", "")
        self._base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self._timeout = config.get("timeout", 60.0)
        self._dimensions = config.get("dimensions", 1536)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        self._client = httpx.Client(timeout=self._timeout, headers=headers)

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
        raise CapabilityNotSupported(f"OpenAI does not support '{capability}'")

    # ── Embed ──

    def _embed(self, text: str) -> list[float]:
        if not text.strip():
            return [0.0] * self._dimensions
        resp = self._client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._api_model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("data", [{}])[0].get("embedding", [])
        return emb

    # ── Extract ──

    _SYSTEM_PROMPT = (
        "Extract named entities from the text. "
        "Types: class, function, module, api, concept. "
        "Return only JSON with no explanation."
    )

    _EXTRACT_USER = (
        'Return {"entities": [{"name": "...", "type": "class|function|module|api|concept"}]}\n\n'
        "TEXT:\n{text}"
    )

    def _extract(self, text: str) -> dict:
        if len(text) < 20:
            return {"entities": []}

        try:
            resp = self._client.post(
                f"{self._base_url}/chat/completions",
                json={
                    "model": self._api_model,
                    "messages": [
                        {"role": "system", "content": self._SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": self._EXTRACT_USER.format(text=text[:2000]),
                        },
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]["content"]
            return json.loads(msg)
        except Exception as e:
            logger.debug("OpenAI extraction failed: %s", e)
            return {"entities": []}

    # ── Rerank ──

    _RERANK_SYSTEM = "Rate relevance (0-10) of each chunk to the query."

    def _rerank(self, query: str, chunks: list[str]) -> list[float]:
        if not chunks:
            return []
        target = chunks[:10]
        user = f"Query: {query}\nChunks:\n" + "\n---\n".join(
            f"[{i}] {c[:200]}" for i, c in enumerate(target)
        )
        try:
            resp = self._client.post(
                f"{self._base_url}/chat/completions",
                json={
                    "model": self._api_model,
                    "messages": [
                        {"role": "system", "content": self._RERANK_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(msg)
            scores = data.get("scores", [1.0] * len(chunks))
            return [float(s) / 10.0 for s in scores[: len(chunks)]]
        except Exception:
            return [1.0] * len(chunks)

    # ── Generate ──

    def _generate(self, prompt: str) -> str:
        try:
            resp = self._client.post(
                f"{self._base_url}/chat/completions",
                json={
                    "model": self._api_model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("OpenAI generation failed: %s", e)
            raise

    def close(self) -> None:
        self._client.close()
