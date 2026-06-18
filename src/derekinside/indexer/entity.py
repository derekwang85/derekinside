"""
derekinside — Entity extraction using LLM (Ollama).

Extracts named entities and their relations from chunk text.
Entities include: classes, functions, modules, APIs, domain concepts.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── LLM Prompt ─────────────────────────────────────────────────

_EXTRACT_PROMPT = """You are an entity extraction system. Given a text passage, extract:

1. **Entities** — important named things (classes, functions, modules, APIs, domain concepts, people)
2. **Relations** — directional connections between entities

Return ONLY a JSON object with "entities" and "relations" arrays.

```json
{
  "entities": [
    {"name": "EntityName", "type": "class|function|module|api|concept|person"},
    {"name": "...", "type": "..."}
  ],
  "relations": [
    {"source": "Entity1", "target": "Entity2", "type": "imports|calls|contains|associated|related"}
  ]
}
```

Rules:
- Entity names should be clean identifiers (e.g. "KYCService", not "@KYCService")
- Types: class, function, module, api, concept, person
- Relation types: imports, calls, contains, associated, related
- Only include meaningful entities (skip common words like "system", "data", "function")
- If no meaningful entities found, return empty arrays
- Be concise: max 10 entities per passage

TEXT:
{text}"""


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str = "concept"


@dataclass
class ExtractedRelation:
    source: str
    target: str
    relation_type: str = "related"


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.entities) == 0 and len(self.relations) == 0


class EntityExtractor:
    """LLM-based entity extractor using Ollama."""

    def __init__(
        self,
        url: str = "http://localhost:11434/api/generate",
        model: str = "qwen2.5-coder:7b",
        enabled: bool = False,
        min_chunk_chars: int = 80,
    ):
        self._url = url
        self._model = model
        self._enabled = enabled
        self._min_chunk_chars = min_chunk_chars
        self._client = httpx.Client(timeout=120.0)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relations from text."""
        if not self._enabled:
            return ExtractionResult()
        if len(text) < self._min_chunk_chars:
            return ExtractionResult()

        prompt = _EXTRACT_PROMPT.format(text=text[:2000])  # limit to 2000 chars

        try:
            resp = self._client.post(
                self._url,
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 512,
                        "temperature": 0.1,
                        "stop": ["```\n", "```"],
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "").strip()

            return self._parse(raw)

        except Exception as e:
            logger.warning("Entity extraction failed: %s — skipping chunk", e)
            return ExtractionResult()

    def extract_batch(
        self, texts: list[str], batch_size: int = 4
    ) -> list[ExtractionResult]:
        """Extract entities from multiple texts in batches."""
        results: list[ExtractionResult] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                try:
                    results.append(self.extract(text))
                except Exception as e:
                    logger.warning("Batch extract failed at %d: %s", i, e)
                    results.append(ExtractionResult())
            if i + batch_size < len(texts):
                time.sleep(0.5)
        return results

    def _parse(self, raw: str) -> ExtractionResult:
        """Parse LLM response into structured result."""
        result = ExtractionResult()

        # Extract JSON from response
        json_str = raw.strip()
        # Find JSON block
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        # Try to find {...} in the text
        brace_start = json_str.find("{")
        brace_end = json_str.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_str[brace_start : brace_end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try with regex fallback
            data = self._parse_with_regex(json_str)

        if not data:
            return result

        # Parse entities
        for e in data.get("entities", []):
            if isinstance(e, dict) and "name" in e:
                name = e["name"].strip()
                if name and len(name) > 1:
                    result.entities.append(
                        ExtractedEntity(
                            name=name,
                            entity_type=e.get("type", "concept"),
                        )
                    )

        # Parse relations
        for r in data.get("relations", []):
            if isinstance(r, dict) and "source" in r and "target" in r:
                result.relations.append(
                    ExtractedRelation(
                        source=r["source"].strip(),
                        target=r["target"].strip(),
                        relation_type=r.get("type", "related"),
                    )
                )

        # Filter out low-quality entities
        result.entities = [e for e in result.entities if self._is_valid_entity(e)]
        result.relations = [
            r for r in result.relations if self._is_valid_relation(r, result.entities)
        ]

        return result

    def _parse_with_regex(self, text: str) -> Optional[dict]:
        """Fallback: regex-based JSON extraction."""
        import re

        entities: list[dict] = []
        relations: list[dict] = []

        # Find entity patterns like {"name": "Foo", "type": "class"}
        entity_matches = re.findall(
            r'name["\s:]+([^"}\]]+)["\s,]+type["\s:]+([^"}\]]+)', text
        )
        for name, etype in entity_matches:
            entities.append({"name": name.strip(), "type": etype.strip()})

        # Find relation patterns
        rel_matches = re.findall(
            r'source["\s:]+([^"}\]]+)["\s,]+target["\s:]+([^"}\]]+)[^}]*type["\s:]+([^"}\]]+)',
            text,
        )
        for src, tgt, rtype in rel_matches:
            relations.append(
                {"source": src.strip(), "target": tgt.strip(), "type": rtype.strip()}
            )

        if entities or relations:
            return {"entities": entities, "relations": relations}
        return None

    def _is_valid_entity(self, entity: ExtractedEntity) -> bool:
        """Filter out weak entities."""
        name = entity.name
        # Too short
        if len(name) < 2:
            return False
        # Generic words
        generic = {
            "system",
            "data",
            "function",
            "code",
            "file",
            "text",
            "information",
            "content",
            "value",
            "method",
            "class",
            "object",
            "name",
            "type",
            "thing",
            "stuff",
            "part",
            "item",
            "path",
            "list",
            "array",
            "map",
            "set",
            "view",
        }
        if name.lower() in generic:
            return False
        # Too long (likely a sentence fragment)
        if len(name) > 80:
            return False
        # Contains special chars only (garbage)
        if not re.search(r"[a-zA-Z0-9_]", name):
            return False
        return True

    def _is_valid_relation(
        self, relation: ExtractedRelation, entities: list[ExtractedEntity]
    ) -> bool:
        """Check both endpoints exist in the entity list."""
        entity_names = {e.name for e in entities}
        return relation.source in entity_names and relation.target in entity_names

    def close(self) -> None:
        self._client.close()
