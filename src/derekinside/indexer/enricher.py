"""
derekinside — Entity Enricher.

Adds descriptions and aliases to graph entities using offline LLM (Phase 1 7B).
Entities are enriched in batch during low-usage hours.

Schema changes (run once):
  ALTER TABLE entities ADD COLUMN IF NOT EXISTS description TEXT;
  ALTER TABLE entities ADD COLUMN IF NOT EXISTS aliases TEXT[] DEFAULT '{}';
  ALTER TABLE entities ADD COLUMN IF NOT EXISTS source_chunks INTEGER[] DEFAULT '{}';
"""

from __future__ import annotations

import json
import logging
import time

from derekinside.storage.graph import KnowledgeGraph
from derekinside.drivers.ollama import OllamaModel

logger = logging.getLogger(__name__)

_PROMPT = (
    "Generate a one-sentence description for this entity in a CTRM/trading system.\n"
    "Entity name: {name}\n"
    "Entity type: {type}\n"
    "Context (source code/document excerpts):\n{context}\n\n"
    'Return JSON: {{"description": "..."}}\n'
    "No explanation. Max 30 words."
)


class EntityEnricher:
    """
    Offline entity enrichment using Phase 1 7B LLM.
    Generates descriptions + collects source chunk references.
    """

    def __init__(self, graph: KnowledgeGraph, model_name: str = "qwen2.5-coder:7b"):
        self._graph = graph
        self._llm = OllamaModel(
            "enricher",
            {
                "api_model": model_name,
                "url": "http://localhost:11434/api/generate",
            },
        )

    def enrich_entity(self, entity_id: int) -> dict | None:
        """Generate description for a single entity."""
        with self._graph.cursor() as cur:
            cur.execute(
                "SELECT name, entity_type FROM entities WHERE id = %s",
                (entity_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            name, etype = row

            # Get context from connected chunks
            cur.execute(
                """
                SELECT c.chunk_text FROM chunks c
                JOIN entity_chunks ec ON ec.chunk_id = c.id
                WHERE ec.entity_id = %s
                ORDER BY ec.relevance DESC
                LIMIT 3
            """,
                (entity_id,),
            )
            contexts = []
            for (text,) in cur.fetchall():
                contexts.append(text[:500])
            context_str = "\n---\n".join(contexts) if contexts else name

        prompt = _PROMPT.format(name=name, type=etype, context=context_str[:1500])
        try:
            resp = self._llm.invoke("generate", prompt=prompt)
            if isinstance(resp, str):
                data = self._parse_json(resp)
            elif isinstance(resp, dict):
                data = resp.get("response", "")
                data = self._parse_json(data)
            else:
                return None

            desc = (data or {}).get("description", "")
            if desc and len(desc) > 5:
                # Update description
                with self._graph.cursor() as cur:
                    cur.execute(
                        "UPDATE entities SET metadata = "
                        "CASE WHEN metadata IS NULL OR metadata = '{}'::jsonb "
                        "THEN jsonb_build_object('description', %s) "
                        "ELSE metadata || jsonb_build_object('description', %s) END "
                        "WHERE id = %s",
                        (desc, desc, entity_id),
                    )
                return {"id": entity_id, "name": name, "description": desc}
        except Exception as e:
            logger.debug("Enrich entity %s failed: %s", name, e)
        return None

    def enrich_batch(
        self, limit: int = 100, offset: int = 0, only_without_desc: bool = True
    ) -> dict:
        """Enrich batch of entities. Returns stats."""
        with self._graph.cursor() as cur:
            if only_without_desc:
                cur.execute(
                    """
                    SELECT id, name, entity_type FROM entities
                    WHERE metadata IS NULL
                       OR metadata->>'description' IS NULL
                       OR metadata->>'description' = ''
                    ORDER BY id LIMIT %s OFFSET %s
                """,
                    (limit, offset),
                )
            else:
                cur.execute(
                    "SELECT id, name, entity_type FROM entities "
                    "ORDER BY id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
            rows = cur.fetchall()

        if not rows:
            return {"enriched": 0, "failed": 0, "total": 0}

        enriched = 0
        failed = 0
        for eid, name, etype in rows:
            result = self.enrich_entity(eid)
            if result:
                enriched += 1
            else:
                failed += 1
            if (enriched + failed) % 10 == 0:
                logger.info("Enriched %d/%d entities", enriched + failed, len(rows))
            time.sleep(0.5)  # cooldown

        return {"enriched": enriched, "failed": failed, "total": len(rows)}

    def _parse_json(self, text: str) -> dict | None:
        """Parse JSON from LLM response (handles wrapping)."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        brace_s = text.find("{")
        brace_e = text.rfind("}")
        if 0 <= brace_s < brace_e:
            text = text[brace_s : brace_e + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
