"""
derekinside — Entity extraction (hybrid: regex + optional LLM).

For code-heavy content, regex extraction handles the bulk (class names,
function names, modules, imports). For domain concept extraction,
an optional LLM pass (1.5B model for speed) adds semantic entities.

This avoids the ~60s/chunk cost of pure LLM extraction on CPU.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


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

    def merge(self, other: ExtractionResult) -> None:
        seen_names = {(e.name, e.entity_type) for e in self.entities}
        for e in other.entities:
            if (e.name, e.entity_type) not in seen_names:
                self.entities.append(e)
                seen_names.add((e.name, e.entity_type))
        self.relations.extend(other.relations)


# ── Regex Patterns for Code Entity Extraction ──────────────────

_PATTERNS = {
    "class": [
        re.compile(r"(?:public\s+)?(?:abstract\s+)?class\s+(\w+)"),
        re.compile(r"(?:export\s+)?(?:default\s+)?class\s+(\w+)"),
        re.compile(r"@Component\s*\n.*?class\s+(\w+)", re.DOTALL),
        re.compile(r"@Service\s*\n.*?class\s+(\w+)", re.DOTALL),
        re.compile(r"@RestController\s*\n.*?class\s+(\w+)", re.DOTALL),
        re.compile(r"@Entity\s*\n.*?class\s+(\w+)", re.DOTALL),
    ],
    "function": [
        re.compile(r"(?:public|private|protected)\s+\w+\s+(\w+)\s*\([^)]*\)"),
        re.compile(r"(?:export\s+)?(?:async\s+)?(?:function\s+)?(\w+)\s*\([^)]*\)\s*{"),
        re.compile(r"(?:def|fn)\s+(\w+)\s*\("),
    ],
    "module": [
        re.compile(r"(?:package|namespace)\s+([\w.]+)"),
        re.compile(r"(?:module|from)\s+['\"]([\w./-]+)['\"]"),
    ],
    "api": [
        re.compile(
            r'@(?:GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping)\(["\']([^"\']+)["\']'
        ),
        re.compile(
            r"(?:router\.(?:get|post|put|delete)|app\.(?:get|post|put|delete))\(['\"](/[\w/{}]+)['\"]"
        ),
    ],
    "variable": [
        re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:require|import)\s*[( ][\"']"),
        re.compile(r"(?:private|public|protected)\s+(?:\w+\s+)?(\w+)\s*=\s*new\s+"),
    ],
}


def extract_regex(text: str, min_name_len: int = 2) -> ExtractionResult:
    """Fast regex-based entity extraction from code/text."""
    result = ExtractionResult()
    seen = set()

    for etype, patterns in _PATTERNS.items():
        for pat in patterns:
            for match in pat.finditer(text):
                name = match.group(1).strip()
                if len(name) >= min_name_len and name not in seen:
                    seen.add(name)
                    # Filter out common false positives
                    if name.lower() not in {
                        "void",
                        "int",
                        "string",
                        "boolean",
                        "long",
                        "double",
                        "true",
                        "false",
                        "null",
                        "this",
                        "return",
                        "if",
                        "else",
                        "for",
                        "while",
                        "class",
                        "function",
                        "import",
                        "export",
                        "default",
                        "extend",
                        "implements",
                        "throws",
                        "new",
                        "type",
                        "default",
                        "extends",
                    }:
                        result.entities.append(
                            ExtractedEntity(name=name, entity_type=etype)
                        )

    return result


def extract_imports(text: str) -> ExtractionResult:
    """Extract import relations."""
    result = ExtractionResult()
    imported_modules = set()
    imported_names = set()

    # Java imports: import com.example.Foo;
    for m in re.finditer(r"^import\s+([\w.]+(?:\.\w+)?)\s*;", text, re.MULTILINE):
        full = m.group(1)
        parts = full.split(".")
        module = parts[-2] if len(parts) >= 2 else parts[0]
        name_parts = [p for p in parts if p[0].isupper()]
        for name in name_parts:
            imported_names.add(name)
            imported_modules.add(module)
            result.entities.append(ExtractedEntity(name=name, entity_type="class"))
            if module:
                result.entities.append(
                    ExtractedEntity(name=module, entity_type="module")
                )

    # Python/JS imports: import X from 'y' or from y import X
    for m in re.finditer(
        r"(?:from\s+['\"](\S+)['\"]\s+import\s+(\w+)|import\s+(\w+)\s+from\s+['\"](\S+)['\"])",
        text,
    ):
        module = m.group(1) or m.group(4) or ""
        name = m.group(2) or m.group(3) or ""
        if name and module:
            mod_name = module.split("/")[-1].split(".")[0]
            if mod_name and mod_name not in imported_modules:
                imported_modules.add(mod_name)
                result.entities.append(
                    ExtractedEntity(name=mod_name, entity_type="module")
                )
            if name and name not in imported_names:
                imported_names.add(name)
                result.entities.append(
                    ExtractedEntity(
                        name=name,
                        entity_type="class" if name[0].isupper() else "function",
                    )
                )

    return result


# ── LLM Prompt (minimal for speed) ────────────────────────────

_LLM_PROMPT = """Extract named entities from this text. Types: class, function, module, api, concept.
Return only JSON: {{"entities":[{{"name":"X","type":"class"}}]}}
No explanation. Empty: {{"entities":[]}}

TEXT:
{text}"""


def _is_valid_entity_name(name: str) -> bool:
    if len(name) < 2 or len(name) > 80:
        return False
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
        "module",
        "api",
        "concept",
        "entity",
        "service",
        "component",
        "config",
        "config",
        "model",
        "utils",
        "helper",
        "util",
        "common",
        "base",
        "abstract",
    }
    if name.lower() in generic:
        return False
    return bool(re.search(r"[a-zA-Z0-9_]", name))


# ── Hybrid Extractor ──────────────────────────────────────────


# ── Mode Configuration ────────────────────────────────────────

_EXTRACTION_MODES = {
    "regex": "Pure regex, ~5ms/chunk, code entities only",
    "1.5b": "LLM-only with qwen2.5-coder:1.5b, ~2.6s/chunk, high recall",
    "7b": "LLM-only with qwen2.5-coder:7b, ~6.6s/chunk, high precision",
    "hybrid-1.5b": "Regex + 1.5B concepts, best balance on CPU",
    "hybrid-7b": "Regex + 7B concepts, highest precision",
}

_VALID_MODES = set(_EXTRACTION_MODES.keys())


class EntityExtractor:
    """
    5-mode configurable entity extractor.

    Modes:
      regex         — Pure regex, ~5ms/chunk
      1.5b          — LLM-only with qwen2.5-coder:1.5b, ~2.6s/chunk
      7b            — LLM-only with qwen2.5-coder:7b, ~6.6s/chunk
      hybrid-1.5b   — Regex + 1.5B concepts (default for CPU)
      hybrid-7b     — Regex + 7B concepts (default)
    """

    def __init__(
        self,
        url: str = "http://localhost:11434/api/generate",
        model: str = "qwen2.5-coder:7b",
        enabled: bool = False,
        use_llm: bool = False,
        llm_min_chars: int = 100,
        mode: str = "",
    ):
        self._url = url
        self._model = model
        self._enabled = enabled
        self._llm_min_chars = llm_min_chars

        # Resolve mode: if mode is explicitly set, use it; else fall back
        # to use_llm/model for backward compatibility
        if mode and mode in _VALID_MODES:
            self._mode = mode
        elif use_llm and "7b" in model.lower():
            self._mode = "hybrid-7b"
        elif use_llm:
            self._mode = "hybrid-1.5b"
        else:
            self._mode = "regex"

        self._client = httpx.Client(timeout=120.0) if enabled else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def mode(self) -> str:
        return self._mode

    @classmethod
    def list_modes(cls) -> dict:
        return dict(_EXTRACTION_MODES)

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities using the configured mode."""
        if not self._enabled or not text.strip():
            return ExtractionResult()

        if self._mode == "regex":
            return self._extract_regex(text)
        elif self._mode == "1.5b":
            return self._extract_llm_only(text, "qwen2.5-coder:1.5b")
        elif self._mode == "7b":
            return self._extract_llm_only(text, "qwen2.5-coder:7b")
        elif self._mode == "hybrid-1.5b":
            return self._extract_hybrid(text, "qwen2.5-coder:1.5b")
        elif self._mode == "hybrid-7b":
            return self._extract_hybrid(text, "qwen2.5-coder:7b")
        else:
            return ExtractionResult()

    def _extract_regex(self, text: str) -> ExtractionResult:
        """Mode: regex only."""
        result = ExtractionResult()
        regex_result = extract_regex(text)
        result.merge(regex_result)
        import_result = extract_imports(text)
        result.merge(import_result)
        return result

    def _extract_llm_only(self, text: str, model: str) -> ExtractionResult:
        """Mode: LLM only (1.5b or 7b)."""
        if len(text) < self._llm_min_chars or not self._client:
            return ExtractionResult()
        return self._extract_llm(text, model)

    def _extract_hybrid(self, text: str, model: str) -> ExtractionResult:
        """Mode: hybrid (regex + LLM concepts)."""
        result = ExtractionResult()

        # Step 1: Regex for code entities
        regex_result = extract_regex(text)
        result.merge(regex_result)
        import_result = extract_imports(text)
        result.merge(import_result)

        # Track what regex found
        regex_names = {e.name for e in result.entities}

        # Step 2: LLM for concepts not found by regex
        if len(text) >= self._llm_min_chars and self._client:
            llm_result = self._extract_llm(text, model)
            # Only add concept/api entities not found by regex
            for e in llm_result.entities:
                if e.name not in regex_names and e.entity_type in ("concept", "api"):
                    result.entities.append(e)
                    regex_names.add(e.name)

        return result

    def _extract_llm(self, text: str, model: str) -> ExtractionResult:
        """LLM-based entity extraction (domain concepts)."""
        if not self._client:
            return ExtractionResult()

        prompt = _LLM_PROMPT.format(text=text[:1500])

        try:
            resp = self._client.post(
                self._url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 200,
                        "temperature": 0.1,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "").strip()
            return self._parse(raw)
        except Exception as e:
            logger.debug("LLM extraction failed: %s", e)
            return ExtractionResult()

    def _parse(self, raw: str) -> ExtractionResult:
        """Parse LLM JSON response."""
        result = ExtractionResult()

        json_str = raw.strip()
        # Handle markdown-wrapped JSON
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        brace_start = json_str.find("{")
        brace_end = json_str.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_str[brace_start : brace_end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return result

        for e in data.get("entities", []):
            if isinstance(e, dict) and "name" in e:
                name = e["name"].strip()
                etype = e.get("type", "concept")
                if _is_valid_entity_name(name) and etype != "variable":
                    result.entities.append(
                        ExtractedEntity(name=name, entity_type=etype)
                    )

        for r in data.get("relations", []):
            if isinstance(r, dict) and "source" in r and "target" in r:
                result.relations.append(
                    ExtractedRelation(
                        source=r["source"].strip(),
                        target=r["target"].strip(),
                        relation_type=r.get("type", "related"),
                    )
                )

        return result

    def close(self) -> None:
        if self._client:
            self._client.close()
