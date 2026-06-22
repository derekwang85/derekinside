"""
derekinside — Relation Inferrer.

Extracts structured relations from code chunks that entity extraction misses.
Works on regex patterns for code-specific relationships:
  - extends / implements (class inheritance)
  - @Autowired / @Inject (dependency injection)
  - @RequestMapping + @PostMapping/... (API routing)
  - Maven/Gradle dependencies
  - @Service / @Component / @Repository (stereotype)
  - has_field (class composition)

These are deterministic, zero-cost (no LLM), and produce high-precision relations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InferredRelation:
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type,
            "weight": self.weight,
        }


# ── Language-specific file extension groups ──

JAVA_FILES = {".java"}
FRONTEND_FILES = {".vue", ".js", ".ts", ".jsx", ".tsx"}
PYTHON_FILES = {".py"}
XML_FILES = {".xml", ".pom.xml"}
YAML_FILES = {".yml", ".yaml"}
ALL_CODE = JAVA_FILES | FRONTEND_FILES | PYTHON_FILES


class RelationInferrer:
    """
    Structured relation inference from code chunks.

    Infers high-precision relations using deterministic patterns.
    Each pattern targets a specific code structure.
    """

    def __init__(self):
        self._patterns = {
            "java_extends": self._java_extends,
            "java_implements": self._java_implements,
            "java_autowired": self._java_autowired,
            "java_request_mapping": self._java_request_mapping,
            "java_stereotype": self._java_stereotype,
            "java_field": self._java_field,
            "java_import": self._java_import,
            "frontend_import": self._frontend_import,
            "python_import": self._python_import,
            "maven_dependency": self._maven_dependency,
        }

    def infer(self, chunk_text: str, chunk_id: int, file_ext: str) -> list[InferredRelation]:
        """
        Run all applicable patterns on a chunk.
        Returns a flat list of InferredRelation.
        """
        results: list[InferredRelation] = []
        ext = file_ext.lower() if file_ext else ""

        # Java patterns
        if ext in JAVA_FILES:
            results.extend(self._java_extends(chunk_text))
            results.extend(self._java_implements(chunk_text))
            results.extend(self._java_autowired(chunk_text))
            results.extend(self._java_request_mapping(chunk_text))
            results.extend(self._java_stereotype(chunk_text))
            results.extend(self._java_field(chunk_text))
            results.extend(self._java_import(chunk_text, chunk_id))

        # Frontend patterns
        if ext in FRONTEND_FILES:
            results.extend(self._frontend_import(chunk_text, chunk_id))

        # Python patterns
        if ext in PYTHON_FILES:
            results.extend(self._python_import(chunk_text, chunk_id))

        # Build/dep patterns
        if ext in XML_FILES or ".pom" in ext:
            results.extend(self._maven_dependency(chunk_text))

        if ext in YAML_FILES:
            results.extend(self._yaml_reference(chunk_text))

        return results

    # ── Pattern: extends (class inheritance) ──

    _PAT_EXTENDS = re.compile(
        r'(?:class|interface|abstract class)\s+(\w+)\s+extends\s+([\w.,\s<>{},()]+?)(?:\s+implements|\s*\{|\s*$)',
        re.DOTALL,
    )

    def _java_extends(self, text: str) -> list[InferredRelation]:
        results = []
        for m in self._PAT_EXTENDS.finditer(text):
            child = m.group(1).strip()
            parents_raw = m.group(2).strip()
            # Split by comma, strip generics
            for parent in re.split(r'\s*,\s*', parents_raw):
                parent = re.sub(r'<[^>]+>', '', parent).strip()
                # Only keep if it looks like a class name (starts uppercase)
                if parent and parent[0].isupper():
                    results.append(InferredRelation(
                        source=child, target=parent,
                        relation_type="extends",
                        weight=1.0,
                    ))
        return results

    # ── Pattern: implements ──

    _PAT_IMPLEMENTS = re.compile(
        r'(?:class|abstract class)\s+(\w+)(?:\s+extends\s+\w+)?\s+implements\s+([\w.,\s<>{},()]+?)\s*\{',
        re.DOTALL,
    )

    def _java_implements(self, text: str) -> list[InferredRelation]:
        results = []
        for m in self._PAT_IMPLEMENTS.finditer(text):
            child = m.group(1).strip()
            ifaces_raw = m.group(2).strip()
            for iface in re.split(r'\s*,\s*', ifaces_raw):
                iface = re.sub(r'<[^>]+>', '', iface).strip()
                if iface and iface[0].isupper():
                    results.append(InferredRelation(
                        source=child, target=iface,
                        relation_type="implements",
                        weight=1.0,
                    ))
        return results

    # ── Pattern: @Autowired (dependency injection) ──

    _PAT_AUTOWIRED = re.compile(
        r'@(?:Autowired|Inject|Resource)\s*\n?\s*'
        r'(?:private|public|protected)\s+(\w+)\s+(\w+)',
    )

    def _java_autowired(self, text: str) -> list[InferredRelation]:
        results = []
        for m in self._PAT_AUTOWIRED.finditer(text):
            type_name = m.group(1).strip()
            field_name = m.group(2).strip()
            if type_name and type_name[0].isupper():
                results.append(InferredRelation(
                    source=type_name, target=field_name,
                    relation_type="depends_on",
                    weight=1.0,
                    metadata={"injection_site": field_name},
                ))
        return results

    # ── Pattern: @RequestMapping + @PostMapping/... (API routes) ──

    _PAT_CLASS_MAPPING = re.compile(
        r'@(?:RequestMapping|RestController)\s*\(\s*(?:"([^"]+)"|\'(?:[^\']+)\')\s*\)',
    )
    _PAT_METHOD_MAPPING = re.compile(
        r'@(GetMapping|PostMapping|PutMapping|DeleteMapping|PatchMapping|RequestMapping)'
        r'\s*\(\s*(?:"([^"]+)"|\'(?:[^\']+)\')\s*\)',
    )

    def _java_request_mapping(self, text: str) -> list[InferredRelation]:
        results = []
        # Find class-level base path
        base_path = ""
        for m in self._PAT_CLASS_MAPPING.finditer(text):
            base_path = m.group(1) if m.lastindex and m.group(1) else ""

        # Find method-level paths
        for m in self._PAT_METHOD_MAPPING.finditer(text):
            http_method = m.group(1).replace("Mapping", "").upper()
            method_path = m.group(2) if m.lastindex and m.group(2) else ""
            full_path = base_path + (method_path if method_path else "")
            if full_path:
                results.append(InferredRelation(
                    source=full_path, target=http_method,
                    relation_type="serves_path",
                    weight=1.0,
                ))

        # Also extract method names near mapping annotations
        pat_method = re.compile(
            r'@(?:Get|Post|Put|Delete|Patch|Request)Mapping.*?\n'
            r'(?:public|private|protected)?\s*(?:\w+\s+)?(\w+)\s*\(',
            re.DOTALL,
        )
        for m in pat_method.finditer(text):
            method = m.group(1).strip()
            results.append(InferredRelation(
                source=method, target=base_path or "/",
                relation_type="serves_path",
                weight=0.8,
            ))

        return results

    # ── Pattern: @Service / @Component / @Repository (stereotype) ──

    _PAT_STEREOTYPE = re.compile(
        r'@(Service|Component|Repository|Controller|RestController|Configuration)\s*\n?\s*'
        r'(?:public\s+)?(?:abstract\s+)?(?:class|interface)\s+(\w+)'
    )

    def _java_stereotype(self, text: str) -> list[InferredRelation]:
        results = []
        for m in self._PAT_STEREOTYPE.finditer(text):
            annotation = m.group(1).strip()
            class_name = m.group(2).strip()
            results.append(InferredRelation(
                source=class_name, target=annotation.lower(),
                relation_type="is_a",
                weight=0.9,
                metadata={"stereotype": annotation},
            ))
        return results

    # ── Pattern: field declarations (class composition) ──

    _PAT_FIELD = re.compile(
        r'(?:private|public|protected)\s+(\w+(?:<[^>]+>)?)\s+(\w+)\s*[=;]',
    )

    def _java_field(self, text: str) -> list[InferredRelation]:
        results = []
        seen = set()
        for m in self._PAT_FIELD.finditer(text):
            type_name = re.sub(r'<[^>]+>', '', m.group(1)).strip()
            field_name = m.group(2).strip()
            # Skip primitives, strings, and common types
            if type_name.lower() in {"int", "long", "double", "float", "boolean",
                                      "string", "void", "list", "set", "map",
                                      "integer", "byte", "short", "char", "object"}:
                continue
            if type_name and type_name[0].isupper():
                key = (type_name, field_name)
                if key not in seen:
                    seen.add(key)
                    results.append(InferredRelation(
                        source=type_name, target=field_name,
                        relation_type="has_field",
                        weight=0.7,
                    ))
        return results

    # ── Pattern: Java imports ──

    _PAT_JAVA_IMPORT = re.compile(r'^import\s+([\w.]+(?:\.\w+)?)\s*;', re.MULTILINE)

    def _java_import(self, text: str, chunk_id: int) -> list[InferredRelation]:
        results = []
        imported = set()
        for m in self._PAT_JAVA_IMPORT.finditer(text):
            full = m.group(1)
            parts = full.split(".")
            # Extract the class name (last uppercase segment)
            for p in reversed(parts):
                if p and p[0].isupper():
                    if p not in imported:
                        imported.add(p)
                        module = parts[-2] if len(parts) >= 2 else ""
                        results.append(InferredRelation(
                            source=p, target=module,
                            relation_type="imports",
                            weight=0.6,
                        ))
                    break
        return results

    # ── Pattern: Frontend imports (ESM / CommonJS) ──

    _PAT_FRONTEND_IMPORT = re.compile(
        r'(?:import\s+(?:\w+\s*,?\s*)?\s*\{?\s*(\w+)|\}?\s*from\s+[\'"]([^\'"]+)[\'"])'
    )

    def _frontend_import(self, text: str, chunk_id: int) -> list[InferredRelation]:
        results = []
        names = set()
        for m in self._PAT_FRONTEND_IMPORT.finditer(text):
            name = m.group(1) or m.group(2)
            if name:
                name = name.strip().split("/")[-1].split(".")[0]
                if name and name not in names:
                    names.add(name)
                    results.append(InferredRelation(
                        source=name, target="",
                        relation_type="imports",
                        weight=0.5,
                    ))
        return results

    # ── Pattern: Python imports ──

    def _python_import(self, text: str, chunk_id: int) -> list[InferredRelation]:
        results = []
        names = set()
        for m in re.finditer(
            r'(?:from\s+(\S+)\s+import\s+(\w+)|import\s+(\w+))', text
        ):
            name = m.group(2) or m.group(3) or ""
            if name and name not in names:
                names.add(name)
                module = m.group(1) or m.group(3) or ""
                results.append(InferredRelation(
                    source=name, target=module,
                    relation_type="imports",
                    weight=0.5,
                ))
        return results

    # ── Pattern: Maven/Gradle dependencies ──

    _PAT_MAVEN_DEP = re.compile(
        r'<dependency>\s*<groupId>([^<]+)</groupId>\s*'
        r'<artifactId>([^<]+)</artifactId>',
        re.DOTALL,
    )

    def _maven_dependency(self, text: str) -> list[InferredRelation]:
        results = []
        seen = set()
        for m in self._PAT_MAVEN_DEP.finditer(text):
            artifact = m.group(2).strip()
            if artifact not in seen:
                seen.add(artifact)
                results.append(InferredRelation(
                    source=artifact, target=m.group(1).strip(),
                    relation_type="depends_on",
                    weight=0.8,
                    metadata={"dep_type": "maven"},
                ))
        return results

    # ── Pattern: YAML references (spring config, bean refs) ──

    def _yaml_reference(self, text: str) -> list[InferredRelation]:
        results = []
        # Match property references like: ${some.property}
        for m in re.finditer(r'\$\{(\w+(?:\.\w+)+)\}', text):
            results.append(InferredRelation(
                source=m.group(1), target="config",
                relation_type="configures",
                weight=0.4,
            ))
        return results
