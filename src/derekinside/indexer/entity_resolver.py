"""
derekinside — Entity Resolver.

Resolves entity name aliases into canonical forms:
  - Exact match
  - Lowercase normalization
  - CamelCase splitting (KYCApplicationService → [KYC, Application, Service])
  - Synonym dictionary
  - Fuzzy match (trigram/levenshtein)

Used during entity extraction to avoid creating duplicate entities for
the same concept expressed differently.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# ── Default synonym dictionary (domain-specific) ──

_DEFAULT_SYNONYMS: dict[str, list[str]] = {
    # KYC domain
    "kyc": ["kyc", "kycapplication", "kyc_application", "客户身份识别", "kyc认证"],
    "kycapplication": [
        "kycapplication",
        "kyc_application",
        "kyc",
        "kycapplicationservice",
    ],
    # User/Security domain
    "user": ["user", "用户", "userinfo", "user_info"],
    "auth": ["auth", "authentication", "认证", "授权"],
    "permission": ["permission", "权限", "permissions"],
    "role": ["role", "角色", "roles"],
    # Trade domain
    "order": ["order", "订单", "trade", "交易"],
    "position": ["position", "持仓", "positions"],
    "margin": ["margin", "保证金", "margins"],
    "settlement": ["settlement", "结算", "清算"],
    # Common abbreviations
    "config": ["config", "configuration", "配置", "cfg"],
    "service": ["service", "services"],
    "dao": ["dao", "repository", "repositories", "mapper"],
    "dto": ["dto", "vo", "request", "response", "command", "query"],
}

# ── Stop words for filtering ──

_STOP_WORDS = {
    "get",
    "set",
    "is",
    "has",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "data",
    "info",
    "base",
    "abstract",
    "utils",
    "util",
    "helper",
    "common",
    "core",
}


class EntityResolver:
    """
    Resolves entity names to canonical forms.
    Used by the extraction pipeline to deduplicate entities.
    """

    def __init__(self, synonyms: dict[str, list[str]] | None = None):
        self._synonyms = synonyms or {}
        # Build reverse lookup: alias → canonical
        self._alias_map: dict[str, str] = {}
        for canonical, aliases in {**_DEFAULT_SYNONYMS, **self._synonyms}.items():
            lower_canon = canonical.lower()
            for alias in aliases:
                self._alias_map[alias.lower()] = lower_canon
            self._alias_map[lower_canon] = lower_canon

    def resolve(self, name: str) -> tuple[str, bool]:
        """
        Resolve entity name to canonical form.
        Returns (canonical_name, was_changed).
        """
        if not name or len(name) < 2:
            return name, False

        # 1. Direct alias lookup
        lower = name.lower()
        if lower in self._alias_map:
            canonical = self._alias_map[lower]
            if canonical.lower() != lower:
                return canonical, True
            return name, False

        # 2. Lowercase normalization
        # If a lowercased version exists, use the original stored casing
        stored = self._find_by_lower(name)
        if stored and stored.lower() == name.lower():
            return stored, stored != name

        # 3. CamelCase splitting
        parts = self._split_camel(name)
        if len(parts) >= 2:
            # Try resolving individual parts
            resolved_parts = [self.resolve(p)[0] for p in parts]
            # Check if any abbreviation matches a canonical
            first_letters = "".join(p[0].lower() for p in parts if p)
            if first_letters in self._alias_map:
                return self._alias_map[first_letters], True

        # 4. Fuzzy match against known entities (done by caller with access to DB)
        # This returns name unchanged; caller should call scan_aliases() separately.

        return name, False

    def scan_aliases(
        self, name: str, known_names: list[str]
    ) -> list[tuple[str, float]]:
        """
        Find fuzzy matches for name among known names.
        Returns [(matched_name, confidence), ...] sorted by confidence.
        """
        if not name or not known_names:
            return []

        results = []
        known_set = set(known_names)
        lower_name = name.lower()

        for known in known_set:
            if known == name:
                continue
            lower_known = known.lower()
            score = self._fuzzy_score(lower_name, lower_known)
            if score >= 0.75:
                results.append((known, score))

        results.sort(key=lambda x: -x[1])
        return results

    def canonicalize_entity_name(self, name: str) -> str:
        """
        Full canonicalization: resolve + normalize.
        Removes suffixes like Service/Impl/DTO/VO.
        """
        resolved, _ = self.resolve(name)
        # Strip common suffixes
        for suffix in [
            "service",
            "impl",
            "dto",
            "vo",
            "dao",
            "mapper",
            "repository",
            "controller",
            "config",
            "utils",
            "request",
            "response",
            "command",
            "query",
            "entity",
            "model",
            "pojo",
            "po",
            "bo",
        ]:
            pattern = re.compile(rf"^(.+?)(?:{suffix})$", re.IGNORECASE)
            m = pattern.match(resolved)
            if m and len(m.group(1)) >= 3:
                return m.group(1)
        return resolved

    # ── Helpers ──

    def _find_by_lower(self, name: str) -> str | None:
        lower = name.lower()
        for canonical, aliases in {**_DEFAULT_SYNONYMS, **self._synonyms}.items():
            if canonical.lower() == lower:
                return canonical
            for alias in aliases:
                if alias.lower() == lower:
                    return canonical
        return None

    def _split_camel(self, name: str) -> list[str]:
        """Split CamelCase into words. 'KYCApplicationService' → ['KYC', 'Application', 'Service']"""
        # Handle acronyms (all caps segments)
        parts = re.findall(
            r"[A-Z]{2,}(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", name
        )
        return [p for p in parts if p.lower() not in _STOP_WORDS and len(p) >= 1]

    def _fuzzy_score(self, a: str, b: str) -> float:
        """Fuzzy match score using SequenceMatcher."""
        if not a or not b:
            return 0.0

        # Quick check: prefix match
        if a.startswith(b) or b.startswith(a):
            return max(len(a), len(b)) / min(len(a), len(b)) * 0.3 + 0.5
            # This gives high score when one is prefix of another

        # Levenshtein-like via SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()
