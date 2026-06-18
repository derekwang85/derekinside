"""
Tests for derekinside Phase 2 — Knowledge Graph.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from derekinside.storage.graph import Entity, Relation
from derekinside.indexer.entity import (
    EntityExtractor,
    extract_regex,
    extract_imports,
)
from derekinside.search.propagation import GraphPropagator


# ── Entity Extractor ───────────────────────────────────────────


def test_extractor_disabled():
    ext = EntityExtractor(enabled=False)
    result = ext.extract("Some text here.")
    assert result.is_empty()


def test_extractor_parsing():
    ext = EntityExtractor(enabled=True)
    result = ext._parse('{"entities": [], "relations": []}')
    assert result.is_empty()

    result = ext._parse("""
        {
            "entities": [
                {"name": "KYCService", "type": "class"},
                {"name": "approveKYC", "type": "function"},
                {"name": "Buyer", "type": "concept"}
            ],
            "relations": [
                {"source": "KYCService", "target": "approveKYC", "type": "calls"},
                {"source": "KYCService", "target": "Buyer", "type": "associated"}
            ]
        }
    """)
    assert len(result.entities) == 3
    assert len(result.relations) == 2
    assert result.entities[0].name == "KYCService"
    assert result.entities[0].entity_type == "class"
    assert result.relations[0].source == "KYCService"
    assert result.relations[0].target == "approveKYC"


def test_extractor_parsing_markdown_json():
    ext = EntityExtractor(enabled=True)
    result = ext._parse("""
    Here's the extraction:
    ```json
    {"entities": [{"name": "TradeOMS", "type": "module"}], "relations": []}
    ```
    """)
    assert len(result.entities) == 1
    assert result.entities[0].name == "TradeOMS"


def test_extractor_filter_generic():
    ext = EntityExtractor(enabled=True)
    # "data" and "api" are in the generic filter list
    result = ext._parse(
        '{"entities": [{"name": "data", "type": "concept"}, {"name": "api", "type": "api"}], "relations": []}'
    )
    assert len(result.entities) == 0  # both filtered


def test_extractor_filter_short():
    ext = EntityExtractor(enabled=True)
    result = ext._parse(
        '{"entities": [{"name": "a", "type": "concept"}], "relations": []}'
    )
    assert len(result.entities) == 0  # too short


def test_extractor_filter_long():
    ext = EntityExtractor(enabled=True)
    long_name = "A" * 100
    result = ext._parse(
        f'{{"entities": [{{"name": "{long_name}", "type": "concept"}}], "relations": []}}'
    )
    assert len(result.entities) == 0  # too long


# ── Regex Extraction ──────────────────────────────────────────


def test_extract_regex_class():
    result = extract_regex("public class KYCService {")
    assert any(
        e.name == "KYCService" and e.entity_type == "class" for e in result.entities
    )


def test_extract_regex_function():
    result = extract_regex("def handle_approval(self):")
    assert any(
        e.name == "handle_approval" and e.entity_type == "function"
        for e in result.entities
    )


def test_extract_regex_api():
    result = extract_regex('@GetMapping("/api/v1/kyc")')
    assert any(e.entity_type == "api" for e in result.entities)


def test_extract_imports():
    result = extract_imports("import com.tradeoms.kyc.KYCService;")
    assert any(e.name == "KYCService" for e in result.entities)


def test_extract_no_false_positives():
    result = extract_regex("function foo() { return true; }")
    # "foo" might be extracted, "true" should not be
    for e in result.entities:
        assert e.name != "true"


# ── Knowledge Graph Dataclasses ────────────────────────────────


def test_entity_to_dict():
    e = Entity(id=1, name="TestService", entity_type="class")
    d = e.to_dict()
    assert d["name"] == "TestService"
    assert d["type"] == "class"
    assert d["id"] == 1


def test_relation_to_dict():
    r = Relation(
        id=1, source_entity_id=1, target_entity_id=2, relation_type="calls", weight=0.8
    )
    d = r.to_dict()
    assert d["source"] == 1
    assert d["target"] == 2
    assert d["type"] == "calls"
    assert d["weight"] == 0.8


# ── Graph Propagator ───────────────────────────────────────────


def test_propagator_disabled():
    pg = GraphPropagator(None, enabled=False)  # type: ignore
    results = pg.propagate([], [])
    assert len(results) == 0


def test_propagator_softmax():
    from derekinside.search.propagation import _softmax

    scores = [1.0, 2.0, 3.0]
    sm = _softmax(scores)
    assert abs(sum(sm) - 1.0) < 0.001
    assert sm[2] > sm[1] > sm[0]
