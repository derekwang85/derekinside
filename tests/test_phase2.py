"""
Tests for derekinside Phase 2 — Knowledge Graph.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from derekinside.storage.graph import Entity, Relation
from derekinside.indexer.entity import EntityExtractor
from derekinside.search.propagation import GraphPropagator


# ── Entity Extractor ───────────────────────────────────────────


def test_extractor_disabled():
    ext = EntityExtractor(enabled=False)
    result = ext.extract("Some text here.")
    assert result.is_empty()


def test_extractor_parsing():
    ext = EntityExtractor(enabled=True)
    # Direct test of parser with mock-like JSON
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
    result = ext._parse(
        '{"entities": [{"name": "data", "type": "concept"}, {"name": "API", "type": "api"}], "relations": []}'
    )
    assert len(result.entities) == 1  # "data" should be filtered out
    assert result.entities[0].name == "API"


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


def test_extractor_missing_relation_endpoint():
    ext = EntityExtractor(enabled=True)
    result = ext._parse("""
    {
        "entities": [{"name": "ServiceA", "type": "class"}],
        "relations": [{"source": "ServiceA", "target": "ServiceB", "type": "calls"}]
    }
    """)
    assert len(result.entities) == 1
    assert len(result.relations) == 0  # "ServiceB" is not in entities


def test_extractor_regex_fallback():
    ext = EntityExtractor(enabled=True)
    result = ext._parse_with_regex(
        'entities: [{"name": "Foo", "type": "class"}] '
        'relations: [{"source": "Foo", "target": "Bar", "type": "calls"}]'
    )
    assert result is not None
    assert len(result["entities"]) == 1
    assert result["entities"][0]["name"] == "Foo"


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
    assert sm[2] > sm[1] > sm[0]  # highest gets most
