"""
Tests for derekinside Phase 1 core modules.
"""

import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from derekinside import config as cfgmod
from derekinside.indexer.chunker import (
    chunk_file,
    detect_strategy,
    chunk_markdown,
    chunk_code,
    chunk_plain,
    chunk_lines,
)
from derekinside.indexer.embedder import Embedder


# ── Config ─────────────────────────────────────────────────────


def test_load_config():
    """Can load config.yaml from project root."""
    cfg = cfgmod.load_config(Path("config.yaml"))
    assert cfg.database.name == "derekinside"
    assert cfg.database.port == 5434
    assert cfg.embedding.model == "bge-m3"
    assert cfg.embedding.dimensions == 1024
    assert cfg.search.rerank.model == "qwen2.5-coder:7b"
    assert len(cfg.sources) > 0


def test_dsn():
    cfg = cfgmod.DatabaseConfig(name="test", user="u", password="p", host="h", port=1)
    assert "dbname=test" in cfg.dsn
    assert "password=p" in cfg.dsn


# ── Chunker ────────────────────────────────────────────────────


def test_detect_strategy():
    assert detect_strategy("foo.md") == "markdown"
    assert detect_strategy("bar.py") == "code"
    assert detect_strategy("baz.java") == "code"
    assert detect_strategy("qux.txt") == "plain"
    assert detect_strategy("conf.yaml") == "config"
    assert detect_strategy("log.log") == "lines"


def test_chunk_markdown():
    text = "# Title\n\n## Section 1\n\nSome content here.\n\n## Section 2\n\nMore content.\n\n### Subsection\n\nDetails.\n"
    chunks = chunk_markdown(text)
    assert len(chunks) >= 1
    # All chunks should have text
    for c in chunks:
        assert len(c.text) > 0
        assert isinstance(c.index, int)


def test_chunk_code():
    code = """import os
import sys

def foo():
    return 1

class Bar:
    def baz(self):
        return 2

def qux():
    return 3
"""
    chunks = chunk_code(code, "python")
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.text) > 0


def test_chunk_plain():
    text = "Word " * 1000
    chunks = chunk_plain(text, max_chars=500)
    assert len(chunks) >= 2  # Should split into multiple chunks


def test_chunk_lines():
    text = "\n".join(f"line {i}" for i in range(100))
    chunks = chunk_lines(text, max_lines=30)
    assert len(chunks) == 4  # 100 / 30 = 4 (ceil)


def test_chunk_file_markdown(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("# Hello\n\n## World\n\nSome text here\n")
    chunks = chunk_file(str(f))
    assert len(chunks) > 0


def test_chunk_file_code(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("def hello():\n    return 'world'\n")
    chunks = chunk_file(str(f))
    assert len(chunks) > 0


# ── Embedder ───────────────────────────────────────────────────


def test_embedder_count_tokens():
    emb = Embedder()  # no real connection needed for token counting
    assert emb.count_tokens("hello world") == 3
    assert emb.count_tokens("a" * 100) == 26


# ── CSV (Config Schema Validation) ─────────────────────────────


def test_config_source_defaults():
    src = cfgmod.SourceConfig(
        name="test", type="filesystem", path="/tmp", wing="w", room="r"
    )
    assert src.patterns == []
    assert src.ignore == []


def test_app_config_defaults():
    c = cfgmod.AppConfig()
    assert c.database.port == 5434
    assert c.embedding.dimensions == 1024
    assert c.search.top_k == 20
    assert c.knowledge_graph.enabled is False
    assert c.mcp_server.enabled is False
