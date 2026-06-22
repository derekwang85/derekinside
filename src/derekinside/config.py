"""
derekinside — Configuration loader.

Reads config.yaml and provides typed access.
Search order: 1) DEREPATH env var  2) ./config.yaml  3) ~/.config/derekinside/config.yaml
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5434
    name: str = "derekinside"
    user: str = "postgres"
    password: str = "postgres"
    schema: str = "public"

    @property
    def dsn(self) -> str:
        return f"dbname={self.name} host={self.host} port={self.port} user={self.user} password={self.password}"

    @property
    def dsn_no_pass(self) -> str:
        return f"dbname={self.name} host={self.host} port={self.port} user={self.user}"


@dataclass
class EmbeddingConfig:
    provider: str = "ollama"
    model: str = "bge-m3"
    dimensions: int = 1024
    url: str = "http://localhost:11434/api/embed"


@dataclass
class RerankConfig:
    enabled: bool = False
    provider: str = "ollama"
    model: str = "qwen2.5-coder:7b"


@dataclass
class TemporalBoostConfig:
    enabled: bool = False
    recent_days: int = 7
    recent_weight: float = 1.5


@dataclass
class SearchConfig:
    top_k: int = 20
    rerank: RerankConfig = field(default_factory=RerankConfig)
    temporal_boost: TemporalBoostConfig = field(default_factory=TemporalBoostConfig)


@dataclass
class EntityExtractionConfig:
    """
    Entity extraction mode configuration.

    Modes:
      regex         — Pure regex (fast, ~5ms/chunk, code entities only)
      1.5b          — LLM-only with qwen2.5-coder:1.5b (~2.6s/chunk, high recall)
      7b            — LLM-only with qwen2.5-coder:7b (~6.6s/chunk, high precision)
      hybrid-1.5b   — Regex + 1.5B concepts (best balance on CPU)
      hybrid-7b     — Regex + 7B concepts (highest precision)
    """
    mode: str = "hybrid-7b"
    ollama_url: str = "http://localhost:11434/api/generate"
    llm_min_chars: int = 100


@dataclass
class KnowledgeGraphConfig:
    enabled: bool = False
    extraction_model: str = "qwen2.5-coder:7b"
    entity_extraction: EntityExtractionConfig = field(default_factory=EntityExtractionConfig)


@dataclass
class MCPConfig:
    enabled: bool = False
    port: int = 18892


@dataclass
class SourceConfig:
    name: str
    type: str  # git | filesystem
    path: str
    wing: str
    room: str
    patterns: list[str] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    level: str = "info"
    file: str = "~/.derekinside/derekinside.log"


@dataclass
class AppConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: dict = field(
        default_factory=lambda: {"provider": "pgvector", "hybrid_search": True}
    )
    search: SearchConfig = field(default_factory=SearchConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    mcp_server: MCPConfig = field(default_factory=MCPConfig)
    sources: list[SourceConfig] = field(default_factory=list)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: dict = field(default_factory=dict)
    _raw: dict = field(default_factory=dict)  # full raw config dict

    def to_dict(self) -> dict:
        """Return full config as a dict (for Engine initialization)."""
        return self._raw or {}


def find_config() -> Path:
    """Locate config.yaml by search order."""
    env_path = os.environ.get("DEREPATH")
    if env_path:
        p = Path(env_path) / "config.yaml"
        if p.exists():
            return p

    for loc in [Path("config.yaml"), Path.home() / ".config/derekinside/config.yaml"]:
        if loc.exists():
            return loc

    raise FileNotFoundError(
        "config.yaml not found. Set DEREPATH or run from project root."
    )


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load and parse config.yaml into AppConfig."""
    if path is None:
        path = find_config()

    raw = yaml.safe_load(path.read_text())

    cfg = AppConfig()

    # Database
    db = raw.get("database", {})
    cfg.database = DatabaseConfig(
        **{k: v for k, v in db.items() if k in DatabaseConfig.__dataclass_fields__}
    )

    # Embedding
    emb = raw.get("embedding", {})
    cfg.embedding = EmbeddingConfig(
        **{k: v for k, v in emb.items() if k in EmbeddingConfig.__dataclass_fields__}
    )

    # Storage
    cfg.storage = raw.get("storage", {"provider": "pgvector", "hybrid_search": True})

    # Search
    s = raw.get("search", {})
    reraw = s.get("rerank", {})
    temraw = s.get("temporal_boost", {})
    cfg.search = SearchConfig(
        top_k=s.get("top_k", 20),
        rerank=RerankConfig(
            **{k: v for k, v in reraw.items() if k in RerankConfig.__dataclass_fields__}
        ),
        temporal_boost=TemporalBoostConfig(
            **{
                k: v
                for k, v in temraw.items()
                if k in TemporalBoostConfig.__dataclass_fields__
            }
        ),
    )

    # Knowledge Graph
    kg = raw.get("knowledge_graph", {})
    ee_raw = kg.get("entity_extraction", {})
    cfg.knowledge_graph = KnowledgeGraphConfig(
        enabled=kg.get("enabled", False),
        extraction_model=kg.get("extraction_model", "qwen2.5-coder:7b"),
        entity_extraction=EntityExtractionConfig(
            mode=ee_raw.get("mode", "hybrid-7b"),
            ollama_url=ee_raw.get("ollama_url", "http://localhost:11434/api/generate"),
            llm_min_chars=ee_raw.get("llm_min_chars", 100),
        ),
    )

    # MCP
    mcp = raw.get("mcp_server", {})
    cfg.mcp_server = MCPConfig(
        **{k: v for k, v in mcp.items() if k in MCPConfig.__dataclass_fields__}
    )

    # Sources
    srcs = raw.get("sources", [])
    for s in srcs:
        cfg.sources.append(
            SourceConfig(
                name=s["name"],
                type=s.get("type", "filesystem"),
                path=s.get("path", ""),
                wing=s.get("wing", "uncategorized"),
                room=s.get("room", "uncategorized"),
                patterns=s.get("patterns", []),
                ignore=s.get("ignore", []),
            )
        )

    # Logging
    log = raw.get("logging", {})
    cfg.logging = LoggingConfig(
        **{k: v for k, v in log.items() if k in LoggingConfig.__dataclass_fields__}
    )

    # Pipeline (new)
    cfg.pipeline = raw.get("pipeline", {})

    # Store raw dict for Engine initialization
    cfg._raw = raw

    return cfg
