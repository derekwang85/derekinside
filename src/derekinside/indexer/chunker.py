"""
derekinside — Smart chunker for code files, markdown, and conversations.

Splits files into chunks with configurable strategies:
- Code: split by function/class boundaries with imports as preamble
- Markdown: split by heading level (## or ###)
- Plain text: fixed-size with overlap
- Conversations: split by speaker turns
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Chunk:
    text: str
    index: int
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


# ── Strategy Detection ─────────────────────────────────────────


def detect_strategy(filepath: str | Path) -> str:
    """Detect chunking strategy from file extension and content."""
    path = Path(filepath)
    ext = path.suffix.lower()

    # Markdown
    if ext in (".md", ".mdx", ".markdown"):
        return "markdown"

    # Code files
    code_exts = {
        ".py": "code",
        ".java": "code",
        ".js": "code",
        ".ts": "code",
        ".vue": "code",
        ".jsx": "code",
        ".tsx": "code",
        ".go": "code",
        ".rs": "code",
        ".rb": "code",
        ".php": "code",
        ".c": "code",
        ".cpp": "code",
        ".h": "code",
        ".hpp": "code",
        ".swift": "code",
        ".kt": "code",
        ".sql": "sql",
        ".yaml": "config",
        ".yml": "config",
        ".json": "config",
        ".xml": "config",
        ".toml": "config",
        ".html": "code",
        ".css": "code",
        ".scss": "code",
    }
    strategy = code_exts.get(ext)
    if strategy:
        return strategy

    # Config/log files
    if ext in (".cfg", ".ini", ".conf", ".log"):
        return "lines"

    # Default
    return "plain"


# ── Markdown Chunker ───────────────────────────────────────────


def chunk_markdown(text: str, max_chars: int = 2000) -> list[Chunk]:
    """Split markdown by ## headings."""
    lines = text.split("\n")
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_size = 0

    for line in lines:
        # Heading level 2 or 3 starts a new chunk
        if re.match(r"^#{2,3}\s", line):
            if current_lines and current_size > 0:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            index=len(chunks),
                            token_count=len(chunk_text) // 4 + 1,
                            metadata={"strategy": "markdown", "boundary": "heading"},
                        )
                    )
            current_lines = [line]
            current_size = len(line)
        else:
            current_lines.append(line)
            current_size += len(line) + 1
            # Hard max_chars boundary
            if current_size >= max_chars:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            index=len(chunks),
                            token_count=len(chunk_text) // 4 + 1,
                            metadata={"strategy": "markdown", "boundary": "size"},
                        )
                    )
                current_lines = []
                current_size = 0

    # Last chunk
    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    token_count=len(chunk_text) // 4 + 1,
                    metadata={"strategy": "markdown", "boundary": "end"},
                )
            )

    return chunks


# ── Code Chunker (Python-aware) ────────────────────────────────

_FUNC_RE = re.compile(r"^(async\s+)?def\s+\w+|^(async\s+)?class\s+\w+")
_IMPORT_RE = re.compile(r"^(import |from \w+ import|from \w+\.)")


def chunk_code(
    text: str, language: str = "python", max_chars: int = 3000, min_chars: int = 100
) -> list[Chunk]:
    """Split code by function/class boundaries."""
    lines = text.split("\n")
    chunks: list[Chunk] = []
    preamble_lines: list[str] = []
    current_lines: list[str] = []
    current_size = 0
    has_preamble = False

    for i, line in enumerate(lines):
        is_func_def = _FUNC_RE.match(line)

        if is_func_def and current_lines:
            # Flush current chunk
            seg_text = "\n".join(current_lines).strip()
            if seg_text:
                chunks.append(
                    Chunk(
                        text=seg_text,
                        index=len(chunks),
                        token_count=len(seg_text) // 4 + 1,
                        metadata={
                            "strategy": "code",
                            "line_start": i - len(current_lines) + 1,
                        },
                    )
                )
            current_lines = [line]
            current_size = len(line)
        elif is_func_def and not current_lines:
            current_lines = [line]
            current_size = len(line)
        else:
            # Collect preamble (imports before first function)
            if not has_preamble and _IMPORT_RE.match(line):
                preamble_lines.append(line)
            elif not has_preamble and line.strip():
                has_preamble = True
                current_lines = preamble_lines + current_lines if current_lines else []
                current_lines.append(line)
                current_size = sum(len(ln) + 1 for ln in current_lines)
            else:
                current_lines.append(line)
                current_size += len(line) + 1

            # Hard boundary
            if current_size >= max_chars and len(current_lines) > 2:
                seg_text = "\n".join(current_lines).strip()
                if seg_text:
                    chunks.append(
                        Chunk(
                            text=seg_text,
                            index=len(chunks),
                            token_count=len(seg_text) // 4 + 1,
                            metadata={"strategy": "code", "boundary": "size"},
                        )
                    )
                current_lines = []
                current_size = 0

    # Last chunk
    if current_lines:
        # If it's small and there's a previous chunk, merge
        seg_text = "\n".join(current_lines).strip()
        if seg_text:
            if chunks and len(seg_text) < min_chars:
                chunks[-1].text += "\n" + seg_text
                chunks[-1].token_count = len(chunks[-1].text) // 4 + 1
            else:
                chunks.append(
                    Chunk(
                        text=seg_text,
                        index=len(chunks),
                        token_count=len(seg_text) // 4 + 1,
                        metadata={"strategy": "code", "boundary": "end"},
                    )
                )

    return chunks


# ── Plain Text Chunker ─────────────────────────────────────────


def chunk_plain(text: str, max_chars: int = 1500, overlap: int = 100) -> list[Chunk]:
    """Fixed-size chunks with overlap."""
    if len(text) <= max_chars:
        return [
            Chunk(
                text=text,
                index=0,
                token_count=len(text) // 4 + 1,
                metadata={"strategy": "plain", "boundary": "single"},
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to break at sentence/line boundary
        if end < len(text):
            # Find last newline or period in range
            break_at = text.rfind("\n\n", start + max_chars // 2, end)
            if break_at == -1:
                break_at = text.rfind(". ", start + max_chars // 2, end)
            if break_at != -1:
                end = break_at + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    token_count=len(chunk_text) // 4 + 1,
                    metadata={"strategy": "plain", "start": start},
                )
            )
        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ── Lines Chunker (configs, logs) ──────────────────────────────


def chunk_lines(text: str, max_lines: int = 50) -> list[Chunk]:
    """Simple line-group chunker."""
    lines = text.split("\n")
    chunks: list[Chunk] = []
    for i in range(0, len(lines), max_lines):
        seg = "\n".join(lines[i : i + max_lines]).strip()
        if seg:
            chunks.append(
                Chunk(
                    text=seg,
                    index=len(chunks),
                    token_count=len(seg) // 4 + 1,
                    metadata={"strategy": "lines", "line_start": i + 1},
                )
            )
    return chunks


# ── Main Dispatcher ────────────────────────────────────────────


def chunk_file(
    filepath: str | Path, strategy: Optional[str] = None, max_chars: int = 2000
) -> list[Chunk]:
    """Chunk a file using auto-detected or explicit strategy."""
    path = Path(filepath)
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []

    if strategy is None:
        strategy = detect_strategy(filepath)

    if strategy == "markdown":
        return chunk_markdown(text, max_chars)
    elif strategy == "code":
        return chunk_code(text, path.suffix.lstrip("."), max_chars)
    elif strategy == "lines":
        return chunk_lines(text)
    else:
        return chunk_plain(text, max_chars)


def chunk_text(
    text: str, strategy: str = "markdown", max_chars: int = 2000
) -> list[Chunk]:
    """Chunk raw text with explicit strategy."""
    if strategy == "markdown":
        return chunk_markdown(text, max_chars)
    elif strategy == "code":
        return chunk_code(text, "text", max_chars)
    elif strategy == "lines":
        return chunk_lines(text)
    else:
        return chunk_plain(text, max_chars)
