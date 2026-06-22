"""
derekinside — Smart dispatch classifier.

Selects the optimal entity extraction mode based on chunk content type.
No external model needed — pure rule-based classification (~70ms/chunk).
"""

from __future__ import annotations
import logging, re

logger = logging.getLogger(__name__)

_CODE_EXTS = {".java", ".py", ".vue", ".js", ".ts", ".jsx", ".tsx",
              ".go", ".rs", ".rb", ".php", ".kt", ".swift", ".c", ".cpp", ".h"}
_DOCS_EXTS = {".md", ".rst", ".txt", ".adoc", ".wiki"}
_CONFIG_EXTS = {".xml", ".yml", ".yaml", ".sql", ".properties", ".conf",
                ".gradle", ".toml", ".ini", ".cfg"}
_LOG_EXTS = {".log", ".out", ".err"}
_SKIP_EXTS = {".lock", ".min.js", ".min.css", ".map", ".svg",
              ".png", ".jpg", ".gif", ".ico", ".woff", ".ttf",
              ".zip", ".tar", ".gz", ".jar", ".war", ".class",
              ".o", ".a", ".so", ".dll", ".dylib",
              ".pyc", ".pyo", ".pkl", ".pickle"}

_TYPE_TO_MODE = {"code": "regex", "docs": "hybrid-1.5b", "config": "1.5b", "log": "", "skip": ""}

_CODE_KEYWORDS = [
    r'\b(?:class|interface|enum|struct|def|fn|function)\s+\w+',
    r'\bextends\s+\w+',
    r'\bimplements\s+\w+',
    r'@\w+\s*(?:\(|$)',
    r'import\s+[\w.*]+',
    r'public|private|protected\s+(?:class|interface|void|\w+)',
    r'\{[\s\S]*?\}',
]

_CONFIG_KEYWORDS = [
    r'<[a-zA-Z]+[\s/>]',
    r'</\w+>',
    r'<\w+>\s*</\w+>',
    r'(?:CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|SELECT)\s+',
    r'<dependency>',
    r'dependency\s*\{',
    r'@(?:Value|ConfigurationProperties)',
    r'\w+[:=]\s*(?:\d+|true|false|null|\[\]|[\w./@$-]+)',
    r'\w+\.\w+\s*[:=]\s*\S+',
]

_LOG_PATTERNS = [
    r'^\d{4}-\d{2}-\d{2}[ T]',
    r'^(INFO|WARN|ERROR|DEBUG|TRACE|FATAL)\s',
    r'^\s*at\s+[\w.]+\(',
]


def _base_ext(ext: str) -> str:
    for known_set in [_CODE_EXTS, _DOCS_EXTS, _CONFIG_EXTS]:
        for known in known_set:
            if ext.endswith(known):
                return known
    return ext


def classify_chunk(chunk_text: str, file_ext: str = "") -> str:
    ext = file_ext.lower() if file_ext else ""
    if ext in _SKIP_EXTS: return "skip"
    if ext in _CODE_EXTS: return "code"
    if ext in _LOG_EXTS: return "log"

    text = chunk_text.strip()
    if not text: return "skip"

    lines = text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    log_count = sum(1 for l in non_empty if any(re.match(p, l) for p in _LOG_PATTERNS))
    if non_empty and log_count / len(non_empty) > 0.5:
        return "log"

    base = _base_ext(ext)
    code_score = _keyword_match(text, _CODE_KEYWORDS)
    config_score = _keyword_match(text, _CONFIG_KEYWORDS)

    if code_score >= 3:
        return "code"
    if config_score >= 1:
        return "config"
    if base in _DOCS_EXTS or _is_doc_like(text):
        return "docs"
    if code_score >= config_score and code_score > 0:
        return "code"
    return "docs"


def select_mode(chunk_text: str, file_ext: str = "", default_mode: str = "hybrid-7b") -> str:
    ctype = classify_chunk(chunk_text, file_ext)
    mode = _TYPE_TO_MODE.get(ctype, "")
    if not mode: return ""
    if ctype == "docs" and mode == "hybrid-1.5b":
        return default_mode
    return mode


def _keyword_match(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def _is_doc_like(text: str) -> bool:
    lines = text.strip().split("\n")
    if not lines: return True
    header_count = sum(1 for l in lines if re.match(r'^#{1,6}\s', l))
    para_count = sum(1 for l in lines if len(l) > 60 and re.search(r'[\u4e00-\u9fff]', l))
    para_count += sum(1 for l in lines if len(l) > 80 and re.search(r'[a-zA-Z]{3,}', l) and not re.search(r'[{}().*]', l))
    return header_count >= 1 or para_count >= 2
