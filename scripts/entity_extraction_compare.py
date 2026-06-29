#!/usr/bin/env python3
"""
Entity Extraction Comparison — 同等条件下对比三种方案

Usage:
    # 默认：100 chunks, qwen2.5-coder:7b
    python3 scripts/entity_extraction_compare.py

    # 指定模型和 chunk 数
    python3 scripts/entity_extraction_compare.py --model qwen2.5-coder:1.5b --chunks 100

    # 全部对比（耗时较长）
    python3 scripts/entity_extraction_compare.py --all
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# ── Add derekinside to path ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from derekinside.cli import DereContext  # noqa: E402

# ── Data Classes ──────────────────────────────────────────────


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    errors: int = 0
    time_ms: float = 0.0


# ── Regex Patterns (from entity.py) ──────────────────────────

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
}

_GENERIC = {
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
    "extends",
    "implements",
    "throws",
    "new",
    "type",
    "default",
    "extends",
}

_LLM_PROMPT = """Extract named entities from this text. Types: class, function, module, api, concept.
Return only JSON: {{"entities":[{{"name":"X","type":"class"}}]}}
No explanation. Empty: {{"entities":[]}}

TEXT:
{text}"""

_LLM_PROMPT_COMBINED = """Extract named entities and relationships from this text.

Entity types: class, function, module, api, concept
- class: Java/Python/TS class names (e.g. UserService, KYCApplication)
- function: method/function names (e.g. processKYC, handleApproval)
- module: package/import module names (e.g. com.tradeoms.kyc, vue-router)
- api: API endpoint paths or system interfaces (e.g. /api/v1/kyc)
- concept: business domain concepts (e.g. KYC流程, 审批, 风险评估, 买方, 合同)

Return JSON: {{"entities":[{{"name":"X","type":"class"}}], "relations":[{{"source":"A","target":"B","type":"related"}}]}}
No explanation. Empty arrays: {{"entities":[],"relations":[]}}

TEXT:
{text}"""

# ── Extraction Methods ───────────────────────────────────────


def _is_valid(name: str) -> bool:
    if len(name) < 2 or len(name) > 80:
        return False
    return bool(re.search(r"[a-zA-Z0-9_]", name))


def extract_regex(text: str) -> ExtractionResult:
    t0 = time.time()
    result = ExtractionResult()
    seen = set()
    for etype, patterns in _PATTERNS.items():
        for pat in patterns:
            for m in pat.finditer(text):
                name = m.group(1).strip()
                if len(name) >= 2 and name not in seen and name.lower() not in _GENERIC:
                    seen.add(name)
                    result.entities.append(ExtractedEntity(name, etype))
    result.time_ms = (time.time() - t0) * 1000
    return result


def extract_llm(
    text: str, model: str, client: httpx.Client, use_combined_prompt: bool = False
) -> ExtractionResult:
    t0 = time.time()
    result = ExtractionResult()
    prompt = _LLM_PROMPT_COMBINED if use_combined_prompt else _LLM_PROMPT
    try:
        resp = client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt.format(text=text[:1500]),
                "stream": False,
                "options": {"num_predict": 300, "temperature": 0.1},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "").strip()
    except Exception:
        result.errors = 1
        result.time_ms = (time.time() - t0) * 1000
        return result

    # Parse JSON
    json_str = raw
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()
    bs = json_str.find("{")
    be = json_str.rfind("}")
    if bs >= 0 and be > bs:
        json_str = json_str[bs : be + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        result.errors = 1
        result.time_ms = (time.time() - t0) * 1000
        return result

    for e in data.get("entities", []):
        if isinstance(e, dict) and "name" in e:
            name = e["name"].strip()
            etype = e.get("type", "concept")
            if _is_valid(name) and etype != "variable":
                result.entities.append(ExtractedEntity(name, etype))
    result.time_ms = (time.time() - t0) * 1000
    return result


def extract_hybrid(
    regex_ents: list[ExtractedEntity], llm_ents: list[ExtractedEntity]
) -> list[ExtractedEntity]:
    """Combination: regex first, LLM adds concepts not already found."""
    seen = set()
    combined = []
    for e in regex_ents:
        key = (e.name, e.entity_type)
        if key not in seen:
            seen.add(key)
            combined.append(e)
    for e in llm_ents:
        key = (e.name, e.entity_type)
        if key not in seen and e.entity_type in ("concept", "api"):
            seen.add(key)
            combined.append(e)
    return combined


# ── Dedup Helper ─────────────────────────────────────────────


def dedup(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    seen = set()
    result = []
    for e in entities:
        key = (e.name, e.entity_type)
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


# ── Run Comparison ────────────────────────────────────────────


def run_comparison(
    test_chunks: list[tuple[int, str]], model: str, client: httpx.Client
) -> dict:
    """Run regex + LLM + hybrid on test chunks, return stats."""
    all_regex: list[ExtractedEntity] = []
    all_llm: list[ExtractedEntity] = []
    all_hybrid: list[ExtractedEntity] = []
    regex_time = 0.0
    llm_time = 0.0
    llm_errors = 0
    chunk_count = len(test_chunks)

    for i, (cid, text) in enumerate(test_chunks):
        # Regex
        rr = extract_regex(text)
        all_regex.extend(rr.entities)
        regex_time += rr.time_ms

        # LLM
        lr = extract_llm(text, model, client)
        all_llm.extend(lr.entities)
        llm_time += lr.time_ms
        llm_errors += lr.errors

        # Hybrid: on the fly
        hr = extract_hybrid(rr.entities, lr.entities)
        all_hybrid.extend(hr)

        if (i + 1) % 20 == 0:
            print(f"  ⏳ {i+1}/{chunk_count} chunks...", file=sys.stderr)

    # Dedup
    stats = {
        "model": model,
        "chunks": chunk_count,
        "regex": {
            "entities": dedup(all_regex),
            "time_ms": round(regex_time, 1),
        },
        "llm": {
            "entities": dedup(all_llm),
            "time_ms": round(llm_time, 1),
            "errors": llm_errors,
        },
        "hybrid": {
            "entities": dedup(all_hybrid),
            "time_ms": round(regex_time + llm_time, 1),
        },
    }

    # Type breakdowns
    for method in ("regex", "llm", "hybrid"):
        breakdown = {}
        for e in stats[method]["entities"]:
            breakdown[e.entity_type] = breakdown.get(e.entity_type, 0) + 1
        stats[method]["type_breakdown"] = breakdown

    return stats


# ── Print Results ─────────────────────────────────────────────


def print_comparison(results: dict, label: str = ""):
    """Pretty-print comparison table."""
    print(f"\n{'='*70}")
    model_name = results["model"]
    print(f"📊 {label or f'Comparison: {model_name}'}")
    print(f"{'='*70}")

    for method in ("regex", "llm", "hybrid"):
        data = results[method]
        ents = data["entities"]
        type_bd = data.get("type_breakdown", {})
        nchunks = results["chunks"]
        time_s = data["time_ms"] / 1000

        print(f"\n  🔹 {method.upper()}")
        print(f"     Total entities: {len(ents)}")
        types_str = ", ".join(
            f"{k}={v}" for k, v in sorted(type_bd.items(), key=lambda x: -x[1])
        )
        if types_str:
            print(f"     Types: {types_str}")
        print(f"     Time: {time_s:.1f}s ({data['time_ms']/nchunks:.0f}ms/chunk)")

        # Show entities by type
        for etype in (
            "concept",
            "api",
            "class",
            "function",
            "module",
            "file",
            "constant",
        ):
            matched = [e.name for e in ents if e.entity_type == etype]
            if matched:
                names = sorted(matched)
                show = names[:12]
                suffix = f" ... +{len(names)-12} more" if len(names) > 12 else ""
                print(f"       {etype}: {', '.join(show)}{suffix}")

    # Overlap analysis
    r_set = {(e.name, e.entity_type) for e in results["regex"]["entities"]}
    l_set = {(e.name, e.entity_type) for e in results["llm"]["entities"]}
    h_set = {(e.name, e.entity_type) for e in results["hybrid"]["entities"]}

    print("\n  🔄 Overlap Analysis:")
    print(f"     Regex-only:  {len(r_set - l_set)}")
    print(f"     LLM-only:    {len(l_set - r_set)}")
    print(f"     Shared:      {len(r_set & l_set)}")
    print(f"     Hybrid total: {len(h_set)}")

    # Show LLM-unique concept/API entities
    llm_unique = [
        e
        for e in results["llm"]["entities"]
        if (e.name, e.entity_type) not in r_set and e.entity_type in ("concept", "api")
    ]
    print("\n  💡 LLM-added concepts/APIs (vs regex):")
    if llm_unique:
        for e in sorted(llm_unique, key=lambda x: x.entity_type + x.name):
            print(f"     [{e.entity_type}] {e.name}")
    else:
        print("     (none)")

    print()


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Entity extraction comparison")
    parser.add_argument(
        "--chunks", type=int, default=100, help="Number of chunks to test"
    )
    parser.add_argument(
        "--model", default="qwen2.5-coder:7b", help="LLM model for comparison"
    )
    parser.add_argument("--all", action="store_true", help="Compare all models")
    args = parser.parse_args()

    # Get chunks from DB
    print("📦 Loading test chunks from DB...", file=sys.stderr)
    dc = DereContext()
    dc.connect()
    with dc.store.cursor() as cur:
        cur.execute(
            "SELECT id, chunk_text FROM chunks "
            "WHERE LENGTH(chunk_text) > 80 ORDER BY RANDOM() LIMIT %s",
            (args.chunks,),
        )
        test_chunks = cur.fetchall()
    ids = [r[0] for r in test_chunks]
    print(
        f"✅ Loaded {len(test_chunks)} chunks (IDs {min(ids)} ~ {max(ids)})",
        file=sys.stderr,
    )

    if args.all:
        models = ["qwen2.5-coder:1.5b", "qwen2.5-coder:7b"]
    else:
        models = [args.model]

    for model in models:
        print(f"\n🧠 Running with model: {model}", file=sys.stderr)
        client = httpx.Client(timeout=120.0)
        results = run_comparison(test_chunks, model, client)
        client.close()
        print_comparison(results, model)

        # Save JSON
        out_path = f"/tmp/entity-compare-{model.replace(':', '-')}.json"
        with open(out_path, "w") as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
                default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
            )
        print(f"📁 Results saved to: {out_path}", file=sys.stderr)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
