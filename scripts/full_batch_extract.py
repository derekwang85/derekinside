#!/usr/bin/env python3
"""
Full batch entity extraction — run one mode on ALL chunks, save result.

Usage:
    python3 scripts/full_batch_extract.py --mode regex
    python3 scripts/full_batch_extract.py --mode 1.5b
    python3 scripts/full_batch_extract.py --mode 7b
    python3 scripts/full_batch_extract.py --mode hybrid-7b

Output: /tmp/kg-full-{mode}.json
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from derekinside.cli import DereContext  # noqa: E402

# ── Regex Patterns (synced with entity.py) ────────────────────

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
        re.compile(r'@(?:GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping)\(["\']([^"\']+)["\']'),
        re.compile(r"(?:router\.(?:get|post|put|delete)|app\.(?:get|post|put|delete))\(['\"](/[\w/{}]+)['\"]"),
    ],
}

_GENERIC = {
    "void", "int", "string", "boolean", "long", "double",
    "true", "false", "null", "this", "return", "if", "else",
    "for", "while", "class", "function", "import", "export",
    "default", "extends", "implements", "throws", "new",
    "type", "default", "extends",
}

_LLM_PROMPT = """Extract named entities from this text. Types: class, function, module, api, concept.
Return only JSON: {{"entities":[{{"name":"X","type":"class"}}]}}
No explanation. Empty: {{"entities":[]}}

TEXT:
{text}"""


def extract_regex(text: str) -> list:
    seen = set()
    result = []
    for etype, patterns in _PATTERNS.items():
        for pat in patterns:
            for m in pat.finditer(text):
                name = m.group(1).strip()
                if len(name) >= 2 and name not in seen and name.lower() not in _GENERIC:
                    seen.add(name)
                    result.append({"name": name, "entity_type": etype})
    return result


def extract_imports(text: str) -> list:
    result = []
    seen = set()
    for m in re.finditer(r"^import\s+([\w.]+(?:\.\w+)?)\s*;", text, re.MULTILINE):
        parts = m.group(1).split(".")
        for p in parts:
            if p[0].isupper() and p not in seen:
                seen.add(p)
                result.append({"name": p, "entity_type": "class"})
    return result


def extract_llm(text: str, model: str, cid: int = 0) -> list:
    """Use curl subprocess to call Ollama (most reliable)."""
    prompt = _LLM_PROMPT.format(text=text[:1500])
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 300, "temperature": 0.1},
    })
    try:
        r = subprocess.run(
            ["curl", "-s", "-X", "POST", "http://localhost:11434/api/generate",
             "-H", "Content-Type: application/json",
             "-d", payload,
             "--max-time", "60"],
            capture_output=True, text=True, timeout=65
        )
        if r.returncode != 0:
            print(f"  ⚠️  Chunk #{cid}: curl error (rc={r.returncode})", file=sys.stderr)
            return None
        raw = json.loads(r.stdout).get("response", "").strip()
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Chunk #{cid}: timeout (>60s)", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ⚠️  Chunk #{cid}: {e}", file=sys.stderr)
        return []

    # Parse JSON from response
    json_str = raw
    for fmt in ["```json", "```"]:
        if fmt in json_str:
            json_str = json_str.split(fmt)[1].split("```")[0].strip()
    bs = json_str.find("{")
    be = json_str.rfind("}")
    if bs >= 0 and be > bs:
        json_str = json_str[bs:be + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    result = []
    for e in data.get("entities", []):
        if isinstance(e, dict) and "name" in e:
            name = e["name"].strip()
            etype = e.get("type", "concept")
            if 2 <= len(name) <= 80 and etype != "variable" and re.search(r"[a-zA-Z0-9_]", name):
                result.append({"name": name, "entity_type": etype})
    return result


def dedup(entities: list) -> list:
    seen = set()
    result = []
    for e in entities:
        key = (e["name"], e["entity_type"])
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["regex", "1.5b", "7b", "hybrid-7b"])
    parser.add_argument("--batch", type=int, default=20, help="Progress report interval")
    args = parser.parse_args()

    mode = args.mode
    batch_size = args.batch
    print(f"🧠 Mode: {mode}", file=sys.stderr)

    # Load all chunks
    dc = DereContext()
    dc.connect()
    with dc.store.cursor() as cur:
        cur.execute(
            "SELECT id, chunk_text FROM chunks "
            "WHERE LENGTH(chunk_text) > 80 ORDER BY id"
        )
        chunks = cur.fetchall()

    total = len(chunks)
    print(f"📦 {total} chunks loaded", file=sys.stderr)

    all_entities = []
    llm_model = "qwen2.5-coder:" + mode.replace("hybrid-", "") if "hybrid" in mode or mode in ("1.5b", "7b") else None
    t0 = time.time()

    for i, (cid, text) in enumerate(chunks):
        # Regex entities (common to regex/hybrid modes)
        regex_ents = extract_regex(text)
        import_ents = extract_imports(text)
        combined_regex = dedup(regex_ents + import_ents)

        if mode == "regex":
            all_entities.extend(combined_regex)
        elif mode in ("1.5b", "7b"):
            llm_ents = extract_llm(text, llm_model, cid)
            if llm_ents is None:
                print(f"  🔄 Chunk #{cid}: retrying...", file=sys.stderr)
                llm_ents = extract_llm(text, llm_model, cid)
                if llm_ents is None:
                    llm_ents = []
            all_entities.extend(llm_ents or [])
        elif mode == "hybrid-7b":
            # Regex first, LLM for concepts not found
            all_entities.extend(combined_regex)
            regex_names = {e["name"] for e in combined_regex}
            llm_ents = extract_llm(text, llm_model, cid)
            if llm_ents is None:
                print(f"  🔄 Chunk #{cid}: retrying...", file=sys.stderr)
                llm_ents = extract_llm(text, llm_model, cid)
                if llm_ents is None:
                    llm_ents = []
            for e in (llm_ents or []):
                if e["name"] not in regex_names and e["entity_type"] in ("concept", "api"):
                    all_entities.append(e)
                    regex_names.add(e["name"])

        # Per-chunk log for LLM modes (slow)
        if mode in ("7b", "1.5b", "hybrid-7b") and (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"  ⏳ {i+1}/{total} (chunk #{cid}, {rate:.1f}/s, ETA {remaining/60:.0f}min)", file=sys.stderr)
        elif (i + 1) % batch_size == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"  ⏳ {i+1}/{total} ({rate:.1f}/s, ETA {remaining/60:.0f}min)...", file=sys.stderr)

    elapsed = time.time() - t0
    final_ents = dedup(all_entities)

    # Type breakdown
    breakdown = {}
    for e in final_ents:
        breakdown[e["entity_type"]] = breakdown.get(e["entity_type"], 0) + 1

    print(f"\n✅ Mode: {mode}", file=sys.stderr)
    print(f"   Total chunks: {total}", file=sys.stderr)
    print(f"   Time: {elapsed:.0f}s ({elapsed/total:.1f}s/chunk)", file=sys.stderr)
    print(f"   Entities: {len(final_ents)}", file=sys.stderr)
    for t, c in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"     {t}: {c}", file=sys.stderr)

    # Save result
    result = {
        "mode": mode,
        "chunks": total,
        "time_s": round(elapsed, 1),
        "entities": final_ents,
        "entity_count": len(final_ents),
        "type_breakdown": breakdown,
    }
    out_path = f"/tmp/kg-full-{mode}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"📁 Saved: {out_path}", file=sys.stderr)
    print(f"\n📊 {mode}: {len(final_ents)} entities in {elapsed:.0f}s")
    for t, c in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"   {t}: {c}")


if __name__ == "__main__":
    main()
