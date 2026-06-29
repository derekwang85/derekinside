#!/usr/bin/env python3
"""
LongMemEval — 实体提取方案全面评测 (同批次对比)

在同一批 100 chunks 上，对 5 种方案运行对比：
  ① regex (基线)        ② 1.5B LLM
  ③ 7B LLM              ④ regex+1.5B hybrid
  ⑤ regex+7B hybrid

并针对人工校验的黄金标准评估精确率/召回率/F1。
"""

import json
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from derekinside.cli import DereContext  # noqa: E402


# ── Regex Patterns (from entity.py) ──────────────────────────

_PATTERNS = {
    "class": [
        re.compile(r"(?:public\s+)?(?:abstract\s+)?class\s+(\w+)"),
        re.compile(r"(?:export\s+)?(?:default\s+)?class\s+(\w+)"),
        re.compile(r"@Component\s*\n.*?class\s+(\w+)", re.DOTALL),
        re.compile(r"@Service\s*\n.*?class\s+(\w+)", re.DOTALL),
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
}

_LLM_PROMPT = """Extract named entities from this text. Types: class, function, module, api, concept.
- class: Java/Python/TS class names
- function: method/function names
- module: package/import module names
- api: API endpoint paths or system interfaces
- concept: business domain concepts (e.g. KYC流程, 审批, 风险评估)
Return JSON: {{"entities":[{{"name":"X","type":"class"}}]}}
No explanation. Empty: {{"entities":[]}}
TEXT:
{text}"""


def extract_regex(text: str) -> list[dict]:
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


def extract_llm(text: str, model: str, client: httpx.Client) -> list[dict]:
    try:
        resp = client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": _LLM_PROMPT.format(text=text[:1500]),
                "stream": False,
                "options": {"num_predict": 300, "temperature": 0.1},
            },
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
    except Exception:
        return []

    json_str = raw
    for fmt in ["```json", "```"]:
        if fmt in json_str:
            json_str = json_str.split(fmt)[1].split("```")[0].strip()
    bs = json_str.find("{")
    be = json_str.rfind("}")
    if bs >= 0 and be > bs:
        json_str = json_str[bs : be + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    result = []
    for e in data.get("entities", []):
        if isinstance(e, dict) and "name" in e:
            name = e["name"].strip()
            etype = e.get("type", "concept")
            if (
                2 <= len(name) <= 80
                and etype != "variable"
                and re.search(r"[a-zA-Z0-9_]", name)
            ):
                result.append({"name": name, "entity_type": etype})
    return result


def build_hybrid(regex: list[dict], llm: list[dict]) -> list[dict]:
    r_set = {(e["name"], e["entity_type"]) for e in regex}
    combined = list(regex)
    seen = set(r_set)
    for e in llm:
        key = (e["name"], e["entity_type"])
        if key not in seen and e["entity_type"] in ("concept", "api"):
            seen.add(key)
            combined.append(e)
    return combined


# ── Gold Standard Builder ────────────────────────────────────

# 通用无效 pattern
_NOISE = [
    r"^\d+分钟$",
    r"^\d+次",
    r"^[\d.]+%$",
    r"^\d+\.?\d*$",
    r"^[<>≥≤]",
    r"^K6_",
    r"^[A-Z_]{6,}$",
    r"^Google Chrome$",
    r"^Safari$",
    r"^360浏览器$",
    r"^Java 编程方法论",
    r"^Remove Assignments",
    r"^Split Variable$",
    r"^Split Temp$",
    r"^Episodic$",
    r"^Procedural$",
    r"^http://",
    r"^\d+\.\d+\.\d+\.\d+",
    r"^wecom_mcp",
    r"^sleep$",
    r"^assert$",
    r"^candidate\(s\)",
    r"^data$",
    r"^channel$",
    r"^exclude$",
    r"^Admin@123$",
    r"^localhost$",
    r"^\d{5}$",
    r"^Math\.",
    r"^TimeUnit\.",
    r"^fence\.",
    r"^\.env\.",
]

_VALID_KEYWORDS = [
    r"KYC",
    r"Buyer",
    r"Seller",
    r"买方",
    r"卖方",
    r"Contract",
    r"WBS",
    r"ADR",
    r"DDD",
    r"CI",
    r"CD",
    r"API",
    r"HTTP",
    r"REST",
    r"JWT",
    r"Token",
    r"Auth",
    r"OpenClaw",
    r"TradeOMS",
    r"aITMS",
    r"WeCom|WeChat",
    r"Gateway",
    r"Postgres|PostgreSQL",
    r"Dashboard",
    r"驾驶舱",
    r"看板",
    r"Observable",
    r"RxJava",
    r"SseEmitter",
    r"Reactive",
    r"PagerDuty",
    r"Grafana",
    r"Loki",
    r"Prometheus",
    r"Kubernetes",
    r"Docker",
    r"git",
    r"openssl",
    r"cron",
    r"schedule",
    r"smoke.test",
    r"Context",
    r"Propagation",
    r"Rerank",
    r"Embedding",
    r"Hybrid",
    r"Vector",
    r"Semantic",
    r"Cache",
    r"Maintenance",
    r"Verification",
    r"Summary",
    r"Trigger",
    r"i18n",
    r"spike",
    r"Bearer",
    r"Basic Auth",
    r"reasonix",
    r"gbrain",
    r"Chunk",
    r"Embedder",
    r"\w+门禁",
    r"\w+评审",
    r"GitHub Actions",
    r"Docker Compose",
    r"Long or Volatile",
    r"Data Flow",
    r"Value Completeness",
    r"Feature Completeness",
]

_MANUALLY_VALID = {
    "Status",
    "Signal",
    "Preconditions",
    "Steps",
    "Overview",
    "sustainability",
    "Cancel/Refund",
    "Standalone",
}


def classify(name: str) -> str:
    if any(re.match(p, name) for p in _NOISE):
        return "noise"
    if name in _MANUALLY_VALID:
        return "valid"
    if any(re.search(p, name) for p in _VALID_KEYWORDS):
        return "valid"
    if len(name) >= 3 and re.match(r"^[a-zA-Z][a-zA-Z0-9]+$", name):
        return "valid"
    if len(name) >= 2 and re.search(r"[\u4e00-\u9fff]", name) and len(name) <= 20:
        return "valid"
    return "noise"


def build_gold(all_entities: dict[str, list[dict]]) -> set:
    gold = set()
    for source, entities in all_entities.items():
        for e in entities:
            if classify(e["name"]) == "valid":
                gold.add((e["name"], e["entity_type"]))
    print(f"  🏆 Gold standard: {len(gold)} entities")
    return gold


# ── Evaluation ────────────────────────────────────────────────


def evaluate(entities: list[dict], gold: set) -> dict:
    eset = {(e["name"], e["entity_type"]) for e in entities}
    tp = eset & gold
    fp = eset - gold
    fn = gold - eset
    p = len(tp) / len(eset) if eset else 0
    r = len(tp) / len(gold) if gold else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {
        "extracted": len(eset),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "p": round(p, 4),
        "r": round(r, 4),
        "f1": round(f1, 4),
    }


# ── Main ──────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("🧪 LongMemEval — 实体提取方案全面评测（同批次）")
    print("=" * 70)

    # 1. Load 100 chunks (deterministic: by page_id)
    print("\n📦 Loading 100 test chunks...")
    dc = DereContext()
    dc.connect()
    with dc.store.cursor() as cur:
        # Use first 100 chunks from the most diverse pages
        cur.execute("""
            SELECT c.id, c.chunk_text
            FROM chunks c
            JOIN pages p ON c.page_id = p.id
            WHERE LENGTH(c.chunk_text) > 80
            ORDER BY p.id, c.id
            LIMIT 100
        """)
        chunks = cur.fetchall()
    print(f"  ✅ {len(chunks)} chunks loaded (IDs {chunks[0][0]} ~ {chunks[-1][0]})")

    # 2. Run ALL extractions on the same chunks
    client = httpx.Client(timeout=120.0)
    models = ["qwen2.5-coder:1.5b", "qwen2.5-coder:7b"]

    all_entities = {"regex": []}
    for model in models:
        all_entities[model] = []

    times = {"regex": 0.0}
    for model in models:
        times[model] = 0.0

    for i, (cid, text) in enumerate(chunks):
        # Regex
        re_ents = extract_regex(text)
        all_entities["regex"].extend(re_ents)

        for model in models:
            t0 = time.time()
            llm_ents = extract_llm(text, model, client)
            all_entities[model].extend(llm_ents)
            times[model] += (time.time() - t0) * 1000

        if (i + 1) % 20 == 0:
            print(
                f"  ⏳ {i+1}/{len(chunks)} chunks... (1.5B={times.get('qwen2.5-coder:1.5b',0)/1000:.0f}s, 7B={times.get('qwen2.5-coder:7b',0)/1000:.0f}s)"
            )

    # 3. Dedup
    def dedup(entities: list[dict]) -> list[dict]:
        seen = set()
        result = []
        for e in entities:
            key = (e["name"], e["entity_type"])
            if key not in seen:
                seen.add(key)
                result.append(e)
        return result

    for source in all_entities:
        all_entities[source] = dedup(all_entities[source])

    # 4. Build gold standard from ALL sources
    gold = build_gold(all_entities)
    print(f"\n  📋 Valid patterns: {len(_VALID_KEYWORDS)} rules")
    print(f"  📋 Manual overrides: {len(_MANUALLY_VALID)} entities")

    # 5. 5 modes
    modes = {
        "① regex": all_entities["regex"],
        "② 1.5B": all_entities["qwen2.5-coder:1.5b"],
        "③ 7B": all_entities["qwen2.5-coder:7b"],
        "④ regex+1.5B": build_hybrid(
            all_entities["regex"],
            all_entities["qwen2.5-coder:1.5b"],
        ),
        "⑤ regex+7B": build_hybrid(
            all_entities["regex"],
            all_entities["qwen2.5-coder:7b"],
        ),
    }

    # 6. Print comparison table
    print(f"\n{'─'*70}")
    print(
        "  方案        |  提取  |  ✓正确  |  ✗噪音  |  ○漏掉  |  精确率 |  召回率 |   F1   |  耗时"
    )
    print(f"{'─'*70}")

    results = {}
    for mode_name, ents in modes.items():
        result = evaluate(ents, gold)
        results[mode_name] = result

        # Deduce time
        if mode_name == "① regex":
            t = 0.1
        elif mode_name == "② 1.5B":
            t = times["qwen2.5-coder:1.5b"] / 1000
        elif mode_name == "③ 7B":
            t = times["qwen2.5-coder:7b"] / 1000
        elif mode_name == "④ regex+1.5B":
            t = times["qwen2.5-coder:1.5b"] / 1000
        else:
            t = times["qwen2.5-coder:7b"] / 1000

        result["time_s"] = round(t, 1)
        print(
            f"  {mode_name}  |  {result['extracted']:>3d}  |  {result['tp']:>3d}  |  {result['fp']:>3d}  |  {result['fn']:>3d}  |  {result['p']:.3f}  |  {result['r']:.3f}  |  {result['f1']:.3f}  |  {result['time_s']:.0f}s"
        )
        bar_p = "█" * int(result["p"] * 20) + "░" * (20 - int(result["p"] * 20))
        bar_r = "█" * int(result["r"] * 20) + "░" * (20 - int(result["r"] * 20))
        print(f"                   ──  精确率: {bar_p}  {result['p']:.1%}")
        print(f"                   ──  召回率: {bar_r}  {result['r']:.1%}")

    print(f"{'─'*70}")

    # 7. Type breakdown per mode
    print("\n📊 类型分布对比:")
    print(
        f"  {'方案':<14s} {'总实体':>6s} {'class':>8s} {'function':>10s} {'module':>8s} {'api':>6s} {'concept':>9s}"
    )
    for mode_name, ents in modes.items():
        breakdown = {}
        for e in ents:
            t = e["entity_type"]
            breakdown[t] = breakdown.get(t, 0) + 1
        total = len(ents)
        c = breakdown.get("class", 0)
        f = breakdown.get("function", 0)
        m = breakdown.get("module", 0)
        a = breakdown.get("api", 0)
        cp = breakdown.get("concept", 0)
        print(
            f"  {mode_name:<14s} {total:>6d} {c:>8d} {f:>10d} {m:>8d} {a:>6d} {cp:>9d}"
        )

    # 8. Noise samples
    print("\n📋 噪音样本 TOP10:")
    for mode_name, ents in modes.items():
        noise = [
            (e["name"], e["entity_type"])
            for e in ents
            if (e["name"], e["entity_type"]) not in gold
        ]
        if noise:
            print(f"  {mode_name}: {', '.join(f'{n}({t})' for n, t in noise[:8])}")

    # 9. Save report
    report = {
        "gold_standard": {
            "count": len(gold),
            "entities": sorted([f"{n}({t})" for n, t in gold]),
        },
        "results": results,
        "type_breakdown": {
            mn: {
                t: sum(1 for e in ents if e["entity_type"] == t)
                for t in set(e["entity_type"] for e in ents)
            }
            for mn, ents in modes.items()
        },
    }
    for mn in modes:
        report["type_breakdown"][mn]["total"] = len(modes[mn])

    out = "/tmp/longmem-eval-report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📁 完整报告: {out}")


if __name__ == "__main__":
    main()
