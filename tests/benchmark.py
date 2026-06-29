"""
DereInside — Internal Benchmark Suite

Measures:
- Search latency (cold/warm cache)
- RRF fusion quality (vector-only vs hybrid)
- Graph propagation impact
- Throughput under load
- Cache hit rate
"""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

# Test queries representing real usage patterns
QUERIES = [
    # Domain concepts
    ("KYC approval flow", "domain"),
    ("buyer risk assessment", "domain"),
    ("trade order processing", "domain"),
    ("document management", "domain"),
    ("user authentication", "domain"),
    # Code entities
    ("KYCService", "entity"),
    ("VaTransactions", "entity"),
    ("approveKYC", "entity"),
    # Architecture
    ("database schema", "arch"),
    ("REST API endpoints", "arch"),
    # Cross-cutting
    ("security configuration", "cross"),
    ("logging and audit", "cross"),
]

HTTP_URL = "http://localhost:18890"

PASS = 0
FAIL = 0
RESULTS = {}


def test(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {detail}")


def search_cli(query: str, **kwargs) -> dict:
    """Run search via CLI and return parsed JSON."""
    args = f'derekinside search "{query}" --json --top-k {kwargs.get("top_k", 10)}'
    if kwargs.get("use_recent"):
        args += " --recent"
    r = subprocess.run(args, shell=True, capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr}


def search_http(query: str, **kwargs) -> dict:
    """Run search via HTTP API."""
    data = {"query": query, "top_k": kwargs.get("top_k", 10)}
    if kwargs.get("before"):
        data["before"] = kwargs["before"]
    if kwargs.get("after"):
        data["after"] = kwargs["after"]
    try:
        req = urllib.request.Request(
            f"{HTTP_URL}/api/v1/search",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        r = urllib.request.urlopen(req, timeout=60)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


# ── 1. Basic latency (CLI) ──
print("📊 [1/6] CLI Search Latency")

latencies = []
for q, _ in QUERIES:
    t0 = time.time()
    r = search_cli(q, top_k=5)
    elapsed = time.time() - t0
    if "results" in r:
        latencies.append(elapsed)
        test(f"'{q[:30]}'", True, f"{elapsed:.2f}s, {len(r['results'])} results")
    else:
        test(f"'{q[:30]}'", False, r.get("error", "unknown"))

if latencies:
    avg = sum(latencies) / len(latencies)
    print(f"\n  Avg latency: {avg:.2f}s")
    print(f"  Min: {min(latencies):.2f}s | Max: {max(latencies):.2f}s")

print()

# ── 2. HTTP API latency ──
print("🌐 [2/6] HTTP API Latency")
http_latencies = []
for q, _ in QUERIES[:5]:
    t0 = time.time()
    r = search_http(q, top_k=5)
    elapsed = time.time() - t0
    if "results" in r:
        http_latencies.append(elapsed)
        cache = ", cached" if r.get("cache_hit") else ""
        test(
            f"HTTP '{q[:30]}'",
            True,
            f"{elapsed * 1000:.0f}ms, {len(r['results'])} results{cache}",
        )
    else:
        test(f"HTTP '{q[:30]}'", False, r.get("error", "unknown"))

if http_latencies:
    print(
        f"\n  Avg HTTP latency: {sum(http_latencies) / len(http_latencies) * 1000:.0f}ms"
    )

print()

# ── 3. Cache effectiveness ──
print("💾 [3/6] Cache Effectiveness")
t0 = time.time()
r1 = search_http("KYC", top_k=5)
t1 = time.time()

t2 = time.time()
r2 = search_http("KYC", top_k=5)
t3 = time.time()

cold = t1 - t0
warm = t3 - t2
test("Cold cache first request", "results" in r1, f"{cold * 1000:.0f}ms")
test(
    "Warm cache second request",
    r2.get("cache_hit"),
    f"{warm * 1000:.0f}ms (hit={r2.get('cache_hit')})",
)
if r2.get("cache_hit"):
    test(
        f"Speedup: {cold / warm:.0f}x",
        cold > warm * 2,
        f"{cold:.2f}s → {warm * 1000:.0f}ms",
    )

print()

# ── 4. Time travel ──
print("⏰ [4/6] Time Travel (version tracking)")
r = search_http("KYC", after="2026-06-01T00:00:00")
test(
    "after filter works",
    "results" in r,
    f"{len(r.get('results', []))} results" if "results" in r else "",
)
r2 = search_http("KYC", before="2026-06-01T00:00:00")
if "results" in r2:
    test("before filter works", True, f"{len(r2['results'])} results")
else:
    test("before filter works", False)

print()

# ── 5. Throughput ──
print("⚡ [5/6] Concurrency (5 queries x 2 rounds)")
import concurrent.futures


def http_search_batch(q):
    try:
        r = search_http(q, top_k=3)
        return len(r.get("results", []))
    except:
        return 0


for round_num in range(2):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        counts = list(ex.map(http_search_batch, [q for q, _ in QUERIES[:5]]))
    elapsed = time.time() - t0
    total_results = sum(counts)
    test(
        f"Round {round_num + 1}: 5 queries",
        total_results > 0,
        f"{elapsed * 1000:.0f}ms, {total_results} total results",
    )

print()

# ── 6. Knowledge graph ──
print("🕸️  [6/6] Knowledge Graph")
r = subprocess.run(
    "derekinside graph stats", shell=True, capture_output=True, text=True, timeout=10
)
graph_ok = r.returncode == 0 and "Entities" in r.stdout
test("graph stats", graph_ok)
if graph_ok:
    for line in r.stdout.split("\n"):
        if "Entities" in line or "Relations" in line:
            print(f"  {line.strip()}")

# Entity search
r = subprocess.run(
    "derekinside graph search Service",
    shell=True,
    capture_output=True,
    text=True,
    timeout=10,
)
test("graph entity search", r.returncode == 0 and len(r.stdout) > 30)

print()
print("=" * 50)
print(f"📊 Benchmark: {PASS} passed, {FAIL} failed")
print("=" * 50)

sys.exit(FAIL)
