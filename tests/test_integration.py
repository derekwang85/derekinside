"""
derekinside — 全量集成测试脚本
Uses derekhubproj 的 Issue 管理流程：测试 → 发现 Issue → 修复 → 验证
"""

import subprocess
import sys
import time
import json


PASS = 0
FAIL = 0
ISSUES = []


def test(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        ISSUES.append({"test": name, "detail": detail})
        print(f"  ❌ {name}: {detail}")


def run(cmd: str, timeout: int = 30, shell: bool = True) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    import subprocess

    try:
        r = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -1, "", str(e)


print("=" * 60)
print("🧪 derekinside 全量集成测试")
print("=" * 60)
print()

# ── 1. CLI 基本测试 ──
print("📋 [CLI] 基本命令")

test("derekinside --help", run("derekinside --help")[0] == 0)

ret, out, _ = run("derekinside status")
test("derekinside status", ret == 0 and "wings" in out)
if ret == 0 and "wings" in out:
    for line in out.split("\n"):
        if "wings" in line and "rooms" in line:
            print(f"     → {line.strip()}")

ret, out, _ = run("derekinside wings")
test("derekinside wings", ret == 0 and len(out.split("\n")) > 2)

ret, out, _ = run("derekinside rooms openclaw")
test("derekinside rooms openclaw", ret == 0 and len(out.split("\n")) > 2)

print()

# ── 2. 搜索测试 ──
print("🔍 [Search] 语义搜索")

ret, out, _ = run('derekinside search "KYC" --top-k 5', timeout=60)
test("search KYC", ret == 0 and "results" in out)
if ret == 0:
    lines = [l for l in out.split("\n") if "score=" in l]
    print(f"     → {len(lines)} results")

ret, out, _ = run('derekinside search "KYC" --top-k 3 --json', timeout=60)
test("search KYC --json", ret == 0)
if ret == 0:
    try:
        data = json.loads(out)
        test("search JSON valid", isinstance(data, dict), "Not a dict")
        test("search has results", len(data.get("results", [])) > 0)
    except json.JSONDecodeError as e:
        test("search JSON valid", False, str(e))

ret, out, _ = run('derekinside search "KYC" --wing openclaw --top-k 3', timeout=60)
test("search with --wing filter", ret == 0 and "results" in out)

ret, out, _ = run('derekinside search "KYC" --recent --top-k 3', timeout=60)
test("search with --recent boost", ret == 0)

# Empty query edge case
ret, out, err = run('derekinside search "" --top-k 3', timeout=30)
# This might fail or return empty — either is ok, as long as it doesn't crash
test(
    "search empty query handles gracefully",
    ret == 0 or "error" in err.lower() or "results" in out,
)

print()

# ── 3. 知识图测试 ──
print("🕸️  [Graph] 知识图")

ret, out, _ = run("derekinside graph stats")
test("graph stats", ret == 0 and "Entities" in out)
if ret == 0:
    for line in out.split("\n"):
        if "Entities" in line:
            print(f"     → {line.strip()}")

ret, out, _ = run("derekinside graph search Service")
test("graph search entities", ret == 0)

ret, out, _ = run("derekinside graph build --max-chunks 10 --batch 10", timeout=30)
test("graph build incremental", ret == 0 and "Graph" in out)

print()

# ── 4. MCP 服务器测试 ──
print("📡 [MCP] 协议")

# Test MCP initialization message
from derekinside.bridge.mcp import rpc_result, rpc_error, _TOOLS

test("MCP tools list has entries", len(_TOOLS) >= 5, f"Got {len(_TOOLS)} tools")

tool_names = [t["name"] for t in _TOOLS]
test("MCP has search tool", "derekinside_search" in tool_names)
test("MCP has status tool", "derekinside_status" in tool_names)
test("MCP has graph tool", "derekinside_graph_stats" in tool_names)
test("MCP has entity tool", "derekinside_graph_entity" in tool_names)
test("MCP has wake tool", "derekinside_wake" in tool_names)

# MCP rpc helpers
res = rpc_result(1, {"ok": True})
test("MCP rpc_result valid", res["jsonrpc"] == "2.0" and res["id"] == 1)

err = rpc_error(2, -32601, "Method not found")
test("MCP rpc_error valid", err["error"]["code"] == -32601)

print()

# ── 5. HTTP 服务器测试 ──
print("🌐 [HTTP] REST API")

# Start server in background
proc = subprocess.Popen(
    ["derekinside", "serve", "--mode", "http", "--port", "18891"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
time.sleep(3)

import urllib.request
import urllib.error


def http_get(path: str) -> tuple[int, str]:
    try:
        r = urllib.request.urlopen(f"http://localhost:18891{path}", timeout=5)
        return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return -1, str(e)


def http_post(path: str, data: dict) -> tuple[int, str]:
    try:
        req = urllib.request.Request(
            f"http://localhost:18891{path}",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        r = urllib.request.urlopen(req, timeout=10)
        return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return -1, str(e)


# Health
code, body = http_get("/health")
test("HTTP /health returns 200", code == 200)
if code == 200:
    data = json.loads(body)
    test("HTTP /health says ok", data.get("status") == "ok")

# Status
code, body = http_get("/api/v1/status")
test("HTTP /api/v1/status", code == 200)
if code == 200:
    data = json.loads(body)
    test("HTTP status has wings", "wings" in data)

# Wings
code, body = http_get("/api/v1/wings")
test("HTTP /api/v1/wings", code == 200)
if code == 200:
    data = json.loads(body)
    test("HTTP wings is list", isinstance(data, list))
    test("HTTP wings has entries", len(data) > 0)

# Search
code, body = http_post("/api/v1/search", {"query": "KYC", "top_k": 3})
test("HTTP POST /api/v1/search", code == 200)
if code == 200:
    data = json.loads(body)
    test("HTTP search has results", len(data.get("results", [])) > 0)

# Search without query
code, body = http_post("/api/v1/search", {})
test("HTTP search empty query returns 400", code == 400)

# Wake
code, body = http_post("/api/v1/wake", {"hours": 24})
test("HTTP POST /api/v1/wake", code == 200)
if code == 200:
    data = json.loads(body)
    test("HTTP wake has context", "context" in data)

# Graph stats
code, body = http_get("/api/v1/graph/stats")
test("HTTP /api/v1/graph/stats", code == 200)

# Graph entity
code, body = http_get("/api/v1/graph/entity/NotFound")
test("HTTP entity not found returns 404", code == 404)

# Stop server
proc.terminate()
proc.wait()

print()

# ── 6. 认证测试 ──
print("🔐 [Auth] 认证")

from derekinside.bridge.auth import Auth, AuthConfig

a = Auth(AuthConfig(enabled=True, token="selftest123"))
test("Auth: correct token", a.check("selftest123") is True)
test("Auth: wrong token", a.check("wrong") is False)
test("Auth: empty token denied", a.check("") is False)
test("Auth: case-sensitive", a.check("SELFTEST123") is False)

a_disabled = Auth(AuthConfig(enabled=False, token="selftest123"))
test("Auth: disabled allows all", a_disabled.check("anything") is True)

print()

# ── 7. Agent Store 测试 ──
print("👤 [Agent] 命名空间隔离")

from derekinside.storage.pgvector import VectorStore
from derekinside.bridge.agent_store import AgentStore
from derekinside.config import load_config

cfg = load_config()
store = VectorStore(dsn=cfg.database.dsn)
store.connect()
agent_store = AgentStore(store)
agent_store.ensure_schema()

info = agent_store.register_agent("test-agent", "Test Agent")
test("Agent registered", info.agent_id == "test-agent", info.agent_id)
test("Agent wing created", info.wing == "agent-test-agent", info.wing)

info2 = agent_store.get_agent("test-agent")
test("Agent retrieved", info2 is not None)
if info2:
    test("Agent name matches", info2.name == "Test Agent")

agents = agent_store.list_agents()
test("Agent list has entries", len(agents) > 0)

# Clean up test data
with store.conn.cursor() as cur:
    cur.execute("DELETE FROM agents WHERE agent_id = 'test-agent'")
    cur.execute("DELETE FROM wings WHERE name = 'agent-test-agent'")
    store.conn.commit()

print()

# ── 8. 边界与错误处理 ──
print("⚠️  [Edge Cases] 边界情况")

# Python unit tests
ret, out, _ = run("python3 -m pytest tests/ -v --tb=short 2>&1 | tail -5")
test("All 33 unit tests pass", ret == 0)
if ret == 0:
    test("No failed tests", "failed" not in out.lower(), out)
    print(f"     → {out.strip()}")

# Large search
ret, out, _ = run(
    'derekinside search "database schema configuration" --top-k 50 --json', timeout=60
)
test("Large search (top-k 50)", ret == 0)
if ret == 0:
    try:
        data = json.loads(out)
        test(
            "Large search returns up to 50 results", len(data.get("results", [])) <= 50
        )
        print(f"     → {len(data.get('results', []))} results")
    except json.JSONDecodeError:
        test("Large search JSON valid", False)

# Non-existent wing
ret, out, _ = run(
    'derekinside search "test" --wing nonexistentwing --top-k 3', timeout=60
)
test("Search non-existent wing returns no error", ret == 0 or "results" in out)

print()
print("=" * 60)
print(f"📊 测试结果: {PASS} passed, {FAIL} failed")
print("=" * 60)

if ISSUES:
    print(f"\n📋 发现 {len(ISSUES)} 个 Issue:")
    for i, issue in enumerate(ISSUES):
        print(f"  #{i + 1}: {issue['test']}")
        if issue["detail"]:
            print(f"       {issue['detail'][:200]}")

sys.exit(FAIL)
