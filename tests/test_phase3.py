"""
Tests for derekinside Phase 3 — Bridge, Auth, MCP, Agent Store.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from derekinside.bridge.auth import Auth, AuthConfig
from derekinside.bridge.agent_store import AgentInfo


# ── Auth ───────────────────────────────────────────────────────


def test_auth_disabled():
    a = Auth(AuthConfig(enabled=False))
    assert a.enabled is False
    assert a.check("anything") is True


def test_auth_enabled():
    a = Auth(AuthConfig(enabled=True, token="secret123"))
    assert a.enabled is True
    assert a.check("secret123") is True
    assert a.check("wrong") is False
    assert a.check("") is False


def test_auth_env_var(monkeypatch):
    monkeypatch.setenv("DEREINSIDE_TOKEN", "env-token")
    a = Auth(AuthConfig(enabled=True))
    assert a.enabled is True
    assert a.check("env-token") is True
    assert a.check("wrong") is False


def test_auth_constant_time():
    a = Auth(AuthConfig(enabled=True, token="secret"))
    assert a.check("secret") is True
    assert a.check("SECRET") is False


# ── Agent Info ─────────────────────────────────────────────────


def test_agent_info():
    info = AgentInfo(agent_id="test", name="Test Agent", wing="agent-test")
    assert info.agent_id == "test"
    assert info.name == "Test Agent"
    assert info.wing == "agent-test"
    assert info.room == "memory"  # default


# ── MCP helpers ────────────────────────────────────────────────


def test_mcp_rpc_helpers():
    from derekinside.bridge.mcp import rpc_result, rpc_error

    res = rpc_result(1, {"ok": True})
    assert res["jsonrpc"] == "2.0"
    assert res["id"] == 1
    assert res["result"]["ok"] is True

    err = rpc_error(2, -32601, "Method not found")
    assert err["error"]["code"] == -32601
    assert err["error"]["message"] == "Method not found"
