"""Harness slash commands, Pi health, pipecat status."""

import json

from fabric.commands import catalog, parse, resolve as slash_resolve
from fabric.intent import resolve
from fabric.voice import pipecat_status

from fastapi.testclient import TestClient

from fabric.main import app


client = TestClient(app)


def test_parse_slash():
    assert parse("/status") == ("status", "")
    assert parse("/code fix the parser") == ("code", "fix the parser")
    assert parse("/") == ("help", "")
    assert parse("hadamard 8") == ("", "hadamard 8")


def test_catalog_has_pi_and_pipecat():
    ids = {c["id"] for c in catalog()["commands"]}
    assert {"code", "goal", "loop", "voice", "pi", "chat", "status", "harness"} <= ids
    lanes = {c["id"]: c["lane"] for c in catalog()["commands"]}
    assert lanes["code"] == "omp" and lanes["voice"] == "pipecat"


def test_intent_slash_status():
    r = resolve("/status")
    assert r["slash"] is True
    assert r["match"]["id"] == "ship"
    assert r["autorun"] is True


def test_intent_slash_code():
    r = resolve("/code add a test")
    assert r["command"] == "code"
    assert r["url"] == "/api/pi/code"
    assert r["body"]["prompt"] == "add a test"
    assert r["lane"] == "omp"


def test_intent_slash_goal_empty_lists():
    r = resolve("/goal")
    assert r["url"] == "/api/goals"


def test_intent_slash_harness():
    r = resolve("/harness")
    assert r["ui"] == "harness"
    assert r["command"] == "harness"
    r2 = resolve("/settings")
    assert r2["ui"] == "harness"


def test_intent_unknown_slash():
    r = resolve("/nope")
    assert r["fallback"] == "help"
    assert any(c["id"] == "code" for c in r["candidates"])


def test_commands_endpoint():
    r = client.get("/api/commands")
    assert r.status_code == 200
    ids = [c["id"] for c in r.json()["commands"]]
    assert "voice" in ids and "code" in ids


def test_voice_endpoint():
    r = client.get("/api/voice")
    assert r.status_code == 200
    body = r.json()
    assert body["lane"] == "pipecat"
    # namespace-package lookalikes (loader=None) must NOT count as installs
    import importlib.util

    spec = importlib.util.find_spec("pipecat")
    real = spec is not None and getattr(spec, "loader", None) is not None
    assert body["installed"] is real
    if body["installed"]:
        assert body["version"]
        assert body["stage"] in ("wired-status", "running")
    else:
        assert body["stage"] == "concept"
    assert any(f["id"] == "barge_in" for f in body["functions"])


def test_pipecat_status_direct():
    s = pipecat_status()
    assert s["running"] is False
    assert "wake" in str(s["functions"])


def test_pi_health_endpoint():
    r = client.get("/api/pi")
    assert r.status_code == 200
    body = r.json()
    assert "up" in body
    assert body.get("bridge") == "olympus-pi-bridge" or body.get("up") is False


def test_pi_code_empty():
    r = client.post("/api/pi/code", json={"prompt": ""})
    assert r.status_code == 200
    assert r.json().get("error")


def test_pi_code_mocked(monkeypatch):
    from fabric import pi_client
    from fabric import tools

    # The pi XML lane was retired 2026-08-27: code_task delegates to
    # OMP (tools.omp_code).  Mock the OMP call, not the old urlopen.
    monkeypatch.setattr(tools, "omp_code",
                        lambda task: {"ok": True, "result": "pong",
                                      "session": "abc123"})
    out = pi_client.code_task("reply pong")
    assert out.get("ok") is True
    assert "pong" in out["reply"]
    assert out.get("lane") == "omp"


def test_parse_pi_tools_shapes():
    from fabric.pi_client import parse_pi_tools

    assert parse_pi_tools("plain text") == []
    duty = parse_pi_tools('<tool name="ls">{"path":"fabric"}</tool>')
    assert duty == [("ls", {"path": "fabric"})]
    qwen = parse_pi_tools(
        "<function=grep><arg_key>pattern</arg_key><arg_value>orbital</arg_value></function>")
    assert qwen[0][0] == "grep" and qwen[0][1]["pattern"] == "orbital"
    param = parse_pi_tools(
        "<function=read><parameter=path>fabric/pi_client.py</parameter></function>")
    assert param == [("read", {"path": "fabric/pi_client.py"})]
    bare = parse_pi_tools("<function=ls>\n<function=grep orbital>")
    names = [n for n, _ in bare]
    assert "ls" in names and "grep" in names
    grep = next(a for n, a in bare if n == "grep")
    assert grep.get("pattern") == "orbital"


def test_pi_dispatch_ls_read(tmp_path, monkeypatch):
    from fabric import pi_client

    (tmp_path / "hello.txt").write_text("orbital probe\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text("print(1)\n")
    monkeypatch.setenv("FABRIC_PI_CWD", str(tmp_path))
    listing = pi_client._dispatch("ls", {"path": "."})
    assert "hello.txt" in listing and "sub/" in listing
    got = pi_client._dispatch("read", {"path": "hello.txt"})
    assert "orbital probe" in got
    hits = pi_client._dispatch("grep", {"pattern": "orbital", "path": "."})
    assert "hello.txt" in hits
    refused = pi_client._dispatch("bash", {"command": "rm -rf /"})
    assert "refused" in refused
    outside = pi_client._dispatch("read", {"path": "/etc/passwd"})
    assert "outside" in outside or "error" in outside
    wrote = pi_client._dispatch("write", {"path": "note2.md", "contents": "hi\n"})
    assert "wrote" in wrote
    edited = pi_client._dispatch("edit", {"path": "note2.md", "old": "hi", "new": "yo"})
    assert "edited" in edited
    assert (tmp_path / "note2.md").read_text() == "yo\n"


def test_pi_tool_loop_mocked(tmp_path, monkeypatch):
    from fabric import pi_client

    monkeypatch.setenv("FABRIC_PI_CWD", str(tmp_path))
    (tmp_path / "note.md").write_text("ok\n")
    monkeypatch.setattr(pi_client, "health", lambda: {
        "up": True, "bridge": "olympus-pi-bridge", "agent": "pi"})
    monkeypatch.setattr(pi_client, "_duty_model", lambda: "default")
    replies = [
        b'{"choices":[{"message":{"content":"<tool name=\\"ls\\">{\\"path\\":\\".\\"}</tool>"}}]}',
        b'{"choices":[{"message":{"content":"tree has note.md"}}]}',
    ]

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
        def read(self):
            return self._payload
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _open(*a, **k):
        return _Resp(replies.pop(0))

    monkeypatch.setattr(pi_client.urllib.request, "urlopen", _open)
    out = pi_client.complete("list the tree")
    assert out.get("ok") is True
    assert out.get("tools") == ["ls"]
    assert "tree has note.md" in out["reply"]
    assert out.get("rounds") == 2


def test_pi_miniapp_closer_mocked(tmp_path, monkeypatch):
    from fabric import pi_client

    monkeypatch.setenv("FABRIC_PI_CWD", str(tmp_path))
    monkeypatch.setattr(pi_client, "health", lambda: {"up": True})
    monkeypatch.setattr(pi_client, "_duty_model", lambda: "default")
    html = ("<!DOCTYPE html><html><body><canvas id=c></canvas>"
            "<script>addEventListener('message',()=>{})</script></body></html>")
    replies = [
        b'{"choices":[{"message":{"content":"I will draft a MiniApp shortly."}}]}',
        json.dumps({"choices": [{"message": {"content":
            '<tool name="generate_microapp">'
            + json.dumps({"query": "atom simulator", "html": html})
            + "</tool>"}}]}).encode(),
        b'{"choices":[{"message":{"content":"Atom simulator is on the wall."}}]}',
    ]

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
        def read(self):
            return self._payload
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _open(*a, **k):
        return _Resp(replies.pop(0))

    monkeypatch.setattr(pi_client.urllib.request, "urlopen", _open)
    monkeypatch.setattr(
        "fabric.microapps.compose",
        lambda *a, **k: {
            "id": "app_atom",
            "spec": {"id": "app_atom", "title": "Atom"},
            "figure": {"type": "html", "html": html, "caption": "Atom",
                       "sandbox": True},
        },
    )
    out = pi_client.complete("develop a 3D atom simulator MiniApp", miniapp=True)
    assert "generate_microapp" in out.get("tools", [])
    assert out.get("id_app") == "app_atom"
    assert out["figure"]["sandbox"] is True


def test_generate_microapp_requires_html_for_unknown_kernel():
    from fabric import pi_client

    out = pi_client._dispatch(
        "generate_microapp", {"query": "3D atom simulator MiniApp"})
    assert isinstance(out, str) and "html" in out.lower()
    dots = pi_client._dispatch(
        "generate_microapp",
        {"query": "atom simulator", "html": "..."})
    assert isinstance(dots, str) and "html" in dots.lower()


def test_harness_roundtrip(tmp_path, monkeypatch):
    from fabric import harness

    monkeypatch.setenv("FABRIC_HARNESS_PATH", str(tmp_path / "harness.json"))
    for name in ("XAI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY", "OPENCODE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    harness.reset_cache()
    pub = harness.public()
    assert pub["orchestrator"] == "local"
    grok = next(p for p in pub["providers"] if p["id"] == "grok")
    assert grok["key_set"] is False
    assert grok["signup"].startswith("https://")
    assert "error" in harness.update({"orchestrator": "grok"})
    out = harness.update({"orchestrator": "grok",
                           "providers": {"grok": {"api_key": "sk-test-abcd"}}})
    assert out["orchestrator"] == "grok"
    grok = next(p for p in out["providers"] if p["id"] == "grok")
    assert grok["key_set"] is True
    assert grok["key_tail"] == "abcd"
    raw = (tmp_path / "harness.json").read_text()
    assert "sk-test-abcd" in raw
    pub2 = harness.public()
    assert "sk-test-abcd" not in json.dumps(pub2)
    orch = harness.orchestrator()
    assert orch["id"] == "grok" and orch["api_key"] == "sk-test-abcd"
    assert orch["base"].startswith("https://api.x.ai")
    assert orch["model"] == "grok-4.6"
    named = harness.update({"providers": {"grok": {"model": "grok-4.5"}}})
    grok = next(p for p in named["providers"] if p["id"] == "grok")
    assert grok["model"] == "grok-4.5"
    r = client.get("/api/harness")
    assert r.status_code == 200
    assert "sk-test-abcd" not in r.text
    bad = harness.update({"orchestrator": "nope"})
    assert "error" in bad
    harness.reset_cache()


def test_zen_api_kind():
    from fabric.harness import api_kind

    assert api_kind("grok", "grok-4.6") == "chat"
    assert api_kind("opencode-zen", "x-preview-f-free") == "chat"
    assert api_kind("opencode-zen", "nemotron-3.5-lightning-free") == "chat"
    assert api_kind("opencode-zen", "grok-4.6") == "responses"
    assert api_kind("opencode-zen", "gpt-5.5") == "responses"
    assert api_kind("opencode-zen", "claude-haiku-4-5") == "messages"
    assert api_kind("opencode-zen", "gemini-3.7-flash") == "unsupported"


def test_harness_endpoint_put(tmp_path, monkeypatch):
    from fabric import harness, main as mainmod

    monkeypatch.setenv("FABRIC_HARNESS_PATH", str(tmp_path / "h.json"))
    for name in ("XAI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY", "OPENCODE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    harness.reset_cache()
    mainmod._agent = object()
    r = client.put("/api/harness", json={
        "orchestrator": "kimi",
        "kimi_key": "moon-secret-99",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["orchestrator"] == "kimi"
    assert "moon-secret" not in r.text
    assert mainmod._agent is None
    ping = client.post("/api/harness/test", json={"provider": "local"})
    assert ping.status_code == 200
    assert "ok" in ping.json()
    harness.reset_cache()


def test_harness_live_models(tmp_path, monkeypatch):
    from fabric import harness

    monkeypatch.setenv("FABRIC_HARNESS_PATH", str(tmp_path / "h.json"))
    harness.reset_cache()

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
        def read(self):
            return self._payload
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _open(req, timeout=0):
        url = getattr(req, "full_url", None) or getattr(req, "get_full_url", lambda: "")()
        if "api.x.ai" in url:
            return _Resp(b'{"data":[{"id":"grok-4.6"},{"id":"grok-4.5"},{"id":"text-embedding-3"}]}')
        if "moonshot" in url:
            return _Resp(b'{"data":[{"id":"kimi-k3"},{"id":"kimi-k2.5"}]}')
        if "opencode.ai" in url:
            return _Resp(b'{"models":[{"id":"glm-5.1"},{"name":"kimi-k3"}]}')
        if ":8000" in url:
            return _Resp(b'{"data":[{"id":"Qwen3.8-27B-GPTQ"}]}')
        raise AssertionError(url)

    monkeypatch.setattr(harness.urllib.request, "urlopen", _open)
    grok = harness.list_models("grok")
    assert grok["live"] is True
    assert grok["models"] == ["grok-4.6", "grok-4.5"]
    assert "text-embedding-3" not in grok["models"]
    all_m = harness.fetch_all_models()
    assert all_m["providers"]["kimi"]["models"] == ["kimi-k3", "kimi-k2.5"]
    assert "glm-5.1" in all_m["providers"]["opencode-zen"]["models"]
    r = client.get("/api/harness/models")
    assert r.status_code == 200
    assert r.json()["providers"]["grok"]["source"] == "live"
    harness.reset_cache()


def test_summarize_loop_replaces_garbage():
    from fabric.pi_client import _summarize_loop

    note = _summarize_loop("II", ["ls", "generate_microapp"], False, True)
    assert "512" in note and "loop" in note
    keep = _summarize_loop("Python files in fabric/: agent.py", ["find"], False,
                           False)
    assert keep.startswith("Python files")
    assert _summarize_loop("pong", [], False, False) == "pong"


def test_goals_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_DB_PATH", str(tmp_path / "g.db"))
    from fabric.log_store import reset_log_store, get_log_store

    reset_log_store()
    store = get_log_store()
    row = store.goal_add("wire slash menu")
    store.goal_update(row["id"], status="done", last_result="ok")
    got = store.goals()
    assert got[0]["text"] == "wire slash menu"
    assert got[0]["status"] == "done"
    r = client.get("/api/goals")
    assert r.status_code == 200
    assert r.json()["goals"][0]["text"] == "wire slash menu"
    reset_log_store()
