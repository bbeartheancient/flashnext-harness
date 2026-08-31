"""Tests for serve.py — OpenAI-compatible endpoint, stream + non-stream."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from neoharness.serve import MODEL_ID, _extract_goal, make_server


class StubLLM:
    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, **kw):
        return self.script.pop(0) if self.script else "<done>fallback</done>"


@pytest.fixture()
def server(tmp_path):
    llm = StubLLM([
        '<tool>write_file {"path": "out.txt", "content": "42"}</tool>',
        "<done>wrote the answer</done>",
    ])
    srv = make_server("127.0.0.1", 0, llm, tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, tmp_path
    srv.shutdown()
    srv.server_close()


def _post(srv, payload: dict, stream: bool = False):
    port = srv.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return urllib.request.urlopen(req, timeout=30)


def test_models_endpoint(server):
    srv, _ = server
    port = srv.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models") as r:
        data = json.loads(r.read())
    assert data["data"][0]["id"] == MODEL_ID


def test_webui_and_apis(server):
    srv, _ = server
    port = srv.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/ui/") as r:
        html = r.read().decode()
        assert r.headers["Content-Type"].startswith("text/html")
    assert "NEOHARNESS" in html and "/v1/chat/completions" in html
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state") as r:
        state = json.loads(r.read())
    assert "### Belief" in state["render"]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/skills") as r:
        skills = json.loads(r.read())
    assert skills == {"skills": []}


def test_non_streaming_completion_runs_goal(server):
    srv, tmp = server
    with _post(srv, {"model": "x", "messages": [
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "write the answer to out.txt"},
    ]}) as r:
        data = json.loads(r.read())
    choice = data["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "wrote the answer" in choice["message"]["content"]
    assert "solved=True" in choice["message"]["content"]
    assert (tmp / "out.txt").read_text() == "42"  # the goal really ran


def test_streaming_completion_emits_step_deltas(server):
    srv, _ = server
    with _post(srv, {"model": "x", "stream": True, "messages": [
        {"role": "user", "content": "write the answer"},
    ]}) as r:
        body = r.read().decode()
    assert "text/event-stream" in r.headers["Content-Type"]
    assert "> t0 [tool] write_file" in body  # step delta
    assert "solved=True" in body             # final delta
    assert body.rstrip().endswith("data: [DONE]")


def test_bad_requests(server):
    srv, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(srv, {"messages": []})
    assert exc.value.code == 400


def test_extract_goal_content_parts():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "world"},
    ]}]
    assert _extract_goal(msgs) == "hello world"
