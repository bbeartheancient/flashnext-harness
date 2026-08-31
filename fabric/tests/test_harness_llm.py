"""harness.post_llm: token cap, length-truncation detection, auto-continue."""

import json
from unittest.mock import patch

from fabric import harness


def _serve(*payloads):
    seq = list(payloads)
    bodies = []
    urls = []

    def fake_open(req, timeout=0):
        bodies.append(json.loads(req.data.decode()))
        urls.append(req.full_url)

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(seq.pop(0)).encode()

        return _Resp()

    return fake_open, bodies, urls


def test_default_cap_in_payload(monkeypatch):
    monkeypatch.delenv("FABRIC_LLM_MAX_TOKENS", raising=False)
    fake_open, bodies, urls = _serve(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
    with patch.object(harness.urllib.request, "urlopen", fake_open):
        out = harness.post_llm("http://x/v1", "k", "m",
                               [{"role": "user", "content": "p"}],
                               orch="local", timeout=5)
    assert out == "hi"
    assert bodies[0]["max_tokens"] == harness._DEFAULT_MAX_TOKENS


def test_env_cap_override(monkeypatch):
    monkeypatch.setenv("FABRIC_LLM_MAX_TOKENS", "512")
    fake_open, bodies, urls = _serve(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
    with patch.object(harness.urllib.request, "urlopen", fake_open):
        harness.post_llm("http://x/v1", "k", "m",
                         [{"role": "user", "content": "p"}],
                         orch="local", timeout=5)
    assert bodies[0]["max_tokens"] == 512


def test_explicit_max_tokens_wins(monkeypatch):
    monkeypatch.setenv("FABRIC_LLM_MAX_TOKENS", "512")
    fake_open, bodies, urls = _serve(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
    with patch.object(harness.urllib.request, "urlopen", fake_open):
        harness.post_llm("http://x/v1", "k", "m",
                         [{"role": "user", "content": "p"}],
                         orch="local", timeout=5, max_tokens=77)
    assert bodies[0]["max_tokens"] == 77


def test_no_continuation_on_stop():
    fake_open, bodies, urls = _serve(
        {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]})
    with patch.object(harness.urllib.request, "urlopen", fake_open):
        out = harness.post_llm("http://x/v1", "k", "m",
                               [{"role": "user", "content": "p"}],
                               orch="local", timeout=5)
    assert out == "done" and len(bodies) == 1


def test_chat_continues_after_length():
    fake_open, bodies, urls = _serve(
        {"choices": [{"message": {"content": "the quick brown"},
                      "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "fox jumps."},
                      "finish_reason": "stop"}]},
    )
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "tell the tale"}]
    with patch.object(harness.urllib.request, "urlopen", fake_open):
        out = harness.post_llm("http://x/v1", "k", "m", msgs,
                               orch="local", timeout=5)
    assert out == "the quick brown fox jumps."
    assert len(bodies) == 2
    roles = [m["role"] for m in bodies[1]["messages"]]
    assert roles[-2:] == ["assistant", "user"]
    assert "Continue exactly where you stopped" in bodies[1]["messages"][-1]["content"]
    assert bodies[1]["messages"][0] == {"role": "system", "content": "sys"}
    assert bodies[1]["messages"][1]["content"] == "tell the tale"


def test_messages_kind_continues_after_max_tokens():
    fake_open, bodies, urls = _serve(
        {"content": [{"type": "text", "text": "alpha"}],
         "stop_reason": "max_tokens"},
        {"content": [{"type": "text", "text": "beta"}],
         "stop_reason": "end_turn"},
    )
    msgs = [{"role": "system", "content": "be brief"},
            {"role": "user", "content": "go"}]
    with patch.object(harness.urllib.request, "urlopen", fake_open):
        out = harness.post_llm("http://x/v1", "k", "claude-x", msgs,
                               orch="opencode-zen", timeout=5)
    assert out == "alpha beta"
    assert len(bodies) == 2
    assert urls[0].endswith("/messages")
    assert bodies[0]["system"] == "be brief"
    assert bodies[1]["messages"][-1]["role"] == "user"


def test_responses_kind_continues_after_incomplete():
    fake_open, bodies, urls = _serve(
        {"status": "incomplete",
         "incomplete_details": {"reason": "max_output_tokens"},
         "output": [{"type": "output_text", "text": "half-"}]},
        {"status": "completed",
         "output": [{"type": "output_text", "text": "sent."}]},
    )
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "go"}]
    with patch.object(harness.urllib.request, "urlopen", fake_open):
        out = harness.post_llm("http://x/v1", "k", "gpt-x", msgs,
                               orch="opencode-zen", timeout=5)
    assert out == "half- sent."
    assert len(bodies) == 2
    assert bodies[0]["max_output_tokens"] == harness._DEFAULT_MAX_TOKENS
    assert [m["role"] for m in bodies[1]["input"]][-2:] == ["assistant", "user"]


def test_leg_cap_bounds_continuations():
    trunc = lambda: {"choices": [{"message": {"content": "x" * 10},
                                  "finish_reason": "length"}]}
    fake_open, bodies, urls = _serve(*[trunc() for _ in range(8)])
    with patch.object(harness.urllib.request, "urlopen", fake_open):
        out = harness.post_llm("http://x/v1", "k", "m",
                               [{"role": "user", "content": "p"}],
                               orch="local", timeout=5)
    assert len(bodies) == harness._MAX_LEGS
    assert out.count("x") == 10 * harness._MAX_LEGS


def test_join_fragments_glue():
    assert harness._join_fragments("", "b") == "b"
    assert harness._join_fragments("a", "") == "a"
    assert harness._join_fragments("end.", "Next") == "end. Next"
    assert harness._join_fragments("word", ", next") == "word, next"
    assert harness._join_fragments("mid", "word") == "mid word"
    assert harness._join_fragments("mid ", "word") == "mid word"


def test_length_truncated_shapes():
    assert harness._length_truncated(
        "chat", {"choices": [{"finish_reason": "length"}]})
    assert not harness._length_truncated(
        "chat", {"choices": [{"finish_reason": "stop"}]})
    assert harness._length_truncated("messages", {"stop_reason": "max_tokens"})
    assert not harness._length_truncated("messages", {"stop_reason": "end_turn"})
    assert harness._length_truncated(
        "responses", {"status": "incomplete",
                      "incomplete_details": {"reason": "max_output_tokens"}})
    assert not harness._length_truncated(
        "responses", {"status": "completed"})
