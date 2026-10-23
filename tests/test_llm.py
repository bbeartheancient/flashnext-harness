"""Tests for llm.py — client defaults (no network calls)."""

from __future__ import annotations

from neoharness.llm import THINKING_SAMPLING, DutyLLM


def test_thinking_sampling_defaults():
    llm = DutyLLM(base_url="http://example.invalid/v1", model="duty")
    assert llm.sampling == {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                            "reasoning_effort": "none"}
    assert llm.base_url == "http://example.invalid/v1"


def test_reasoning_effort_is_env_overridable(monkeypatch):
    # Strata defaults reasoning_effort to "high", which for a tool-calling duty
    # model costs ~347 completion tokens where "none" costs ~14, and can return
    # a null content. The harness therefore sends it explicitly.
    monkeypatch.setenv("NEOH_REASONING_EFFORT", "low")
    assert DutyLLM(base_url="http://example.invalid/v1", model="duty"
                   ).sampling["reasoning_effort"] == "low"


def test_null_content_never_returns_none():
    import json as _json
    llm = DutyLLM(base_url="http://example.invalid/v1", model="duty")
    payload = _json.dumps({"choices": [{"message": {"content": None,
                                                   "reasoning_content": "thinking..."}}],
                           "usage": {}})
    import unittest.mock as m
    resp = m.MagicMock()
    resp.read.return_value = payload.encode()
    resp.__enter__.return_value = resp
    with m.patch("urllib.request.urlopen", return_value=resp):
        out = llm.chat([{"role": "user", "content": "hi"}])
    assert out is not None and "thinking..." in out


def test_sampling_override():
    llm = DutyLLM(base_url="http://example.invalid/v1",
                  sampling={"temperature": 0.7})
    assert llm.sampling == {"temperature": 0.7, "reasoning_effort": "none"}
    assert THINKING_SAMPLING["temperature"] == 1.0  # module default untouched
