"""Tests for llm.py — client defaults (no network calls)."""

from __future__ import annotations

from neoharness.llm import THINKING_SAMPLING, DutyLLM


def test_thinking_sampling_defaults():
    llm = DutyLLM(base_url="http://example.invalid/v1", model="duty")
    assert llm.sampling == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    assert llm.base_url == "http://example.invalid/v1"


def test_sampling_override():
    llm = DutyLLM(base_url="http://example.invalid/v1",
                  sampling={"temperature": 0.7})
    assert llm.sampling == {"temperature": 0.7}
    assert THINKING_SAMPLING["temperature"] == 1.0  # module default untouched
