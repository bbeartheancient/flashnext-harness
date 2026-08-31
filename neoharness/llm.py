"""Thin OpenAI-compatible client for the duty LLM (vLLM chat completions).

Sampling defaults follow the model-card THINKING set used on the M1 stack
(temperature 1.0, top_p 0.95, top_k 20) — temperature=0.2 fights the card
and sharpens repetition failure modes (M1 AGENTS.md 2026-08-29).
"""

from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

# Model-card THINKING sampling set.
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


class DutyLLM:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        sampling: dict | None = None,
        max_tokens: int = 2048,
        timeout: float = 600.0,
    ):
        self.base_url = (
            base_url or os.environ.get("NEOH_LLM_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.model = model or os.environ.get("NEOH_LLM_MODEL", "duty")
        self.api_key = api_key or os.environ.get("NEOH_LLM_API_KEY", "local")
        self.sampling = dict(THINKING_SAMPLING if sampling is None else sampling)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def chat(self, messages: list[dict], **overrides) -> str:
        """One chat completion; returns the assistant message content."""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **self.sampling,
            **overrides,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        u = payload.get("usage") or {}
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        self.usage["calls"] += 1
        return payload["choices"][0]["message"]["content"]
