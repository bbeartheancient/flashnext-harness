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

# Strata's `reasoning_effort` (docs/DETAILS.md). It DEFAULTS TO "high", which for a
# tool-calling duty model is mostly waste: measured against Strata on this box the
# same request cost 347 completion tokens at the default and 14 at "none". It is
# also a correctness hazard — thinking can exhaust max_tokens and leave
# `message.content` null, which used to crash the consolidation stage.
DEFAULT_REASONING_EFFORT = os.environ.get("NEOH_REASONING_EFFORT", "none")


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
        eff = os.environ.get("NEOH_REASONING_EFFORT")
        self.sampling.setdefault("reasoning_effort", eff if eff is not None else DEFAULT_REASONING_EFFORT)
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
        content = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
        if content is None:
            # every token went to reasoning (or the reply was cut off). Say so
            # rather than returning None into a string concatenation.
            reasoning = (payload["choices"][0]["message"].get("reasoning_content")
                         or payload["choices"][0]["message"].get("reasoning") or "")
            return (reasoning.strip() or
                    f"[no content: the reply used all {self.max_tokens} max_tokens on reasoning "
                    f"(reasoning_effort={self.sampling.get('reasoning_effort')})]")
        return content
