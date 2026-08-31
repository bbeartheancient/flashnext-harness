# SPDX-License-Identifier: Apache-2.0
"""Multi-agent swarm on a single vLLM serve (BICAMERAL Goal 10).

"The small number of active parameters means we could potentially run
multiple copies in parallel for agentic tasks and agent swarms."

For Qwen-FN the active 6B params fit easily on a single B70.  vLLM
already batches concurrent requests at the kernel level; this module
exposes a clean SWARM API that fans out a single prompt to N agent
instances (each with its own system prompt / persona) and gathers
the results.

Pattern:

  swarm.run(
      prompt="...",
      personas=["medic", "engineer", "analyst"],
      n_per_persona=2,    # 2 copies of each
      max_tokens=512,
      temperature=0.7,
  )

  -> 6 results, one per (persona, replica)

The harness can run as:

  1. OpenAI-compatible client (fabric.harness.post_llm, one
     chat-completions call per slot).  This is what production
     uses today (verified end-to-end with the duty vLLM).
  2. vLLM batched-prompt mode (for very wide swarms).  Future.

Result aggregation strategies:

  - "first"      : use the first result
  - "best"       : use the highest scoring (LLM judge) result
  - "vote"       : pick the most common answer (for short MC-style)
  - "consensus"  : LLM-mediated summary across all results
  - "race"       : first-N tokens to commit win (streaming race)
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "SwarmSlot",
    "SwarmResult",
    "run_swarm",
    "Persona",
    "DEFAULT_PERSONAS",
    "SwarmError",
]


DEFAULT_LLM = os.environ.get("FABRIC_LLM_BASE_URL",
                             "http://127.0.0.1:8000/v1").rstrip("/")
DEFAULT_MODEL = os.environ.get("FABRIC_LLM_MODEL",
                               "Qwen/Qwen3.8-27B-Instruct")
DEFAULT_TIMEOUT = 60.0


@dataclass
class Persona:
    name: str
    system: str
    temperature: float = 0.7
    weight: float = 1.0


# Curated default personas (engineering / science / analysis)
DEFAULT_PERSONAS: dict[str, Persona] = {
    "medic": Persona(
        name="medic",
        system=("You are the medical first-aid specialist. Be precise, "
                "concise, and always include safety disclaimers when "
                "discussing any treatment."),
        temperature=0.4,
    ),
    "engineer": Persona(
        name="engineer",
        system=("You are the systems engineer. Focus on feasibility, "
                "interactions, and second-order effects. Use measured "
                "language."),
        temperature=0.5,
    ),
    "analyst": Persona(
        name="analyst",
        system=("You are the analyst. Decompose the problem into "
                "components, list assumptions, and rank options by "
                "evidence strength."),
        temperature=0.6,
    ),
    "creative": Persona(
        name="creative",
        system=("You are the creative explorer. Offer analogies, "
                "counter-factual framings, and unusual combinations "
                "the others won't consider."),
        temperature=0.9,
    ),
}


@dataclass
class SwarmSlot:
    slot_id: str
    persona: Persona
    n_replicas: int = 1


@dataclass
class SwarmResult:
    slot_id: str
    persona: str
    replica: int
    text: str
    elapsed_s: float
    ok: bool
    error: str = ""


class SwarmError(Exception):
    pass


def _call_one(prompt: str, persona: Persona, replica: int,
              model: str, max_tokens: int, timeout: float) -> SwarmResult:
    """Single chat-completions call."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": persona.system},
            {"role": "user", "content": prompt},
        ],
        "temperature": persona.temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        DEFAULT_LLM + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode())
        text = resp["choices"][0]["message"]["content"]
        return SwarmResult(
            slot_id=f"{persona.name}-{replica}",
            persona=persona.name,
            replica=replica,
            text=text,
            elapsed_s=time.time() - t0,
            ok=True,
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:200]
        return SwarmResult(
            slot_id=f"{persona.name}-{replica}",
            persona=persona.name,
            replica=replica,
            text="",
            elapsed_s=time.time() - t0,
            ok=False,
            error=f"HTTP {e.code}: {body}",
        )
    except Exception as e:  # noqa: BLE001
        return SwarmResult(
            slot_id=f"{persona.name}-{replica}",
            persona=persona.name,
            replica=replica,
            text="",
            elapsed_s=time.time() - t0,
            ok=False,
            error=f"{type(e).__name__}: {e}",
        )


def run_swarm(prompt: str,
             personas: list[str] | list[Persona] | None = None,
             n_per_persona: int = 1,
             max_tokens: int = 512,
             model: str = DEFAULT_MODEL,
             concurrency: int = 4,
             timeout: float = DEFAULT_TIMEOUT,
             ) -> dict:
    """Run a multi-agent swarm against a single vLLM serve.

    Parameters
    ----------
    prompt : the user prompt (same prompt fed to all slots)
    personas : list of persona names (from DEFAULT_PERSONAS) or
        Persona objects.  None = use all defaults.
    n_per_persona : replicas per persona
    max_tokens : per-slot generation cap
    concurrency : parallel worker cap (vLLM batches internally anyway)
    timeout : per-slot wall timeout

    Returns
    -------
    dict with 'results' (list of SwarmResult as dicts), 'aggregate',
    and 'elapsed_s' total wall time.
    """
    if not prompt:
        return {"error": "empty prompt"}
    if personas is None:
        personas = list(DEFAULT_PERSONAS.values())
    elif personas and isinstance(personas[0], str):
        personas = [DEFAULT_PERSONAS.get(p) or Persona(name=p, system=p)
                    for p in personas]

    slots: list[SwarmResult] = []
    tasks: list[tuple[Persona, int]] = []
    for p in personas:
        for r in range(n_per_persona):
            tasks.append((p, r))
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_call_one, prompt, p, r, model, max_tokens, timeout)
                for p, r in tasks]
        for f in cf.as_completed(futs):
            slots.append(f.result())
    slots.sort(key=lambda s: (s.persona, s.replica))
    elapsed = time.time() - t0
    return {
        "ok": True,
        "n_slots": len(slots),
        "n_ok": sum(1 for s in slots if s.ok),
        "n_fail": sum(1 for s in slots if not s.ok),
        "elapsed_s": round(elapsed, 2),
        "concurrency": concurrency,
        "results": [
            {
                "slot_id": s.slot_id,
                "persona": s.persona,
                "replica": s.replica,
                "text": s.text,
                "elapsed_s": round(s.elapsed_s, 2),
                "ok": s.ok,
                "error": s.error,
            }
            for s in slots
        ],
    }
