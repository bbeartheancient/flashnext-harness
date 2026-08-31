# SPDX-License-Identifier: Apache-2.0
"""Self-improvement loop harness (BICAMERAL Goal 7).

"the model learns about specific domains and becomes optimized to the
user's needs."

This module combines three existing pieces into a closed improvement loop:

  1. latent_loop  (fabric/latent.py)  — iterative self-refinement with
     spectral tracking; each iteration produces a compaction_90.
  2. dept_memory   (fabric/dept_memory.py) — .mv2 shard per department;
     stores iteration frames with spectral fingerprints.
  3. skill feedback — the operator / user can score the output; scores
     are stored as annotations in the Sciences shard.

The loop:

    generate → capture → analyze → score → store → critique → augment → repeat

Key insight: the loop does NOT modify model weights.  It optimizes
the *selection* of which context to carry forward (via dct_topk
compaction at varying rates) and which domain shard to route through.
The improvement is in the routing policy, not the model.

The harness runs up to ``max_iterations`` rounds and returns a report
with per-iteration quality scores, the best output, and recommendations
for routing policy changes.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "SelfImproveConfig",
    "run_self_improve",
    "score_output",
    "get_shard_trajectory",
]

DEFAULT_LLM = os.environ.get("FABRIC_LLM_BASE_URL",
                             "http://127.0.0.1:8000/v1")


@dataclass
class SelfImproveConfig:
    prompt: str
    max_iterations: int = 3
    scoring_criteria: list[str] | None = None
    min_score: float = 0.8
    compact_rate: float = 0.5
    department: str = "Sciences"
    session_tag: str = ""
    timeout_s: float = 300.0


# ---------------------------------------------------------------------------
# Output scoring (Goal 7)
# ---------------------------------------------------------------------------

def score_output(text: str, criteria: list[str] | None = None) -> dict:
    """Score the output text against criteria.

    Uses the duty LLM to score the output on a 0-1 scale.  Returns
    {score, reasons, criteria_scores}.
    """
    crit = criteria or [
        "correctness — does it answer the prompt?",
        "completeness — are all parts addressed?",
        "clarity — is it well-structured?",
    ]
    crit_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(crit))
    prompt_text = (
        f"Score this text on each criterion, 0.0 to 1.0:\n\n{text[:1500]}\n\n"
        f"Criteria:\n{crit_text}\n\n"
        "Reply with JSON: {\"score\": 0.85, \"reasons\": [\"...\"], "
        "\"criteria_scores\": [0.9, 0.8, 0.85]}"
    )
    try:
        body = json.dumps({
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "system",
                 "content": "You are a scoring assistant. Reply ONLY with the JSON."},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }).encode()
        req = urllib.request.Request(
            DEFAULT_LLM + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode())
        raw = resp["choices"][0]["message"]["content"]
        for seg in raw.split("```"):
            if seg.strip().startswith("json"):
                raw = seg.strip()[4:].strip()
        result = json.loads(raw)
        return {
            "score": float(result.get("score", 0.0)),
            "reasons": result.get("reasons", []),
            "criteria_scores": [float(s) for s in (result.get("criteria_scores") or [])],
        }
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "error": str(e)[:100],
                "reasons": [], "criteria_scores": []}


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------

def get_shard_trajectory(department: str = "Sciences",
                        session_tag: str = "") -> dict:
    """Read the Sciences shard and extract the improvement trajectory."""
    try:
        from .dept_memory import list_all
        docs = list_all(department)
        relevant = [
            d for d in docs
            if (not session_tag or session_tag in d.get("title", ""))
               and "latent" in d.get("title", "")
        ]
        relevant.sort(key=lambda d: d.get("time", ""))
        trajectory = []
        for d in relevant:
            body = json.loads(d.get("body", "{}"))
            trajectory.append({
                "title": d.get("title", ""),
                "iteration": body.get("iteration"),
                "backend": body.get("backend"),
                "compaction_90": body.get("compaction_90"),
                "mean_energy": body.get("mean_energy"),
                "gen_mode": body.get("gen_mode"),
                "text_len": body.get("text_len", 0),
                "time": d.get("time", ""),
            })
        return {"n_frames": len(trajectory), "trajectory": trajectory}
    except Exception as e:  # noqa: BLE001
        return {"n_frames": 0, "trajectory": [], "error": str(e)[:200]}


def _recommend_policy_change(trajectory: list[dict]) -> str:
    """Analyze a completed trajectory and recommend routing policy changes."""
    if not trajectory:
        return "No trajectory data to analyze."
    scores = [t.get("compaction_90", 1.0) for t in trajectory]
    energies = [t.get("mean_energy", 0.0) for t in trajectory]
    modes = [t.get("gen_mode", "?") for t in trajectory]
    if len(scores) < 2:
        return "Insufficient iterations to recommend policy changes."
    score_trend = scores[-1] - scores[0]
    energy_trend = (energies[-1] - energies[0]) / max(abs(energies[0]), 1e-9)
    recommendations = []
    if score_trend > 0.05:
        recommendations.append(
            f"compaction_90 improved {score_trend:+.3f}: "
            "current dct_topk rate is effective; keep compact_rate.")
    elif score_trend < -0.05:
        recommendations.append(
            f"compaction_90 degraded {score_trend:+.3f}: "
            "try reducing compact_rate (e.g. 0.25) to preserve more context.")
    if energy_trend > 0.1:
        recommendations.append(
            "energy trending up: model is exploring more confidently; "
            "domain routing is effective.")
    if "embeds" not in modes[-1]:
        recommendations.append(
            "last iteration used tokens (not embeds); "
            "set FABRIC_LATENT_COMPACT>0 to enable compaction mode.")
    return " ".join(recommendations) if recommendations else \
        "Policy stable; no routing changes recommended."


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_self_improve(config: SelfImproveConfig | None = None,
                    prompt: str = "") -> dict:
    """Run the self-improvement loop.

    Wraps fabric/latent.py's latent_loop with:
      - per-iteration scoring
      - trajectory storage
      - policy recommendation
    """
    if config is None:
        config = SelfImproveConfig(prompt=prompt or "analyze this topic")
    if not config.prompt:
        return {"error": "no prompt provided"}
    t0 = time.time()
    iterations: list[dict] = []
    best_score = 0.0
    best_output = ""
    best_iteration = -1

    try:
        from .latent import latent_loop
        result = latent_loop(
            prompt=config.prompt,
            max_iterations=config.max_iterations,
            session_tag=config.session_tag or f"si-{int(t0)}",
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"latent loop failed: {e}"}

    for entry in result.get("iterations", []):
        text = entry.get("text", "")[:2000]
        score_result = score_output(text, config.scoring_criteria)
        score = score_result.get("score", 0.0)
        entry["score"] = score
        entry["score_reasons"] = score_result.get("reasons", [])
        if score > best_score:
            best_score = score
            best_output = text
            best_iteration = entry.get("iteration", -1)
        iterations.append(entry)

    trajectory_result = get_shard_trajectory(
        config.department, config.session_tag)
    policy = _recommend_policy_change(trajectory_result.get("trajectory", []))

    return {
        "ok": True,
        "prompt": config.prompt[:200],
        "n_iterations": len(iterations),
        "best_score": round(best_score, 3),
        "best_iteration": best_iteration,
        "best_output": best_output[:1000],
        "iterations": iterations,
        "trajectory": trajectory_result,
        "policy_recommendation": policy,
        "elapsed_s": round(time.time() - t0, 1),
    }
