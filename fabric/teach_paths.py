# SPDX-License-Identifier: Apache-2.0
"""Teacher-example library for component-system prompts (BICAMERAL Goal 4).

"instead of learning the code lines verbatim, the model learns the
function paths to match data throughput from a teacher example."

This module builds a structured prompt library that:

  1. TEACHING_EXAMPLES — curated (input, assembly_diagram, description)
     pairs showing the component model.  These are injected into the
     system prompt so the model learns to REASON about data flow,
     not just repeat tokens.
  2. PATH_TEMPLATES — recipe-like skeletons for common tasks (smoothing
     sensor data, building a chart, routing by domain).  These are
     reference assemblies the model can adapt.
  3. build_prompt(domain) — assembles a system prompt from the examples
     relevant to a domain.
  4. prompt_for_intent(text) — routes the text to the best-matching
     examples and returns a complete system prompt.

The key design insight: the model should learn "a gated clock feeds
a counter whose output smooths before driving a chart" — not the
specific EEL2 source.  The assembly description is the level of
abstraction we teach.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "TeachingExample",
    "PATH_TEMPLATES",
    "TEACHING_EXAMPLES",
    "build_prompt",
    "prompt_for_intent",
    "register_example",
]


@dataclass
class TeachingExample:
    id: str
    domain: str
    difficulty: str
    description: str
    assembly_mermaid: str
    inputs: list[str]
    outputs: list[str]
    blocks_used: list[str]
    tags: list[str]


# ---------------------------------------------------------------------------
# Canonical teaching examples (curated for the model to learn from)
# ---------------------------------------------------------------------------

TEACHING_EXAMPLES: list[TeachingExample] = [
    TeachingExample(
        id="gated_counter_chart",
        domain="general",
        difficulty="beginner",
        description="A gated clock fires a counter; the count output "
                    "is smoothed with a one-pole filter before driving "
                    "a rolling chart.",
        assembly_mermaid=(
            "flowchart LR\n"
            "    C[Clock] -->|trig| G[Gate]\n"
            "    G -->|q| A[Accum]\n"
            "    A -->|acc| S[Smooth]\n"
            "    S -->|cv| V[(Chart)]"
        ),
        inputs=["clock_bpm", "gate_open"],
        outputs=["rolling chart"],
        blocks_used=["clock_bpm", "gate_buffer", "accum", "smooth", "viz_series"],
        tags=["clock", "counter", "chart", "beginner"],
    ),
    TeachingExample(
        id="sensor_threshold_meter",
        domain="engineering",
        difficulty="beginner",
        description="A live sensor feeds a hysteresis threshold; "
                    "the gate output drives a meter visualization.",
        assembly_mermaid=(
            "flowchart LR\n"
            "    S[Sensor] -->|cv| T[Threshold]\n"
            "    T -->|gate| V[(Meter)]"
        ),
        inputs=["sensor_topic"],
        outputs=["meter"],
        blocks_used=["sensor", "threshold", "viz_series"],
        tags=["sensor", "threshold", "meter", "engineering"],
    ),
    TeachingExample(
        id="xy_scope_lfo",
        domain="general",
        difficulty="intermediate",
        description="An LFO modulates X; a fader scales Y; "
                    "the XY pair drives a vectorscope.",
        assembly_mermaid=(
            "flowchart LR\n"
            "    L[LFO] -->|cv| X[X Scale]\n"
            "    F[Fader] -->|out| Y[Y Scale]\n"
            "    X -->|x| V[(XY Scope)]\n"
            "    Y -->|y| V"
        ),
        inputs=["lfo_rate", "x_gain", "y_gain"],
        outputs=["xy_scope"],
        blocks_used=["sine_lfo", "fader", "gain", "viz_xy"],
        tags=["xy", "scope", "lfo", "intermediate"],
    ),
    TeachingExample(
        id="hadamard_3d_scope",
        domain="hoa64",
        difficulty="advanced",
        description="An audio signal feeds a H4 Hadamard transform; "
                    "the W/X/Y/Z components drive a 3D Lissajous scope "
                    "in spherical-harmonic space.",
        assembly_mermaid=(
            "flowchart LR\n"
            "    A[Audio] -->|cv| H4[Hadamard-4]\n"
            "    H4 -->|w| V[(3D Scope)]\n"
            "    H4 -->|x| V\n"
            "    H4 -->|y| V\n"
            "    H4 -->|z| V"
        ),
        inputs=["audio_stream"],
        outputs=["3d_lissajous"],
        blocks_used=["sensor", "hadamard4", "viz_wxyz3d"],
        tags=["hadamard", "3d", "hoa64", "advanced"],
    ),
    TeachingExample(
        id="live_gauge",
        domain="general",
        difficulty="beginner",
        description="A live bus signal (e.g. tok/s) is smoothed and "
                    "rendered as both a gauge and a rolling trace.",
        assembly_mermaid=(
            "flowchart LR\n"
            "    B[Bus] -->|sig| S[Smooth]\n"
            "    S -->|cv| V[(Gauge)]\n"
            "    S -->|cv| T[(Trace)]"
        ),
        inputs=["bus_signal"],
        outputs=["gauge", "rolling_trace"],
        blocks_used=["sensor", "smooth", "viz_series"],
        tags=["live", "gauge", "bus", "beginner"],
    ),
    TeachingExample(
        id="domain_router_signal",
        domain="signal_processing",
        difficulty="intermediate",
        description="A text input selects a domain shard; "
                    "the shard feeds the signal path; "
                    "the result is charted.",
        assembly_mermaid=(
            "flowchart LR\n"
            "    T[Text] -->|intent| R[Domain Router]\n"
            "    R -->|shard_id| I[igram fetch]\n"
            "    I -->|fp16| F[FWHT]\n"
            "    F -->|flux| V[(Chart)]"
        ),
        inputs=["user_text"],
        outputs=["domain_chart"],
        blocks_used=["sensor", "mdct_flux", "viz_series"],
        tags=["router", "domain", "igram", "advanced"],
    ),
    TeachingExample(
        id="bicameral_preprocess_active",
        domain="bicameral",
        difficulty="advanced",
        description="GPU1 preprocesses the input into a spectral "
                    "fingerprint; GPU0 makes the active decision "
                    "from that fingerprint; the decision drives a chart.",
        assembly_mermaid=(
            "flowchart LR\n"
            "    subgraph GPU1\n"
            "        P[Preprocess]\n"
            "    end\n"
            "    subgraph GPU0\n"
            "        A[Active]\n"
            "    end\n"
            "    I[Input] -->|raw| P\n"
            "    P -->|fp| A\n"
            "    A -->|decision| V[(Chart)]"
        ),
        inputs=["raw_signal"],
        outputs=["decision_chart"],
        blocks_used=["sensor", "mdct_flux", "threshold", "viz_series"],
        tags=["bicameral", "gpu1", "gpu0", "advanced"],
    ),
    TeachingExample(
        id="metronome_pattern",
        domain="music",
        difficulty="intermediate",
        description="A clock generates a regular beat; "
                    "the accent fader changes every N beats; "
                    "the beat count drives a pattern chart.",
        assembly_mermaid=(
            "flowchart LR\n"
            "    C[Clock BPM] -->|trig| A[Accum]\n"
            "    A -->|acc| M[Mod]\n"
            "    M -->|beat| V[(Pattern)]\n"
            "    F[Fader] -->|accent| M"
        ),
        inputs=["clock_bpm", "accent_value"],
        outputs=["pattern_chart"],
        blocks_used=["clock_bpm", "accum", "gain", "smooth", "viz_series"],
        tags=["metronome", "beat", "pattern", "intermediate"],
    ),
]


# ---------------------------------------------------------------------------
# Path templates (skeleton assemblies for common tasks)
# ---------------------------------------------------------------------------

PATH_TEMPLATES: dict[str, dict] = {
    "smooth_then_chart": {
        "description": "Smooth a noisy signal then chart it",
        "blocks": [
            {"id": "s", "primitive": "sensor", "label": "Source"},
            {"id": "sm", "primitive": "smooth", "params": {"alpha": 0.2}},
            {"id": "v", "primitive": "viz_series", "label": "Chart"},
        ],
        "wires": [
            {"from": "s.cv", "to": "sm.in"},
            {"from": "sm.cv", "to": "v.in"},
        ],
        "blocks_used": ["sensor", "smooth", "viz_series"],
    },
    "gated_clock_chart": {
        "description": "A clock fires a counter to a chart",
        "blocks": [
            {"id": "c", "primitive": "clock_bpm", "params": {"bpm": 60}},
            {"id": "a", "primitive": "accum", "params": {"per_tick": 1}},
            {"id": "v", "primitive": "viz_series"},
        ],
        "wires": [
            {"from": "c.trig", "to": "a.acc"},
            {"from": "a.acc", "to": "v.in"},
        ],
        "blocks_used": ["clock_bpm", "accum", "viz_series"],
    },
    "threshold_gate": {
        "description": "Gate a signal with a hysteresis threshold",
        "blocks": [
            {"id": "s", "primitive": "sensor"},
            {"id": "t", "primitive": "threshold", "params": {"lo": -0.5, "hi": 0.5}},
            {"id": "g", "primitive": "gain"},
            {"id": "v", "primitive": "viz_series"},
        ],
        "wires": [
            {"from": "s.cv", "to": "t.in"},
            {"from": "t.gate", "to": "g.in"},
            {"from": "s.cv", "to": "g.in"},
            {"from": "g.cv", "to": "v.in"},
        ],
        "blocks_used": ["sensor", "threshold", "gain", "viz_series"],
    },
    "3d_hoa_scope": {
        "description": "Audio to H4 WXYZ 3D scope",
        "blocks": [
            {"id": "a", "primitive": "sensor", "label": "Audio"},
            {"id": "h", "primitive": "hadamard4", "label": "H4"},
            {"id": "v", "primitive": "viz_wxyz3d", "label": "3D Scope"},
        ],
        "wires": [
            {"from": "a.cv", "to": "h.in"},
            {"from": "h.w", "to": "v.w"},
            {"from": "h.x", "to": "v.x"},
            {"from": "h.y", "to": "v.y"},
            {"from": "h.z", "to": "v.z"},
        ],
        "blocks_used": ["sensor", "hadamard4", "viz_wxyz3d"],
    },
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PREFIX = """You are a signal-processing programmer using a modular
component-based system.  Programs are ASSEMBLIES of typed blocks:

  BLOCK TYPES:
    source  — emits a stream: const, clock_bpm, sine_lfo, sensor
    function — transforms one stream: gain, bias, smooth, threshold,
               moving_avg, clamp, accum, hadamard4, mdct_flux
    visualizer — displays: viz_series (chart), viz_xy (scope),
                 viz_wxyz3d (3D scope)

  CONNECTIONS: wires go from an output port to an input port.
  PORT TYPES: cv (continuous value), trig (rising-edge trigger),
               series (rolling buffer).

  The model learns FUNCTION PATHS, not code lines.  Given a goal,
  decompose it into SOURCE → FUNCTION(S) → VISUALIZER and wire them.

  Canonical patterns (learn these structures):
"""


def _examples_section() -> str:
    lines = [""]
    for ex in TEACHING_EXAMPLES:
        lines.append(f"  [{ex.difficulty.upper()}] {ex.id}")
        lines.append(f"    {ex.description}")
        lines.append(f"    inputs: {', '.join(ex.inputs)}")
        lines.append(f"    outputs: {', '.join(ex.outputs)}")
        lines.append(f"    blocks: {', '.join(ex.blocks_used)}")
        # Inline mermaid so the model sees the data flow shape
        for ml in ex.assembly_mermaid.splitlines():
            lines.append(f"    {ml}")
    return "\n".join(lines)


def _templates_section() -> str:
    lines = ["\n  RECIPE TEMPLATES (adapt to the task):"]
    for name, t in PATH_TEMPLATES.items():
        lines.append(f"\n  {name}: {t['description']}")
        lines.append(f"    blocks: {', '.join(t['blocks_used'])}")
    return "\n".join(lines)


def build_prompt(domain: str = "general", include_templates: bool = True) -> str:
    """Build a system prompt for the component model.

    domain filters the examples to relevant ones; "all" returns everything.
    """
    if domain == "all":
        examples = TEACHING_EXAMPLES
    else:
        examples = [e for e in TEACHING_EXAMPLES
                    if e.domain == domain or domain in e.tags]
    if not examples:
        examples = TEACHING_EXAMPLES[:3]  # always show at least 3
    parts = [SYSTEM_PREFIX]
    parts.append(_examples_section())
    if include_templates:
        parts.append(_templates_section())
    parts.append("\n\nWhen given a goal, respond with:\n"
                "  1. the signal path (SOURCE → ... → VISUALIZER)\n"
                "  2. a brief description of what each block does\n"
                "  3. the wire list")
    return "\n".join(parts)


def prompt_for_intent(text: str) -> str:
    """Route the intent text to the best-matching domain and return
    the appropriate system prompt.  Uses the domain_router's richer
    keyword lexicon (medical / engineering / code / finance / legal /
    navigation / media / hoa64) when available.
    """
    domain = "general"
    try:
        from . import domain_router
        routes = domain_router.route_intent(text, top_k=1)
        if routes and routes[0][1] > 0.0:
            domain = routes[0][0]
    except Exception:  # noqa: BLE001
        # Fallback: simple tag overlap on the teaching examples
        text_l = text.lower()
        scores: dict[str, int] = {}
        for ex in TEACHING_EXAMPLES:
            for tag in ex.tags:
                if tag in text_l:
                    scores[ex.domain] = scores.get(ex.domain, 0) + 1
        if scores:
            domain = max(scores, key=scores.get)
    return build_prompt(domain=domain)


def register_example(example: TeachingExample) -> str:
    """Add a teaching example to the in-process library."""
    existing = [i for i, e in enumerate(TEACHING_EXAMPLES) if e.id == example.id]
    if existing:
        TEACHING_EXAMPLES[existing[0]] = example
    else:
        TEACHING_EXAMPLES.append(example)
    return example.id
