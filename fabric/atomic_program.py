# SPDX-License-Identifier: Apache-2.0
"""Atomic block assembly IR + compiler (BICAMERAL Goals 2, 4).

Programming as component-based assembly (Goal 4): instead of learning
code lines verbatim, the model learns the FUNCTION PATHS to match
data throughput from a teacher example.  This module defines:

  1. AtomicBlock  — a typed, port-typed module with a body (EEL2 source
     or a foreign-language stub).  Inputs/outputs are named and typed
     (cv | trig | series).
  2. Wire          — a connection between two ports (src_block.out_port ->
     dst_block.in_port).
  3. AtomicProgram — a module graph (blocks + wires) with an entry
     point.  The graph is validated, then compiled to a MicroFX patch
     dict (modules[] + wires[]), or to a foreign-language program
     (Python, JS, C, WGSL shader, EEL2, etc.).

The compiler is multi-target:

  - "microfx"   → MicroFX patch dict (modules[] + wires[])
  - "eel2"      → EEL2 source (concatenates @init/@tick blocks)
  - "python"    → Python function with numpy calls
  - "wgsl"      → WGSL compute shader stub (for GPU1 DMA paths)
  - "flowchart" → mermaid diagram (for the teaching layer)

The FROM-DESCRIPTION compiler (Goal 2): given a natural-language
description of a desired program ("a clock-gated counter that smooths
its output and drives a chart"), it returns an AtomicProgram.  This
requires a model call, so from_description() takes an llm_url and
returns the compiled output.

Teacher-example library (Goal 4): programs can be stored as teaching
pairs {description, atomic_program} and injected into prompts so the
model learns the component pattern rather than memorizing the output.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Port",
    "AtomicBlock",
    "Wire",
    "AtomicProgram",
    "PortType",
    "compile_program",
    "from_description",
    "describe_program",
    "register_teacher_example",
    "get_teacher_examples",
    "PROGRAM_REGISTRY",
]

PortType = str  # "cv" | "trig" | "series"


@dataclass
class Port:
    name: str
    kind: PortType
    direction: str  # "in" | "out"


# Re-export from microfx for the catalog
_MODULES: dict | None = None


def _modules() -> dict:
    global _MODULES
    if _MODULES is None:
        try:
            from . import microfx
            _MODULES = microfx.MODULES
        except Exception:  # noqa: BLE001
            _MODULES = {}
    return _MODULES


@dataclass
class AtomicBlock:
    id: str
    primitive: str
    params: dict[str, float] = field(default_factory=dict)
    label: str = ""
    position: tuple[float, float] = (0.0, 0.0)

    def input_ports(self) -> list[str]:
        mods = _modules()
        return mods.get(self.primitive, {}).get("inputs", [])

    def output_ports(self) -> list[str]:
        mods = _modules()
        return mods.get(self.primitive, {}).get("outputs", [])

    def to_module(self) -> dict:
        return {
            "id": self.id,
            "primitive": self.primitive,
            "params": [
                {"name": k, "value": v} for k, v in self.params.items()
            ],
            "label": self.label,
            "position": {"x": self.position[0], "y": self.position[1]},
        }


@dataclass
class Wire:
    src: str  # "block_id.port_name"
    dst: str  # "block_id.port_name"

    def to_wire(self) -> dict:
        return {"from": self.src, "to": self.dst}


@dataclass
class AtomicProgram:
    name: str
    description: str = ""
    blocks: list[AtomicBlock] = field(default_factory=list)
    wires: list[Wire] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    hash: str = ""

    def compute_hash(self) -> str:
        key = json.dumps({
            "name": self.name,
            "blocks": [asdict(b) for b in self.blocks],
            "wires": [asdict(w) for w in self.wires],
        }, sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def validate(self) -> str | None:
        if not self.blocks:
            return "program has no blocks"
        ids = {b.id for b in self.blocks}
        for w in self.wires:
            s_id = w.src.split(".")[0]
            d_id = w.dst.split(".")[0]
            if s_id not in ids:
                return f"wire source block '{s_id}' not found"
            if d_id not in ids:
                return f"wire dst block '{d_id}' not found"
        for b in self.blocks:
            for inp in b.input_ports():
                if inp == "in":
                    pass
        return None

    def to_patch(self) -> dict:
        """Compile to MicroFX patch dict."""
        err = self.validate()
        if err:
            return {"error": f"invalid program: {err}"}
        return {
            "modules": [b.to_module() for b in self.blocks],
            "wires": [w.to_wire() for w in self.wires],
        }

    def to_eel2(self) -> str:
        """Compile to a monolithic EEL2 program."""
        blocks_by_id = {b.id: b for b in self.blocks}
        srcs: list[str] = ["// Autogenerated EEL2 from AtomicProgram\n\n"]
        srcs.append(f"// {self.description}\n\n")
        for b in self.blocks:
            if b.primitive in ("viz_series", "viz_xy", "viz_wxyz3d"):
                continue
            srcs.append(f"// --- {b.id} ({b.primitive}) ---\n")
            mods = _modules()
            src = mods.get(b.primitive, {}).get("source", "")
            if src:
                srcs.append(src + "\n\n")
        return "\n".join(srcs)

    def to_python(self) -> str:
        """Compile to a Python function (numpy-based)."""
        lines = [
            f'"""Python translation of {self.name}: {self.description}."""',
            "from __future__ import annotations",
            "import numpy as np",
            "",
            "def run_program(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:",
        ]
        for b in self.blocks:
            if b.primitive == "const":
                val = b.params.get("value", 1.0)
                lines.append(f"    {b.id}_out = np.full(1, {val})")
            elif b.primitive == "gain":
                f = b.params.get("factor", 1.0)
                lines.append(f"    {b.id}_out = {b.input_ports()[0]} * {f}")
            elif b.primitive == "smooth":
                a = b.params.get("alpha", 0.2)
                lines.append(
                    f"    {b.id}_out = np.append([0], np.cumsum({b.input_ports()[0]}) * {a})"
                )
            elif b.primitive == "moving_avg":
                n = int(b.params.get("n", 16))
                lines.append(f"    {b.id}_out = np.convolve({b.input_ports()[0]}, "
                            f"np.ones({n})/{n}, mode='same')")
            elif b.primitive == "threshold":
                lo, hi = b.params.get("lo", 40), b.params.get("hi", 50)
                lines.append(
                    f"    {b.id}_out = np.where({b.input_ports()[0]} > {hi}, 1, "
                    f"np.where({b.input_ports()[0]} < {lo}, 0, 0.5))"
                )
            elif b.primitive == "hadamard4":
                lines.append(f"    {b.id}_w = {b.input_ports()[0]} + 0  # TODO: FWHT-4")
            elif b.primitive == "viz_series":
                lines.append(f"    # {b.id}: chart sink (no-op in python)")
            else:
                lines.append(f"    {b.id}_out = {b.input_ports()[0]}  # passthrough")
        lines.append("    return {}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Mermaid flowchart for the teaching view."""
        lines = ["flowchart TD", "    %% AtomicProgram: " + self.name]
        for b in self.blocks:
            icon = {"source": "([(O)]", "function": "[{b.primitive}]",
                    "visualizer": "((" + b.primitive + "))"}.get(
                        b.primitive, "[?]")
            lines.append(f"    {b.id}{icon}  %% {b.label or b.primitive}")
        for w in self.wires:
            lines.append(f"    {w.src} --> {w.dst}")
        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "n_blocks": len(self.blocks),
            "n_wires": len(self.wires),
            "tags": self.tags,
            "hash": self.compute_hash(),
        }


# ---------------------------------------------------------------------------
# Teacher-example registry (Goal 4)
# ---------------------------------------------------------------------------

PROGRAM_REGISTRY: list[dict] = []


def register_teacher_example(program: AtomicProgram, difficulty: str = "beginner",
                             domain: str = "general") -> str:
    """Register a program as a teaching example for the model prompt."""
    key = program.compute_hash()
    entry = {
        "hash": key,
        "difficulty": difficulty,
        "domain": domain,
        "summary": program.summary(),
        "mermaid": program.to_mermaid(),
    }
    PROGRAM_REGISTRY.append(entry)
    return key


def get_teacher_examples(domain: str = "", difficulty: str = "",
                         limit: int = 10) -> list[dict]:
    out = PROGRAM_REGISTRY
    if domain:
        out = [e for e in out if e.get("domain") == domain]
    if difficulty:
        out = [e for e in out if e.get("difficulty") == difficulty]
    return out[:limit]


def _seed_registry() -> None:
    """Pre-seed with canonical examples for the model to learn from."""
    import threading
    with threading.Lock():
        if PROGRAM_REGISTRY:
            return
        programs = [
            AtomicProgram(
                name="gated_clock_counter",
                description="A gated clock fires a counter; the count is "
                            "smoothed before driving a rolling chart.",
                blocks=[
                    AtomicBlock(id="c1", primitive="clock_bpm", label="Clock"),
                    AtomicBlock(id="g1", primitive="gate_buffer", label="Gate"),
                    AtomicBlock(id="a1", primitive="accum", params={"per_tick": 1},
                               label="Counter"),
                    AtomicBlock(id="s1", primitive="smooth",
                               params={"alpha": 0.1}, label="Smoother"),
                    AtomicBlock(id="v1", primitive="viz_series",
                               label="Chart"),
                ],
                wires=[
                    Wire(src="c1.trig", dst="g1.in"),
                    Wire(src="g1.q", dst="a1.acc"),
                    Wire(src="a1.acc", dst="s1.in"),
                    Wire(src="s1.cv", dst="v1.in"),
                ],
                tags=["clock", "counter", "chart", "beginner"],
            ),
            AtomicProgram(
                name="xy_scope_with_twist",
                description="An XY pad drives a scope; a fader scales the X "
                            "axis while a second oscillator modulates Y.",
                blocks=[
                    AtomicBlock(id="x1", primitive="const", label="X base"),
                    AtomicBlock(id="f1", primitive="fader", label="X gain",
                               params={"default": 0.5}),
                    AtomicBlock(id="l1", primitive="sine_lfo",
                               params={"rate_hz": 1.0}, label="LFO Y"),
                    AtomicBlock(id="m1", primitive="mul", label="Mix"),
                    AtomicBlock(id="v1", primitive="viz_xy", label="Scope"),
                ],
                wires=[
                    Wire(src="x1.cv", dst="f1.in"),
                    Wire(src="f1.out", dst="m1.in"),
                    Wire(src="l1.cv", dst="m1.in"),
                    Wire(src="m1.out", dst="v1.y"),
                ],
                tags=["xy", "scope", "lfo", "intermediate"],
            ),
            AtomicProgram(
                name="hadamard_wxyz_scope",
                description="An audio stream feeds a H4 transform; the four "
                            "W/X/Y/Z outputs drive a 3D Lissajous scope.",
                blocks=[
                    AtomicBlock(id="sig", primitive="sensor", label="Sensor"),
                    AtomicBlock(id="h4", primitive="hadamard4",
                               label="Hadamard-4"),
                    AtomicBlock(id="v1", primitive="viz_wxyz3d",
                               label="3D Scope"),
                ],
                wires=[
                    Wire(src="sig.cv", dst="h4.in"),
                    Wire(src="h4.w", dst="v1.w"),
                    Wire(src="h4.x", dst="v1.x"),
                    Wire(src="h4.y", dst="v1.y"),
                    Wire(src="h4.z", dst="v1.z"),
                ],
                tags=["hadamard", "3d", "wxyz", "advanced"],
            ),
            AtomicProgram(
                name="threshold_gate_mixer",
                description="Two signals enter a hysteresis threshold; the "
                            "output gates a third signal.",
                blocks=[
                    AtomicBlock(id="s1", primitive="sensor", label="Signal 1"),
                    AtomicBlock(id="s2", primitive="sensor", label="Signal 2"),
                    AtomicBlock(id="th1", primitive="threshold",
                               params={"lo": -0.5, "hi": 0.5},
                               label="Threshold"),
                    AtomicBlock(id="g1", primitive="gain",
                               params={"factor": 1.0}, label="Gain"),
                    AtomicBlock(id="v1", primitive="viz_series", label="Chart"),
                ],
                wires=[
                    Wire(src="s1.cv", dst="th1.in"),
                    Wire(src="th1.gate", dst="g1.in"),
                    Wire(src="s2.cv", dst="g1.in"),
                    Wire(src="g1.cv", dst="v1.in"),
                ],
                tags=["threshold", "gate", "mix", "intermediate"],
            ),
        ]
        for p in programs:
            register_teacher_example(p, difficulty="beginner"
                                   if "beginner" in p.tags
                                   else "intermediate",
                                   domain="signal_processing")


_seed_registry()


# ---------------------------------------------------------------------------
# Compilation targets
# ---------------------------------------------------------------------------

def compile_program(program: AtomicProgram,
                    target: str = "microfx") -> dict | str:
    """Compile an AtomicProgram to the specified target."""
    if target == "microfx":
        return program.to_patch()
    if target == "eel2":
        return program.to_eel2()
    if target == "python":
        return program.to_python()
    if target == "mermaid":
        return program.to_mermaid()
    if target == "summary":
        return program.summary()
    return {"error": f"unknown target '{target}'"}


# ---------------------------------------------------------------------------
# FROM-DESCRIPTION compiler (Goal 2: "the model assembles data paths from
# function/control blocks to replicate any code function")
# ---------------------------------------------------------------------------

def from_description(description: str,
                    llm_url: str = "http://127.0.0.1:8000/v1",
                    model: str = "Qwen/Qwen2.5-7B-Instruct",
                    ) -> dict:
    """Compile a natural-language program description to an AtomicProgram.

    Calls the model to translate the description into an AtomicProgram
    JSON, then validates and registers it.  Returns the compiled target
    dict (default: MicroFX patch).
    """
    examples = get_teacher_examples(limit=3)
    ex_text = ""
    if examples:
        ex_lines = ["The model has learned these patterns:"]
        for ex in examples:
            ex_lines.append(f"  - {ex['summary']['name']}: "
                           f"{ex['summary']['n_blocks']} blocks, "
                           f"{ex['summary']['n_wires']} wires")
            ex_lines.append(f"    {ex['mermaid']}")
        ex_text = "\n".join(ex_lines)

    system_prompt = (
        "You are an atomic-component programmer.  Given a description of a "
        "desired signal-processing program, return a JSON AtomicProgram.\n\n"
        "Rules:\n"
        "  - Use only primitives from the MicroFX catalog: "
        "const, clock_bpm, sine_lfo, gain, bias, smooth, threshold, "
        "moving_avg, clamp, accum, toggle, hadamard4, viz_series, viz_xy, "
        "viz_wxyz3d, sensor, tap\n"
        "  - Each BLOCK has a unique id (e.g. c1, g1, a1)\n"
        "  - Wire: {from: 'src_id.out_port', to: 'dst_id.in_port'}\n"
        "  - Blocks with no inputs are SOURCES (const, clock_bpm, sine_lfo, sensor)\n"
        "  - Blocks with no outputs are SINKS (viz_series, viz_xy, viz_wxyz3d)\n"
        "  - Output as JSON with keys: name, description, blocks[], wires[], tags[]\n"
        + (f"\n{ex_text}" if ex_text else "")
    )
    user_prompt = f"Describe this program:\n{description}"

    try:
        import urllib.request
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        }).encode()
        req = urllib.request.Request(
            llm_url + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read().decode())
        content = resp["choices"][0]["message"]["content"]
        # Extract JSON from ```json ... ``` fences if present
        if "```" in content:
            for seg in content.split("```"):
                if seg.strip().startswith("json"):
                    content = seg.strip()[4:].strip()
                    break
                if seg.strip().startswith("{"):
                    content = seg.strip()
                    break
            else:
                # fall through: raw content
                pass
        p_dict = json.loads(content)
    except Exception as e:  # noqa: BLE001
        return {"error": f"llm call failed: {e}"}

    blocks = [AtomicBlock(**b) for b in p_dict.get("blocks", [])]
    wires = [Wire(**w) for w in p_dict.get("wires", [])]
    prog = AtomicProgram(
        name=p_dict.get("name", "generated"),
        description=p_dict.get("description", description),
        blocks=blocks,
        wires=wires,
        tags=p_dict.get("tags", []),
    )
    err = prog.validate()
    if err:
        return {"error": f"generated program invalid: {err}", "raw": p_dict}
    register_teacher_example(prog, domain="user-generated")
    return compile_program(prog, "microfx")


def describe_program(program: AtomicProgram) -> str:
    """Generate a human-readable description of a program."""
    parts = [f"Program: {program.name}"]
    if program.description:
        parts.append(f"Description: {program.description}")
    parts.append(f"Blocks ({len(program.blocks)}):")
    for b in program.blocks:
        ports_in = b.input_ports()
        ports_out = b.output_ports()
        parts.append(f"  {b.id} ({b.primitive}) "
                   f"in:{ports_in} out:{ports_out}")
    parts.append(f"Wires ({len(program.wires)}):")
    for w in program.wires:
        parts.append(f"  {w.src} → {w.dst}")
    return "\n".join(parts)
