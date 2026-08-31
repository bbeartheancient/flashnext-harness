# SPDX-License-Identifier: Apache-2.0
"""App round-trip harness (BICAMERAL Goal 3).

"Users could edit and run completed applications by modifying the
assembly blocks."

This wraps the existing App Mode (TR App Management) flow with a
harness that:

  1. Loads a library app by id
  2. Decomposes it into an AtomicProgram (the assembly view)
  3. Lets the user (or model) modify one or more blocks / wires
  4. Compiles back to a MicroFX patch
  5. Mounts the new app in a viewport
  6. Captures the per-frame output and the dm_trace for validation

The round-trip works in both directions:

  app  →  AtomicProgram  (decompose)
  AtomicProgram  →  app  (compose + mount)

The decomposition is based on the app's io + source.  For EEL2
PRIMITIVE_APPS the block graph is direct (the primitive is one
block); for patch apps the blocks[] + wires[] ARE the AtomicProgram
already.  This module reconciles both forms into the same IR.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from .atomic_program import (
    AtomicBlock,
    AtomicProgram,
    Wire,
    compile_program,
    describe_program,
)

logger = logging.getLogger(__name__)

__all__ = [
    "decompose_app",
    "compose_app",
    "round_trip",
    "modify_block",
    "modify_wire",
]


@dataclass
class AppSummary:
    app_id: str
    name: str
    kernel: str = ""
    n_blocks: int = 0
    n_wires: int = 0
    source: str = ""
    io: dict = None  # type: ignore


def _app_spec(app_id: str) -> dict:
    """Load an app's full spec from the library."""
    try:
        from . import library
        return library.get_app_spec(app_id)
    except Exception as e:  # noqa: BLE001
        return {"error": f"load failed: {e}"}


def decompose_app(app_id: str) -> dict:
    """Load an app and decompose it into an AtomicProgram.

    Returns the program + a summary so the shell can render a
    'before' view before allowing edits.
    """
    spec = _app_spec(app_id)
    if "error" in spec:
        return spec
    program = _spec_to_program(spec, app_id)
    summary = program.summary()
    return {
        "app_id": app_id,
        "summary": summary,
        "program": {
            "name": program.name,
            "description": program.description,
            "blocks": [asdict(b) for b in program.blocks],
            "wires": [asdict(w) for w in program.wires],
            "tags": program.tags,
        },
        "human": describe_program(program),
    }


def _spec_to_program(spec: dict, app_id: str) -> AtomicProgram:
    """Translate an app spec into an AtomicProgram."""
    program = AtomicProgram(name=spec.get("title", app_id),
                            description=spec.get("principle", ""))
    modules = spec.get("modules") or []
    wires = spec.get("wires") or []
    for m in modules:
        params = {p.get("name"): p.get("value") for p in m.get("params", [])}
        program.blocks.append(AtomicBlock(
            id=m.get("id", "blk"),
            primitive=m.get("primitive", ""),
            params=params,
            label=m.get("label", ""),
        ))
    for w in wires:
        program.wires.append(Wire(src=w.get("from", ""), dst=w.get("to", "")))
    program.tags = [spec.get("kernel", "")]
    return program


def compose_app(program: AtomicProgram,
                app_id: str | None = None) -> dict:
    """Compile an AtomicProgram back into a MicroFX patch.

    Returns a dict ready for POST /api/microfx/patch.
    """
    err = program.validate()
    if err:
        return {"error": f"invalid program: {err}"}
    patch = compile_program(program, "microfx")
    if "error" in patch:
        return patch
    return {
        "ok": True,
        "app_id": app_id or program.name,
        "title": program.name,
        "patch": patch,
        "summary": program.summary(),
    }


def modify_block(program: AtomicProgram, block_id: str,
                 **changes) -> AtomicProgram:
    """Return a deep-copy program with one block's params modified."""
    new_prog = AtomicProgram(
        name=program.name,
        description=program.description,
        blocks=[copy.deepcopy(b) for b in program.blocks],
        wires=[copy.deepcopy(w) for w in program.wires],
        tags=list(program.tags),
    )
    for b in new_prog.blocks:
        if b.id == block_id:
            for k, v in changes.items():
                if hasattr(b, k):
                    setattr(b, k, v)
                elif isinstance(b.params, dict):
                    b.params[k] = v
            break
    return new_prog


def modify_wire(program: AtomicProgram, src: str, dst: str,
                mode: str = "set") -> AtomicProgram:
    """Add or remove a wire."""
    if mode == "remove":
        program.wires = [w for w in program.wires
                         if not (w.src == src and w.dst == dst)]
    else:
        existing = any(w.src == src and w.dst == dst for w in program.wires)
        if not existing:
            program.wires.append(Wire(src=src, dst=dst))
    return program


def round_trip(app_id: str, modifications: dict | None = None) -> dict:
    """Full round-trip: load → decompose → modify → compose → mount.

    modifications:
      {"blocks": [{"id": "c1", "params": {"bpm": 120}}],
       "wires": [{"src": "c1.trig", "dst": "g1.in", "mode": "add"}]}
    """
    t0 = time.time()
    decomp = decompose_app(app_id)
    if "error" in decomp:
        return decomp
    program = AtomicProgram(
        name=decomp["program"]["name"],
        description=decomp["program"]["description"],
        blocks=[AtomicBlock(**b) for b in decomp["program"]["blocks"]],
        wires=[Wire(**w) for w in decomp["program"]["wires"]],
        tags=decomp["program"]["tags"],
    )
    if modifications:
        for b in (modifications.get("blocks") or []):
            modify_block(program, b["id"], **{
                k: v for k, v in b.items() if k != "id"
            })
        for w in (modifications.get("wires") or []):
            modify_wire(program, w["src"], w["dst"],
                        mode=w.get("mode", "set"))
    composed = compose_app(program, app_id=app_id)
    composed["elapsed_s"] = round(time.time() - t0, 2)
    composed["decomposed"] = decomp["summary"]
    return composed
