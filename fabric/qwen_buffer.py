# SPDX-License-Identifier: Apache-2.0
"""Qwen-Flash-Next buffer events wired into MicroFX (BICAMERAL §6.6).

The buffer is not a separate runtime — it is a *patch* in MicroFX.
This module publishes three signals to the existing bus:

  1. ``compaction_90``  — a SENSOR input the buffer publishes to gmem[].
                          dct_topk rate, per-block energy, QSA indexer
                          top-k — all readable as ``input('compaction_90')``
                          from any EEL2 patch.
  2. ``layer_swap``     — a TRIGGER (rising-edge in jsfx.js) fired when
                          a per-layer hot-swap event arrives from the
                          active card.
  3. ``mv2_tap``        — a TRIGGER fired per latent-loop iteration,
                          mirroring the buffer's compaction trajectory
                          to the Sciences shard.

The catalog entries below are drop-in to MODULES.  Patch wiring code
that wants the live value can use ``input('compaction_90')`` from an
EEL2 program; shell + per-iteration logic can call ``fire_layer_swap()``
or ``fire_mv2_tap()`` to broadcast the event.

Landmine (BICAMERAL §9 #6): EEL2 ``compile()`` must wrap parse output
in block node; ``Proxy.has`` must return true for all globals.  Our
buffer patch variables are real module outputs, not ad-hoc globals,
so this does not regress.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Module name -> latest known value (process-global, MicroFX-bus mirror).
# The real source-of-truth is the buffer loop in fabric/latent.py; this
# is the read-side cache that the EEL2 patch sees.
_BUS: dict[str, float] = {
    "compaction_90": 1.0,   # 1.0 == fully compact (sentinel)
    "qsa_keep_frac": 0.25,  # 512/2048 default
    "igram_block_load": 0.0,
    "layer_idx": -1,
}

# Rolling event log: layer_swap and mv2_tap are rising-edge TRIGGERS
# in jsfx.js, so we keep a small ring of recent events with timestamps
# for shell inspection.  The actual trigger pulse is computed in the
# front-end from the ring (delta > 0 within a frame = rising edge).
_EVENTS: list[dict] = []  # [{t, name, value}, ...]
_MAX_EVENTS = 256


def publish_compaction(value: float) -> None:
    """Buffer publishes its current compaction_90 (Phase 3 spectral
    compaction signal: lower = more compressible)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if v != v or v < 0:  # NaN / negative → drop
        return
    _BUS["compaction_90"] = v


def publish_qsa_keep(keep: int, budget: int = 2048) -> None:
    """Buffer publishes its current QSA keep fraction (QSA indexer KEEP
    of BUDGET blocks).  This is the second layer of the stack: QSA
    prunes ``r=4`` then dct_topk@0.5 prunes another 2x, total ~8x."""
    try:
        k, b = int(keep), int(budget)
    except (TypeError, ValueError):
        return
    if b <= 0:
        return
    _BUS["qsa_keep_frac"] = max(0.0, min(1.0, k / b))


def publish_igram_progress(loaded: int, total: int) -> None:
    """Buffer publishes the current n-gram shard load progress
    (e.g. 7 of 10 shards prefetched ahead of the decode window)."""
    try:
        l, t = int(loaded), int(total)
    except (TypeError, ValueError):
        return
    if t <= 0:
        return
    _BUS["igram_block_load"] = max(0.0, min(1.0, l / t))


def fire_layer_swap(layer_idx: int) -> None:
    """Buffer signals a per-layer hot-swap event (TRIGGER: rising edge
    in jsfx.js).  The XPU graph capture for the per-layer kernel is
    triggered off this event."""
    try:
        li = int(layer_idx)
    except (TypeError, ValueError):
        return
    _BUS["layer_idx"] = li
    _push_event("layer_swap", {"layer": li})


def fire_mv2_tap(note: str = "") -> None:
    """Buffer signals a latent-loop iteration complete (TRIGGER: the
    .mv2 Sciences shard write fired for this iteration)."""
    _push_event("mv2_tap", {"note": str(note)[:200]})


def _push_event(name: str, payload: dict) -> None:
    _EVENTS.append({"t": time.time(), "name": name, **payload})
    del _EVENTS[:-_MAX_EVENTS]
    logger.debug("qwen_buffer: %s %s", name, payload)


def snapshot() -> dict:
    """Return the current buffer-state for the shell / API.  Used by
    ``/api/buffer/state`` and the LCARS deck pane."""
    return {
        "bus": dict(_BUS),
        "events": list(_EVENTS[-16:]),
        "ts": time.time(),
    }


# ---------------------------------------------------------------------------
# MicroFX module catalog entries (drop into fabric/microfx.py MODULES)
# ---------------------------------------------------------------------------

BUFFER_MODULES: dict[str, dict] = {
    "compaction_90": {
        "category": "source",
        "title": "Buffer: compaction_90",
        "params": [],
        "inputs": [], "outputs": ["cv"],
        "host": True,  # buffer publishes via fabric.qwen_buffer
        "source": "@tick\noutput('cv', input('compaction_90'));",
    },
    "qsa_keep_frac": {
        "category": "source",
        "title": "Buffer: QSA keep fraction",
        "params": [],
        "inputs": [], "outputs": ["cv"],
        "host": True,
        "source": "@tick\noutput('cv', input('qsa_keep_frac'));",
    },
    "igram_progress": {
        "category": "source",
        "title": "Buffer: n-gram shard load progress",
        "params": [],
        "inputs": [], "outputs": ["cv"],
        "host": True,
        "source": "@tick\noutput('cv', input('igram_block_load'));",
    },
    "layer_swap": {
        "category": "source",
        "title": "Buffer: layer_swap trigger",
        "params": [],
        "inputs": [], "outputs": ["trig"],
        "host": True,  # host-fed via fabric.qwen_buffer.fire_layer_swap()
        "source": "@tick\noutput('trig', input('layer_swap'));",
    },
    "mv2_tap": {
        "category": "source",
        "title": "Buffer: mv2 write tap",
        "params": [],
        "inputs": [], "outputs": ["trig"],
        "host": True,
        "source": "@tick\noutput('trig', input('mv2_tap'));",
    },
}
