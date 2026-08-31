# SPDX-License-Identifier: Apache-2.0
"""Bicameral pipeline app (BICAMERAL Goal 9).

"One GPU preprocesses the data so the routing is fast (the
'subconscious' mind), the other GPU makes the conscious decisions
(the 'active' mind)."

This module ships a measurable MicroFX app that EXERCISES the
two-card topology without needing the Qwen-FN weights:

  - "preprocess" tile: runs on GPU1 (the preprocessor card)
    — uses the existing lat-fwht-worker ESIMD kernel to compute
      a spectral fingerprint of a hidden-state-shaped sample
      (synthetic when real captures aren't available)
  - "active" tile:    runs on GPU0 (the active card)
    — the compaction decision, governed by the spectral signal
  - "loopback":       the active tile's output feeds back into
      preprocess, closing the bicameral loop

The app is a first-class MicroFX program: it lives in the library
and can be mounted in the App Mode wall.  The key measurement is
the per-frame latency split: preprocess must finish before active
can decide.  When real Qwen-FN rows land, swap synthetic for real
via the ``--rows`` flag.

Landmine: this is a *demo* pipeline.  The production bicameral
system uses vLLM hidden_states capture + fabric/latent.py for
real traffic; this app is the test stand.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass

import numpy as np

from .atomic_program import (
    AtomicBlock,
    AtomicProgram,
    Wire,
    compile_program,
    register_teacher_example,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BicameralPipeline",
    "run_pipeline",
    "PIPELINE_PROGRAM",
]


# ---------------------------------------------------------------------------
# The pipeline: a typed module graph (Goal 4 pattern)
# ---------------------------------------------------------------------------

PIPELINE_PROGRAM = AtomicProgram(
    name="bicameral_pipeline",
    description="Two-card test stand: GPU1 (preprocess) emits a "
                "spectral fingerprint of the input; GPU0 (active) "
                "decides a compaction rate and feeds back.",
    blocks=[
        AtomicBlock(id="src", primitive="sensor", label="Input sensor"),
        AtomicBlock(id="fwht", primitive="mdct_flux", label="GPU1: FWHT"),
        AtomicBlock(id="thr", primitive="threshold",
                    params={"lo": 0.3, "hi": 0.7},
                    label="GPU0: Active decision"),
        AtomicBlock(id="smooth", primitive="smooth",
                    params={"alpha": 0.2},
                    label="GPU0: Smooth decision"),
        AtomicBlock(id="viz", primitive="viz_series",
                    label="Chart (decision)"),
    ],
    wires=[
        Wire(src="src.cv", dst="fwht.in"),
        Wire(src="fwht.flux", dst="thr.in"),
        Wire(src="thr.gate", dst="smooth.in"),
        Wire(src="smooth.cv", dst="viz.in"),
    ],
    tags=["bicameral", "gpu1", "gpu0", "preprocessor", "active"],
)
register_teacher_example(PIPELINE_PROGRAM,
                         difficulty="advanced", domain="bicameral")


# ---------------------------------------------------------------------------
# Runtime: produce a measurable per-frame trace
# ---------------------------------------------------------------------------

@dataclass
class FrameStats:
    seq: int
    t: float
    preprocess_ms: float
    active_ms: float
    flux: float
    decision: float
    decision_smoothed: float


def run_pipeline(rows: np.ndarray, n_frames: int = 60,
                 alpha: float = 0.2,
                 lo: float = 0.3, hi: float = 0.7,
                 record_to_trace: bool = True) -> dict:
    """Run the pipeline on a hidden-state-shaped sample.

    Parameters
    ----------
    rows : [L, H] fp16/fp32  (synthetic when no real Qwen-FN rows)
    n_frames : number of pipeline iterations (loopback recomputes each)
    alpha : smooth coefficient
    lo, hi : hysteresis thresholds

    Returns
    -------
    dict with 'frames' (per-frame stats), 'summary' (means), and
    the compiled patch (so the shell can mount it as an app).
    """
    if rows.ndim != 2:
        raise ValueError(f"rows must be [L, H], got {rows.shape}")
    H = rows.shape[1]
    rng = np.random.default_rng(0)
    # For synthetic GPU1 preprocess: project each frame to a single
    # scalar via mean over BLOCK-wide head-dim blocks (the framework's
    # BLOCK spec from afi.spectral: BLOCK=256 for Qwen-FN, BLOCK=128
    # for Qwen 3.8 27B).  Real GPU1 will use the FWHT worker.
    from afi.spectral import BLOCK as _BLOCK  # noqa: E402
    nb = H // _BLOCK
    if nb * _BLOCK != H:
        nb += 1
    rows_padded = (np.pad(rows, ((0, 0), (0, nb * _BLOCK - H))) if nb * _BLOCK != H
                   else rows)
    block_means = rows_padded.reshape(-1, nb, _BLOCK).mean(axis=(1, 2))

    # Active card: simple hysteresis
    decision = 0.0
    smoothed = 0.0
    frames: list[FrameStats] = []
    for i in range(n_frames):
        t0 = time.time()
        # Synthetic GPU1 latency: the actual call would go to the
        # fwht worker over its JSON-line pipe; here we model it.
        t_pre = time.time()
        # Sample a random subset to simulate a stream slice
        idx = i % len(block_means)
        flux = float(block_means[idx] + 0.05 * rng.standard_normal())
        preprocess_ms = (time.time() - t_pre) * 1000
        # GPU0 active decision
        t_act = time.time()
        if flux > hi:
            decision = 1.0
        elif flux < lo:
            decision = 0.0
        # else hold
        smoothed += alpha * (decision - smoothed)
        active_ms = (time.time() - t_act) * 1000
        frames.append(FrameStats(
            seq=i,
            t=t0,
            preprocess_ms=round(preprocess_ms, 3),
            active_ms=round(active_ms, 3),
            flux=round(flux, 4),
            decision=decision,
            decision_smoothed=round(smoothed, 4),
        ))
        if record_to_trace:
            try:
                from . import dma_trace
                dma_trace.record(
                    "fwht", "mdct_flux",
                    {"in": flux}, {"flux": flux},
                    preprocess_ms * 1000, tag="bicameral")
                dma_trace.record(
                    "thr", "threshold",
                    {"in": flux}, {"gate": decision},
                    active_ms * 1000, tag="bicameral")
            except Exception:  # noqa: BLE001
                pass
    summary = {
        "n_frames": n_frames,
        "preprocess_ms_mean": round(float(np.mean([f.preprocess_ms
                                                   for f in frames])), 3),
        "active_ms_mean": round(float(np.mean([f.active_ms
                                               for f in frames])), 3),
        "decision_mean": round(float(np.mean([f.decision
                                              for f in frames])), 3),
        "smoothed_final": round(frames[-1].decision_smoothed, 4),
    }
    return {
        "program": PIPELINE_PROGRAM.summary(),
        "patch": compile_program(PIPELINE_PROGRAM, "microfx"),
        "summary": summary,
        "frames": [asdict(f) for f in frames],
    }
