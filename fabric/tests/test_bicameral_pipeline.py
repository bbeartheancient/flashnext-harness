# SPDX-License-Identifier: Apache-2.0
"""Tests for fabric/bicameral_pipeline.py (BICAMERAL Goal 9)."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
from fabric import bicameral_pipeline
import importlib


def _fresh():
    importlib.reload(bicameral_pipeline)
    return bicameral_pipeline


def test_run_pipeline_returns_summary():
    bp = _fresh()
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((64, 2560)).astype(np.float16)
    result = bp.run_pipeline(rows, n_frames=5, record_to_trace=False)
    assert "summary" in result
    assert "frames" in result
    assert "program" in result
    assert result["summary"]["n_frames"] == 5


def test_pipeline_frame_structure():
    bp = _fresh()
    rows = np.zeros((8, 256), dtype=np.float32)
    result = bp.run_pipeline(rows, n_frames=3, record_to_trace=False)
    frames = result["frames"]
    assert len(frames) == 3
    assert all("flux" in f for f in frames)
    assert all("preprocess_ms" in f for f in frames)
    assert all("active_ms" in f for f in frames)


def test_pipeline_program_registered():
    bp = _fresh()
    assert bp.PIPELINE_PROGRAM.name == "bicameral_pipeline"
    assert len(bp.PIPELINE_PROGRAM.blocks) == 5
    assert len(bp.PIPELINE_PROGRAM.wires) == 4


def test_pipeline_compiles_to_patch():
    bp = _fresh()
    patch = bp.compile_program(bp.PIPELINE_PROGRAM, "microfx")
    assert "modules" in patch
    assert "wires" in patch
    assert len(patch["modules"]) == 5


def test_pipeline_smoothed_converges():
    bp = _fresh()
    rows = np.full((16, 256), 0.9, dtype=np.float32)  # all-high signal
    # alpha=0.2: after 20 frames y=1-(0.8)^20≈0.99 — tight bound works
    result = bp.run_pipeline(rows, n_frames=20, alpha=0.2, lo=0.3, hi=0.7,
                           record_to_trace=False)
    last = result["frames"][-1]
    assert last["decision"] == 1.0
    assert 0.85 <= last["decision_smoothed"] <= 1.0


if __name__ == "__main__":
    n = 0
    for name in dir(sys.modules[__name__]):
        if name.startswith("test_"):
            fn = getattr(sys.modules[__name__], name)
            fn()
            print(f"  {name} ok")
            n += 1
    print(f"bicameral_pipeline: {n}/{n} pass")
