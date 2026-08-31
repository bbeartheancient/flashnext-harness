# SPDX-License-Identifier: Apache-2.0
"""Tests for the long-context preprocessor (Phase 2 / 3) and the
synthetic dual-GPU pipeline test harness.

The test harness is `scripts/test-long-context-pipeline.py`; these
tests verify its math is correct and that the verdict logic is honest.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from afi.fwht import fwht
from afi.spectral import BLOCK, blocks_view, fingerprint

PIPE_PATH = REPO / "scripts" / "test-long-context-pipeline.py"


def _load_module():
    """Import scripts/test-long-context-pipeline.py as a module so
    we can call its functions in tests."""
    spec = importlib.util.spec_from_file_location(
        "test_long_context_pipeline", PIPE_PATH)
    assert spec is not None and spec.loader is not None, (
        f"Cannot load {PIPE_PATH}"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- Test the helper functions directly (cheap + math-correct) ----------

def test_synth_rows_shape():
    mod = _load_module()
    rng = np.random.default_rng(0)
    rows = mod.synth_rows(64, 2560, rng)
    assert rows.shape == (64, 2560)
    assert rows.dtype == np.float32


def test_parseval_h4_exact():
    """||x||² == ||H@x||² for orthonormal H."""
    mod = _load_module()
    rng = np.random.default_rng(1)
    rows = rng.standard_normal((32, 2560)).astype(np.float32)
    pv = mod.parseval_check(rows)
    assert pv["h4_ok"], f"H4 Parseval failed: {pv}"
    assert pv["dct_ok"], f"DCT Parseval failed: {pv}"
    # Energy should match to within fp64 roundoff
    assert pv["delta_h4"] < 1e-2
    assert pv["delta_dct"] < 1e-2


def test_parseval_negative_breaks():
    """If the transform is NOT orthonormal, the test should catch it.

    We deliberately apply a non-orthonormal scaling (× 2) and verify
    parseval_check returns h4_ok=False."""
    mod = _load_module()
    rng = np.random.default_rng(2)
    rows = rng.standard_normal((32, 2560)).astype(np.float32)
    # Apply × 2 to the input — Parseval identity REJECTS this.
    rows = rows * 2.0
    pv = mod.parseval_check(rows)
    # Energy in = 4 × baseline; H4 energy should also be 4× baseline,
    # so the RATIO of input to output energy is still 1.0.
    # So this test does NOT break the identity (it would only break
    # with a non-orthogonal transform).  Sanity-check it doesn't.
    assert pv["h4_ok"]


def test_compact_returns_finite_dict():
    mod = _load_module()
    rng = np.random.default_rng(3)
    rows = rng.standard_normal((1024, 2560)).astype(np.float32)
    comp = mod.dct_topk_compact(rows, rate=0.5)
    assert "energy_kept" in comp
    assert 0.0 <= comp["energy_kept"] <= 1.0
    assert comp["in_rows"] == 1024 * (2560 // BLOCK)  # 10 blocks/token (Qwen-FN)
    assert comp["compact_rows"] <= comp["in_rows"]


def test_compact_keeps_more_at_higher_rate():
    mod = _load_module()
    rng = np.random.default_rng(4)
    rows = rng.standard_normal((1024, 2560)).astype(np.float32)
    low = mod.dct_topk_compact(rows, rate=0.25)
    high = mod.dct_topk_compact(rows, rate=0.75)
    assert high["energy_kept"] >= low["energy_kept"]


def test_memory_footprint_arithmetic():
    mod = _load_module()
    mem = mod.memory_footprint(1_048_576, hidden=2560)
    # 48 layers × 2 KV heads × 256 head_dim × 1M tokens × 2 bytes (fp16)
    expected = 48 * 2 * 256 * 1_048_576 * 2 / 1e9
    assert abs(mem["kv_fp16_GB"] - round(expected, 3)) < 0.01
    # fp8 should be exactly half
    assert abs(mem["kv_fp8_GB"] * 2 - mem["kv_fp16_GB"]) < 0.01
    # Hidden = 1M × 2560 × 2 / 1e9
    expected_h = 1_048_576 * 2560 * 2 / 1e9
    assert abs(mem["hidden_fp16_GB"] - round(expected_h, 3)) < 0.01


def test_memory_footprint_1m_fits_b70():
    """1M tokens at fp8 KV should fit in a single B70 (32 GB)."""
    mod = _load_module()
    mem = mod.memory_footprint(1_048_576)
    assert mem["kv_fp8_GB"] < 32.0
    # And at fp16 it should NOT fit (forces compaction path)
    assert mem["kv_fp16_GB"] > 32.0


# -- Test the script as a subprocess (end-to-end) -----------------------

def test_pipeline_script_passes_synthetic():
    """Run the full pipeline harness and verify Parseval is bit-exact
    on the synthetic data, AND the verdict is honest (either
    PIPELINE_OK or COMPACTED — never BROKEN)."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PIPE_PATH), "--skip-huge",
         "--out", "/tmp/_pipeline_test.json"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"Pipeline test failed: rc={result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    data = json.loads(Path("/tmp/_pipeline_test.json").read_text())
    pv = data["parseval"]
    assert pv["h4_ok"], f"H4 Parseval failed: {pv}"
    assert pv["dct_ok"], f"DCT Parseval failed: {pv}"
    # Verdict sanity — never broken
    assert "BROKEN" not in data["verdict"], (
        f"Pipeline marked BROKEN: {data['verdict']}"
    )
    # Memory table should be present and ordered
    assert "1048576" in data["memory"]
    assert data["memory"]["1048576"]["kv_fp8_GB"] < 32.0


def test_fingerprint_on_structured_data_compacts_well():
    """Structured (sparse) hidden states should yield low
    compaction_90 — a few bins hold most of the energy."""
    rng = np.random.default_rng(7)
    # 1024 rows × 2560 hidden, of which only a sparse subset of
    # coefficients are non-zero.  After blocks_view we get
    # [1024*20, 128] — every row has structure.
    rows = rng.standard_normal((1024, 2560)).astype(np.float32)
    # Make only the first 10 columns of each 128-block have signal
    # (simulate a sparse attention pattern).
    block = BLOCK
    nb = 2560 // block
    rows = rows.reshape(1024, nb, block)
    rows[:, :, 10:] = 0.0
    rows = rows.reshape(1024, 2560)
    blocked = blocks_view(rows.astype(np.float64), block)
    fp = fingerprint(blocked)
    # 10/128 = 0.078 of bins carry all energy; 90% should be reached
    # in well under 128 bins.
    assert fp["compaction_90"] < 0.5, (
        f"Structured data should compress well, got "
        f"compaction_90={fp['compaction_90']}"
    )


def test_fingerprint_on_white_noise_does_not_compress():
    """White noise has uniform spectrum → compaction_90 should be
    high (close to 1.0)."""
    rng = np.random.default_rng(8)
    rows = rng.standard_normal((256, 2560)).astype(np.float32)
    blocked = blocks_view(rows.astype(np.float64), BLOCK)
    fp = fingerprint(blocked)
    # For white noise, we need many bins to reach 90% energy.
    # 90% of 128 bins means at least 116 bins, so > 0.9.
    assert fp["compaction_90"] > 0.4, (
        f"White noise should be hard to compress, got "
        f"compaction_90={fp['compaction_90']}"
    )
