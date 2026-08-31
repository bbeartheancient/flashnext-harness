"""Tests for vectors.py — dct_topk gate, worker protocol, XYZ mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from neoharness import vectors
from neoharness.vectors import BufferWorker, XYZBuffer, dct_topk_roundtrip, passes_gate

FAKE_WORKER = str(Path(__file__).parent / "fake_buffer_worker.py")


def test_dct_topk_smooth_signal_passes_gate():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, 256)
    arr = np.stack([np.sin(t + p) for p in rng.uniform(0, 1, 16)])
    _, row_cos = dct_topk_roundtrip(arr, rate=0.5)
    assert row_cos >= 0.95


def test_dct_topk_white_noise_fails_gate():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((16, 256))
    ok, row_cos = passes_gate(arr, rate=0.25, gate=0.95)
    assert not ok
    assert row_cos < 0.95


def test_dct_topk_requires_2d():
    with pytest.raises(ValueError):
        dct_topk_roundtrip(np.zeros(5), 0.5)


def test_buffer_worker_protocol_roundtrip():
    w = BufferWorker(cmd=[sys.executable, FAKE_WORKER], spawn_timeout=30,
                     request_timeout=10)
    arr = np.zeros((4, 5120), dtype=np.float16)
    resp = w.store("bpe-x", "mug at shelf", arr, mode="merge", persist=False)
    assert resp["ok"] and resp["session"] == "bpe-x"
    assert resp["span_tokens"] == 4
    act = w.activate("bpe-x")
    assert act["table_rows"] == 4
    stats = w.stats()
    assert stats["sessions"]["bpe-x"]["rows"] == 4
    w._proc.kill()


def test_buffer_worker_failsoft_after_max_fails():
    w = BufferWorker(cmd=[sys.executable, "-c", "import sys; sys.exit(1)"],
                     spawn_timeout=5, request_timeout=2)
    assert w.stats() is None
    assert w.stats() is None
    assert w.dead
    assert w.stats() is None  # no further spawn attempts


def test_xyz_buffer_store_axis(monkeypatch):
    captured = {}

    def fake_capture(text, base_url=None):
        captured["text"] = text
        t = np.linspace(0, 4 * np.pi, 5120)
        return np.sin(t)[None, :].repeat(8, axis=0)  # smooth -> passes gate

    monkeypatch.setattr(vectors, "capture_hidden", fake_capture)
    buf = XYZBuffer(
        worker=BufferWorker(cmd=[sys.executable, FAKE_WORKER],
                            spawn_timeout=30, request_timeout=10),
        enabled=True,
    )
    resp = buf.store_axis("y", "[done] find mug", persist=False)
    assert resp is not None
    assert resp["axis"] == "y" and resp["session"] == "bpe-y"
    assert resp["row_cos"] >= 0.95
    assert captured["text"] == "[done] find mug"
    buf.worker._proc.kill()


def test_xyz_buffer_gate_rejection(monkeypatch):
    monkeypatch.setattr(
        vectors, "capture_hidden",
        lambda text, base_url=None: np.random.default_rng(2).standard_normal((8, 5120)),
    )
    monkeypatch.setattr(vectors, "ROW_COS_GATE", 0.99)
    buf = XYZBuffer(
        worker=BufferWorker(cmd=[sys.executable, FAKE_WORKER],
                            spawn_timeout=30, request_timeout=10),
        enabled=True, rate=0.25,  # hard-compact: noise must fail the gate
    )
    assert buf.store_axis("x", "noisy state") is None
    assert buf.gate_rejections and buf.gate_rejections[0]["axis"] == "x"
    assert buf.worker._proc is None  # rejected before the worker was needed


def test_xyz_buffer_disabled_by_default():
    buf = XYZBuffer(enabled=False)
    assert buf.store_axis("z", "anything") is None


def test_xyz_buffer_bad_axis():
    buf = XYZBuffer(enabled=False)
    with pytest.raises(ValueError):
        buf.store_axis("w", "nope")
