# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ple-buffer-worker SessionBuffer frequency eviction.

Imports the worker module directly (module top level has no
torch/transformers deps — those load inside main()/BufferState)."""

import importlib.util
from pathlib import Path

import pytest

_WORKER = Path(__file__).resolve().parents[2] / "scripts" / "ple-buffer-worker.py"
_spec = importlib.util.spec_from_file_location("ple_buffer_worker", _WORKER)
pbw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pbw)


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setattr(pbw, "MAX_ROWS", 10)
    monkeypatch.setattr(pbw, "FREQ_MIN", 3)
    monkeypatch.setattr(pbw, "DECAY", 0)
    return pbw


def _store(buf, ids):
    return buf.upsert(ids, [f"row-{t}" for t in ids], "merge")


def test_pure_fifo_when_threshold_unreachable(cfg, monkeypatch):
    # Threshold above any plausible span count: nothing ever pins.
    monkeypatch.setattr(pbw, "FREQ_MIN", 1000)
    buf = cfg.SessionBuffer(prior={999: "prior"})
    for i in range(5):
        _store(buf, [i * 4 + j for j in range(4)])  # 4 fresh ids per span
    # 20 unique ids > cap 10: oldest spans evicted, newest kept
    assert len(buf.overlay) <= 10
    assert 18 in buf.overlay and 19 in buf.overlay
    assert 0 not in buf.overlay
    assert buf.rows()[999] == "prior"  # prior survives untouched


def test_frequent_ids_pin_through_eviction(cfg):
    buf = cfg.SessionBuffer(prior={})
    # id 7 appears in 3 spans -> pinned at FREQ_MIN=3
    _store(buf, [7, 1])
    _store(buf, [7, 2])
    _store(buf, [7, 3])
    assert buf.pinned() == 1
    # Flood with fresh ids to force eviction of the early spans
    for i in range(6):
        _store(buf, [100 + i, 200 + i])
    assert len(buf.overlay) <= 10
    assert buf.overlay[7] == "row-7"   # pinned id persists
    assert 1 not in buf.overlay        # sub-threshold ids faded FIFO
    assert buf.freq[7] == 3


def test_decay_unpins_faded_ids(cfg, monkeypatch):
    monkeypatch.setattr(pbw, "DECAY", 1)
    buf = cfg.SessionBuffer(prior={})
    _store(buf, [7, 1])
    _store(buf, [7, 2])
    _store(buf, [7, 3])          # freq[7] = 3 -> pinned
    _store(buf, [10, 11])        # 7 untouched: decays to 2
    assert buf.freq[7] == 2
    assert buf.pinned() == 0
    # Re-reference twice: back to threshold, re-pins
    _store(buf, [7, 12])
    _store(buf, [7, 13])
    assert buf.freq[7] == 4
    assert buf.pinned() == 1


def test_replace_resets_frequency(cfg):
    buf = cfg.SessionBuffer(prior={})
    _store(buf, [7, 1])
    _store(buf, [7, 2])
    _store(buf, [7, 3])
    buf.upsert([7, 50], ["new-7", "new-50"], "replace")
    assert buf.freq[7] == 1        # history reset, not pinned anymore
    assert buf.pinned() == 0
    assert buf.overlay == {7: "new-7", 50: "new-50"}


def test_over_cap_only_from_pins_terminates(cfg, monkeypatch):
    # More pinned ids than the cap: eviction must stop when the span
    # log empties instead of looping forever.
    monkeypatch.setattr(pbw, "MAX_ROWS", 2)
    buf = cfg.SessionBuffer(prior={})
    for i in range(4):           # 3 refs each -> all four pin
        for _ in range(3):
            _store(buf, [i])
    assert len(buf.overlay) == 4  # over cap but all pinned: kept
    assert buf.pinned() == 4
