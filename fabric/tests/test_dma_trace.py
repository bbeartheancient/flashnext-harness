# SPDX-License-Identifier: Apache-2.0
"""Tests for fabric/dma_trace.py (BICAMERAL Goals 1, 9)."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from fabric import dma_trace
import importlib


def _fresh():
    importlib.reload(dma_trace)
    return dma_trace


def test_record_returns_entry():
    d = _fresh()
    d._trace._active = True
    e = d.record("m1", "gain", {"in": 0.5}, {"cv": 0.25}, 12.0)
    assert e is not None
    assert e.module_id == "m1"
    assert e.out_ports == {"cv": 0.25}
    assert e.latency_us == 12.0


def test_inactive_returns_none():
    d = _fresh()
    d._trace._active = False
    e = d.record("m1", "gain", {}, {}, 0)
    assert e is None


def test_snapshot_structure():
    d = _fresh()
    d._trace._active = True
    for i in range(3):
        d.record(f"m{i}", "gain", {"in": float(i)}, {"cv": float(i * 2)}, 1.0)
    snap = d.snapshot()
    assert snap["n_frames"] == 3
    assert snap["active"] is True
    assert snap["seq"] == 3


def test_clear():
    d = _fresh()
    d._trace._active = True
    d.record("m1", "gain", {}, {"cv": 1.0}, 1.0)
    d.record("m1", "gain", {}, {"cv": 2.0}, 1.0)
    d.clear()
    snap = d.snapshot()
    assert snap["n_frames"] == 0
    assert snap["seq"] == 0


def test_replay():
    importlib.reload(dma_trace)
    d = dma_trace
    frames = [
        {"seq": 1, "t": 1.0, "module_id": "s", "in_ports": {"cv": 0.5},
         "out_ports": {},},
        {"seq": 2, "t": 1.1, "module_id": "a", "in_ports": {"cv": 0.5},
         "out_ports": {"acc": 1.0},},
    ]
    evs = dma_trace.replay(frames)
    # 1 in event (frame 1) + 1 in + 1 out (frame 2) = 3 events
    assert len(evs) == 3, evs
    assert evs[0]["port"] == "cv" and evs[0]["value"] == 0.5
    assert evs[0]["direction"] == "in"
    # Last event is the out_port of frame 2
    assert evs[-1]["port"] == "acc" and evs[-1]["value"] == 1.0
    assert evs[-1]["direction"] == "out"


if __name__ == "__main__":
    for name in dir(sys.modules[__name__]):
        if name.startswith("test_"):
            fn = getattr(sys.modules[__name__], name)
            fn()
            print(f"  {name} ok")
    print(f"dma_trace: {sum(1 for n in dir(sys.modules[__name__]) if n.startswith('test_'))}/{sum(1 for n in dir(sys.modules[__name__]) if n.startswith('test_'))} pass")
