# SPDX-License-Identifier: Apache-2.0
"""Tests for fabric/self_improve.py and fabric/round_trip.py (Goals 3, 7)."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from fabric import self_improve, round_trip, atomic_program
import importlib


def _fresh_si():
    importlib.reload(self_improve)
    return self_improve


def _fresh_rt():
    importlib.reload(round_trip)
    return round_trip


# ---- self_improve ----

def test_score_output_degrades_on_short_text():
    si = _fresh_si()
    # LLM call will likely fail in test (no network), check error dict
    result = si.score_output("")
    assert "score" in result
    assert isinstance(result["score"], float)


def test_recommend_policy_change():
    si = _fresh_si()
    # Empty trajectory
    policy = si._recommend_policy_change([])
    assert "No trajectory" in policy
    # Single iteration
    policy2 = si._recommend_policy_change([
        {"compaction_90": 0.5, "mean_energy": 1.0, "gen_mode": "tokens"},
    ])
    assert "Insufficient" in policy2
    # Two iterations
    policy3 = si._recommend_policy_change([
        {"compaction_90": 0.4, "mean_energy": 1.0, "gen_mode": "tokens"},
        {"compaction_90": 0.6, "mean_energy": 1.2, "gen_mode": "embeds"},
    ])
    assert len(policy3) > 0


def test_run_self_improve_empty_prompt():
    si = _fresh_si()
    cfg = si.SelfImproveConfig(prompt="")
    result = si.run_self_improve(cfg)
    assert "error" in result


# ---- round_trip ----

def test_round_trip_decompose_invalid():
    rt = _fresh_rt()
    res = rt.decompose_app("no-such-app-id")
    assert "error" in res


def test_round_trip_compose_invalid_program():
    rt = _fresh_rt()
    ap = importlib.reload(atomic_program)
    p = ap.AtomicProgram(
        name="bad",
        blocks=[ap.AtomicBlock(id="m", primitive="const")],
        wires=[ap.Wire(src="missing.cv", dst="m.in")],
    )
    res = rt.compose_app(p)
    assert "error" in res


def test_modify_block():
    ap = importlib.reload(atomic_program)
    rt = _fresh_rt()
    p = ap.AtomicProgram(
        name="t",
        blocks=[ap.AtomicBlock(id="c1", primitive="clock_bpm",
                               params={"bpm": 60})],
        wires=[],
    )
    p2 = rt.modify_block(p, "c1", bpm=120)
    assert p2.blocks[0].params.get("bpm") == 120
    assert p.blocks[0].params.get("bpm") == 60  # original unchanged


def test_modify_wire_add():
    ap = importlib.reload(atomic_program)
    rt = _fresh_rt()
    p = ap.AtomicProgram(
        name="t",
        blocks=[ap.AtomicBlock(id="a", primitive="const"),
                ap.AtomicBlock(id="b", primitive="smooth")],
        wires=[],
    )
    p2 = rt.modify_wire(p, "a.cv", "b.in", mode="add")
    assert len(p2.wires) == 1
    assert p2.wires[0].src == "a.cv"
    # adding same wire again is idempotent
    p3 = rt.modify_wire(p2, "a.cv", "b.in", mode="add")
    assert len(p3.wires) == 1


def test_modify_wire_remove():
    ap = importlib.reload(atomic_program)
    rt = _fresh_rt()
    p = ap.AtomicProgram(
        name="t",
        blocks=[ap.AtomicBlock(id="a", primitive="const"),
                ap.AtomicBlock(id="b", primitive="smooth")],
        wires=[ap.Wire(src="a.cv", dst="b.in")],
    )
    p2 = rt.modify_wire(p, "a.cv", "b.in", mode="remove")
    assert len(p2.wires) == 0


if __name__ == "__main__":
    n = 0
    for name in dir(sys.modules[__name__]):
        if name.startswith("test_"):
            fn = getattr(sys.modules[__name__], name)
            fn()
            print(f"  {name} ok")
            n += 1
    print(f"self_improve+round_trip: {n}/{n} pass")
