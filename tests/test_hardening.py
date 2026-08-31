"""Tests for the hardening changes: arg caps + atlas rate default."""

from __future__ import annotations

from neoharness import vectors
from neoharness.bpe import BPEState
from neoharness.vectors import XYZBuffer


def test_commit_caps_long_subgoal():
    s = BPEState()
    entry = s.progress.commit("x" * 500, "done")
    assert len(entry["subgoal"]) == 160


def test_note_caps_long_insight():
    s = BPEState()
    n = s.experience.note("y" * 900)
    assert len(n["text"]) == 500


def test_atlas_default_rate_resolution():
    # tap layer 63 is hot in the cache map -> rate 0.75
    assert vectors._atlas_default_rate() == 0.75


def test_xyzbuffer_defaults_to_atlas_rate_when_enabled():
    buf = XYZBuffer(enabled=True)  # no explicit rate, no worker spawn yet
    assert buf.rate == 0.75


def test_xyzbuffer_explicit_rate_wins():
    buf = XYZBuffer(enabled=True, rate=0.5)
    assert buf.rate == 0.5


def test_xyzbuffer_disabled_stays_rateless():
    buf = XYZBuffer(enabled=False)
    assert buf.rate is None
