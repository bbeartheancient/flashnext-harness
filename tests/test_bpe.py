"""Tests for bpe.py — store semantics, caps, eviction, persistence."""

from __future__ import annotations

import pytest

from neoharness.bpe import (
    BPEState,
    BeliefStore,
    ExperienceStore,
    ProgressStore,
)


def test_belief_update_and_track():
    b = BeliefStore()
    b.update("Mug", state="clean", location="countertop")
    b.update("mug", location="cabinet")  # case-insensitive merge
    assert "mug" in b.facts and "Mug" not in b.facts
    assert b.facts["mug"]["state"] == "clean"
    assert b.facts["mug"]["location"] == "cabinet"
    assert "mug" in b.track("mug")
    assert "unknown" in b.track("plate")
    assert "mug" in b.track("world")


def test_belief_lru_eviction():
    b = BeliefStore(max_facts=2)
    b.update("a", state="1")
    b.update("b", state="2")
    b.update("c", state="3")
    assert set(b.facts) == {"b", "c"}


def test_belief_empty_object_rejected():
    with pytest.raises(ValueError):
        BeliefStore().update("  ", state="x")


def test_progress_commit_upsert_and_status_validation():
    p = ProgressStore()
    p.commit("find mug", "pending")
    p.commit("find mug", "done")
    assert len(p.entries) == 1
    assert p.entries[0]["status"] == "done"
    with pytest.raises(ValueError):
        p.commit("x", "weird")


def test_progress_eviction_prefers_completed():
    p = ProgressStore(max_subgoals=2)
    p.commit("old done", "done")
    p.commit("open work", "pending")
    p.commit("new work", "pending")
    subs = [e["subgoal"] for e in p.entries]
    assert "old done" not in subs
    assert "open work" in subs


def test_experience_recall_bumps_usage():
    e = ExperienceStore()
    e.add("general", "search order", "check countertops before cabinets")
    e.add("mistake", "hot mug", "never place a heated mug on wood")
    hits = e.recall("countertops cabinets search")
    assert hits and hits[0]["title"] == "search order"
    assert hits[0]["usage_count"] == 1
    assert e.recall("unrelated xyzzy") == []


def test_experience_note_buffer():
    e = ExperienceStore()
    n = e.note("knives are usually in drawers", kind="prior")
    assert e.note_buffer == [n]


def test_experience_eviction_preserves_frequently_used():
    e = ExperienceStore(max_skills=2)
    hot = e.add("general", "hot", "used a lot")
    e.add("general", "cold1", "rarely")
    for _ in range(3):
        e.recall("hot used")
    e.add("general", "cold2", "also rarely")
    assert hot["skill_id"] in e.skills
    assert len(e.skills) == 2


def test_invalid_kind_rejected():
    e = ExperienceStore()
    with pytest.raises(ValueError):
        e.add("bogus", "t", "x")
    with pytest.raises(ValueError):
        e.note("x", kind="bogus")


def test_bpe_state_roundtrip(tmp_path):
    s = BPEState()
    s.belief.update("mug", location="shelf")
    s.progress.commit("find mug", "done")
    s.experience.add("task", "clean mug", "use the sink")
    p = s.save(tmp_path / "state.json")
    s2 = BPEState.load(p)
    assert s2.belief.track("mug").startswith("mug")
    assert s2.progress.entries[0]["subgoal"] == "find mug"
    assert s2.experience.recall("sink")
    assert "### Belief" in s2.render()


def test_bpe_state_load_missing(tmp_path):
    s = BPEState.load(tmp_path / "nope.json")
    assert s.render()
