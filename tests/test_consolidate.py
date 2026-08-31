"""Tests for consolidate.py — add/update/remove, usage counts, note merge."""

from __future__ import annotations

from neoharness.bpe import ExperienceStore
from neoharness.consolidate import consolidate


def test_novel_notes_become_skills():
    e = ExperienceStore()
    e.note("knives are usually in the kitchen drawers", kind="prior")
    e.note("check the sink before the dishwasher", kind="general")
    r = consolidate(e)
    assert r.notes_consumed == 2
    assert len(r.added) == 2 and not r.updated
    assert e.note_buffer == []
    assert len(e.skills) == 2


def test_redundant_note_updates_existing_skill():
    e = ExperienceStore()
    existing = e.add("prior", "knives location", "knives are in kitchen drawers")
    e.note("knives are in the kitchen drawers near the sink", kind="prior")
    r = consolidate(e)
    assert r.updated == [existing["skill_id"]] and not r.added
    assert existing["usage_count"] == 1
    assert "sink" in existing["text"]
    assert len(e.skills) == 1


def test_trajectory_summary_added_as_task_skill():
    e = ExperienceStore()
    r = consolidate(e, trajectory_summary="clean the mug then place it in cabinet")
    assert len(r.added) == 1
    entry = next(iter(e.skills.values()))
    assert entry["kind"] == "task"


def test_prune_removes_low_usage_preserves_hot():
    e = ExperienceStore()
    hot = e.add("general", "hot skill", "used every episode")
    cold = e.add("general", "cold skill", "never useful")
    for _ in range(3):
        e.recall("hot")
    r = consolidate(e, prune_below_usage=1)
    assert r.removed == [cold["skill_id"]]
    assert hot["skill_id"] in e.skills


def test_consolidation_preserves_kinds():
    e = ExperienceStore()
    e.note("always verify the object is clean first", kind="mistake")
    consolidate(e)
    assert next(iter(e.skills.values()))["kind"] == "mistake"
