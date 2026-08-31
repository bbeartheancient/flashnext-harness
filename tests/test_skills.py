"""Tests for skills.py — SKILL.md IO, experience mirror, trajectory distillation."""

from __future__ import annotations

import pytest

from neoharness.bpe import BPEState
from neoharness.harness import EpisodeResult, StepRecord
from neoharness.skills import SkillStore


@pytest.fixture()
def store(tmp_path):
    return SkillStore(BPEState(), directory=tmp_path)


def test_save_load_roundtrip(store):
    store.save("clean-and-place", "how to clean and place",
               "1. find it\n2. clean it\n3. place it")
    skill = store.load("clean-and-place")
    assert skill["name"] == "clean-and-place"
    assert "clean it" in skill["body"]
    assert store.list()[0]["description"] == "how to clean and place"


def test_bad_name_rejected(store):
    with pytest.raises(ValueError):
        store.save("Bad Name!", "x", "y")


def test_mirrored_into_experience(store):
    store.save("mug-search", "where mugs hide", "countertops first")
    hits = store.state.experience.recall("countertops mugs")
    assert hits and hits[0]["skill_id"] == "skill:mug-search"
    assert hits[0]["kind"] == "task"


def test_load_all_into_experience(store, tmp_path):
    (tmp_path / "a.md").write_text(
        "---\nname: alpha\ndescription: first skill\n---\nbody a")
    n = store.load_all_into_experience()
    assert n == 1
    assert "skill:alpha" in store.state.experience.skills


def _ep(solved: bool) -> EpisodeResult:
    steps = [
        StepRecord(0, "<action>recall[mug]</action>", "harness", "recall", "mug", "tip"),
        StepRecord(1, "<action>take mug</action>", "env", "take mug", "", "ok"),
        StepRecord(2, "<action>clean mug</action>", "env", "clean mug", "", "ok"),
        StepRecord(3, "<action>note[prior | sinks clean fast]</action>", "harness",
                   "note", "prior | sinks clean fast", "noted", "z",
                   {"note": {"text": "sinks clean fast"}}),
    ]
    return EpisodeResult("clean the mug", steps, solved, 2, 2, 0)


def test_create_from_solved_trajectory(store):
    path = store.create_from_trajectory(_ep(True))
    assert path is not None
    skill = store.load("clean-the-mug")
    assert "1. take mug" in skill["body"]
    assert "2. clean mug" in skill["body"]
    assert "sinks clean fast" in skill["body"]


def test_create_from_failed_trajectory_is_noop(store):
    assert store.create_from_trajectory(_ep(False)) is None
