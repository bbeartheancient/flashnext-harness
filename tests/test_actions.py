"""Tests for actions.py — protocol parse and dispatch."""

from __future__ import annotations

from neoharness.actions import dispatch, parse_action
from neoharness.bpe import BPEState


def test_parse_last_tag_wins():
    text = '<action>track[world]</action> thinking... <action>commit[a | done]</action>'
    assert parse_action(text) == ("commit", "a | done")


def test_parse_missing_tag():
    assert parse_action("no action here") is None


def test_parse_env_action_passthrough():
    assert parse_action("<action>go to countertop 1</action>") == (
        "go to countertop 1",
        "",
    )


def test_dispatch_track_world_and_object():
    s = BPEState()
    s.belief.update("mug", location="shelf")
    r = dispatch("<action>track[mug]</action>", s)
    assert (r.kind, r.verb, r.axis) == ("harness", "track", "x")
    assert "mug" in r.observation
    r2 = dispatch("<action>track[world]</action>", s)
    assert "mug" in r2.observation


def test_dispatch_commit_writes_and_fires_on_write():
    s = BPEState()
    writes = []
    r = dispatch(
        "<action>commit[find mug | done]</action>",
        s,
        on_write=lambda axis, verb, text, detail: writes.append((axis, verb, text)),
    )
    assert r.kind == "harness" and r.axis == "y"
    assert s.progress.entries[0]["status"] == "done"
    assert writes and writes[0][0] == "y" and "find mug" in writes[0][2]


def test_dispatch_commit_default_status_and_bad_status():
    s = BPEState()
    r = dispatch("<action>commit[open cabinet]</action>", s)
    assert r.detail["entry"]["status"] == "done"
    r2 = dispatch("<action>commit[x | sideways]</action>", s)
    assert r2.kind == "invalid"


def test_dispatch_recall_and_note():
    s = BPEState()
    s.experience.add("prior", "knives", "knives are in drawers")
    r = dispatch("<action>recall[knives]</action>", s)
    assert "drawers" in r.observation
    assert r.detail["hits"]
    r2 = dispatch("<action>note[prior | forks are in drawers]</action>", s)
    assert r2.kind == "harness" and r2.axis == "z"
    assert s.experience.note_buffer[0]["kind"] == "prior"


def test_dispatch_env_and_invalid():
    s = BPEState()
    r = dispatch("<action>take mug 1 from countertop 2</action>", s)
    assert r.kind == "env"
    r2 = dispatch("I will now do a thing without tags", s)
    assert r2.kind == "invalid"


def test_on_write_exception_is_recorded_not_raised():
    s = BPEState()

    def boom(axis, verb, text, detail):
        raise RuntimeError("buffer down")

    r = dispatch("<action>commit[a | done]</action>", s, on_write=boom)
    assert r.kind == "harness"
    assert "buffer down" in r.detail["vector_error"]
